"""Implementation of the cost function for a frenét trajectory."""
# Standard imports
import math
import os
import sys

# Third party imports
from commonroad.scenario.scenario import Scenario
from shapely.geometry import LineString, Point, Polygon
import numpy as np
from commonroad.planning.goal import Interval
import commonroad_dc.pycrcc as pycrcc
from shapely.geometry import MultiPoint

# Custom imports
module_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(module_path)
from planner.utils.timers import ExecTimer
from planner.Frenet.utils.calc_occlusion_costs import (
    calc_distance_to_occlusion,
    calc_occluded_area_vs_velocity,
)
from planner.Frenet.utils.helper_functions import distance, is_in_interval
from planner.Frenet.utils.calc_occlusion_costs import get_visible_area
from risk_assessment.risk_costs import get_bayesian_costs, get_ego_costs, get_equality_costs, get_maximin_costs, get_responsibility_cost

# Global variables
# 风险相关的代价项 key
# 这些 key 会在后面用于判断：当轨迹有效性不足(validity_level < 10)且启用了 multiple_cost_functions 时，
# 是否只保留风险相关代价项的权重，而将其它舒适性/效率类代价权重置零。
RISK_COSTS_KEY = ["bayes", "equality", "maximin", "responsibility", "ego", "risk_cost"]


def calc_trajectory_costs(
    traj,
    global_path,
    planning_problem,
    ego_state,
    validity_level,
    scenario: Scenario,
    params: dict,
    ego_id: int,
    dt: float,
    predictions: dict,
    vehicle_params,
    sensor_radius: float = 30.0,
    exec_timer=None,
    mode=None,
    reach_set=None,
    mode_num=3
):
    """
    Calculate the total cost of a frenét trajectory.

    Args:
        mode:
        global_path:
        params:
        traj (FrenetTrajectory): Considered Trajectory.
        planning_problem (PlanningProblem): Considered planning problem.
        ego_state (State): Current state of the ego vehicle.
        scenario (Scenario): Considered Scenario.
        ego_id (int): ID of the ego vehicle.
        dt (float): Time step size of the scenario.
        predictions (dict): Predictions of the visible objects.
        goal_area (ShapeGroup): Shape group of the goal area.
        vehicle_params (VehicleParameters): Parameters of the considered vehicle.
        occluded_area (float): The currently visible area.
        sensor_radius (float): Sensor radius for the sensor model. Defaults to 30.0.
        exec_timer (dict): Dictionary for the execution times.

    Returns:
        float: Total costs of the trajectory.
        dict: Dictionary with the execution times.
        dict: Dictionary with ego harms for every timestep concerning every obstacle
        dict: Dictionary with obstacle harms for every timestep concerning every obstacle
    """
    # 如果外部没有传入计时器，则新建一个默认不启用 timing 的计时器；
    # 否则直接复用外部传入的 exec_timer。
    timer = ExecTimer(timing_enabled=False) if exec_timer is None else exec_timer

    # 启动“总代价计算”这一大模块的计时
    timer.start_timer("simulation/sort trajectories/calculate costs/total")

    # 从参数字典中读取：
    # 1. 各类代价项对应的权重 weights
    # 2. 模式相关配置 modes
    weights = params['weights']
    modes = params['modes']

    # =========================
    # 1. 计算 cost factor
    # =========================
    # factor 是一个额外乘数，不是某个单独代价项，而是对最终总代价进行缩放。
    # 例如：
    # - 某条轨迹更接近满足目标/时序要求，可能得到奖励（factor 更小）
    # - 某条轨迹偏离约束或不合理，可能得到惩罚（factor 更大）
    with timer.time_with_cm(
        "simulation/sort trajectories/calculate costs/get cost factor/total"
    ):
        # 获取代价缩放因子
        # the final costs are multiplied with this factor
        # it rewards certain trajectories (e. g. reaches goal in time) and penalises others (e. g. trajectory leaves goal area)
        factor = get_cost_factor(
            ego_state=ego_state,
            traj=traj,
            planning_problem=planning_problem,
            dt=dt,
            exec_timer=timer,
        )

    # =========================
    # 2. 计算风险相关代价
    # =========================
    with timer.time_with_cm(
        "simulation/sort trajectories/calculate costs/calculate risk/total"
    ):
        # 只有当 risk_cost 权重大于 0 且存在 predictions 时，才计算风险代价
        # 否则直接将风险字典置空。
        if weights["risk_cost"] > 0.0 and predictions is not None:

            # 计算 Bayesian 风险代价
            # 输入包括：
            # - ego_risk_max: 自车对各障碍物的最大风险
            # - obst_risk_max: 障碍物对自车的最大风险
            # - boundary_harm: 边界碰撞/越界相关伤害
            bayes_cost = get_bayesian_costs(
                ego_risk_max=traj.ego_risk_dict, obst_risk_max=traj.obst_risk_dict, boundary_harm=traj.bd_harm
            )

            # 计算 equality 风险代价
            # 通常表示风险分配的“平等性”或“公平性”度量
            equality_cost = get_equality_costs(
                ego_risk_max=traj.ego_risk_dict, obst_risk_max=traj.obst_risk_dict
            )

            # 计算 maximin 风险代价
            # 一般体现“最差情况”的风险最优化思想
            maximin_cost = get_maximin_costs(
                ego_risk_max=traj.ego_risk_dict, obst_risk_max=traj.obst_risk_dict,
                ego_harm_max=traj.ego_harm_dict, obst_harm_max=traj.obst_harm_dict, boundary_harm=traj.bd_harm
            )

            # 计算 ego 风险代价
            # 主要关注自车自身风险与边界伤害
            ego_cost = get_ego_costs(ego_risk_max=traj.ego_risk_dict, boundary_harm=traj.bd_harm)

            # 计算 responsibility 风险代价
            # 同时返回 bool_contain_cache，后面会再次用于责任修正
            responsibility_cost, bool_contain_cache = get_responsibility_cost(
                scenario=scenario,
                traj=traj,
                ego_state=ego_state,
                obst_risk_max=traj.obst_risk_dict,
                predictions=predictions,
                reach_set=reach_set
            )

            # 将所有风险代价加权求和，形成总风险代价
            # 这里要注意：responsibility 项又额外乘了 weights["bayes"]
            # 说明责任风险与 bayes 风险在当前设计中存在耦合。
            total_risk_cost = (
                weights["bayes"] * bayes_cost
                + weights["equality"] * equality_cost
                + weights["maximin"] * maximin_cost
                + weights["responsibility"] * weights["bayes"] * responsibility_cost
                + weights["ego"] * ego_cost
            )

            # 将各风险代价值存入 traj.risk_dict
            # 后续可用于可视化、debug、dashboard 显示等
            traj.risk_dict = {
                "bayes": bayes_cost,
                "equality": equality_cost,
                "maximin": maximin_cost,
                "responsibility": responsibility_cost,
                "ego": ego_cost,
                "total": total_risk_cost,
            }

            # 将“总风险代价”作为一个总代价项写入 cost_dict
            # 注意：这里 cost_dict 里存的是 risk_cost，
            # 具体分项 bayes/equality/maximin 等保存在 traj.risk_dict 中
            cost_dict = {"risk_cost": traj.risk_dict["total"]}

        else:
            # 若不满足风险代价计算条件，则风险字典和代价字典都置空
            traj.risk_dict = {}
            cost_dict = {}

    # =========================
    # 3. 责任风险修正
    # =========================
    with timer.time_with_cm(
        "simulation/sort trajectories/calculate costs/calculate responsibility/total"
    ):
        # 只有 responsibility 权重大于 0 时，才做进一步责任修正
        if weights["responsibility"] > 0.0:
            # 如果缓存 bool_contain_cache 存在，优先使用缓存结果
            if bool_contain_cache is not None:
                # 取出当前 ego_state.time_step 下所有 reach set 的对象 id 列表
                key_list = list(reach_set.reach_sets[ego_state.time_step].keys())
                for i in range(len(key_list)):
                    # 如果 bool_contain_cache[i] 中不存在 1，
                    # 表示当前对象对应可达域未“包含”自车轨迹点，即自车对该对象不承担责任
                    if 1 not in bool_contain_cache[i]:
                        obj_id = key_list[i]
                        # 从 responsibility 和 total 中减去该障碍物风险
                        traj.risk_dict["responsibility"] -= traj.obst_risk_dict[obj_id]
                        traj.risk_dict["total"] -= traj.obst_risk_dict[obj_id]

            else:
                # 如果没有缓存，则直接遍历 reach set 进行逐对象、逐时刻判断
                responsibility_cost = 0.0
                for obj_id, rs in reach_set.reach_sets[ego_state.time_step].items():
                    # 默认认为自车对该对象“有责任”
                    # time_steps = [float(list(entry.keys())[0]) for entry in rs]
                    responsibility = True
                    for part_set in rs:
                        # part_set 的 key 是时间，value 是对应时刻的可达域多边形点集
                        time_t = list(part_set.keys())[0]
                        # 根据连续时间 time_t 映射回离散轨迹索引
                        time_step = int(time_t / dt - 1)

                        # 自车在该离散时刻的位置
                        ego_pos = Point(traj.x[time_step], traj.y[time_step])
                        # 构造该障碍物在对应时刻的可达域多边形
                        obj_rs = Polygon(list(part_set.values())[0])

                        # 如果自车轨迹点落在障碍物可达域内，则认为该对象与自车存在责任关联
                        if obj_rs.contains(ego_pos):
                            responsibility = False
                            break

                    # 若整段检查后 responsibility 仍为 True，
                    # 则从责任项和总风险项中扣除该障碍物风险
                    if responsibility:
                        traj.risk_dict["responsibility"] -= traj.obst_risk_dict[obj_id]
                        traj.risk_dict["total"] -= traj.obst_risk_dict[obj_id]

    # =========================
    # 4. 计算可见区域 / 遮挡相关代价
    # =========================
    with timer.time_with_cm(
        "simulation/sort trajectories/calculate costs/calculate visible area"
    ):
        # 只有 visible_area 权重大于 0 时才进入该部分
        if weights["visible_area"] > 0.0:

            try:
                # 获取当前时刻的可见区域 visible_area
                # 以及期望可见区域 desired_visible_area
                visible_area, desired_visible_area = get_visible_area(
                    scenario,
                    ego_pos=ego_state.position,
                    time_step=ego_state.time_step,
                    sensor_radius=int(sensor_radius),
                )

                # 收集整条轨迹上的所有离散点，并构造其凸包
                # 用于近似表示该轨迹会经过的整体空间范围
                traj_hull = MultiPoint(
                    [(traj.x[i], traj.y[i]) for i in range(len(traj.x))]
                ).convex_hull

                # 构造多个“危险区”
                # 越靠近轨迹核心区域，通常越重要
                danger_zones = [
                    traj_hull.buffer(sensor_radius * 0.2),
                    traj_hull.buffer(sensor_radius * 0.5),
                ]

                # 计算每个危险区内：
                # 1. 实际可见区域
                # 2. 理想应可见区域
                areas_in_danger_zone = [
                    (
                        visible_area.intersection(dz),
                        desired_visible_area.intersection(dz),
                    )
                    for dz in danger_zones
                ]

                # 计算可见性比例
                # 值越大表示可见性越好；接近 0 表示遮挡严重
                visibility_ratios = [
                    va.area / (dva.area + 0.01) for (va, dva) in areas_in_danger_zone
                ]

                # 再追加一个全局可见性比例
                visibility_ratios.append(
                    visible_area.area / (desired_visible_area.area + 0.01)
                )
            except Exception as e:  # TopologicalError or AttributeError:
                # 如果可见区域计算失败，则打印错误
                # 注意：此处不会中断程序，后续可能继续沿用未定义或旧变量，存在潜在风险
                print(e)

            # 根据不同的 occlusion_mode 计算遮挡代价
            if modes["occlusion_mode"] == "area":
                # 基于可见面积比例来计算遮挡代价
                cost_dict["visible_area"] = calc_occluded_area_vs_velocity(
                    traj=traj,
                    visible_area=visible_area,
                    desired_visible_area=desired_visible_area,
                    visibility_ratios=visibility_ratios,
                    sensor_radius=sensor_radius,
                )
            elif modes["occlusion_mode"] == "distance":
                # 基于轨迹到遮挡区域的距离计算代价
                cost_dict["visible_area"] = calc_distance_to_occlusion(
                    traj=traj,
                    visible_area=visible_area,
                    desired_visible_area=desired_visible_area,
                )
            elif modes["occlusion_mode"] == "mixed":
                # 混合模式：面积项和距离项平均融合
                cost_area = calc_occluded_area_vs_velocity(
                    traj=traj,
                    visible_area=visible_area,
                    desired_visible_area=desired_visible_area,
                    visibility_ratios=visibility_ratios,
                    sensor_radius=sensor_radius,
                )
                cost_dist = calc_distance_to_occlusion(
                    traj=traj,
                    visible_area=visible_area,
                    desired_visible_area=desired_visible_area,
                )
                w1 = 1
                w2 = 1
                cost_dict["visible_area"] = (cost_area * w1 + cost_dist * w2) / (
                    w1 + w2
                )
            else:
                # 若 occlusion_mode 不在已实现模式中，则抛出异常
                raise NotImplementedError(
                    f"The given mode {modes['occlusion_mode']} is not implemented."
                )

    # =========================
    # 5. 计算纵向/横向 jerk 代价
    # =========================
    with timer.time_with_cm(
        "simulation/sort trajectories/calculate costs/calculate jerk"
    ):
        # 只要任一 jerk 权重大于 0，就先统一计算纵横向 jerk
        if weights["lon_jerk"] > 0.0 or weights["lat_jerk"] > 0.0:
            lon_jerk, lat_jerk = get_jerk(traj=traj)
            if weights["lon_jerk"] > 0.0:
                # 纵向 jerk 反映加速度变化剧烈程度，通常与舒适性相关
                cost_dict["lon_jerk"] = lon_jerk
            if weights["lat_jerk"] > 0.0:
                # 横向 jerk 通常反映转向平顺性
                cost_dict["lat_jerk"] = lat_jerk

    # =========================
    # 6. 计算速度代价
    # =========================
    with timer.time_with_cm(
        "simulation/sort trajectories/calculate costs/calculate velocity"
    ):
        if weights["velocity"] > 0.0:
            # 速度代价可能综合考虑期望速度、限速、动态可行性、到达目标等
            cost_dict["velocity"] = velocity_costs(
                traj=traj,
                dt=dt,
                ego_state=ego_state,
                planning_problem=planning_problem,
                scenario=scenario,
            )

    # =========================
    # 7. 计算到全局路径的平均距离代价
    # =========================
    with timer.time_with_cm(
        "simulation/sort trajectories/calculate costs/calculate distance to global path"
    ):
        if weights["dist_to_global_path"] > 0.0:
            # 原始代码中保留了一句注释：
            # if mode_num != 3:
            # 说明这里过去可能有模式分支控制，但当前已被注释掉
            cost_dict["dist_to_global_path"] = calc_avg_dist_to_global_path(traj=traj)

    # =========================
    # 8. 计算轨迹行驶距离代价
    # =========================
    with timer.time_with_cm(
        "simulation/sort trajectories/calculate costs/calculate travelled distance"
    ):
        if weights["travelled_dist"] > 0.0:
            # 轨迹实际行驶长度
            cost_dict["travelled_dist"] = calc_travelled_dist(traj=traj)

    # =========================
    # 9. 计算到目标位置的距离代价
    # =========================
    with timer.time_with_cm(
        "simulation/sort trajectories/calculate costs/calculate distance to goal pos"
    ):
        if weights["dist_to_goal_pos"] > 0.0:
            # 轨迹终点与规划目标区域/位置之间的距离代价
            cost_dict["dist_to_goal_pos"] = calc_dist_to_goal_pos(
                traj=traj,
                lanelet_network=scenario.lanelet_network,
                planning_problem=planning_problem,
            )

    # =========================
    # 10. 计算到车道中心线的距离代价
    # =========================
    with timer.time_with_cm(
        "simulation/sort trajectories/calculate costs/calculate distance to lane center"
    ):
        if weights["dist_to_lane_center"] > 0.0:
            # 轨迹偏离所在车道中心的程度，常用于鼓励沿车道中心平稳行驶
            cost_dict["dist_to_lane_center"] = calc_dist_to_center_line(
                traj=traj, lanelet_network=scenario.lanelet_network
            )

    # =========================
    # 11. 权重处理 + 最终代价加权求和
    # =========================
    with timer.time_with_cm(
        "simulation/sort trajectories/calculate costs/multiply weights and costs"
    ):
        curr_weights = {}

        # 若轨迹有效性低于 10 且启用了 multiple_cost_functions，
        # 则只保留风险相关项的权重，其他项权重设为 0。
        # 这通常意味着：对于低有效性轨迹，仅从风险角度进行评价，而不再考虑舒适性、效率等指标。
        if validity_level < 10 and modes["multiple_cost_functions"]:
            for key in weights:
                if key in RISK_COSTS_KEY:
                    curr_weights[key] = weights[key]
                else:
                    curr_weights[key] = 0
        else:
            # 否则直接使用原始权重
            curr_weights = weights

        # 按照当前权重对已有代价项进行加权求和
        cost = 0.0
        for key in list(cost_dict.keys()):
            cost += curr_weights[key] * cost_dict[key]

    # 将代价相关信息打包输出
    # 包括：
    # - risk_cost_dict: 风险分项
    # - weights: 实际参与本次加权的权重
    # - factor: 最终总代价缩放因子
    # - unweighted_cost: 未乘权重的各分代价
    # - total_cost: factor * cost
    output_dict = {
        'risk_cost_dict': traj.risk_dict,
        'weights': curr_weights,
        'factor': factor,
        'unweighted_cost': cost_dict,
        'total_cost': factor * cost,
    }

    # 停止总代价计算计时
    timer.stop_timer("simulation/sort trajectories/calculate costs/total")

    # 返回：
    # 1. cost：当前已加权但尚未乘 factor 的总代价
    # 2. output_dict：包含详细分项信息的字典
    #
    # 注意这里返回的是 cost，而不是 factor * cost
    # 虽然 output_dict["total_cost"] 中保存的是 factor * cost
    # 下方被注释掉的旧代码说明这里曾考虑直接返回 factor * cost
    # return factor * cost, output_dict
    return cost, output_dict



def get_cost_factor(
    ego_state,
    traj,
    planning_problem,
    dt: float,
    exec_timer=None,
):
    """
    Get a factor with which the costs are multiplied.

    Args:
        ego_state (State): Current state of the ego vehicle.
        traj (FrenetTrajectory): Considered trajectory.
        planning_problem (PlanningProblem): Considered planning problem.
        dt (float): Time step size of the scenario.
        goal_area (ShapeGroup): Shape group representing the goal area.
        exec_times_dict (dict): Dictionary with the execution times. Defaults to None

    Returns:
        float: Multiplication factor for the costs of the trajectory.
        dict: Dictionary with the updated execution times.

    """
    timer = ExecTimer(timing_enabled=False) if exec_timer is None else exec_timer
    # default cost factor is 1.0
    factor = 1.0

    with timer.time_with_cm(
        "simulation/sort trajectories/calculate costs/get cost factor/goal reached?"
    ):
        # check if the trajectory reaches a goal state
        # goal_reached = reached_goal_state(traj, planning_problem, goal_area)
        goal_reached = False

    with timer.time_with_cm(
        "simulation/sort trajectories/calculate costs/get cost factor/goal reached in time?"
    ):
        check_good = True
        # factor is low if the trajectory reaches the goal  TODO: Why?
        if goal_reached is True:
            factor = 0.01
            # factor is even lower, if the trajectory reaches the goal in time
            for i in range(len(traj.x)):
                if reached_target_time(
                    ego_state_time=ego_state.time_step,
                    planning_problem=planning_problem,
                    dt=dt,
                    t=traj.t[i],
                ):
                    check_good = False
                    break
    if check_good is False:
        factor = 0.0001
        return factor

    '''
    with timer.time_with_cm(
        "simulation/sort trajectories/calculate costs/get cost factor/goal reached and left?"
    ):
        # factor is high if the goal was reached but the trajectory left the goal area again
        goal_pos_reached_and_left_again = goal_position_reached_and_left_again(
            pos=ego_state.position, traj=traj, goal_area=goal_area
        )

    if goal_pos_reached_and_left_again is True:
        factor = 10.0
    '''

    return factor


def get_jerk(traj):
    """
    Calculate the longitudinal and lateral jerk cost for the given trajectory.

    Args:
        traj (FrenetTrajectory): Considered trajectory.

    Returns:
        float: Mean square sum of longitudinal jerk.
        float: mean square sum of lateral jerk.

    """
    # longitudinal jerk is equal to the third derivation of the covered arc length
    lon_jerk = sum(np.power(traj.s_ddd, 2)) / len(traj.s)
    # lateral jerk is equal to the third derivation of the lateral offset
    lat_jerk = sum(np.power(traj.d_ddd, 2)) / len(traj.d)

    return lon_jerk, lat_jerk


def velocity_costs(traj, ego_state, planning_problem, scenario, dt: float):
    """
    Calculate the costs for the velocity of the given trajectory.

    Args:
        traj (FrenetTrajectory): Considered trajectory.
        ego_state (State): Current state of the ego vehicle.
        planning_problem (PlanningProblem): Considered planning problem.
        scenario (Scenario): Considered scenario.
        dt (float): Time step size of the scenario.

    Returns:
        float: Costs for the velocity of the trajectory.

    """

    '''
    # if the goal area is reached then just consider the goal velocity
    if reached_target_position(np.array([traj.x[0], traj.y[0]]), goal_area):
        # if the planning problem has a target velocity
        if hasattr(planning_problem.goal.state_list[0], "velocity"):
            return abs(
                (
                    planning_problem.goal.state_list[0].velocity.start
                    + (
                        planning_problem.goal.state_list[0].velocity.end
                        - planning_problem.goal.state_list[0].velocity.start
                    )
                    / 2
                )
                - np.mean(traj.v)
            )
        # otherwise prefer slow trajectories
        else:
            return np.mean(traj.v)

    # if the goal is not reached yet, try to reach it
    # get the center points of the possible goal positions
    goal_centers = []
    # get the goal lanelet ids if they are given directly in the planning problem
    if (
        hasattr(planning_problem.goal, "lanelets_of_goal_position")
        and planning_problem.goal.lanelets_of_goal_position is not None
    ):
        goal_lanelet_ids = planning_problem.goal.lanelets_of_goal_position[0]
        for lanelet_id in goal_lanelet_ids:
            lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
            n_center_vertices = len(lanelet.center_vertices)
            goal_centers.append(lanelet.center_vertices[int(n_center_vertices / 2.0)])
    elif hasattr(planning_problem.goal.state_list[0], "position"):
        # get lanelet id of the ending lanelet (of goal state), this depends on type of goal state
        if hasattr(planning_problem.goal.state_list[0].position, "center"):
            goal_centers.append(planning_problem.goal.state_list[0].position.center)
    # if it is a survival scenario with no goal areas, no velocity can be proposed
    else:
        return 0.0

    # get the distances to the previous found goal positions
    distances = []
    for goal_center in goal_centers:
        distances.append(distance(goal_center, ego_state.position))

    # calculate the average distance to the goal positions
    avg_dist = np.mean(distances)

    # get the remaining time
    _, max_remaining_time_steps = calc_remaining_time_steps(
        planning_problem=planning_problem,
        ego_state_time=ego_state.time_step,
        t=0.0,
        dt=dt,
    )
    remaining_time = max_remaining_time_steps * dt

    # if there is time remaining, calculate the difference between the average desired velocity and the velocity of
    # the trajectory
    if remaining_time > 0.0:
        avg_desired_velocity = avg_dist / remaining_time
        avg_v = np.mean(traj.v)
        cost = abs(avg_desired_velocity - avg_v)
    # if the time limit is already exceeded, prefer fast velocities
    else:
        cost = 30.0 - np.mean(traj.v)
    
    '''

    cost = 10 - np.mean(traj.v)

    return cost


def goal_position_reached_and_left_again(pos: np.array, traj, goal_area):
    """
    Check if a trajectory starts in the goal area but leaves it.

    Args:
        pos (np.array): Current position of the ego vehicle.
        traj (FrenetTrajectory): Considered frenét trajectory.
        goal_area (ShapeGroup): Shape group representing the goal area.

    Returns:
        bool: True if the trajectory starts in the goal area but leaves it.
    """
    # check if the initial position is in the goal area
    reached_target_pos = reached_target_position(pos=pos, goal_area=goal_area)
    # if the target position is never reached return False
    if reached_target_pos is False:
        return False
    # if the target position is reached by the trajectory, check if it leaves the goal area again
    else:
        for i in range(len(traj.x)):
            if (
                reached_target_position(
                    pos=np.array([traj.x[i], traj.y[i]]), goal_area=goal_area
                )
                is False
            ):
                return True

    return False


def reached_goal_state(traj, planning_problem, goal_area):
    """
    Check if the goal state (position, velocity and yaw) is reached.

    Args:
        traj (FrenetTrajectory): Considered frenét trajectory.
        planning_problem (PlanningProblem): Considered planning problem.
        goal_area (ShapeGroup): Shape group representing the goal area.

    Returns:
        bool: True if goal state (position, velocity and yaw) is reached.
    """
    # check if position, velocity and target yaw is reached
    for i in range(len(traj.x)):
        if reached_target_position(np.array([traj.x[i], traj.y[i]]), goal_area):
            if reached_target_velocity(traj.v[i], planning_problem):
                if reached_target_yaw(traj.yaw[i], planning_problem):
                    return True
    return False


def reached_target_position(pos: np.array, goal_area):
    """
    Check if the given position is in the goal area of the planning problem.

    Args:
        pos (np.array): Position to be checked.
        goal_area (ShapeGroup): Shape group representing the goal area.

    Returns:
        bool: True if the given position is in the goal area.
    """
    # if there is no goal area (survival scenario) return True
    if goal_area is None:
        return True

    point = pycrcc.Point(x=pos[0], y=pos[1])

    # check if the point of the position collides with the goal area
    if point.collide(goal_area) is True:
        return True

    return False


def reached_target_velocity(vel: float, planning_problem):
    """
    Check if the target velocity is reached.

    Args:
        vel (float): Current velocity.
        planning_problem (PlanningProblem): Considered planning problem.

    Returns:
        bool: True if the target velocity is reached.
    """
    if calc_final_diff("velocity", vel, planning_problem) == 0.0:
        return True
    else:
        return False


def reached_target_yaw(yaw: float, planning_problem):
    """
    Check if the target yaw angle is reached.

    Args:
        yaw (float): Current yaw angle.
        planning_problem (PlanningProblem): Considered planning problem.

    Returns:
        bool: True if the target yaw angle is reached.
    """
    if calc_final_diff("orientation", yaw, planning_problem) == 0.0:
        return True
    else:
        return False


def reached_target_time(ego_state_time: float, t: float, planning_problem, dt: float):
    """
    Check if the target time is reached.

    Args:
        ego_state_time (float): Current time ego vehicle.
        t (float): Time to be checked.
        planning_problem (PlanningProblem): Considered planning problem.
        dt (float): Time step size of the scenario.

    Returns:
        bool: True if the target time is reached.
    """
    min_remaining_time, max_remaining_time = calc_remaining_time_steps(
        ego_state_time, t, planning_problem, dt
    )
    if min_remaining_time <= 0.0 and max_remaining_time >= 0.0:
        return True
    else:
        return False


def calc_final_diff(prop: str, value: float, planning_problem):
    """
    Calculate the difference of the given value and the goal value for this property.

    Args:
        prop (str): Which property should be considered (velocity or orientation).
        value (float): Checked value.
        planning_problem (PlanningProblem): Considered planning problem.

    Returns:
        float: Difference between the value and the goal value.
    """
    # check if the planning problem has a goal velocity or orientation
    if hasattr(planning_problem.goal.state_list[0], prop):
        if prop == "velocity":
            interval = planning_problem.goal.state_list[0].velocity
            if is_in_interval(value, interval):
                return 0.0
            # return the minimum distance to the interval
            else:
                if value < interval.start:
                    return interval.start - value
                elif value > interval.end:
                    return value - interval.end
        elif prop == "orientation":
            interval = planning_problem.goal.state_list[0].orientation
            if check_orientation(value, interval):
                return 0.0
            # return the minimum distance to the interval
            else:
                if value < interval.start:
                    return interval.start - value
                elif value > interval.end:
                    return value - interval.end
        else:
            raise ValueError(
                "Selected property hast to be 'velocity' 'or orientation'."
            )
    # return 0.0 if the planning problem has no goal value for the chosen property
    else:
        return 0.0


def calc_remaining_time_steps(
    ego_state_time: float, t: float, planning_problem, dt: float
):
    """
    Get the minimum and maximum amount of remaining time steps.

    Args:
        ego_state_time (float): Current time of the state of the ego vehicle.
        t (float): Checked time.
        planning_problem (PlanningProblem): Considered planning problem.
        dt (float): Time step size of the scenario.

    Returns:
        int: Minimum remaining time steps.
        int: Maximum remaining time steps.
    """
    considered_time_step = int(ego_state_time + t / dt)
    if hasattr(planning_problem.goal.state_list[0], "time_step"):
        min_remaining_time = (
            planning_problem.goal.state_list[0].time_step.start - considered_time_step
        )
        max_remaining_time = (
            planning_problem.goal.state_list[0].time_step.end - considered_time_step
        )
        return min_remaining_time, max_remaining_time
    else:
        return False


def calc_avg_dist_to_global_path(traj):
    """
    Calculate the average distance of the frenét trajectory to the global path.

    Args:
        traj (FrenetTrajectory): Frenét trajectory to be checked.

    Returns:
        float: Average distance of the trajectory to the given path.
    """
    return np.mean(np.abs(traj.d))


def calc_travelled_dist(traj):
    """
    Get the travelled distance for a trajectory.

    Args:
        traj (FrenetTrajectory): Considered trajectory.

    Returns:
        float: Travelled distance at the end of the trajectory.
    """
    # calculate the distance between two consecutive points of the trajectory and add them up
    dist = 0.0
    for i in range(len(traj.x) - 1):
        delta_dist = distance(
            np.array([traj.x[i], traj.y[i]]), np.array([traj.x[i + 1], traj.y[i + 1]])
        )
        dist += delta_dist

    return dist


def calc_dist_to_goal_pos(traj, planning_problem, lanelet_network):
    """
    Calculate the distance of the last position of the trajectory to the goal positions.

    Args:
        traj (FrenetTrajectory): Considered trajectory.
        planning_problem (PlanningProblem): Considered planning problem.
        lanelet_network (LaneletNetwork): Considered lanelet network.

    Returns:
        float: Distance from the end of the given trajectory to the closest goal position.
    """
    min_dist = np.inf

    # if the goal position is given in lanelets, use their center points
    if (
        hasattr(planning_problem.goal, "lanelets_of_goal_position")
        and planning_problem.goal.lanelets_of_goal_position is not None
    ):
        goal_lanelet_ids = planning_problem.goal.lanelets_of_goal_position
        for i in range(len(goal_lanelet_ids[0])):
            goal_lanelet_obj = lanelet_network.find_lanelet_by_id(
                goal_lanelet_ids[0][i]
            )
            dist = dist_to_nearest_point(
                goal_lanelet_obj.center_vertices, np.array([traj.x[-1], traj.y[-1]])
            )
            if dist < min_dist:
                min_dist = dist
    # if the goal position is given as a shape
    elif hasattr(planning_problem.goal.state_list[0], "position"):
        if hasattr(planning_problem.goal.state_list[0].position, "center"):
            for i in range(len(planning_problem.goal.state_list)):
                dist = distance(
                    planning_problem.goal.state_list[i].position.center,
                    np.array([traj.x[-1], traj.y[-1]]),
                )
                if dist < min_dist:
                    min_dist = dist
    # survival scenario
    else:
        return 0

    return min_dist


def calc_dist_to_center_line(traj, lanelet_network):
    """
    Calculate the average distance of the trajectory to the center line of a lane.

    Args:
        traj (FrenetTrajectory): Considered trajectory.
        lanelet_network (LaneletNetwork): Considered lanelet network.

    Returns:
        float: Average distance from the trajectory to the center line of a lane.
    """
    dist = 0.0
    for i in range(len(traj.t)):
        # find the lanelet of every position
        pos = [traj.x[i], traj.y[i]]
        lanelet_ids = lanelet_network.find_lanelet_by_position([np.array(pos)])
        if len(lanelet_ids[0]) > 0:
            lanelet_id = lanelet_ids[0][0]
            lanelet_obj = lanelet_network.find_lanelet_by_id(lanelet_id)
            # find the distance of the current position to the center line of the lanelet
            dist = dist + dist_to_nearest_point(lanelet_obj.center_vertices, pos)
        # theirs should always be a lanelet for the current position
        # otherwise the trajectory should not be valid and no costs are calculated
        else:
            dist = dist + 5

    return dist / len(traj.x)


def dist_to_nearest_point(center_vertices: np.ndarray, pos: np.array):
    """
    Find the closest distance of a given position to a polyline.

    Args:
        center_vertices (np.ndarray): Considered polyline.
        pos (np.array): Conisdered position.

    Returns:
        float: Closest distance between the polyline and the position.
    """
    # create a point and a line, project the point on the line and find the nearest point
    # shapely used
    point = Point(pos)
    linestring = LineString(center_vertices)
    project = linestring.project(point)
    nearest_point = linestring.interpolate(project)

    return distance(pos, np.array([nearest_point.x, nearest_point.y]))


def get_relative_velocity(
    velocity_obstacle_1: float,
    yaw_obstacle_1: float,
    velocity_obstacle_2: float,
    yaw_obstacle_2: float,
):
    """
    Get the maximum relative velocity of two obstacles.

    Args:
        velocity_obstacle_1 (float): Velocity of the first obstacle.
        yaw_obstacle_1 (float): Yaw of the first obstacle.
        velocity_obstacle_2 (float): Velocity of the second obstacle.
        yaw_obstacle_2 (float): Yaw of the second obstacle.

    Returns:
        float: Maximum relative velocity between the two obstacles.
    """
    yaw_diff = yaw_obstacle_1 - yaw_obstacle_2
    rel_velocity1 = abs(velocity_obstacle_1 - velocity_obstacle_2 * np.cos(yaw_diff))
    rel_velocity2 = abs(velocity_obstacle_2 - velocity_obstacle_1 * np.cos(yaw_diff))

    return max(rel_velocity1, rel_velocity2)


def check_orientation(value_to_check: float, interval: Interval):
    """
    Check if the orientation is in the given interval.

    Normalize every used orientation in the interval from - pi to pi.

    Args:
        value_to_check (float): Orientation to be checked.
        interval (Interval): The interval to be checked.

    Returns:
        Interval: The valid interval
    """
    # get the values of the interval
    values = [interval.start, interval.end]

    # counter for how many boundaries stay unchanged
    n_in_range = 0

    # create a new interval ranging from -pi to pi
    new_interval = []

    # check start and end of the interval
    for value in values:
        if value > math.pi:
            new_value = value
            # subtract 2 pi until new value is in the valid interval
            while new_value > math.pi:
                new_value -= 2 * math.pi
            new_interval.append(new_value)
        elif value < -math.pi:
            new_value = value
            # add 2 pi until new value is in the valid interval
            while new_value < -math.pi:
                new_value += 2 * math.pi
            new_interval.append(new_value)
        # count unchanged values
        else:
            n_in_range += 1
            new_interval.append(value)

    new_interval.sort()
    new_interval = Interval(start=new_interval[0], end=new_interval[1])

    # if only one value of the interval was changed it is now necessary to check for values outside of this interval
    if n_in_range == 1:
        if is_in_interval(value=value_to_check, interval=new_interval) is False:
            return True
    else:
        if is_in_interval(value=value_to_check, interval=new_interval):
            return True

    return False


# EOF
