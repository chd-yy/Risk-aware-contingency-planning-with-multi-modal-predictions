"""Risk cost function and principles of ethics of risk."""

from beliefplanning.planner.Frenet.utils.validity_checks import create_collision_object
from commonroad_dc.collision.trajectory_queries import trajectory_queries

from beliefplanning.risk_assessment.harm_estimation import get_harm
from beliefplanning.risk_assessment.collision_probability import (
    get_collision_probability_fast,
    get_inv_mahalanobis_dist
)
from beliefplanning.risk_assessment.helpers.timers import ExecTimer
from beliefplanning.planner.utils.responsibility import assign_responsibility_by_action_space, \
    calc_responsibility_reach_set
from beliefplanning.risk_assessment.utils.logistic_regression_symmetrical import \
    get_protected_inj_prob_log_reg_ignore_angle


def calc_risk(
        traj,
        ego_state,
        predictions: dict,
        scenario,
        ego_id: int,
        vehicle_params,
        params,
        reach_set=None,
        exec_timer=None,
        start_idx=0,
        mode_num=100,
        belief=None,
):
    """
    Calculate the risk for the given trajectory.

    Args:
        traj (FrenetTrajectory): Considered frenét trajectory.
        predictions (dict): Predictions for the visible obstacles.
        scenario (Scenario): Considered scenario.
        ego_id (int): ID of the ego vehicle.
        vehicle_params (VehicleParameters): Vehicle parameters of the
            considered vehicle.
        weights (Dict): Weighing factors. Read from weights.json.
        modes (Dict): Risk modes. Read from risk.json.
        coeffs (Dict): Risk parameters. Read from risk_parameters.json.
        exec_timer (ExecTimer): Timer for the exec_timing.json.

    Returns:
        float: Weighed risk costs.
        dict: Dictionary with ego harms for every time step concerning every
            obstacle
        dict: Dictionary with obstacle harms for every time step concerning
            every obstacle

    """
    """
    计算给定轨迹 traj 的风险。

    参数说明
    --------
    traj : FrenetTrajectory
        当前正在评估的候选轨迹（ego车的规划轨迹）

    ego_state :
        ego车辆当前状态

    predictions : dict
        其他车辆的未来预测结果
        结构通常为：
        {
            obstacle_id :
                {
                    pos_list : [...],
                    cov_list : [...],
                    orientation_list : [...],
                    v_list : [...]
                }
        }

    scenario :
        当前仿真场景

    ego_id : int
        ego车辆ID

    vehicle_params :
        ego车辆参数（长宽等）

    params :
        风险模型参数
        包含：
            modes : 风险计算模式
            harm  : 伤害模型参数

    reach_set :
        可达集（当前代码未使用）

    exec_timer :
        用于记录执行时间

    start_idx :
        从预测轨迹的哪个时间步开始评估

    mode_num :
        场景模式数量
        默认100表示 shared planning

    belief :
        对不同行为模式的概率信念
    """
    timer = ExecTimer(timing_enabled=False) if exec_timer is None else exec_timer

    # 读取风险计算模式
    modes = params['modes']

    # 读取伤害模型参数
    coeffs = params['harm']

    with timer.time_with_cm(
            "simulation/sort trajectories/calculate costs/calculate risk/"
            + "collision probability"
    ):

        if modes["fast_prob_mahalanobis"]:
            coll_prob_dict = get_inv_mahalanobis_dist(
                traj=traj,
                predictions=predictions,
                vehicle_params=vehicle_params,
            )

        else:
            # 使用快速碰撞概率估计
            coll_prob_dict = get_collision_probability_fast(
                traj=traj,
                predictions=predictions,
                vehicle_params=vehicle_params,
                start_idx=start_idx,
                mode_num=mode_num,
            )

    # ---------------------------------------------------------
    # Step 2：计算碰撞伤害
    # ---------------------------------------------------------
    ego_harm_traj, obst_harm_traj = get_harm(
        scenario, traj, predictions, ego_id, vehicle_params, modes, coeffs, timer, start_idx, mode_num
    )

    # Calculate risk out of harm and collision probability
    # ---------------------------------------------------------
    # Step 3：初始化风险存储结构
    # ---------------------------------------------------------

    ego_risk_max = {}   # ego车最大风险
    ego_harm_max = {}   # ego最大伤害
    obst_risk_max = {}  # 障碍车最大风险
    obst_harm_max = {}  # 障碍车最大伤害

    # ---------------------------------------------------------
    # Step 4：遍历每个障碍物
    # ---------------------------------------------------------
    for key in ego_harm_traj:
        # 每个mode对应一条轨迹风险
        ego_risk_traj_list = [[None]] * len(ego_harm_traj[key])
        obst_risk_traj_list = [[None]] * len(ego_harm_traj[key])
        # iterate over the modes per obstacle
        # -----------------------------------------------------
        # 遍历每个行为模式
        # -----------------------------------------------------
        for mode in range(len(ego_harm_traj[key])):
            '''
            ego_risk_traj_list[mode] = [
                ego_harm_traj[key][mode][t] * coll_prob_dict[key][mode][t]
                for t in range(len(ego_harm_traj[key][mode]))
            ]
            '''
            '''
            原本风险计算应该是：

            risk = collision_probability x harm

            ego_risk_traj_list[mode] = [
                ego_harm_traj[key][mode][t] * coll_prob_dict[key][mode][t]
                for t in range(len(ego_harm_traj[key][mode]))
            ]

            但当前代码把 harm 注释掉了
            '''
            # 当前实现：风险 = 碰撞概率
            ego_risk_traj_list[mode] = [
                coll_prob_dict[key][mode][t]
                for t in range(len(ego_harm_traj[key][mode]))
            ]
            # 障碍车风险（同样只用碰撞概率）
            obst_risk_traj_list[mode] = [
                coll_prob_dict[key][mode][t]
                for t in range(len(obst_harm_traj[key][mode]))
            ]
        # 保存伤害值（不变）
        ego_harm_traj_list = ego_harm_traj[key]
        obst_harm_traj_list = obst_harm_traj[key]

        # For the shared plan, we need to weight the probability of collision of the shared
        # plan based on the beliefs over the modes
        # -----------------------------------------------------
        # Step 5：belief 加权（scenario tree）
        # -----------------------------------------------------

        # 如果是 shared plan（mode_num=100），则根据 belief 对风险进行加权
        if mode_num == 100:
            belief_idx = 0
            # This means this is the shared part of the plan
            for mode in range(len(ego_harm_traj[key])):
                # 每3个mode属于同一行为
                if mode % 3 == 0:
                    belief_idx += 1
                # 乘以 belief 权重
                ego_risk_traj_list[mode] = [belief[belief_idx - 1] * num for num in ego_risk_traj_list[mode]]
                ego_harm_traj_list[mode] = [belief[belief_idx - 1] * num for num in ego_harm_traj_list[mode]]
                obst_harm_traj_list[mode] = [belief[belief_idx - 1] * num for num in obst_harm_traj_list[mode]]
                obst_risk_traj_list[mode] = [belief[belief_idx - 1] * num for num in obst_risk_traj_list[mode]]
                '''
              
                ego_risk_traj_list[mode] = [belief[mode + 3] * num for num in ego_risk_traj_list[mode]]
                ego_harm_traj_list[mode] = [belief[mode + 3] * num for num in ego_harm_traj_list[mode]]
                obst_harm_traj_list[mode] = [belief[mode + 3] * num for num in obst_harm_traj_list[mode]]
                obst_risk_traj_list[mode] = [belief[mode + 3] * num for num in obst_risk_traj_list[mode]]
                '''

        # Take max as representative for the whole trajectory
        # -----------------------------------------------------
        # Step 6：取整条轨迹的最大风险
        # -----------------------------------------------------

        # print("\n" + "=" * 80)
        # print(f"[DEBUG] obstacle key = {key}")

        # print("\n[1] ego_risk_traj_list =")
        # for i, row in enumerate(ego_risk_traj_list):
        #     print(f"  mode {i}: {row}")

        # print("\n[2] obst_risk_traj_list =")
        # for i, row in enumerate(obst_risk_traj_list):
        #     print(f"  mode {i}: {row}")

        # print("\n[3] ego_harm_traj_list =")
        # for i, row in enumerate(ego_harm_traj_list):
        #     print(f"  mode {i}: {row}")

        # print("\n[4] obst_harm_traj_list =")
        # for i, row in enumerate(obst_harm_traj_list):
        #     print(f"  mode {i}: {row}")

        # print("\n[5] ego_risk_max (before assign) =")
        # print(ego_risk_max)

        # print("\n[6] obst_risk_max (before assign) =")
        # print(obst_risk_max)

        # print("\n[7] ego_harm_max (before assign) =")
        # print(ego_harm_max)

        # print("\n[8] obst_harm_max (before assign) =")
        # print(obst_harm_max)

        ego_risk_max[key] = max(max(row) for row in ego_risk_traj_list)
        ego_harm_max[key] = max(max(row) for row in ego_harm_traj_list)
        obst_harm_max[key] = max(max(row) for row in obst_harm_traj_list)
        obst_risk_max[key] = max(max(row) for row in obst_risk_traj_list)

        
        # print("\n[AFTER ASSIGN]")
        # print(f"ego_risk_max[{key}] = {ego_risk_max[key]}")
        # print(f"ego_harm_max[{key}] = {ego_harm_max[key]}")
        # print(f"obst_harm_max[{key}] = {obst_harm_max[key]}")
        # print(f"obst_risk_max[{key}] = {obst_risk_max[key]}")
        # print("=" * 80 + "\n")
    # calculate boundary harm
    # col_obj = create_collision_object(traj, vehicle_params, ego_state)
    # ---------------------------------------------------------
    # Step 7：道路边界风险
    # ---------------------------------------------------------

    '''
    下面代码本来用于检测车辆是否撞到道路边界
    并计算边界碰撞伤害

    当前被注释掉
    '''
    '''
    leaving_road_at = trajectory_queries.trajectories_collision_static_obstacles(
        trajectories=[col_obj],
        static_obstacles=road_boundary,
        method="grid",
        num_cells=32,
        auto_orientation=True,
    )

    if leaving_road_at[0] != -1:
        coll_time_step = leaving_road_at[0] - ego_state.time_step
        coll_vel = traj.v[coll_time_step]

        boundary_harm = get_protected_inj_prob_log_reg_ignore_angle(
            velocity=coll_vel, coeff=coeffs
        )

    else:
        boundary_harm = 0
    '''
    boundary_harm = 0
    # ---------------------------------------------------------
    # 返回风险结果
    # ---------------------------------------------------------
    return ego_risk_max, obst_risk_max, ego_harm_max, obst_harm_max, boundary_harm


def get_bayesian_costs(ego_risk_max, obst_risk_max, boundary_harm):
    """
    Bayesian Principle.

    Calculate the risk cost via the Bayesian Principle for the given
    trajectory.

    Args:
        ego_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        obst_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.

    Returns:
        Dict: Risk costs for the considered trajectory according to the
            Bayesian Principle
    """
    if len(ego_risk_max) == 0:
        return 0

    return (sum(ego_risk_max.values()) + sum(obst_risk_max.values()) + boundary_harm) / (
            len(ego_risk_max) * 2
    )


def get_equality_costs(ego_risk_max, obst_risk_max):
    """
    Equality Principle.

    Calculate the risk cost via the Equality Principle for the given
    trajectory.

    Args:
        ego_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        obst_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        timestep (Int): Currently considered time step.
        equality_mode (Str): Select between normalized ego risk
            ("normalized") and partial risks ("partial").

    Returns:
        float: Risk costs for the considered trajectory according to the
            Equality Principle
    """
    if len(ego_risk_max) == 0:
        return 0

    return sum(
        [abs(ego_risk_max[key] - obst_risk_max[key]) for key in ego_risk_max]
    ) / len(ego_risk_max)


def get_maximin_costs(ego_risk_max, obst_risk_max, ego_harm_max, obst_harm_max, boundary_harm, eps=10e-10,
                      scale_factor=10):
    """
    Maximin Principle.

    Calculate the risk cost via the Maximin principle for the given
    trajectory.

    Args:
        ego_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        obst_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        timestep (Int): Currently considered time step.
        maximin_mode (Str): Select between normalized ego risk
            ("normalized") and partial risks ("partial").

    Returns:
        float: Risk costs for the considered trajectory according to the
            Maximum Principle
    """
    if len(ego_harm_max) == 0:
        return 0

    # Set maximin to 0 if probability (or risk) is 0
    maximin_ego = [a * int(b < eps) for a, b in zip(ego_harm_max.values(), ego_risk_max.values())]
    maximin_obst = [a * int(bool(b < eps)) for a, b in zip(obst_harm_max.values(), obst_risk_max.values())]

    return max(maximin_ego + maximin_obst + [boundary_harm]) ** scale_factor


def get_ego_costs(ego_risk_max, boundary_harm):
    """
    Calculate the utilitarian ego cost for the given trajectory.

    Args:
        ego_harm_traj (Dict): Dictionary with collision data for all
            obstacles and all time steps.
        timestep (Int): Currently considered time step.

    Returns:
        Dict: Utilitarian ego risk cost
    """
    if len(ego_risk_max) == 0:
        return 0

    return sum(ego_risk_max.values()) + boundary_harm


def get_responsibility_cost(scenario, traj, ego_state, obst_risk_max, predictions, reach_set, mode="reach_set"):
    """Get responsibility cost.

    Args:
        obst_risk_max (_type_): _description_
        predictions (_type_): _description_
        mode (str) : "reach set" for reachable set mode, else assignement by space of action

    Returns:
        _type_: _description_

    """
    bool_contain_cache = None
    if "reach_set" in mode and reach_set is not None:
        resp_cost, bool_contain_cache = calc_responsibility_reach_set(traj, ego_state, reach_set)

    else:
        # Assign responsibility to predictions
        predictions = assign_responsibility_by_action_space(
            scenario, ego_state, predictions
        )
        resp_cost = 0

        for key in predictions:
            resp_cost -= predictions[key]["responsibility"] * obst_risk_max[key]

    return resp_cost, bool_contain_cache
