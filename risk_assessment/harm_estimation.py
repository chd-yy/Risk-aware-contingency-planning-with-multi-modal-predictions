"""Harm estimation function calling models based on risk json."""

import numpy as np
from commonroad.scenario.obstacle import ObstacleType

from beliefplanning.risk_assessment.helpers.harm_parameters import HarmParameters
from beliefplanning.risk_assessment.helpers.properties import calc_crash_angle, get_obstacle_mass
from beliefplanning.risk_assessment.utils.logistic_regression import (
    get_protected_log_reg_harm,
    get_unprotected_log_reg_harm,
)
from beliefplanning.risk_assessment.utils.reference_speed import (
    get_protected_ref_speed_harm,
    get_unprotected_ref_speed_harm,
)
from beliefplanning.risk_assessment.utils.reference_speed_symmetrical import (
    get_protected_inj_prob_ref_speed_complete_sym,
    get_protected_inj_prob_ref_speed_ignore_angle,
    get_protected_inj_prob_ref_speed_reduced_sym,
)
from beliefplanning.risk_assessment.utils.reference_speed_asymmetrical import (
    get_protected_inj_prob_ref_speed_complete,
    get_protected_inj_prob_ref_speed_reduced,
)
from beliefplanning.risk_assessment.utils.gidas import (
    get_protected_gidas_harm,
    get_unprotected_gidas_harm,
)
from beliefplanning.risk_assessment.utils.logistic_regression_symmetrical import (
    get_protected_inj_prob_log_reg_complete_sym,
    get_protected_inj_prob_log_reg_ignore_angle,
    get_protected_inj_prob_log_reg_reduced_sym,
)
from beliefplanning.risk_assessment.utils.logistic_regression_asymmetrical import (
    get_protected_inj_prob_log_reg_complete,
    get_protected_inj_prob_log_reg_reduced,
)

# Dictionary for existence of protective crash structure.
obstacle_protection = {
    ObstacleType.CAR: True,
    ObstacleType.TRUCK: True,
    ObstacleType.BUS: True,
    ObstacleType.BICYCLE: False,
    ObstacleType.PEDESTRIAN: False,
    ObstacleType.PRIORITY_VEHICLE: True,
    ObstacleType.PARKED_VEHICLE: True,
    ObstacleType.TRAIN: True,
    ObstacleType.MOTORCYCLE: False,
    ObstacleType.TAXI: True,
    ObstacleType.ROAD_BOUNDARY: None,
    ObstacleType.PILLAR: None,
    ObstacleType.CONSTRUCTION_ZONE: None,
    ObstacleType.BUILDING: None,
    ObstacleType.MEDIAN_STRIP: None,
    ObstacleType.UNKNOWN: False,
}


def harm_model(
        scenario,
        ego_vehicle_id: int,
        vehicle_params,
        ego_velocity: float,
        ego_yaw: float,
        obstacle_id: int,
        obstacle_size: float,
        obstacle_velocity: float,
        obstacle_yaw: float,
        pdof: float,
        ego_angle: float,
        obs_angle: float,
        modes,
        coeffs,
):
    """
    Get the harm for two possible collision partners.

    Args:
        scenario (Scenario): Considered scenario.
        ego_vehicle_id (Int): ID of ego vehicle.
        vehicle_params (Dict): Parameters of ego vehicle (1, 2 or 3).
        ego_velocity (Float): Velocity of ego vehicle [m/s].
        ego_yaw (Float): Yaw of ego vehicle [rad].
        obstacle_id (Int): ID of considered obstacle.
        obstacle_size (Float): Size of obstacle in [m²] (length * width)
        obstacle_velocity (Float): Velocity of obstacle [m/s].
        obstacle_yaw (Float): Yaw of obstacle [rad].
        pdof (float): Crash angle between ego vehicle and considered
            obstacle [rad].
        ego_angle (float): Angle of impact area for the ego vehicle.
        obs_angle (float): Angle of impact area for the obstacle.
        modes (Dict): Risk modes. Read from risk.json.
        coeffs (Dict): Risk parameters. Read from risk_parameters.json

    Returns:
        float: Harm for the ego vehicle.
        float: Harm for the other collision partner.
        HarmParameters: Class with independent variables for the ego
            vehicle
        HarmParameters: Class with independent variables for the obstacle
            vehicle
    """
    # create dictionaries with crash relevant parameters
    ego_vehicle = HarmParameters()
    obstacle = HarmParameters()

    # assign parameters to dictionary
    ego_vehicle.type = scenario.obstacle_by_id(ego_vehicle_id).obstacle_type
    obstacle.type = scenario.obstacle_by_id(obstacle_id).obstacle_type
    ego_vehicle.protection = obstacle_protection[ego_vehicle.type]
    obstacle.protection = obstacle_protection[obstacle.type]
    if ego_vehicle.protection is not None:
        ego_vehicle.mass = vehicle_params.m
        ego_vehicle.velocity = ego_velocity
        ego_vehicle.yaw = ego_yaw
        ego_vehicle.size = vehicle_params.w * vehicle_params.l
    else:
        ego_vehicle.mass = None
        ego_vehicle.velocity = None
        ego_vehicle.yaw = None
        ego_vehicle.size = None

    if obstacle.protection is not None:
        obstacle.velocity = obstacle_velocity
        obstacle.yaw = obstacle_yaw
        obstacle.size = obstacle_size
        obstacle.mass = get_obstacle_mass(
            obstacle_type=obstacle.type, size=obstacle.size
        )
    else:
        obstacle.mass = None
        obstacle.velocity = None
        obstacle.yaw = None
        obstacle.size = None

    # get model based on selection
    if modes["harm_mode"] == "log_reg":
        # select case based on protection structure
        if obstacle.protection is True:
            ego_vehicle.harm, obstacle.harm = get_protected_log_reg_harm(
                ego_vehicle=ego_vehicle,
                obstacle=obstacle,
                pdof=pdof,
                ego_angle=ego_angle,
                obs_angle=obs_angle,
                modes=modes,
                coeffs=coeffs,
            )
        elif obstacle.protection is False:
            ego_vehicle.harm, obstacle.harm = get_unprotected_log_reg_harm(
                ego_vehicle=ego_vehicle, obstacle=obstacle, pdof=pdof, coeff=coeffs
            )
        else:
            ego_vehicle.harm = 1
            obstacle.harm = 1

    elif modes["harm_mode"] == "ref_speed":
        # select case based on protection structure
        if obstacle.protection is True:
            ego_vehicle.harm, obstacle.harm = get_protected_ref_speed_harm(
                ego_vehicle=ego_vehicle,
                obstacle=obstacle,
                pdof=pdof,
                ego_angle=ego_angle,
                obs_angle=obs_angle,
                modes=modes,
                coeffs=coeffs,
            )
        elif obstacle.protection is False:
            ego_vehicle.harm, obstacle.harm = get_unprotected_ref_speed_harm(
                ego_vehicle=ego_vehicle, obstacle=obstacle, pdof=pdof, coeff=coeffs
            )
        else:
            ego_vehicle.harm = 1
            obstacle.harm = 1

    elif modes["harm_mode"] == "gidas":
        # select case based on protection structure
        if obstacle.protection is True:
            ego_vehicle.harm, obstacle.harm = get_protected_gidas_harm(
                ego_vehicle=ego_vehicle, obstacle=obstacle, pdof=pdof, coeff=coeffs
            )
        elif obstacle.protection is False:
            ego_vehicle.harm, obstacle.harm = get_unprotected_gidas_harm(
                ego_vehicle=ego_vehicle, obstacle=obstacle, pdof=pdof, coeff=coeffs
            )
        else:
            ego_vehicle.harm = 1
            obstacle.harm = 1

    else:
        raise ValueError(
            "Please select a valid mode for harm estimation "
            "(log_reg, ref_speed, gidas)"
        )

    return ego_vehicle.harm, obstacle.harm, ego_vehicle, obstacle


def _resolve_obstacle_mode(mode_num, obstacle_id, mode_count):
    if mode_num == 100:
        return None
    if isinstance(mode_num, dict):
        return min(mode_num.get(obstacle_id, 0), mode_count - 1)
    return min(mode_num, mode_count - 1)


def get_harm(scenario, traj, predictions, ego_id, vehicle_params, modes, coeffs, timer, start_idx, mode_num):
    """Get harm.
    """
    # ----------------------------------------------------------------------
    # 函数目标
    # ----------------------------------------------------------------------
    # 这个函数的作用是：
    # 对当前候选 ego 轨迹 traj，与所有预测到的障碍物 predictions，
    # 逐障碍物、逐 mode 地计算碰撞发生时的 harm（伤害程度）。
    #
    # 返回两个字典：
    # - ego_harm_traj  : ego 车辆在与各障碍物碰撞时的伤害
    # - obst_harm_traj : 对应障碍物在碰撞时的伤害
    #
    # 这两个字典的结构一般都是：
    #   dict[obstacle_id] = [mode_0_harm_list, mode_1_harm_list, ...]
    #
    # 其中每个 mode_harm_list 又是一个按时间步排列的 harm 值列表。
    # ----------------------------------------------------------------------

    # get the IDs of the predicted obstacles
    # Important
    # ----------------------------------------------------------------------
    # 取出所有被预测到的障碍物 ID
    # predictions 一般形如：
    #   predictions[obstacle_id] = {
    #       'pos_list': ...,
    #       'cov_list': ...,
    #       'orientation_list': ...,
    #       'v_list': ...,
    #       'shape': ...
    #   }
    # ----------------------------------------------------------------------
    obstacle_ids = list(predictions.keys())

    # max_pred_length = 0
    # （注释掉的变量）原本可能用于记录最大预测长度

    # 保存 ego 对每个 obstacle 的 harm 结果
    ego_harm_traj = {}

    # 保存 obstacle 对应的 harm 结果
    obst_harm_traj = {}

    # get the ego vehicle size
    # ego_vehicle_size = vehicle_params.w * vehicle_params.l
    # （这里注释掉了 ego 车尺寸计算，当前函数内未用到）

    # iterate over the obstacle
    # ----------------------------------------------------------------------
    # 遍历所有障碍物，逐个计算它们与当前 ego 轨迹的碰撞伤害
    # ----------------------------------------------------------------------
    for obstacle_id in obstacle_ids:

        # 用于保存“当前 obstacle 的所有 mode 下 ego harm 列表”
        ego_harm_obst_list = []

        # 用于保存“当前 obstacle 的所有 mode 下 obstacle harm 列表”
        obst_harm_obst_list = []

        # ------------------------------------------------------------------
        # 根据 mode_num 和当前 obstacle 的 mode 数量，解析出应该使用哪个 mode
        #
        # _resolve_obstacle_mode(...) 的作用通常是：
        # - 如果 mode_num 表示“遍历所有 mode”，则返回 None
        # - 如果 mode_num 表示“只使用指定 mode”，则返回具体的 mode 索引
        # ------------------------------------------------------------------
        selected_mode = _resolve_obstacle_mode(
            mode_num=mode_num,
            obstacle_id=obstacle_id,
            mode_count=len(predictions[obstacle_id]['pos_list']),
        )

        # iterate over the modes per obstacle
        # ------------------------------------------------------------------
        # 遍历当前 obstacle 的所有预测 mode
        # ------------------------------------------------------------------
        for mode in range(len(predictions[obstacle_id]['pos_list'])):

            # 如果 selected_mode 不是 None，说明外部只要求评估某一个 mode
            # 那么这里强制把 mode 改成 selected_mode
            if selected_mode is not None:
                mode = selected_mode

            # choose which model should be used to calculate the harm
            # ------------------------------------------------------------------
            # 根据 modes 配置和 obstacle 类型，选择用于 harm 计算的模型函数
            # 返回两个函数：
            # - ego_harm_fun      : 用来计算 ego 伤害
            # - obstacle_harm_fun : 用来计算对方障碍物伤害
            # ------------------------------------------------------------------
            ego_harm_fun, obstacle_harm_fun = get_model(modes, obstacle_id, scenario)

            # only calculate the risk as long as both obstacles are in the scenario
            # ------------------------------------------------------------------
            # 从 start_idx 开始取当前 mode 的预测轨迹
            # pred_path 是当前 obstacle 在该 mode 下未来的位置序列
            # ------------------------------------------------------------------
            pred_path = predictions[obstacle_id]['pos_list'][mode][start_idx:]

            # 实际参与 harm 计算的时间长度
            # 注意这里使用 len(traj.t) - 1，通常是因为 traj 末端/首端索引和预测数组的
            # 对齐方式要求减 1
            pred_length = min(len(traj.t) - 1, len(pred_path))

            # 如果没有任何可比较的时间步，直接跳过该 mode
            if pred_length == 0:
                continue

            # get max prediction length
            # if pred_length > max_pred_length:
            #     max_pred_length = pred_length
            # （当前未使用）

            # get the size, the velocity and the orientation of the predicted
            # vehicle
            # ------------------------------------------------------------------
            # 取出当前预测障碍物的“面积近似值”
            # 这里用 length * width 作为 size
            # ------------------------------------------------------------------
            pred_size = (
                    predictions[obstacle_id]['shape']['length']
                    * predictions[obstacle_id]['shape']['width']
            )

            # 当前 mode 下，从 start_idx 开始的障碍物速度序列
            pred_v = np.array(predictions[obstacle_id]['v_list'][mode][start_idx:], dtype=float)

            # 当前 mode 下，从 start_idx 开始的障碍物朝向序列
            pred_yaw = np.array(predictions[obstacle_id]['orientation_list'][mode][start_idx:], dtype=float)

            # lists to save ego and obstacle harm as well as ego and obstacle risk
            # one list per obstacle
            # ------------------------------------------------------------------
            # 当前 obstacle、当前 mode 下，逐时间步存储 harm 的列表
            # ------------------------------------------------------------------
            ego_harm_obst = []
            obst_harm_obst = []

            # replace get_obstacle_mass() by get_obstacle_mass()
            # get the predicted obstacle vehicle mass
            # ------------------------------------------------------------------
            # 原本这里可能希望根据 obstacle type 和尺寸估算障碍物质量，
            # 但相关逻辑被注释掉了。
            #
            # 现在直接把 obstacle_mass 固定成一个常数 3264.8413。
            # 这意味着当前 harm 计算中，对方障碍物质量不随具体目标而变化。
            # ------------------------------------------------------------------
            '''
            obstacle_mass = get_obstacle_mass(
                obstacle_type=scenario.obstacle_by_id(obstacle_id).obstacle_type, size=pred_size
            )
            '''
            obstacle_mass = 3264.8413

            # calc crash angle if comprehensive mode selected
            # ------------------------------------------------------------------
            # 如果没有启用“简化碰撞角模型”，则走更完整的 crash angle 计算流程
            # ------------------------------------------------------------------
            if modes["crash_angle_simplified"] is False:

                with timer.time_with_cm(
                        "simulation/sort trajectories/calculate costs/calculate risk/"
                        + "calculate harm/calculate PDOF comp"
                ):
                    # ----------------------------------------------------------
                    # 计算碰撞几何相关角度：
                    # - pdof      : principal direction of force，碰撞主受力方向
                    # - ego_angle : 碰撞作用在 ego 车体上的受撞区域角度
                    # - obs_angle : 碰撞作用在障碍物车体上的受撞区域角度
                    #
                    # 这个 calc_crash_angle(...) 一般会考虑更复杂的碰撞几何关系。
                    # ----------------------------------------------------------
                    pdof, ego_angle, obs_angle = calc_crash_angle(
                        traj=traj,
                        predictions=predictions,
                        scenario=scenario,
                        obstacle_id=obstacle_id,
                        modes=modes,
                        vehicle_params=vehicle_params,
                    )

                # 对每一个可比较时间步计算 harm
                for i in range(pred_length):
                    with timer.time_with_cm(
                            "simulation/sort trajectories/calculate costs/calculate risk/"
                            + "calculate harm/harm_model"
                    ):
                        # get the harm ego harm and the harm of the collision opponent
                        # ------------------------------------------------------
                        # 通过统一的 harm_model(...) 计算：
                        # - ego_harm : ego 在该时刻碰撞下的 harm
                        # - obst_harm: 障碍物在该时刻碰撞下的 harm
                        # 以及一些附加数据 ego_harm_data / obst_harm_data
                        #
                        # 输入包含：
                        # - ego 速度与朝向
                        # - obstacle 的尺寸、速度与朝向
                        # - pdof / ego_angle / obs_angle
                        # - modes / coeffs 等模型配置
                        # ------------------------------------------------------
                        ego_harm, obst_harm, ego_harm_data, obst_harm_data = harm_model(
                            scenario=scenario,
                            ego_vehicle_id=ego_id,
                            vehicle_params=vehicle_params,
                            ego_velocity=traj.v[i],
                            ego_yaw=traj.yaw[i],
                            obstacle_id=obstacle_id,
                            obstacle_size=pred_size,
                            obstacle_velocity=pred_v[i],
                            obstacle_yaw=pred_yaw[i],
                            pdof=pdof,
                            ego_angle=ego_angle,
                            obs_angle=obs_angle,
                            modes=modes,
                            coeffs=coeffs,
                        )

                        # store information to calculate harm and harm value in list
                        # 将当前时间步的 harm 保存下来
                        ego_harm_obst.append(ego_harm)
                        obst_harm_obst.append(obst_harm)

            else:
                # calc the risk for every time step
                # ------------------------------------------------------------------
                # 如果启用了“简化碰撞角模型”，则使用一个更轻量的近似方法：
                # 直接通过 ego 和 obstacle 的相对位置、朝向、速度来计算简化的碰撞角，
                # 再推导 delta-v，最后根据 delta-v 和碰撞角计算 harm。
                # ------------------------------------------------------------------
                with timer.time_with_cm(
                        "simulation/sort trajectories/calculate costs/calculate risk/"
                        + "calculate harm/calculate PDOF simple"
                ):
                    # crash angle between ego vehicle and considered obstacle [rad]
                    # ----------------------------------------------------------
                    # pdof_array：
                    # ego 和障碍物的相对碰撞方向（简化表达）
                    #
                    # 这里用：
                    # obstacle_yaw - ego_yaw + pi
                    # 来近似碰撞主方向相关角度
                    # ----------------------------------------------------------
                    pdof_array = predictions[obstacle_id]["orientation_list"][mode][start_idx:pred_length + start_idx] - traj.yaw[
                                                                                              :pred_length] + np.pi

                    # ----------------------------------------------------------
                    # rel_angle_array：
                    # 从 ego 指向 obstacle 的相对方位角
                    # 用 obstacle 位置减 ego 位置，再 atan2 得到
                    # ----------------------------------------------------------
                    rel_angle_array = np.arctan2(
                        predictions[obstacle_id]["pos_list"][mode][start_idx:pred_length + start_idx, 1] - traj.y[:pred_length],
                        predictions[obstacle_id]["pos_list"][mode][start_idx:pred_length + start_idx, 0] - traj.x[:pred_length])

                    # angle of impact area for the ego vehicle
                    # ----------------------------------------------------------
                    # ego_angle_array：
                    # 碰撞作用到 ego 车体上的相对碰撞角
                    # = relative angle - ego yaw
                    # ----------------------------------------------------------
                    ego_angle_array = rel_angle_array - traj.yaw[:pred_length]

                    # angle of impact area for the obstacle
                    # ----------------------------------------------------------
                    # obs_angle_array：
                    # 碰撞作用到 obstacle 车体上的相对碰撞角
                    # ----------------------------------------------------------
                    obs_angle_array = np.pi + rel_angle_array - predictions[obstacle_id]["orientation_list"][mode][start_idx:pred_length + start_idx]

                    # calculate the difference between pre-crash and post-crash speed
                    # ----------------------------------------------------------
                    # 根据两车速度和相对碰撞方向，近似计算碰撞前后速度变化量 delta_v
                    #
                    # 公式形式类似两速度向量夹角下的合速度差：
                    # sqrt(v_ego^2 + v_obs^2 + 2 v_ego v_obs cos(pdof))
                    # ----------------------------------------------------------
                    delta_v_array = np.sqrt(
                        np.power(traj.v[:pred_length], 2)
                        + np.power(pred_v[:pred_length], 2)
                        + 2 * traj.v[:pred_length] * pred_v[:pred_length] * np.cos(pdof_array)
                    )

                    # ----------------------------------------------------------
                    # 按两车质量比，把总 delta-v 分摊到 ego 和 obstacle
                    # 这里相当于用一个非常简化的动量交换近似
                    # ----------------------------------------------------------
                    ego_delta_v = obstacle_mass / (vehicle_params.m + obstacle_mass) * delta_v_array
                    obstacle_delta_v = vehicle_params.m / (vehicle_params.m + obstacle_mass) * delta_v_array

                    # calculate harm based on selected model
                    # ----------------------------------------------------------
                    # 利用选择的 harm 模型，根据：
                    # - 自身 delta-v
                    # - 碰撞角
                    # - 模型系数 coeffs
                    # 来估算伤害
                    #
                    # 这里返回的通常是一个数组，对应 pred_length 个时间步的 harm
                    # ----------------------------------------------------------
                    ego_harm_obst = ego_harm_fun(velocity=ego_delta_v, angle=ego_angle_array, coeff=coeffs)
                    obst_harm_obst = obstacle_harm_fun(velocity=obstacle_delta_v, angle=obs_angle_array, coeff=coeffs)

            # store harm list for the obstacles in dictionary for current frenét
            # trajectory
            # make it a list of the modes per obstacle
            # ------------------------------------------------------------------
            # 将当前 obstacle 的当前 mode 的 harm 序列加入 mode 列表中
            # ------------------------------------------------------------------
            ego_harm_obst_list.append(ego_harm_obst)
            obst_harm_obst_list.append(obst_harm_obst)

            # ------------------------------------------------------------------
            # 如果 selected_mode 不为 None，说明当前本来就只想算某一个 mode
            # 那么这一轮算完之后就 break，不再继续遍历其他 mode
            # ------------------------------------------------------------------
            if selected_mode is not None:
                break

        # ------------------------------------------------------------------
        # 当前 obstacle 的所有 mode harm 列表保存到最终结果字典中
        # key 是 obstacle_id
        # value 是 [mode_0_harm, mode_1_harm, ...]
        # ------------------------------------------------------------------
        ego_harm_traj[obstacle_id] = ego_harm_obst_list
        obst_harm_traj[obstacle_id] = obst_harm_obst_list

    # 返回：
    # - ego_harm_traj  : ego 对每个 obstacle / mode / time 的 harm
    # - obst_harm_traj : obstacle 对每个 obstacle / mode / time 的 harm
    return ego_harm_traj, obst_harm_traj


def get_model(modes, obstacle_id, scenario):
    """Get harm model according to settings.

    Args:
        modes (_type_): _description_
        obstacle_id (_type_): _description_
        scenario (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # obstacle protection type
    # obs_protection = obstacle_protection[scenario.obstacle_by_id(obstacle_id).obstacle_type]
    obs_protection = True

    if modes["harm_mode"] == "log_reg":
        # select case based on protection structure
        if obs_protection is True:
            # calculate harm based on angle mode
            if modes["ignore_angle"] is False:
                if modes["sym_angle"] is False:
                    if modes["reduced_angle_areas"] is False:
                        # use log reg complete
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_log_reg_complete

                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_log_reg_complete

                    else:
                        # use log reg reduced
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_log_reg_reduced

                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_log_reg_reduced
                else:
                    if modes["reduced_angle_areas"] is False:
                        # use log reg sym complete
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_log_reg_complete_sym

                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_log_reg_complete_sym
                    else:
                        # use log reg sym reduced
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_log_reg_reduced_sym

                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_log_reg_reduced_sym
            else:
                # use log reg delta v
                # calculate harm for the ego vehicle
                ego_harm = get_protected_inj_prob_log_reg_ignore_angle

                # calculate harm for the obstacle vehicle
                obstacle_harm = get_protected_inj_prob_log_reg_ignore_angle

        elif obs_protection is False:
            # calc ego harm
            ego_harm = get_protected_inj_prob_log_reg_ignore_angle

            # calculate obstacle harm
            # logistic regression model
            obstacle_harm = lambda velocity, angle, coeff: 1 / (  # noqa E731
                    1
                    + np.exp(
                coeff["pedestrian"]["const"]
                - coeff["pedestrian"]["speed"] * velocity
            )
            )
        else:
            ego_harm = lambda velocity, angle, coeff: 1  # noqa E731
            obstacle_harm = lambda velocity, angle, coeff: 1  # noqa E731

    elif modes["harm_mode"] == "ref_speed":
        # select case based on protection structure
        if obs_protection is True:
            # calculate harm based on angle mode
            if modes["ignore_angle"] is False:
                if modes["sym_angle"] is False:
                    if modes["reduced_angle_areas"] is False:
                        # use log reg complete
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_ref_speed_complete

                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_ref_speed_complete

                    else:
                        # use log reg reduced
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_ref_speed_reduced

                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_ref_speed_reduced
                else:
                    if modes["reduced_angle_areas"] is False:
                        # use log reg sym complete
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_ref_speed_complete_sym

                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_ref_speed_complete_sym
                    else:
                        # use log reg sym reduced
                        # calculate harm for the ego vehicle
                        ego_harm = get_protected_inj_prob_ref_speed_reduced_sym

                        # calculate harm for the obstacle vehicle
                        obstacle_harm = get_protected_inj_prob_ref_speed_reduced_sym
            else:
                # use log reg delta v
                # calculate harm for the ego vehicle
                ego_harm = get_protected_inj_prob_ref_speed_ignore_angle

                # calculate harm for the obstacle vehicle
                obstacle_harm = get_protected_inj_prob_ref_speed_ignore_angle

        elif obs_protection is False:
            # calc ego harm
            ego_harm = get_protected_inj_prob_ref_speed_ignore_angle

            # calculate obstacle harm
            # logistic regression model
            obstacle_harm = lambda velocity, angle, coeff: 1 / (  # noqa E731
                    1
                    + np.exp(
                coeff["pedestrian"]["const"]
                - coeff["pedestrian"]["speed"] * velocity
            )
            )
        else:
            ego_harm = lambda velocity, angle, coeff: 1  # noqa E731
            obstacle_harm = lambda velocity, angle, coeff: 1  # noqa E731

    elif modes["harm_mode"] == "gidas":
        # select case based on protection structure
        if obs_protection is True:
            ego_harm = lambda velocity, angle, coeff: 1 / (  # noqa E731
                    1 + np.exp(-coeff["gidas"]["const"] - coeff["gidas"]["speed"] * velocity)
            )

            obs_harm = lambda velocity, angle, coeff: 1 / (  # noqa E731
                    1
                    + np.exp(-coeff["gidas"]["const"] - coeff["gidas"]["speed"] * velocity)
            )
        elif obs_protection is False:
            # calc ego harm
            ego_harm = lambda velocity, angle, coeff: 1 / (  # noqa E731
                    1 + np.exp(-coeff["gidas"]["const"] - coeff["gidas"]["speed"] * velocity)
            )

            # calculate obstacle harm
            # logistic regression model
            obstacle_harm = lambda velocity, angle, coeff: 1 / (  # noqa E731
                    1
                    + np.exp(
                coeff["pedestrian_MAIS2+"]["const"]
                - coeff["pedestrian_MAIS2+"]["speed"] * velocity
            )
            )
        else:
            ego_harm = lambda velocity, angle, coeff: 1  # noqa E731
            obstacle_harm = lambda velocity, angle, coeff: 1  # noqa E731

    else:
        raise ValueError(
            "Please select a valid mode for harm estimation "
            "(log_reg, ref_speed, gidas)"
        )

    return ego_harm, obstacle_harm
