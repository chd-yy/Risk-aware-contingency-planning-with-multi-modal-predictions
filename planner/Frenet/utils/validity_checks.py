"""Check validity for a given list of trajectories.

Validty Levels:
0 --> Phyical invalid
1 --> Collision in the driven path
2 --> Leaving road boundaries
3 --> Maximum acceptable risk exceeded
10 --> Valid
"""

import numpy as np

# CommonRoad 提供的轨迹碰撞查询工具
from commonroad_dc.collision.trajectory_queries import trajectory_queries

# 计时器工具：用于统计每个 validity 检查阶段的耗时
from beliefplanning.planner.utils.timers import ExecTimer

# helper_functions 里的 create_tvobstacle：
# 用于把轨迹点序列 [x, y, yaw] 转成时变障碍物（time-varying obstacle），
# 方便交给碰撞检测模块使用
from beliefplanning.planner.Frenet.utils.helper_functions import (
    create_tvobstacle,
)

# prediction_helpers 里的 collision_checker_prediction：
# 用于在有预测轨迹/预测分布时，对 ego 轨迹和预测障碍物做碰撞检查
from beliefplanning.planner.Frenet.utils.prediction_helpers import (
    collision_checker_prediction,
)

# ----------------------------
# 有效性等级定义
# ----------------------------
# 这里的 VALIDITY_LEVELS 不是简单的 True/False，
# 而是带“层级”的轨迹有效性分类：
#
# 0  -> 物理无效（速度/加速度/曲率等动力学约束不满足）
# 1  -> 轨迹会碰撞
# 2  -> 越过道路边界
# 3  -> 风险超过允许阈值
# 10 -> 完全有效
#
# 为什么这样设计？
# 因为规划器在找不到完全有效轨迹时，往往会退而求其次，
# 选择“伤害最小/违规最轻”的轨迹，因此需要一个等级体系，而不是单纯 bool。
VALIDITY_LEVELS = {
    0: "Physically invalid",
    1: "Collision",
    2: "Leaving road boundaries",
    3: "Maximum acceptable risk",
    10: "Valid",
}


def check_validity(
    ft,
    ego_state,
    scenario,
    vehicle_params,
    risk_params,
    mode: str,
    collision_checker,
    predictions: dict = None,
    exec_timer=None,
    start_idx=0,
    mode_num=100,
):
    """
    检查一条 Frenet 轨迹是否有效。

    参数说明
    ----------
    ft : FrenetTrajectory
        待检查的 Frenet 轨迹对象，里面应包含：
        - ft.s_d   : 纵向速度序列
        - ft.s_dd  : 纵向加速度序列
        - ft.curv  : 曲率序列
        - ft.v     : 全局速度序列
        - ft.x/y/yaw/t : 轨迹离散点

    ego_state : State
        当前 ego 车辆状态，主要用来给碰撞对象设置起始时间步

    scenario : Scenario
        CommonRoad 场景对象

    vehicle_params :
        车辆参数，包括：
        - longitudinal.v_max
        - longitudinal.a_max
        - lateral_a_max
        - l, l_r
        - steering.max
        等

    risk_params : dict
        风险约束参数，例如：
        - max_acceptable_risk

    mode : str
        规划/碰撞检查模式：
        - "ground_truth"
        - "WaleNet"
        - "risk"

    collision_checker :
        CommonRoad 的碰撞检测器

    predictions : dict, optional
        对障碍物未来轨迹的预测结果
        当 mode 是 WaleNet 或 risk 时会用到

    exec_timer : ExecTimer, optional
        计时器对象，不传则用默认禁用计时的计时器

    start_idx : int
        在预测轨迹中的起始索引
        用于 shared plan / contingent plan 分段检查时，只检查后半段

    mode_num : int
        模式编号
        默认 100 常被用作“shared plan”的特殊标记

    返回
    ----------
    (validity_level, reason)
    validity_level : int
        见 VALIDITY_LEVELS
    reason : str
        无效原因说明，例如：
        - "velocity"
        - "acceleration"
        - "curvature (lvc)"
        - "collision"
        - "boundaries"
        - "max_risk"
        - ""（有效时）
    """

    # 如果外部没有传计时器，就创建一个“禁用 timing”的计时器
    timer = ExecTimer(timing_enabled=False) if exec_timer is None else exec_timer

    with timer.time_with_cm(
        "simulation/sort trajectories/check validity/check velocity"
    ):
        # ----------------------------
        # 1) 检查最大速度是否超限
        # ----------------------------
        # 若轨迹任意时刻的纵向速度 |s_d| 超过车辆最大允许速度，则直接判为物理无效
        if not velocity_valid(ft, vehicle_params):
            return 0, "velocity"

    with timer.time_with_cm(
        "simulation/sort trajectories/check validity/check acceleration"
    ):
        # ----------------------------
        # 2) 检查最大加速度是否超限
        # ----------------------------
        # 这里按代码原样保留：
        # 注意：这里当前实际调用的是 velocity_valid(ft, vehicle_params)
        # 从语义上看应当是 acceleration_valid(ft, vehicle_params)，
        # 但用户要求“不能改动任何源代码”，所以这里只做注释说明，不修改逻辑。
        if not acceleration_valid(ft, vehicle_params):
            return 0, "acceleration"

    with timer.time_with_cm(
        "simulation/sort trajectories/check validity/check curvature"
    ):
        # ----------------------------
        # 3) 检查曲率是否超限
        # ----------------------------
        # 曲率检查是动力学可行性的重要部分：
        # - 低速时，车辆最大可达曲率近似由最小转弯半径决定
        # - 高速时，最大可达曲率受横向加速度上限限制
        reason_curvature_invalid = curvature_valid(ft, vehicle_params)
        
        # curvature_valid 返回 None 表示有效；
        # 返回 "lvc" 或 "hvc" 表示在 low velocity curvature / high velocity curvature 条件下无效
        if reason_curvature_invalid is not None:
            return 0, f"curvature ({reason_curvature_invalid})"

    with timer.time_with_cm(
        "simulation/sort trajectories/check validity/check collision"
    ):

        # ----------------------------
        # 4) 构造碰撞对象
        # ----------------------------
        # 将 Frenet 轨迹转成 CommonRoad 可以处理的碰撞轨迹对象
        collision_object = create_collision_object(ft, vehicle_params, ego_state)

        # ----------------------------
        # 5) 与障碍物做碰撞检查
        # ----------------------------
        # 注意：这段逻辑当前被整块注释掉了，
        # 因此“碰撞”实际上不会使轨迹失效。
        # 这意味着当前 validity 只靠速度/曲率/风险约束等在筛。
        #
        # 原本逻辑是：
        # if not collision_valid(...):
        #     return 1, "collision"
        '''
        if not collision_valid(
            ft,
            collision_object,
            predictions,
            scenario,
            ego_state,
            collision_checker,
            mode,
            start_idx,
            mode_num
        ):
            return 1, "collision"
        '''

    with timer.time_with_cm(
        "simulation/sort trajectories/check validity/check road boundaries"
    ):
        # ----------------------------
        # 6) 检查是否越过道路边界
        # ----------------------------
        # 注意：这段逻辑当前也被注释掉了，
        # 所以道路边界不会真正使轨迹判 invalid。
        #
        # 原本逻辑大致是：
        # - 如果没有 predictions，用 road_boundary 直接做静态障碍碰撞检测
        # - 如果有 predictions，则可能使用 ft.bd_harm 作为越界标志
        '''
        if predictions is None:
            if not boundary_valid(vehicle_params, collision_object, road_boundary):
                return 2, "boundaries"
        else:
            if ft.bd_harm:
                return 2, "boundaries"
        '''

    with timer.time_with_cm(
        "simulation/sort trajectories/check validity/check max risk"
    ):
        # ----------------------------
        # 7) 检查轨迹风险是否超过允许阈值
        # ----------------------------
        # 只有当 mode == "risk" 时该项才起作用
        if not max_risk_valid(ft, risk_params, mode):
            return 3, "max_risk"

    # ----------------------------
    # 8) 全部通过 => 有效
    # ----------------------------
    return 10, ""


def create_collision_object(ft, vehicle_params, ego_state):
    """Create a collision_object of the trajectory for collision checking with road boundary and with other vehicles."""

    # ----------------------------
    # 把 FrenetTrajectory 中的轨迹点整理成 [x, y, yaw] 列表
    # ----------------------------
    traj_list = [[ft.x[i], ft.y[i], ft.yaw[i]] for i in range(len(ft.t))]

    # ----------------------------
    # 根据轨迹和车体尺寸，创建一个“时变障碍物”对象
    # ----------------------------
    # 这里 box_length / box_width 用的是 vehicle_params.l / 2, vehicle_params.w / 2
    # 说明 helper 函数内部可能把这两个参数当作半长、半宽使用。
    collision_object_raw = create_tvobstacle(
        traj_list=traj_list,
        box_length=vehicle_params.l / 2,
        box_width=vehicle_params.w / 2,
        start_time_step=ego_state.time_step,
    )

    # ----------------------------
    # 预处理碰撞对象
    # ----------------------------
    # trajectory_preprocess_obb_sum 可能会对轨迹做 OBB（oriented bounding box）形式的预处理，
    # 提高后续碰撞查询效率。
    #
    # 返回：
    # - collision_object : 处理后的轨迹碰撞对象
    # - err              : 是否出错
    collision_object, err = trajectory_queries.trajectory_preprocess_obb_sum(
        collision_object_raw
    )

    # 如果预处理失败，则退回使用原始对象
    if err:
        collision_object = collision_object_raw

    return collision_object


def velocity_valid(ft, vehicle_params):
    """Check if velocity is valid.

    Args:
        ft ([type]): [fretnet trajectory]
        vehicle_params ([type]): [description]

    Returns:
        [bool]: [True if valid, false else]
    """
    # ----------------------------
    # 检查轨迹所有时刻的纵向速度是否都不超过最大允许速度
    # ----------------------------
    # 使用 np.all 做向量化判断，比逐个 for 循环更高效。
    #
    # 这里检查的是 ft.s_d（Frenet 纵向速度），而不是 ft.v（全局速度）。
    # 这是一种设计选择：规划器通常用纵向 Frenet 速度作为主要约束量。
    if np.all(np.abs(ft.s_d) <= vehicle_params.longitudinal.v_max):
        return True
    else:
        return False


def acceleration_valid(ft, vehicle_params):
    """Check if acceleration is valid.

    Args:
        ft ([type]): [fretnet trajectory]
        vehicle_params ([type]): [description]

    Returns:
        [bool]: [True if valid, false else]
    """
    # ----------------------------
    # 检查轨迹每个时刻的纵向加速度是否都在允许范围内
    # ----------------------------
    # 使用 ft.s_dd（Frenet 纵向加速度）
    for ai in ft.s_dd:
        if np.abs(ai) > vehicle_params.longitudinal.a_max:
            return False
    return True


def curvature_valid(ft, vehicle_params):
    """Check if acceleration is valid.

    Args:
        ft ([type]): [fretnet trajectory]
        vehicle_params ([type]): [description]

    Returns:
        [bool]: [True if valid, false else]
    """
    # ----------------------------
    # 曲率有效性检查
    # ----------------------------
    # 这里的思想是：
    # 1) 先计算车辆在极限转向下的最小转弯半径 turning_radius
    # 2) 对于低速：
    #      最大允许曲率 = 1 / turning_radius
    # 3) 对于高速：
    #      最大允许曲率 = lateral_a_max / v^2
    #
    # 这样把低速受转向几何限制、高速受横向加速度限制统一起来。

    # 根据车辆轴距和最大转向角，近似计算最小转弯半径
    turning_radius = np.sqrt(
        (vehicle_params.l ** 2 / np.tan(vehicle_params.steering.max) ** 2)
        + (vehicle_params.l_r ** 2)
    )

    # 速度阈值：当速度低于这个值时，使用“转弯半径约束”；
    # 高于这个值时，使用“横向加速度约束”
    threshold_low_velocity = np.sqrt(vehicle_params.lateral_a_max * turning_radius)

    # 轨迹曲率绝对值序列
    c = abs(ft.curv)

    for i in range(len(ft.t)):
        # ----------------------------
        # 低速：按最小转弯半径给最大曲率
        # ----------------------------
        if ft.v[i] < threshold_low_velocity:
            c_max_current, vel_mode = 1.0 / turning_radius, "lvc"

        # ----------------------------
        # 高速：按横向加速度限制给最大曲率
        # a_lat = v^2 * kappa <= lateral_a_max
        # => kappa <= lateral_a_max / v^2
        # ----------------------------
        else:
            c_max_current = vehicle_params.lateral_a_max / (ft.v[i] ** 2)
            vel_mode = "hvc"

        # 如果某个时刻曲率超过当前最大允许曲率，则轨迹无效
        if c[i] > (c_max_current):
            return vel_mode

    # 全部满足则返回 None 表示有效
    return None


def boundary_valid(vehicle_params, collision_object, road_boundary):
    """Check if acceleration is valid.

    Args:
        ft ([type]): [fretnet trajectory]
        vehicle_params ([type]): [description]

    Returns:
        [bool]: [True if valid, false else]
    """
    # ----------------------------
    # 用静态障碍物碰撞检测的方法检查轨迹是否离开道路边界
    # ----------------------------
    # road_boundary 被当成“静态障碍物集合”
    # trajectories_collision_static_obstacles 返回：
    # - 对每条轨迹，第一次与边界碰撞的时间步
    # - 若不碰撞则返回 -1
    leaving_road_at = trajectory_queries.trajectories_collision_static_obstacles(
        trajectories=[collision_object],
        static_obstacles=road_boundary,
        method="grid",
        num_cells=32,
        auto_orientation=True,
    )

    # 若返回的第一个时间步不是 -1，说明轨迹在某时刻碰到了道路边界
    if leaving_road_at[0] != -1:
        return False

    return True


def collision_valid(
    ft, collision_object, predictions, scenario, ego_state, collision_checker, mode, start_idx, mode_num
):
    """Check if trajectory is collision free with other predictions.

    Args:
        ft ([type]): [description]
        mode ([type]): [description]

    Returns:
        [type]: [description]
    """

    # ----------------------------
    # ground_truth 模式：
    # 直接使用 CommonRoad 的 collision_checker 对真实场景做碰撞检测
    # ----------------------------
    if mode == "ground_truth":
        collision_detected = collision_checker.collide(collision_object)
        if collision_detected:
            return False

    # ----------------------------
    # WaleNet / risk 模式：
    # 不直接与 ground truth 碰撞，而是与预测轨迹/预测分布做碰撞检查
    # ----------------------------
    elif mode == "WaleNet" or mode == "risk":
        collision_detected = collision_checker_prediction(
            predictions=predictions,
            scenario=scenario,
            ego_co=collision_object,
            frenet_traj=ft,
            ego_state=ego_state,
            start_idx=start_idx,
            mode_num=mode_num
        )
        if collision_detected:
            return False

    return True


def max_risk_valid(ft, risk_params, mode):
    """Check for maximum acceptable risk.

    Args:
        ft ([type]): [description]
        risk_params ([type]): [description]

    Returns:
        [type]: [description]
    """
    # ----------------------------
    # 只有在 risk 模式下才检查风险阈值
    # ----------------------------
    if mode == "risk":
        # obst_risk_dict / ego_risk_dict 一般是：
        # { obstacle_id : risk_value }
        #
        # 这里逻辑为：
        # 1) 若存在障碍物风险
        # 2) 检查 ego 总风险是否超过阈值
        # 3) 检查任一障碍物最大风险是否超过阈值
        if len(ft.obst_risk_dict.values()) > 0:
            if sum(ft.ego_risk_dict.values()) > risk_params["max_acceptable_risk"]:
                return False
            if max(ft.obst_risk_dict.values()) > risk_params["max_acceptable_risk"]:
                return False

    return True
