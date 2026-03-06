"""
This file contains relevant functions for the Frenet planner.

【整体用途】
- 这是一个 Frenet(弗雷内)轨迹采样与评估模块的核心实现之一:
  1) 给定参考线(cubic spline / global path),以及当前状态(s, d 及其导数)
  2) 在不同的目标时域 tT、目标速度 vT、目标横向偏移 dT 上采样生成候选 Frenet 轨迹
  3) 将 Frenet 轨迹转换为全局坐标(x, y, yaw, curvature, v)
  4) 对轨迹做可行性/安全性检查(validity)
  5) 对可行轨迹计算代价(cost),按代价排序
  6) 也支持风险评估(calc_risk)并在 validity/cost 中使用

【重要概念回顾】
- Frenet 坐标:沿参考线的弧长方向 s(纵向) + 垂向偏移 d(横向)
- 纵向多用 quartic(4次)多项式:给定初始位置/速度/加速度 + 目标速度 + 目标加速度(通常为0),得到 s(t)
- 横向多用 quintic(5次)多项式:给定初始 d、d_dot、d_ddot + 目标 dT、0、0,得到 d(t) 或 d(s)

【速度模式】
- 高速:横向多项式按时间 t 生成 d(t),然后转换为 d'(s), d''(s)
- 低速:横向多项式按弧长 s 生成 d(s),再换算回时间域的 d_dot(t), d_ddot(t)
"""

# Standard imports
import os
import sys
import math

# Third party imports
import numpy as np
from scipy.stats import beta
# CommonRoad 的 2D cubic spline，用来表示参考路径(global path)
from commonroad_helper_functions.utils.cubicspline import CubicSpline2D
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.scenario import Scenario
import concurrent.futures
from functools import partial

# Custom imports
module_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(module_path)

# 纵向 quartic / 横向 quintic 多项式工具
from planner.Frenet.utils.polynomials import quartic_polynomial, quintic_polynomial
# 轨迹有效性检查(碰撞、动力学约束、道路边界等)
from planner.Frenet.utils.validity_checks import VALIDITY_LEVELS, check_validity
# 代价函数 + 距离计算
from planner.Frenet.utils.calc_trajectory_cost import (
    calc_trajectory_costs,
    distance,
)
# 计时器(统计每个阶段耗时)
from planner.utils.timers import ExecTimer
# 最大曲率工具:通常基于车辆参数、速度、轮胎侧向极限等推一个允许曲率上限
from planner.Frenet.utils.helper_functions import get_max_curvature
# 风险评估:根据预测、碰撞概率、伤害模型等给 ego/obstacle 风险字典
from beliefplanning.risk_assessment.risk_costs import calc_risk


class ContingentTrajectory:
    """
    【用途】
    - 用于“应急/分支”轨迹(contingency planning)存储。
    - shared_plan:共享部分(比如前一段一致执行)
    - opt_plan1 / opt_plan2:两个分支方案(例如:对方车不同意图时的备选轨迹)
    """
    def __init__(self):
        self.shared_plan = []
        self.opt_plan1 = []
        self.opt_plan2 = []


class FrenetTrajectory:
    """
    Trajectory in frenet Coordinates with longitudinal and lateral position and up to 3rd derivative.
    It also includes the global pose and curvature.

    【字段说明(很关键，后面 validity/cost/risk 都依赖这些)】
    - t:时间序列
    - s, s_d, s_dd, s_ddd:沿参考线弧长方向位置/速度/加速度/jerk(通常由 quartic 多项式生成)
    - d, d_d, d_dd, d_ddd:横向偏移及其导数(注意:这里 d_d/d_dd 有时是时间域，有时是弧长域，代码里会区分)
    - x, y, yaw:转换到全局坐标后的轨迹
    - v:全局速度(由 s_d 和几何关系转换得到，不一定等于 s_d)
    - curv:全局曲率
    - d_dcurv / d_ddcurv:看命名像“用于曲率计算的 d'(s), d''(s)”之类的保存(供后续用)
    - valid_level / reason_invalid:有效性检查结果
    - cost / cost_dict:代价值以及分项字典
    - ego_risk_dict / obst_risk_dict:风险评估结果(字典或列表)
    """
    def __init__(
            self,
            t: [float] = None,
            d: [float] = None,
            d_d: [float] = None,
            d_dd: [float] = None,
            d_ddd: [float] = None,
            s: [float] = None,
            s_d: [float] = None,
            s_dd: [float] = None,
            s_ddd: [float] = None,
            x: [float] = None,
            y: [float] = None,
            yaw: [float] = None,
            v: [float] = None,
            curv: [float] = None,
            d_dcurv: [float] = None,
            d_ddcurv: [float] = None,
    ):
        """
        Initialize a frenét trajectory.

        Args:
            t ([float]): List for the time. Defaults to None.
            d ([float]): List for the lateral offset. Defaults to None.
            d_d: ([float]): List for the lateral velocity. Defaults to None.
            d_dd ([float]): List for the lateral acceleration. Defaults to None.
            d_ddd ([float]): List for the lateral jerk. Defaults to None.
            s ([float]): List for the covered arc length of the spline. Defaults to None.
            s_d ([float]): List for the longitudinal velocity. Defaults to None.
            s_dd ([float]): List for the longitudinal acceleration. Defaults to None.
            s_ddd ([float]): List for the longitudinal jerk. Defaults to None.
            x ([float]): List for the x-position. Defaults to None.
            y ([float]): List for the y-position. Defaults to None.
            yaw ([float]): List for the yaw angle. Defaults to None.
            v([float]): List for the velocity. Defaults to None.
            curv ([float]): List for the curvature. Defaults to None.
        """
        # time vector
        # ===== 时间 =====
        self.t = t

        # frenet coordinates
        # ===== Frenet 坐标（横向/纵向） =====
        self.d = d
        self.d_d = d_d
        self.d_dd = d_dd
        self.d_ddd = d_ddd
        self.s = s
        self.s_d = s_d
        self.s_dd = s_dd
        self.s_ddd = s_ddd

        # Global coordinates
        # ===== 全局坐标 =====
        self.x = x
        self.y = y
        self.yaw = yaw
        # Velocity
        self.v = v
        # Curvature
        self.curv = curv

        # 这里看起来用于保存“曲率计算所需的 d'(s), d''(s)”
        # NOTE: 你后面有时候会把 d_dcurv 塞 d'(s)，也会塞 d_dot(t)；命名略混乱，建议统一。
        self.d_dcurv = d_dcurv
        self.d_ddcurv = d_ddcurv

        # Validity
        # ===== 有效性（validity） =====
        # valid_level：离散等级（越大越“有效”），reason_invalid：无效原因（字符串/结构）
        self.valid_level = 0
        self.reason_invalid = None

        # Cost
        self.cost = 0

        # Risk
        self.ego_risk_dict = []
        self.obst_risk_dict = []


def check_curvature_of_global_path(
        global_path: np.ndarray, planning_problem, vehicle_params, ego_state
):
    """
    Check the curvature of the global path.

    If the curvature is to high, points of the global path are removed to smooth the global path. In addition, a new point is added which ensures the initial orientation.

    Args:
        global_path (np.ndarray): Coordinates of the global path.

    Returns:
        np.ndarray: Coordinates of the new, smooth global path.

    """
    """
    Check the curvature of the global path.

    【作用】
    - 你拿到的 global_path 是离散点，可能拐得太急（局部曲率过大）。
    - 这里通过“删除某些点”来让离散路径更平滑，从而降低曲率。
    - 同时插入一个点确保起始朝向（orientation）一致。

    【为什么要这样做】
    - Frenet 规划器通常依赖参考线的平滑性（曲率/曲率导数会被用在坐标变换与可行性检查）。
    - 起点附近若曲率突变，会导致 yaw/curv 计算数值不稳，甚至让轨迹生成直接“爆炸”。

    Args:
        global_path (np.ndarray): 形如 (N,2) 的全局路径点序列
        planning_problem: CommonRoad 规划问题（取初速度）
        vehicle_params: 车辆参数（用于推可用最大曲率）
        ego_state: 当前 ego 状态（位置、朝向用于插入点）

    Returns:
        np.ndarray: 平滑后的 global_path
    """
    global_path_curvature_ok = False

    # get start velocity of the planning problem
    # 起始速度：用于根据速度推最大允许曲率（速度越高可用曲率越小）
    start_velocity = planning_problem.initial_state.velocity

    # calc max curvature for the initial velocity
    # 初始速度下的最大曲率（车辆可实现的最大转弯能力）
    max_initial_curvature, _ = get_max_curvature(
        vehicle_params=vehicle_params, v=start_velocity
    )

    # get x and y from the global path
    # 拆出 x,y 便于操作（list 便于 pop/insert）
    global_path_x = global_path[:, 0].tolist()
    global_path_y = global_path[:, 1].tolist()

    # add a point to the global path to ensure the initial orientation of the planning problem
    # never delete this point or the initial point
    # 插入一个“第二个点”以强制起始方向（不允许删掉）
    # 该点位于 ego 朝向方向上 0.1m，目的是让离散路径在起点处方向与 ego orientation 一致
    # NOTE: 这个点很关键：后面删除点时明确不会删它，同时它也为后续的曲率检查提供了一个“稳定的起点”，避免起始处曲率过大导致数值不稳。
    new_x = ego_state.position[0] + np.cos(ego_state.orientation) * 0.1
    new_y = ego_state.position[1] + np.sin(ego_state.orientation) * 0.1
    global_path_x.insert(1, new_x)
    global_path_y.insert(1, new_y)

    # check if the curvature of the global path is ok
    # 反复检查并删除过大曲率点，直到满足条件
    while global_path_curvature_ok is False:
        # calc the already covered arc length for the points of global path
        # 计算每个离散点对应的累计弧长 s（用来做 gradient 的自变量）
        global_path_s = [0.0]

        for i in range(len(global_path_x) - 1):
            p_start = np.array([global_path_x[i], global_path_y[i]])
            p_end = np.array([global_path_x[i + 1], global_path_y[i + 1]])
            global_path_s.append(distance(p_start, p_end) + global_path_s[-1])

        # calculate the curvature of the global path
        # 用弧长作为“步长”计算一阶/二阶导，得到曲率
        # dx/ds, dy/ds
        dx = np.gradient(global_path_x, global_path_s)
        dy = np.gradient(global_path_y, global_path_s)
        # d2x/ds2, d2y/ds2
        ddx = np.gradient(dx, global_path_s)
        ddy = np.gradient(dy, global_path_s)
        # 曲率公式：|x' y'' - y' x''| / (x'^2 + y'^2)^(3/2)
        # NOTE: 这里对曲率取 abs，表示“转弯程度”不关心左右，只要过大就删点。后续在 Frenet 轨迹生成时会根据左右转弯分别计算 yaw_diff。
        curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5

        # loop through every curvature of the global path
        # 第一轮：按“初速度允许曲率”检查（并 *2 放宽，因为 spline 转换后可能更大）
        global_path_curvature_ok = True
        for i in range(len(curvature)):
            # check if the curvature of the global path is too big
            # be generous (* 2.) since the curvature might increase again when converting to a cubic spline
            # *2 放宽：经验上离散点->cubic spline 后曲率可能增大，所以这里先放宽一些，如果后续生成的 spline 仍然过大，后续的检查会再删点。
            if (curvature[i] * 2.0) > max_initial_curvature:
                # if the curvature is too big, then delete the global path point to smooth the global path
                # never remove the first (starting) point of the global path
                # and never remove the second point of the global path to keep the initial orientation
                # 要删除的点索引：至少从 2 开始（保护第0点和第1点：起点+朝向点）
                index_closest_path_point = max(2, i)
                # only consider the first part of the global path, later on it gets smoothed by the frenét planner
                # itself
                # 只处理起点附近 10m 内：后面 Frenet 自己会平滑更远处
                if global_path_s[index_closest_path_point] <= 10.0:
                    global_path_x.pop(index_closest_path_point)
                    global_path_y.pop(index_closest_path_point)
                    global_path_curvature_ok = False
                    break

        # also check if the curvature is smaller than the turning radius anywhere
        # 第二轮：更严格/更物理的检查——任何地方不能超过 v=0 时的最大曲率（最小转弯半径）
        # NOTE: v=0 时能达到最大曲率（极限转向），若还超出，说明几何本身不可能实现，必须删点。
        for i in range(len(curvature)):
            # check if the curvature of the global path is too big
            # be generous (* 2.) since the curvature might increase again when converting to a cubic spline
            if (curvature[i] * 2.0) > get_max_curvature(
                    vehicle_params=vehicle_params, v=0.0
            )[0]:
                # if the curvature is too big, then delete the global path point to smooth the global path
                # never remove the first (starting) point of the global path
                # and never remove the second point of the global path to keep the initial orientation
                index_closest_path_point = max(2, i)
                # only consider the first part of the global path, later on it gets smoothed by the frenét planner
                # itself
                global_path_x.pop(index_closest_path_point)
                global_path_y.pop(index_closest_path_point)
                global_path_curvature_ok = False
                break
        # NOTE: 潜在风险
        # - 如果 global_path 过短或曲率处处过大，可能 pop 到只剩很少点，甚至导致 gradient 不稳定
        # - 建议加最小点数保护（例如 < 4 就 break/raise）

    # create the new global path
    # 重新拼回 (N,2) ndarray
    new_global_path = np.array([np.array([global_path_x[0], global_path_y[0]])])
    for i in range(1, len(global_path_y)):
        new_global_path = np.concatenate(
            (
                new_global_path,
                np.array([np.array([global_path_x[i], global_path_y[i]])]),
            )
        )

    return new_global_path


def calc_shared_plan(
        c_s: float,
        c_s_d: float,
        c_s_dd: float,
        c_d: float,
        c_d_d: float,
        c_d_dd: float,
        d_list: [float],
        t_list: [float],
        v_list: [float],
        dt: float,
        csp: CubicSpline2D,
        v_thr: float = 3.0,
        exec_timer=None,
        v_min: float = 0,
        v_max: float = 50,
        v_cur: float = 10,
        v_goal_min: float = None,
        v_goal_max: float = None,
        n_samples: int = 3,
        mode: str = "linspace",
):
    """
    【用途】
    - 看起来是“共享计划”生成入口：内部直接调用 calc_frenet_trajectories
    - 主要就是参数透传 + 命名更语义化（shared_plan）

    Args:
        c_s, c_s_d, c_s_dd: 当前纵向状态 s, ds/dt, d2s/dt2
        c_d, c_d_d, c_d_dd: 当前横向状态 d, dd/dt, d2d/dt2
        d_list, t_list, v_list: 横向目标、时域目标、速度目标的采样集合
        dt: 离散时间步长
        csp: 参考线 cubic spline
        v_thr: 速度阈值：区分“低速模式/高速模式”
        其余 v_min/v_max/v_cur/v_goal_* 这里并没被直接用（只是传下去）
    """
    ft_shared_plan = calc_frenet_trajectories(c_s=c_s,
                                              c_s_d=c_s_d,
                                              c_s_dd=c_s_dd,
                                              c_d=c_d,
                                              c_d_d=c_d_d,
                                              c_d_dd=c_d_dd,
                                              d_list=d_list,
                                              t_list=t_list,
                                              v_list=v_list,
                                              dt=dt,
                                              csp=csp,
                                              v_thr=v_thr,
                                              exec_timer=exec_timer,
                                              v_min=v_min,
                                              v_max=v_max,
                                              v_cur=v_cur,
                                              v_goal_min=v_goal_min,
                                              v_goal_max=v_goal_max,
                                              mode=mode,
                                              n_samples=n_samples,
                                              )
    return ft_shared_plan


def calc_frenet_trajectories(
        c_s: float,
        c_s_d: float,
        c_s_dd: float,
        c_d: float,
        c_d_d: float,
        c_d_dd: float,
        d_list: [float],
        t_list: [float],
        v_list: [float],
        dt: float,
        csp: CubicSpline2D,
        v_thr: float = 3.0,
        exec_timer=None,
        t_min: float = 0,
        t_max: float = 4,
        max_acceleration: float = 2,
        max_velocity: float = 50,
        v_goal_min: float = None,
        v_goal_max: float = None,
        n_samples: int = 3,
        mode: str = "linspace",
        contin=False
):
    """
    Calculate all possible frenet trajectories.

    【核心功能】
    - 三层循环采样（vT x tT x dT），对每个组合：
      1) 纵向用 quartic：给出 s(t), s_dot, s_ddot, s_ddd
      2) 计算参考线在 s(t) 的位置与几何导数（dx/ds, dy/ds, curvature, curvature'）
      3) 横向用 quintic：
         - 高速：d(t)，再换成 d'(s), d''(s) 参与坐标变换
         - 低速：d(s)，再换算 d_dot(t), d_ddot(t) 存入轨迹
      4) 用 Frenet->global 变换得到 x,y,yaw,v,curv
      5) 打包成 FrenetTrajectory 放入列表

    【参数注意】
    - t_min/t_max/max_acceleration/max_velocity 这里虽然作为参数出现，但你当前实现没用它们做过滤
      NOTE: 这可能是“规划器外层”过滤的，也可能是遗留参数。建议明确：不用就删，用就加约束。
    - contin：看起来想区分“连续/应急”轨迹，但这里两个分支创建对象完全一样（重复代码）
      NOTE: 你可以删掉 if contin 分支，或者让 contin 真正改变某些字段。
    """

    """
    Calculate all possible frenet trajectories from a given starting point and target lateral deviations, times and velocities.

    Args:
        c_s (float): Start longitudinal position.
        c_s_d (float): Start longitudinal velocity.
        c_s_dd (float): Start longitudinal acceleration
        c_d (float): Start lateral position.
        c_d_d (float): Start lateral velocity.
        c_d_dd (float): Start lateral acceleration.
        d_list ([float]): List of target lateral offsets to the reference spline.
        t_list ([float]): List of target end-times.
        v_list ([float]): List of target end-velocities.
        dt (float): Time step size of the trajectories.
        csp (CubicSpline2D): Reference spline of the global path.
        v_thr (float): Threshold velocity to distinguish slow and fast trajectories.
        exec_times_dict (dict): Dictionary for execution times. Defaults to None.

    Returns:
        [FrenetTrajectory]: List with all frenét trajectories.
        :param t_min:
        :param mode:
        :param n_samples:
        :param v_goal_max:
        :param v_goal_min:
        :param v_cur:
        :param v_max:
        :param dt:
        :param v_list:
        :param t_list:
        :param d_list:
        :param c_d_dd:
        :param c_d_d:
        :param c_d:
        :param c_s_dd:
        :param c_s_d:
        :param c_s:
        :param csp:
        :param v_thr:
        :param exec_timer:
        :param v_min:
    """
    # 如果没传计时器，就创建一个“禁用计时”的默认计时器
    timer = ExecTimer(timing_enabled=False) if exec_timer is None else exec_timer
    # list of all generated frenet trajectories
    # 保存所有生成的候选轨迹
    fp_list = []

    number_of_shared_trajectories = 0
    contingent_plans_numbers_list = []

    # ========== 第一层：遍历目标末速度 vT ==========
    # all end velocities
    for vT in v_list:
        # 速度阈值：低速时用“横向关于 s 的多项式”更稳；高速用“横向关于 t 的多项式”更自然
        if abs(c_s_d) < v_thr or abs(vT) < v_thr:
            lat_mode = "low_velocity"
        else:
            lat_mode = "high_velocity"
        # all end times
        # ========== 第二层：遍历目标时域 tT ==========
        for tT in t_list:
            # quartic polynomial in longitudinal direction
            # ---------- 纵向 quartic ----------
            # 纵向：给定初始 s, s_dot, s_ddot；目标速度 vT；目标加速度=0；时长 T=tT-dt
            # NOTE: 这里用 T=tT-dt 而不是 tT，通常是为了让 t=[0,tT) 的最后一个点对应多项式末端
            with timer.time_with_cm(
                    "simulation/calculate trajectories/initialize quartic polynomial"
            ): 
                # 用一个四次多项式（quartic polynomial）在“纵向 Frenet 坐标 s”上生成一条速度/位移随时间变化的轨迹，
                # 满足“起点状态 + 终点速度约束”。 五个边界条件分别是：初始位置 c_s、初始速度 c_s_d、初始加速度 c_s_dd、目标速度 vT、目标加速度 0。
                # 生成的多项式对象 qp_long 可以计算任意时间点的 s(t)、s_dot(t)、s_ddot(t)、s_ddd(t)。
                # NOTE: 这里用 T=tT-dt 而不是 tT，通常是为了让 t=[0,tT) 的最后一个点对应多项式末端
                qp_long = quartic_polynomial(
                    xs=c_s, vxs=c_s_d, axs=c_s_dd, vxe=vT, axe=0.0, T=tT - dt
                )
                # qp_long.print_polynomial()
            with timer.time_with_cm(
                    "simulation/calculate trajectories/calculate quartic polynomial"
            ):
                # use universal function feature to perform array operation
                # time vector
                # 时间采样：np.arange(0, tT, dt) -> [0, dt, ..., tT-dt]
                t = np.arange(0.0, tT, dt)
                # longitudinal position and derivatives
                # 纵向状态
                s = qp_long.calc_point(t)
                s_d = qp_long.calc_first_derivative(t)
                s_dd = qp_long.calc_second_derivative(t)
                s_ddd = qp_long.calc_third_derivative(t)

            # ---------- 参考线位置 ----------
            with timer.time_with_cm(
                    "simulation/calculate trajectories/calculate global trajectory/calculate reference points"
            ):
                # use CubicSpineLine's internal function to perform array operation
                # calculate the position of the reference path
                # 参考线在弧长 s 处的全局位置（向量化调用）
                global_path_x, global_path_y = csp.calc_position(s)

            # move gradient calculation and deviation calculation out from calc_global_trajectory(),
            # avoid unnecessary repeat calculation
            # ---------- 参考线导数（用于 yaw / curvature） ----------
            with timer.time_with_cm(
                    "simulation/calculate trajectories/calculate global trajectory/calculate reference gradients"
            ):
                # calculate derivations necessary to get the curvature
                # 用 s 作为自变量求导：dx/ds 等
                # NOTE: 如果 s 有重复/非单调（例如 s_d 很小还可能回头？），gradient 可能数值不稳
                dx = np.gradient(global_path_x, s)
                ddx = np.gradient(dx, s)
                dddx = np.gradient(ddx, s)
                dy = np.gradient(global_path_y, s)
                ddy = np.gradient(dy, s)
                dddy = np.gradient(ddy, s)

            with timer.time_with_cm(
                    "simulation/calculate trajectories/calculate global trajectory/calculate reference yaw"
            ):
                # calculate yaw of the global path
                # 参考线切向角（全局 yaw）
                global_path_yaw = np.arctan2(dy, dx)

            with timer.time_with_cm(
                    "simulation/calculate trajectories/calculate global trajectory/calculate reference curvature"
            ):
                # calculate the curvature of the global path
                # 参考线曲率 κ(s)
                # κ = (x' y'' - x'' y') / (x'^2 + y'^2)^(3/2)
                # NOTE: 你这里写成 (dx*ddy - ddx*dy) / (dx^2 + dy^2)^(3/2) 是正确形式
                global_path_curv = (np.multiply(dx, ddy) - np.multiply(ddx, dy)) / (
                        np.power(dx, 2) + np.power(dy, 2) ** (3 / 2)
                )

            with timer.time_with_cm(
                    "simulation/calculate trajectories/calculate global trajectory/calculate reference curvature "
                    "derivation"
            ):
                # 计算曲率对 s 的导数 κ'(s)
                # 写法来自对 κ = z/n 的求导：κ' = (z' n - z n') / n^2
                # calculate the derivation of the global path's curvature
                z = np.multiply(dx, ddy) - np.multiply(ddx, dy)
                z_d = np.multiply(dx, dddy) - np.multiply(dddx, dy)

                # n = (dx^2 + dy^2)^(3/2)
                n = (np.power(dx, 2) + np.power(dy, 2)) ** (3 / 2)

                # n' 的推导：n = (r)^(3/2), r = dx^2 + dy^2
                # n' = (3/2) * r^(1/2) * r'
                # r' = 2*dx*ddx + 2*dy*ddy
                n_d = (3 / 2) * np.multiply(
                    np.power((np.power(dx, 2) + np.power(dy, 2)), 0.5),
                    (2 * np.multiply(dx, ddx) + 2 * np.multiply(dy, ddy)),
                )
                global_path_curv_d = (np.multiply(z_d, n) - np.multiply(z, n_d)) / (
                    np.power(n, 2)
                )

            # ---------- 为低速模式准备 ds ----------
            s0 = s[0]
            ds = s[-1] - s0 # 纵向总“弧长增量”（近似路程）
            # all lateral distances
            # ========== 第三层：遍历目标横向偏移 dT ==========
            for dT in d_list:
                # quintic polynomial in lateral direction
                # for high velocities we have ds/dt and dd/dt
                
                # if ds <= abs(dT):
                #     continue
    
                # quintic polynomial in lateral direction

                # 你这里有一段被注释掉的过滤：
                # if ds <= abs(dT): continue
                # NOTE: 这类过滤常用来避免“横向偏移比纵向推进还大”导致横向多项式很奇怪
                # 但是直接用 ds<=|dT| 可能过粗暴，因为某些场景（出车道）可能 dT 会大
                # 建议改为：根据横向速度/加速度上限来过滤，而不是几何比例硬切。

                # ---------- 高速：横向按时间生成 d(t) ----------
                if lat_mode == "high_velocity":

                    with timer.time_with_cm(
                            "simulation/calculate trajectories/initialize quintic polynomial"
                    ):  
                        # 横向 quintic：初始 d, d_dot, d_ddot；目标 dT, 0, 0；时长 T=tT-dt
                        qp_lat = quintic_polynomial(
                            xs=c_d,
                            vxs=c_d_d,
                            axs=c_d_dd,
                            xe=dT,
                            vxe=0.0,
                            axe=0.0,
                            T=tT - dt,
                        )

                    with timer.time_with_cm(
                            "simulation/calculate trajectories/calculate quintic polynomial"
                    ):

                        # use universal function feature to perform array operation
                        # lateral distance and derivatives
                        # 横向轨迹（时间域）
                        d = qp_lat.calc_point(t)
                        d_d = qp_lat.calc_first_derivative(t)
                        d_dd = qp_lat.calc_second_derivative(t)
                        d_ddd = qp_lat.calc_third_derivative(t)

                    # 保存时间域的横向速度/加速度（后续要存进 FrenetTrajectory）
                    d_d_time = d_d
                    d_dd_time = d_dd

                    # 接下来要做 Frenet->global 变换，它用到的是 d'(s), d''(s)
                    # 因为公式里是“横向对弧长的导数”，不是对时间的导数
                    # d'(s) = (dd/dt) / (ds/dt) = d_dot / s_dot
                    # d''(s) = (d_ddot - d'(s)*s_ddot) / s_dot^2  （链式法则）
                    # NOTE: 如果 s_d 里有接近 0 的值，会出现除零/数值爆炸，需要保护
                    d_d = d_d / np.where(np.abs(s_d) < 1e-6, 1e-6, s_d)
                    d_dd = (d_dd - d_d * s_dd) / np.power(s_d, 2)

                # for low velocities, we have ds/dt and dd/ds
                # ---------- 低速：横向按弧长生成 d(s) ----------
                elif lat_mode == "low_velocity":
                    # singularity
                    # ds==0 代表纵向几乎没动，d(s) 的定义域会退化导致多项式不好解
                    if ds == 0:
                        ds = 0.00001  # NOTE: 强行给一个很小的 ds，避免除零；更稳的是直接跳过该轨迹

                    with timer.time_with_cm(
                            "simulation/calculate trajectories/initialize quintic polynomial"
                    ):
                        # the quintic polynomial shows dd/ds, so d(c_s)/ds and dd(c_s)/dds is needed
                        # 低速模式：横向多项式的自变量用 s（弧长），因此初始条件也应是对 s 的导数
                        # c_d_d_not_time = dd/ds = (dd/dt) / (ds/dt) = c_d_d / c_s_d
                        # c_d_dd_not_time = d2d/ds2 = (d_ddot - s_ddot*dd/ds) / s_dot^2
                        if c_s_d != 0.0:
                            c_d_d_not_time = c_d_d / c_s_d
                            c_d_dd_not_time = (c_d_dd - c_s_dd * c_d_d_not_time) / (
                                    c_s_d ** 2
                            )
                        else:
                            # NOTE: s_dot=0 时无法换算，直接置0属于启发式；也可以跳过这条轨迹
                            c_d_d_not_time = 0.0
                            c_d_dd_not_time = 0.0

                        # Upper boundary for ds to avoid bad lat polynoms (solved by  if ds > abs(dT)?)
                        # ds = max(ds, 0.1)
                        # 构造 quintic：自变量范围是 ds（总弧长增量）
                        qp_lat = quintic_polynomial(
                            xs=c_d,
                            vxs=c_d_d_not_time,
                            axs=c_d_dd_not_time,
                            xe=dT,
                            vxe=0.0,
                            axe=0.0,
                            T=ds,
                        )

                    with timer.time_with_cm(
                            "simulation/calculate trajectories/calculate quintic polynomial"
                    ):
                        # use universal function feature to perform array operation
                        # lateral distance and derivatives
                        # 这里用 (s - s0) 作为自变量，因为多项式从 0 开始定义，而 s 的起点是 s0
                        d = qp_lat.calc_point(s - s0)
                        d_d = qp_lat.calc_first_derivative(s - s0)     # dd/ds
                        d_dd = qp_lat.calc_second_derivative(s - s0)   # d2d/ds2
                        d_ddd = qp_lat.calc_third_derivative(s - s0)   # d3d/ds3（这里其实是对 s 的三阶导）

                    # since dd/ds, a conversion to dd/dt is needed
                    # 把 dd/ds 换回 dd/dt，便于存入轨迹（时间域横向速度/加速度）
                    # d_dot = s_dot * dd/ds
                    # d_ddot = s_ddot*dd/ds + s_dot^2 * d2d/ds2
                    d_d_time = s_d * d_d
                    d_dd_time = s_dd * d_d + np.power(s_d, 2) * d_dd

                # with timer.time_with_cm(
                #     "simulation/calculate trajectories/calculate global trajectory/total"
                # ):
                # ---------- Frenet -> Global 变换 ----------
                # 这部分把 (s, d, d'(s), d''(s)) + 参考线几何 (x_ref, y_ref, yaw_ref, kappa_ref, kappa'_ref)
                # 转成车辆轨迹 (x, y, yaw, v, curvature)
                with timer.time_with_cm(
                        "simulation/calculate trajectories/calculate global trajectory/calculate trajectory states"
                ):
                    # yaw_diff = atan( d'(s) / (1 - kappa_ref * d) )
                    # 解释：车体相对参考线切线的偏航差，来自 Frenet 坐标变换推导
                    yaw_diff_array = np.arctan(d_d / (1 - global_path_curv * d))

                    # 车辆 yaw = 参考线 yaw + yaw_diff
                    yaw = yaw_diff_array + global_path_yaw
                    # 位置变换：沿法向偏移 d 后的全局坐标
                    x = global_path_x - d * np.sin(global_path_yaw)
                    y = global_path_y + d * np.cos(global_path_yaw)
                    # 真实速度：v = (s_dot * (1 - kappa_ref*d)) / cos(yaw_diff)
                    # NOTE: 当 yaw_diff 接近 +/- 90° 时 cos->0，会爆；通常 validity 会过滤，但最好提前保护一下，例如加个 min_cos = 0.1 来限制 cos(yaw_diff) 的最小值，避免数值不稳。
                    v = (s_d * (1 - global_path_curv * d)) / np.cos(yaw_diff_array)
                    # 曲率变换（标准 Frenet->global 曲率公式）
                    # 这里比较复杂，但结构是：
                    # kappa = [ (d'' + (kappa'_ref*d + kappa_ref*d')*tan(yaw_diff)) * cos^2/(1-kappa_ref*d) + kappa_ref ] * cos/(1-kappa_ref*d)
                    curv = (
                                   (
                                           (
                                                   d_dd
                                                   + (global_path_curv_d * d + global_path_curv * d_d)
                                                   * np.tan(yaw_diff_array)
                                           )
                                           * (np.power(np.cos(yaw_diff_array), 2) / (1 - global_path_curv * d))
                                   )
                                   + global_path_curv
                           ) * (np.cos(yaw_diff_array) / (1 - global_path_curv * d))

                with timer.time_with_cm(
                        "simulation/calculate trajectories/initialize trajectory"
                ):
                    # create frenet trajectory
                    if contin == True:
                        fp = FrenetTrajectory(
                            t=t,
                            d=d,
                            d_d=d_d_time,     # 存时间域横向速度
                            d_dd=d_dd_time,   # 存时间域横向加速度
                            d_ddd=d_ddd,
                            s=s,
                            s_d=s_d,
                            s_dd=s_dd,
                            s_ddd=s_ddd,
                            x=x,
                            y=y,
                            yaw=yaw,
                            v=v,
                            curv=curv,
                            # 这里存的是 d'(s), d''(s)（用于曲率或其他检查）
                            d_dcurv=d_d,
                            d_ddcurv=d_dd,
                        )
                    else:
                        fp = FrenetTrajectory(
                            t=t,
                            d=d,
                            d_d=d_d_time,
                            d_dd=d_dd_time,
                            d_ddd=d_ddd,
                            s=s,
                            s_d=s_d,
                            s_dd=s_dd,
                            s_ddd=s_ddd,
                            x=x,
                            y=y,
                            yaw=yaw,
                            v=v,
                            curv=curv,
                            d_dcurv=d_d,
                            d_ddcurv=d_dd,
                        )
                # 收集该候选轨迹
                fp_list.append(fp)
    return fp_list


def calc_contingent_plans():
    print('Hi')


def calc_contingent_trajectories(c_s: float,
                                 c_s_d: float,
                                 c_s_dd: float,
                                 c_d: float,
                                 c_d_d: float,
                                 c_d_dd: float,
                                 d_list: [float],
                                 t_list: [float],
                                 v_list: [float],
                                 dt: float,
                                 csp: CubicSpline2D,
                                 v_thr: float = 3.0,
                                 exec_timer=None, ):
    """
    【用途】
    - 这是另一个“轨迹生成器”，逻辑与 calc_frenet_trajectories 非常相似，但实现上：
      1) 少了计时器分段
      2) 多了 ds<=|dT| 的过滤（避免极端横向）
      3) 结构上有一些缩进/分支位置不同（注意：这里存在明显潜在 bug，见 NOTE）

    NOTE: 这个函数中，`yaw_diff_array ... fp = FrenetTrajectory(...)` 的代码块
          被缩进在 low_velocity 分支里，导致 high_velocity 时 `fp` 可能未定义就 append。
          你后面 `fp_list.append(fp)` 不在分支内，会在 high_velocity 下直接报 UnboundLocalError。
          （除非你原文件缩进不同，但按你贴的就是这个问题）
    """
    # list of all generated frenet trajectories
    fp_list = []
    # all end velocities
    for vT in v_list:
        if abs(c_s_d) < v_thr or abs(vT) < v_thr:
            lat_mode = "low_velocity"
        else:
            lat_mode = "high_velocity"
        # all end times
        for tT in t_list:
            # 纵向 quartic
            qp_long = quartic_polynomial(
                xs=c_s, vxs=c_s_d, axs=c_s_dd, vxe=vT, axe=0.0, T=tT - dt
            )

            # use universal function feature to perform array operation
            # time vector
            t = np.arange(0.0, tT, dt)
            # longitudinal position and derivatives
            s = qp_long.calc_point(t)
            s_d = qp_long.calc_first_derivative(t)
            s_dd = qp_long.calc_second_derivative(t)
            s_ddd = qp_long.calc_third_derivative(t)

            # use CubicSpineLine's internal function to perform array operation
            # calculate the position of the reference path
            global_path_x, global_path_y = csp.calc_position(s)

            # move gradient calculation and deviation calculation out from calc_global_trajectory(), avoid unnecessary repeat calculation

            # calculate derivations necessary to get the curvature
            dx = np.gradient(global_path_x, s)
            ddx = np.gradient(dx, s)
            dddx = np.gradient(ddx, s)
            dy = np.gradient(global_path_y, s)
            ddy = np.gradient(dy, s)
            dddy = np.gradient(ddy, s)

            # calculate yaw of the global path
            global_path_yaw = np.arctan2(dy, dx)

            # calculate the curvature of the global path
            global_path_curv = (np.multiply(dx, ddy) - np.multiply(ddx, dy)) / (
                    np.power(dx, 2) + np.power(dy, 2) ** (3 / 2)
            )

            # calculate the derivation of the global path's curvature
            z = np.multiply(dx, ddy) - np.multiply(ddx, dy)
            z_d = np.multiply(dx, dddy) - np.multiply(dddx, dy)
            n = (np.power(dx, 2) + np.power(dy, 2)) ** (3 / 2)
            n_d = (3 / 2) * np.multiply(
                np.power((np.power(dx, 2) + np.power(dy, 2)), 0.5),
                (2 * np.multiply(dx, ddx) + 2 * np.multiply(dy, ddy)),
            )
            global_path_curv_d = (np.multiply(z_d, n) - np.multiply(z, n_d)) / (
                np.power(n, 2)
            )

            s0 = s[0]
            ds = s[-1] - s0
            # all lateral distances
            for dT in d_list:
                # quintic polynomial in lateral direction
                # for high velocities we have ds/dt and dd/dt
                # 过滤：纵向推进太小但横向偏移太大时跳过（避免横向多项式奇异/极端）
                if ds <= abs(dT):
                    continue

                if lat_mode == "high_velocity":
                    qp_lat = quintic_polynomial(
                        xs=c_d,
                        vxs=c_d_d,
                        axs=c_d_dd,
                        xe=dT,
                        vxe=0.0,
                        axe=0.0,
                        T=tT - dt,
                    )

                    # use universal function feature to perform array operation
                    # lateral distance and derivatives
                    d = qp_lat.calc_point(t)
                    d_d = qp_lat.calc_first_derivative(t)
                    d_dd = qp_lat.calc_second_derivative(t)
                    d_ddd = qp_lat.calc_third_derivative(t)

                    d_d_time = d_d
                    d_dd_time = d_dd

                    # 转换到对 s 的导数，供变换使用
                    d_d = d_d / s_d
                    d_dd = (d_dd - d_d * s_dd) / np.power(s_d, 2)

                # for low velocities, we have ds/dt and dd/ds
                elif lat_mode == "low_velocity":
                    # singularity
                    if ds == 0:
                        ds = 0.00001

                    # the quintic polynomial shows dd/ds, so d(c_s)/ds and dd(c_s)/dds is needed
                    if c_s_d != 0.0:
                        c_d_d_not_time = c_d_d / c_s_d
                        c_d_dd_not_time = (c_d_dd - c_s_dd * c_d_d_not_time) / (
                                c_s_d ** 2
                        )
                    else:
                        c_d_d_not_time = 0.0
                        c_d_dd_not_time = 0.0

                    # Upper boundary for ds to avoid bad lat polynoms (solved by  if ds > abs(dT)?)
                    # ds = max(ds, 0.1)

                    qp_lat = quintic_polynomial(
                        xs=c_d,
                        vxs=c_d_d_not_time,
                        axs=c_d_dd_not_time,
                        xe=dT,
                        vxe=0.0,
                        axe=0.0,
                        T=ds,
                    )

                    # use universal function feature to perform array operation
                    # lateral distance and derivatives
                    d = qp_lat.calc_point(s - s0)
                    d_d = qp_lat.calc_first_derivative(s - s0)
                    d_dd = qp_lat.calc_second_derivative(s - s0)
                    d_ddd = qp_lat.calc_third_derivative(s - s0)

                    # since dd/ds, a conversion to dd/dt is needed
                    d_d_time = s_d * d_d
                    d_dd_time = s_dd * d_d + np.power(s_d, 2) * d_dd

                    # NOTE: 这里的 Frenet->global 变换只写在 low_velocity 分支里（这是个大坑）
                    yaw_diff_array = np.arctan(d_d / (1 - global_path_curv * d))
                    yaw = yaw_diff_array + global_path_yaw
                    x = global_path_x - d * np.sin(global_path_yaw)
                    y = global_path_y + d * np.cos(global_path_yaw)
                    v = (s_d * (1 - global_path_curv * d)) / np.cos(yaw_diff_array)
                    curv = (
                                   (
                                           (
                                                   d_dd
                                                   + (global_path_curv_d * d + global_path_curv * d_d)
                                                   * np.tan(yaw_diff_array)
                                           )
                                           * (np.power(np.cos(yaw_diff_array), 2) / (1 - global_path_curv * d))
                                   )
                                   + global_path_curv
                           ) * (np.cos(yaw_diff_array) / (1 - global_path_curv * d))

                    # create frenet trajectory
                    fp = FrenetTrajectory(
                        t=t,
                        d=d,
                        d_d=d_d_time,
                        d_dd=d_dd_time,
                        d_ddd=d_ddd,
                        s=s,
                        s_d=s_d,
                        s_dd=s_dd,
                        s_ddd=s_ddd,
                        x=x,
                        y=y,
                        yaw=yaw,
                        v=v,
                        curv=curv,
                        d_dcurv=d_d,
                        d_ddcurv=d_dd,
                    )

                # if ds > abs(dT):
                # NOTE: 如果走 high_velocity，上面没有创建 fp，这里会直接报错
                fp_list.append(fp)

    return fp_list


def sort_frenet_trajectories(
        ego_state,
        fp_list: [FrenetTrajectory],
        global_path: np.ndarray,
        predictions: dict,
        mode: str,
        params: dict,
        planning_problem: PlanningProblem,
        scenario: Scenario,
        vehicle_params,
        ego_id: int,
        dt: float,
        sensor_radius: float,
        collision_checker,
        exec_timer=None,
        start_idx=0,
        mode_num=100,
        belief=None,
        reach_set=None,
):
    """Sort the frenet trajectories. Check validity of all frenet trajectories in fp_list and sort them by increasing cost.

    Args:
        ego_state (State): Current state of the ego vehicle.
        fp_list ([FrenetTrajectory]): List with all frenét trajectories.
        global_path (np.ndarray): Global path.
        predictions (dict): Predictions of the visible obstacles.
        mode (Str): Mode of the frenét planner.
        planning_problem (PlanningProblem): Planning problem of the scenario.
        scenario (Scenario): Scenario.
        vehicle_params (VehicleParameters): Parameters of the ego vehicle.
        ego_id (int): ID of the ego vehicle.
        dt (float): Delta time of the scenario.
        sensor_radius (float): Sensor radius for the sensor model.
        road_boundary (ShapeGroup): Shape group representing the road boundary.
        collision_checker (CollisionChecker): Collision checker for the scenario.
        goal_area (ShapeGroup): Shape group of the goal area.
        exec_times_dict (dict): Dictionary for the execution times. Defaults to None.

    Returns:
        [FrenetTrajectory]: List of sorted valid frenét trajectories.
        [FrenetTrajectory]: List of sorted invalid frenét trajectories.
        dict: Dictionary with execution times.
    """
    """
    Sort the frenet trajectories:
    1) （可选）先给每条轨迹算风险（calc_risk）
    2) validity 检查：碰撞、动力学约束、道路边界、风险阈值等
    3) 取最高 validity_level 的那一批轨迹
    4) 对这些轨迹算 cost（calc_trajectory_costs）
    5) 返回：最高有效轨迹列表、无效轨迹列表、按有效性分桶的字典

    Args:
        ego_state: 自车状态（CommonRoad State）
        fp_list: 候选轨迹列表
        global_path: 全局参考路径点
        predictions: 其他车预测（轨迹分布/占据等）
        mode: 规划模式（ground_truth / WaleNet / 其他）
        params: 参数字典（包含 cost/risk/validity 的权重/阈值等）
        planning_problem/scenario/vehicle_params: CommonRoad 场景信息
        collision_checker: 碰撞检查器
        start_idx/mode_num/belief/reach_set: 看起来用于 belief planning / 多模态风险计算

    Returns:
        ft_list_highest_validity: 最高有效等级的轨迹
        ft_list_invalid: 其他较低有效等级的轨迹（展平后的列表）
        validity_dict: {valid_level: [traj,...]}
    """
    timer = ExecTimer(timing_enabled=False) if exec_timer is None else exec_timer

    # 以每个有效等级做桶
    validity_dict = {key: [] for key in VALIDITY_LEVELS}

    # cost 计算时是否使用 predictions：
    # - ground_truth / WaleNet 模式下 cost_predictions=None（可能表示不用预测去算某些代价项）
    if mode == "ground_truth" or mode == "WaleNet":
        cost_predictions = None
    else:
        cost_predictions = predictions

    # ========== 1) 风险计算（若有 predictions） ==========
    # NOTE: 风险计算通常很耗时，你这里对每条轨迹都算一次，可能是主要瓶颈
    if predictions is not None:
        risk_loop_total = len(fp_list)
        print(f"[risk] for fp in fp_list 总循环次数: {risk_loop_total}")
        for risk_loop_count, fp in enumerate(fp_list, start=1):
            print(f"[risk] 已循环: {risk_loop_count}/{risk_loop_total}")
            # calc_risk 返回多个结构：ego_risk_dict/obst_risk_dict/ego_harm_dict/obst_harm_dict/bd_harm
            fp.ego_risk_dict, fp.obst_risk_dict, fp.ego_harm_dict, fp.obst_harm_dict, fp.bd_harm = calc_risk(
                traj=fp,
                ego_state=ego_state,
                predictions=predictions,
                scenario=scenario,
                ego_id=ego_id,
                vehicle_params=vehicle_params,
                params=params,
                exec_timer=timer,
                start_idx=start_idx,
                mode_num=mode_num,
                belief=belief
            )
        print(f"[risk] for fp in fp_list 循环结束，总计: {risk_loop_total}")
        
    # ========== 2) validity 检查 ==========
    for fp in fp_list:
        with timer.time_with_cm("simulation/sort trajectories/check validity/total"):
            # check validity
            fp.valid_level, fp.reason_invalid = check_validity(
                ft=fp,
                ego_state=ego_state,
                scenario=scenario,
                vehicle_params=vehicle_params,
                risk_params=params['modes'],  # NOTE: 这里 risk_params 取 params['modes']，要确认结构是否合理
                predictions=predictions,
                mode=mode,
                collision_checker=collision_checker,
                exec_timer=timer,
                start_idx=start_idx,
                mode_num=mode_num,
            )

            validity_dict[fp.valid_level].append(fp)

    # 取“存在轨迹的最高有效等级”
    validity_level = max(
        [lvl for lvl in VALIDITY_LEVELS if len(validity_dict[lvl])]
    )

    # 把比最高等级低的全部视为 invalid（并展开）
    ft_list_highest_validity = validity_dict[validity_level]
    ft_list_invalid = [
        validity_dict[inv] for inv in VALIDITY_LEVELS if inv < validity_level
    ]
    # 把二维 list 拍平成一维 list
    ft_list_invalid = [item for sublist in ft_list_invalid for item in sublist]

    # ========== 3) cost 计算（只对最高有效等级轨迹） ==========
    for fp in ft_list_highest_validity:
        (
            fp.cost,
            fp.cost_dict,
        ) = calc_trajectory_costs(
            traj=fp,
            global_path=global_path,
            ego_state=ego_state,
            validity_level=validity_level,
            planning_problem=planning_problem,
            params=params,
            scenario=scenario,
            ego_id=ego_id,
            dt=dt,
            predictions=cost_predictions,
            sensor_radius=sensor_radius,
            vehicle_params=vehicle_params,
            exec_timer=timer,
            mode=mode,
            reach_set=reach_set,
            mode_num=mode_num
        )

    return ft_list_highest_validity, ft_list_invalid, validity_dict


def calc_global_trajectory(
        csp: CubicSpline2D,
        s: [float],
        s_d: [float],
        s_dd: [float],
        d: [float],
        d_d_lat: [float],
        d_dd_lat: [float],
        lat_mode: str,
        exec_timer=None,
):
    """
    Calculate the global trajectory with a cubic spline reference and a frenet trajectory.

    Args:
        csp (CubicSpline2D): 2D cubic spline representing the global path.
        s ([float]): List with the values for the covered arc length of the spline.
        s_d ([float]): List with the values for the longitudinal velocity.
        s_dd ([float]): List with the values for the longitudinal acceleration.
        d ([float]): List with the values of the lateral offset from the spline.
        d_d_lat ([float]): List with the lateral velocity (defined over s or t, depending on lat_mode).
        d_dd_lat ([float]): List with the lateral acceleration (defined over s or t, depending on lat_mode)
        lat_mode (str): Determines if it is a high speed or low speed trajectory.
        exec_times_dict (dict): Dictionary with the execution times.

    Returns:
        [float]: x-position of the global trajectory.
        [float]: y-position of the global trajectory.
        [float]: Yaw angle of the global trajectory.
        [float]: Curvature of the global trajectory.
        [float]: Velocity of the global trajectory.
        [float]: Acceleration position of the global trajectory.
        dict: Dictionary with the execution times.
    """
    """
    【用途】
    - 这是一个“更通用/更清晰”的 Frenet->global 转换函数（比你上面内联那段更模块化）。
    - 输入：参考线 csp + 纵向（s, s_d, s_dd）+ 横向（d, d' or d_dot, d'' or d_ddot）以及 lat_mode
    - 输出：全局轨迹 x, y, yaw, curv, v, a

    【lat_mode 区别】
    - high_velocity: 输入的 d_d_lat, d_dd_lat 认为是时间域（d_dot, d_ddot），需要先换成 d'(s), d''(s)
    - low_velocity: 输入的 d_d_lat, d_dd_lat 认为已经是弧长域（d'(s), d''(s)）

    NOTE: 你在 calc_frenet_trajectories 里自己做了一遍类似计算（向量化），这里用循环做（更慢但更直观）。
    """
    timer = ExecTimer(timing_enabled=False) if exec_timer is None else exec_timer

    x = []
    y = []
    yaw = []
    curv = []
    v = []
    a = []

    # ========== 0) 把横向导数统一为 d'(s), d''(s) ==========
    with timer.time_with_cm(
            "simulation/calculate trajectories/calculate global trajectory/convert to ds-dt"
    ):
        # if it is a high velocity frenét trajectory, dd/dt and ddd/ddt need to be converted to dd/ds and ddd/dds
        if lat_mode == "high_velocity":
            # d'(s) = d_dot / s_dot
            d_d = [d_d_lat[i] / s_d[i] for i in range(len(s))]
            # d''(s) = (d_ddot - d'(s)*s_ddot)/s_dot^2
            d_dd = [
                (d_dd_lat[i] - d_d[i] * s_dd[i]) / (s_d[i] ** 2) for i in range(len(s))
            ]
        elif lat_mode == "low_velocity":
            d_d = d_d_lat
            d_dd = d_dd_lat
    # ========== 1) 参考线位置 ==========
    with timer.time_with_cm(
            "simulation/calculate trajectories/calculate global trajectory/calculate reference points"
    ):
        # calculate the position of the reference path
        global_path_x = []
        global_path_y = []

        for si in s:
            global_path_x.append(csp.sx(si))
            global_path_y.append(csp.sy(si))
    # ========== 2) 参考线导数 ==========
    with timer.time_with_cm(
            "simulation/calculate trajectories/calculate global trajectory/calculate reference gradients"
    ):
        # calculate derivations necessary to get the curvature
        dx = np.gradient(global_path_x, s)
        ddx = np.gradient(dx, s)
        dddx = np.gradient(ddx, s)
        dy = np.gradient(global_path_y, s)
        ddy = np.gradient(dy, s)
        dddy = np.gradient(ddy, s)

    with timer.time_with_cm(
            "simulation/calculate trajectories/calculate global trajectory/calculate reference yaw"
    ):
        # calculate yaw of the global path
        global_path_yaw = np.arctan2(dy, dx)

    with timer.time_with_cm(
            "simulation/calculate trajectories/calculate global trajectory/calculate reference curvature"
    ):
        # calculate the curvature of the global path
        global_path_curv = (np.multiply(dx, ddy) - np.multiply(ddx, dy)) / (
                np.power(dx, 2) + np.power(dy, 2) ** (3 / 2)
        )

    with timer.time_with_cm(
            "simulation/calculate trajectories/calculate global trajectory/calculate reference curvature derivation"
    ):
        # calculate the derivation of the global path's curvature
        z = np.multiply(dx, ddy) - np.multiply(ddx, dy)
        z_d = np.multiply(dx, dddy) - np.multiply(dddx, dy)
        n = (np.power(dx, 2) + np.power(dy, 2)) ** (3 / 2)
        n_d = (3 / 2) * np.multiply(
            np.power((np.power(dx, 2) + np.power(dy, 2)), 0.5),
            (2 * np.multiply(dx, ddx) + 2 * np.multiply(dy, ddy)),
        )
        global_path_curv_d = (np.multiply(z_d, n) - np.multiply(z, n_d)) / (
            np.power(n, 2)
        )
    # ========== 3) 每个点做 Frenet->global 变换 ==========
    with timer.time_with_cm(
            "simulation/calculate trajectories/calculate global trajectory/calculate trajectory states"
    ):
        # transform every point of the trajectory from the frenét frame to the global coordinate system
        for i in range(len(s)):
            # information from the global path necessary for the transformation
            curvature_si = global_path_curv[i]
            yaw_si = global_path_yaw[i]
            curvature_d_si = global_path_curv_d[i]
            pos_si = [global_path_x[i], global_path_y[i]]

            # transform yaw, position and velocity
            # yaw_diff = atan( d'(s) / (1 - kappa*d) )
            yaw_diff = math.atan(d_d[i] / (1 - curvature_si * d[i]))
            iyaw = yaw_diff + yaw_si

            # 位置偏移
            sx, sy = pos_si
            ix = sx - d[i] * math.sin(yaw_si)
            iy = sy + d[i] * math.cos(yaw_si)
            # 速度
            iv = (s_d[i] * (1 - curvature_si * d[i])) / math.cos(yaw_diff)

            # transform curvature
            # 曲率（同上公式）
            icurv = (
                            (
                                    (
                                            d_dd[i]
                                            + (curvature_d_si * d[i] + curvature_si * d_d[i])
                                            * np.tan(yaw_diff)
                                    )
                                    * ((np.cos(yaw_diff) ** 2) / (1 - curvature_si * d[i]))
                            )
                            + curvature_si
                    ) * (np.cos(yaw_diff) / (1 - curvature_si * d[i]))

            # transform acceleration
            # 加速度（同样来自 Frenet->global 推导）
            ia = s_dd[i] * ((1 - curvature_si * d[i]) / np.cos(yaw_diff)) + (
                    (s_d[i] ** 2) / (np.cos(yaw_diff))
            ) * (
                         (1 - curvature_si * d[i])
                         * np.tan(yaw_diff)
                         * (
                                 icurv * ((1 - curvature_si * d[i]) / np.cos(yaw_diff))
                                 - curvature_si
                         )
                         - (curvature_d_si * d[i] + curvature_si * d_d[i])
                 )

            x.append(ix)
            y.append(iy)
            yaw.append(iyaw)
            curv.append(icurv)
            v.append(iv)
            a.append(ia)

    return x, y, yaw, curv, v, a


def get_v_list(
        v_min: float,
        v_max: float,
        v_cur: float,
        v_goal_min: float = None,
        v_goal_max: float = None,
        n_samples: int = 3,
        mode: str = "linspace",
):
    """
    Get a list of end velocities for the frenet planner.

    【用途】
    - 用来构造候选末速度集合 v_list(会在 calc_frenet_trajectories 里遍历)
    - 支持三种策略:
      1) linspace:均匀采样(外加当前速度 v_cur)
      2) deterministic:基于 beta 分布拟合后取分位点(更“覆盖”中间区域)
      3) random:从 beta 分布随机采样(探索性更强)

    Args:
        v_min/v_max: 速度取值范围
        v_cur: 当前速度(通常强制加入候选集,保证不必强制加减速)
        v_goal_min/v_goal_max: 目标速度范围(例如终点/限速要求),两者要同时 None 或同时给出
        n_samples: 期望采样数量
        mode: "linspace" / "deterministic" / "random"
    """
    """
    Get a list of end velocities for the frenét planner.

    Args:
        v_min (float): Minimum velocity.
        v_max (float): Maximum velocity.
        v_cur (float): Velocity at the current state.
        v_goal_min (float): Minimum goal velocity. Defaults to None.
        v_goal_max (float): Maximum goal velocity. Defaults to None.
        n_samples (int): Number of desired velocities. Defaults to 3.
        mode (Str): Chosen mode. (linspace, deterministic, or random).
            Defaults to linspace.

    Returns:
        list(float): A list of velocities.

    """
    # modes: 0 = linspace, 1 = deterministic, 2 = random

    # check if n_samples is valid
    if n_samples <= 0:
        raise ValueError("Number of samples must be at least 1")

    # 目标速度边界必须同时为空或同时有值
    # NOTE: 这里逻辑写法有点绕: (v_goal_min is None or v_goal_max is None) and v_goal_max != v_goal_min
    #       目的是:一个 None 一个非 None 时触发异常
    if (v_goal_min is None or v_goal_max is None) and v_goal_max != v_goal_min:
        raise AttributeError("Both goal velocities must be None or both must be filled")
    # mode 必须合法
    if mode not in ["linspace", "deterministic", "random"]:
        raise ValueError("V-list mode must be linspace, deterministic, or random")
    
    # ========== 1) linspace:简单均匀采样 ==========
    if mode == "linspace":
        if v_goal_min is None:
            # 采样 n_samples-1 个点,再把 v_cur append 进去 -> 最终 n_samples 个
            v_list = np.linspace(v_min, v_max, n_samples - 1)
        else:
            # 若有 goal 区间,则把采样范围扩到 [max(min(v_goal_min, v_min), 0.001), max(v_goal_max, v_max)]
            # NOTE: 这里看起来是在“合并约束范围”和“搜索范围”
            v_list = np.linspace(
                max(min(v_goal_min, v_min), 0.001), max(v_goal_max, v_max), n_samples - 1
            )
        return np.append(v_list, v_cur)
    
    # ========== 2) deterministic / random:先做一些校验 ==========
    # check if n_samples is valid for the chosen mode
    if mode == "deterministic":
        # deterministic 模式限制采样数不超过 10/11(防止生成太多分位点)
        if v_goal_min is None:
            if n_samples > 10:
                raise ValueError(
                    "n_samples can not be greater than 10 in deterministic mode"
                )
        else:
            if n_samples > 10:
                raise ValueError(
                    "n_samples can not be greater than 11 in deterministic mode"
                )
    # 目标速度均值(若有目标区间)
    # calculate mean of the goal velocities
    if v_goal_min is not None:
        v_goal = v_goal_min + ((v_goal_max - v_goal_min) / 2)
    # 先把一些关键速度塞进去:v_cur、边界、以及 v_goal(如果有)
    # create the order in which the velocities should be appended
    if v_goal_min is None:
        v_append_list = [v_cur, v_min, v_max]
    else:
        v_append_list = [v_goal, v_min, v_max, v_cur]

    v_list = []

    # add the velocities from the append list, further points are added from the density distribution
    for i in range(n_samples):
        if i < len(v_append_list):
            v_list.append(v_append_list[i])
        else:
            break

    # 如果需要的采样数 <= 已经填入的关键点数,直接排序返回
    # calculate the density distribution if necessary
    if n_samples <= len(v_append_list):
        v_list.sort()
        return v_list
    else:
        # 还需要补充的样本数
        n_remaining_samples = n_samples - len(v_append_list)
        # loc is the lower limit, scale the upper limit
        # Beta 分布参数:loc 下界,scale 上界(注意:beta 的 scale 是长度,不是上界本身)
        # 你这里写的 scale = max(v_max, v_goal_max)
        # NOTE: scipy.stats.beta 的参数是 loc + scale*X,其中 X in [0,1]
        #       所以 scale 应该是 (upper-lower),你这里直接用 upper 值,等价于把下界当 0 处理
        #       但你也同时传了 floc/fscale 给 beta.fit,fit 会尝试兼容；依然建议明确 upper-lower。
        loc = min(v_min, v_goal_min)
        scale = max(v_max, v_goal_max)
        # get the beta-distribution (depends on the available velocities)
        # 用少量“代表点”拟合 beta 分布形状(a,b)
        if v_goal_min is not None:
            a, b, floc, fscale = beta.fit(
                floc=loc,
                fscale=scale,
                data=[
                    loc + ((v_goal - loc) / 2),
                    v_goal,
                    scale - ((scale - v_goal) / 2),
                ],
            )
        else:
            a, b, floc, fscale = beta.fit(
                floc=loc,
                fscale=scale,
                data=[loc + ((v_cur - loc) / 2), v_cur, scale - ((scale - v_cur) / 2)],
            )
        # 中位数(0.5 分位点)
        median = beta.median(a=a, b=b, loc=loc, scale=scale)
        # ========== deterministic :取固定分位点 ==========
        # for the deterministic mode, add the following quartiles
        if mode == "deterministic":
            # 偶数/奇数采样时用不同的 alpha,保证分位点分布均匀
            alpha_even = [0.2, 0.4, 0.6, 0.8]
            alpha_odd = [0.3, 0.5, 0.7, 0.9]

            # for an odd number of samples, add the median and the quartile from alpha_odd
            # 若剩余样本数为奇数:先加 median,再加 alpha_odd 的区间两端
            if n_remaining_samples % 2 == 1:
                v_list.append(median)
                alpha = alpha_odd
            # for an even number of samples, add the quartile from alpha_even
            else:
                # 若为偶数:为了成对取 interval(α) 的两端,需要把数量+1 变成奇数对称
                n_remaining_samples += 1
                alpha = alpha_even
            # interval(alpha) 返回 (lower, upper) 两端点,每个 alpha 取两端点
            for i in range(1, int(n_remaining_samples / 2) + 1):
                interv = beta.interval(alpha=alpha[i], a=a, b=b, loc=loc, scale=scale)
                v_list.append(interv[0])
                v_list.append(interv[1])

        # for the random mode, just add random velocities from the beta distribution
        # ========== random:随机采样 ==========
        elif mode == "random":
            random_samples = beta.rvs(
                a=a, b=b, loc=loc, scale=scale, size=n_remaining_samples
            )
            v_list = [*v_list, *random_samples]

    v_list.sort()

    return v_list
