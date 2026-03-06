#!/user/bin/env python

"""
Sampling-based trajectory planning in a frenet frame considering ethical implications.

【文件作用(从名字+导入+代码结构推断)】
- 这是 FrenetPlanner 的主入口之一,负责每个仿真步:
  1) 获取当前 ego 状态(速度/加速度/在参考线上的 Frenet 坐标)
  2) 生成末速度 v_list、横向偏移 d_list、时域 t_list 的采样组合
  3) 调用 calc_frenet_trajectories 生成候选轨迹
  4) 调用预测(这里你用 branching MPC + scenario tree 预测)
  5) 对候选轨迹做 validity 检查 + cost 计算并排序
  6) 基于分支概率(belief/branch_w)做 contingent planning:
     - 对每条 shared plan 再采样生成 contingent trajectories
     - 对每个模式(mode_num)挑最优 contingent 轨迹
     - 把 shared + contingent 组合成一个“全计划”,计算加权总代价
  7) 可视化/日志/输出 best plan

【关键词】
- Frenet frame:沿参考线弧长 s + 横向偏移 d
- Jerk-optimal:纵向 quartic + 横向 quintic 多项式(最小 jerk 等)
- Ethical implications:在你的工程里主要体现在 risk/harm/responsibility 的 cost/validity 中
"""
# =========================
# Standard imports
# =========================

import os
import sys
import copy
import warnings
import json
from inspect import currentframe, getframeinfo
import pathlib
import pickle
import time
import math

# Third party imports
import numpy as np
import matplotlib
# CommonRoad 核心对象
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State
# CommonRoad-DC 碰撞检测
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import (
    create_collision_checker,
)
# 自定义异常:用于超时控制
from commonroad_helper_functions.exceptions import (
    ExecutionTimeoutError,
)
# 预测网络/工具(这里你最终没用 WaleNet,而是走了 branching MPC 预测)
from prediction import WaleNet

# 新增search目录
from pathlib import Path
PLANNING_DIR = str(Path(__file__).parent.parent.parent)
if PLANNING_DIR not in sys.path:
    sys.path.append(PLANNING_DIR)

# Branching MPC 相关
from Init_MPC import initBranchMPC
from MPC_branch import BranchMPC
from highway_branch_dyn import *
import Highway_env_branch

from utils_baseline import Branch_constants

# Custom imports
# 某些 numpy 操作可能触发弃用警告；在批量采样时关闭这些噪声信息
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# 追加仓库根路径,便于导入 beliefplanning.* 模块
mopl_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(mopl_path)

# Planner 基类 + Timeout 工具
from beliefplanning.planner.planning import Planner
from beliefplanning.planner.utils.timeout import Timeout

# 可视化工具(绘制 Frenet轨迹、contingent轨迹、计划树等)
from beliefplanning.planner.Frenet.utils.visualization import draw_frenet_trajectories, \
    draw_contingent_trajectories, draw_all_contingent_trajectories, draw_all_plans

# 预测/可见障碍提取/信念更新等工具
from beliefplanning.planner.Frenet.utils.prediction_helpers import (
    add_static_obstacle_to_prediction,
    get_dyn_and_stat_obstacles,
    get_ground_truth_prediction,
    get_obstacles_in_radius,
    get_orientation_velocity_and_shape_of_prediction,
    belief_updater,
    get_obstacles_prediction_overtake,
    get_prediction_from_scenario_tree, # 你在 _step_planner 里用这个把 zPred -> predictions
)

# 读取 json 配置:伤害模型/规划参数/风险参数/权重/应急规划参数
from beliefplanning.planner.Frenet.configs.load_json import (
    load_harm_parameter_json,
    load_planning_json,
    load_risk_json,
    load_weight_json,
    load_contingency_json,
)

# Frenet 轨迹生成 / 排序(validity+cost)等
from beliefplanning.planner.Frenet.utils.frenet_functions import (
    calc_frenet_trajectories,
    calc_contingent_plans, # NOTE: 这里 import 了但没调用,因为你在 _step_planner 里直接调用了 calc_frenet_trajectories 生成备选轨迹；如果 contingency planning 逻辑更复杂了,可以考虑把它封装成一个函数
    get_v_list,
    sort_frenet_trajectories,
)

# 日志
from beliefplanning.planner.Frenet.utils.logging import FrenetLogging
# 可达集(用于责任/责任敏感 cost)
from beliefplanning.planner.utils import reachable_set

# 风险可视化/仪表盘工具
from beliefplanning.risk_assessment.visualization.risk_visualization import (
    create_risk_files,
)
from beliefplanning.risk_assessment.visualization.risk_dashboard import risk_dashboard


class FrenetPlanner(Planner):
    """
    Jerk optimal planning in frenet coordinates with quintic polynomials in lateral direction
    and quartic polynomials in longitudinal direction.

    【类职责】
    - 继承 Planner 基类,核心重载 _step_planner:
      每个 time_step 生成候选轨迹 -> 风险/有效性检查 -> 代价排序 -> 输出最佳轨迹/应急计划

    【关键成员变量(后面会用到)】
    - self.frenet_parameters / self.contingency_parameters:采样参数与应急规划参数
    - self.params_dict:风险/权重/伤害模型综合字典(传给 validity/cost/risk)
    - self.predictor:预测模块(WaleNet 或 ground_truth 或 risk 模式下使用)
    - self.collision_checker:碰撞检测器(CommonRoad-DC)
    - self.reach_set:可达集模块(责任相关)
    - self.driven_traj:执行过的轨迹历史,用于可视化与日志
    """

    def __init__(
            self,
            scenario: Scenario,
            planning_problem: PlanningProblem,
            ego_id: int,
            vehicle_params,
            mode,
            exec_timer=None,
            frenet_parameters: dict = None,
            contingency_parameters: dict = None,
            sensor_radius: float = 55.0,
            plot_frenet_trajectories: bool = False,
            weights=None,
            settings=None,
    ):
        """
        Initialize a frenét planner.

        Args:
            scenario / planning_problem: CommonRoad 的场景与规划任务
            ego_id: ego vehicle ID
            vehicle_params: 车辆参数(尺寸、动力学上限等)
            mode: 'ground_truth'/'WaleNet'/'risk'(控制预测/风险计算逻辑)
            exec_timer: 计时器(统计性能)
            frenet_parameters: Frenet采样参数(t_list/d_list/v_list采样等)
            contingency_parameters: 应急规划参数(第二阶段采样)
            sensor_radius: 传感器半径(用于可见障碍/预测过滤)
            plot_frenet_trajectories: 是否可视化
            weights/settings: 可覆盖默认权重与风险配置
        """
        # 基类初始化:通常会设置 self.scenario/self.planning_problem/self.ego_state/self.reference_spline 等
        super().__init__(scenario, planning_problem, ego_id, vehicle_params, exec_timer)
        # 分支 MPC/环境相关(你用于 overtaking 分支预测)
        self.N_lane = None
        self.mpc = None

        # 记录各时刻的轨迹/状态/预测,便于复现或可视化
        self.traj_rec = []
        self.state_rec = []
        self.zPred_rec = []
        self.exec_time = []
        self.branch_w_rec = []

        self.long_jerk = []  # 纵向 jerk 记录(调试舒适性)
        self.lat_jerk = []  # 横向 jerk 记录

        # Set up logger
        # 日志输出到 results/logs/{benchmark_id}.csv
        self.logger = FrenetLogging(
            log_path=f"./planner/Frenet/results/logs/{scenario.benchmark_id}.csv"
        )

        try:
            # 使用上下文计时与异常捕获,防止初始化阶段卡死拖垮整体仿真
            # TODO(yanjun): 目前暂时关闭超时限制,正常使用记得打开
            with Timeout(100000, "Frenet Planner initialization"):

                self.exec_timer.start_timer("initialization/total")
                if frenet_parameters is None:
                    print(
                        "No frenet parameters found. Swichting to default parameters."
                    )
                    # 若没给 frenet_parameters,用默认参数(较粗采样)
                    # 默认采样参数:沿横向 d 方向均匀采样,速度采用 linspace
                    frenet_parameters = {
                        "t_list": [2.0],                     # 规划时域(秒)
                        "v_list_generation_mode": "linspace", # v_list 采样方式
                        "n_v_samples": 5,                     # 末速度样本数量
                        "d_list": np.linspace(-3.5, 3.5, 15), # 横向偏移采样(左右)
                        "dt": 0.1,                            # 离散步长
                        "v_thr": 3.0,                         # 高低速切换阈值
                    }

                # 保存 Frenet 规划参数 & 应急规划参数
                # parameters for frenet planner
                self.frenet_parameters = frenet_parameters
                # parameters for contingency planner
                self.contingency_parameters = contingency_parameters

                # vehicle parameters
                self.p = vehicle_params

                # ========== 加载风险/伤害/权重配置 ==========
                self.params_harm = load_harm_parameter_json()
                if weights is None:
                    self.params_weights = load_weight_json()
                else:
                    # 允许外部传入覆盖默认权重(比如从命令行参数或上层决策模块动态调整)
                    self.params_weights = weights
                # settings 可覆盖风险字典(risk_dict)
                if settings is not None:
                    if "risk_dict" in settings:
                        self.params_mode = settings["risk_dict"]
                    else:
                        self.params_mode = load_risk_json()
                else:
                    self.params_mode = load_risk_json()
                # NOTE: 如果 settings is None,这里 self.params_mode 没被设置 -> 但后面会用 self.params_mode
                # BUG: 应该加 else: self.params_mode = load_risk_json()
                
                # 统一封装为 params_dict,传给 sort_frenet_trajectories/calc_cost/calc_risk 等函数使用
                self.params_dict = {
                    'weights': self.params_weights,
                    'modes': self.params_mode,
                    'harm': self.params_harm,
                }

                # 目标速度范围(可用于速度采样约束),默认 None
                self.v_goal_min = None
                self.v_goal_max = None

                self.cost_dict = {}

                # 如果场景未提供加速度信息,默认置零,防止多项式构建时报错
                if not hasattr(self.ego_state, "acceleration"):
                    self.ego_state.acceleration = 0.0

                # driven_traj:用于记录实际执行轨迹(用于绘制/日志)；初始时包含一个状态,即仿真环境提供的初始状态
                self.driven_traj = [
                    State(
                        position=self.ego_state.position,
                        orientation=self.ego_state.orientation,
                        time_step=self.ego_state.time_step,
                        velocity=self.ego_state.velocity,
                        acceleration=self.ego_state.acceleration,
                    )
                ]

                # 传感器半径 & 模式(影响预测/风险计算)
                # 记录感知范围与模式,用于后续过滤障碍物以及选择预测器
                self.sensor_radius = sensor_radius
                self.mode = mode

                # get visualization marker
                # 可视化开关
                self.plot_frenet_trajectories = plot_frenet_trajectories

                # 根据模式加载预测器:WaleNet/风险模式共享同一网络,ground_truth 直接读取仿真真值
                if self.mode == "WaleNet" or self.mode == "risk":

                    prediction_config_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "configs",
                        "prediction.json",
                    )
                    with open(prediction_config_path, "r") as f:
                        online_args = json.load(f)

                    self.predictor = WaleNet(scenario=scenario, online_args=online_args, verbose=True)
                    # print("WaleNet model_path =", getattr(self.predictor, "model_path", None))
                elif self.mode == "ground_truth":
                    self.predictor = None  # 直接使用仿真环境提供的真实轨迹
                else:
                    raise ValueError("mode must be ground_truth, WaleNet, or risk")

                # 若风险权重中含“responsibility”项,则额外计算可达集用于责任分析
                if (
                        'responsibility' in self.params_weights
                        and self.params_weights['responsibility'] > 0
                ):
                    self.responsibility = True
                    self.reach_set = reachable_set.ReachSet(
                        scenario=self.scenario,
                        ego_id=self.ego_id,
                        ego_length=self.p.l,
                        ego_width=self.p.w,
                    )
                else:
                    self.responsibility = False
                    self.reach_set = None

                # 碰撞检测器需要移除 ego 车辆(否则会与自身碰撞)
                with self.exec_timer.time_with_cm(
                        "initialization/initialize collision checker"
                ):
                    # deep copy 场景,移除 ego obstacle,避免自碰撞
                    cc_scenario = copy.deepcopy(self.scenario)
                    cc_scenario.remove_obstacle(
                        obstacle=[cc_scenario.obstacle_by_id(ego_id)]
                    )
                    try:
                        self.collision_checker = create_collision_checker(cc_scenario)
                    except Exception:
                        raise BrokenPipeError("Collision Checker fails.") from None
                self.exec_timer.stop_timer("initialization/total")
        except ExecutionTimeoutError:
            raise TimeoutError

    # ============================================================
    # 分支预测:模拟超车(overtake)场景的 Branching MPC
    # ============================================================
    def sim_overtake(self):
        """调用高速公路分支 MPC 仿真环境,生成超车场景的预测树与权重。"""
        """
        【作用】
        - 初始化并运行 Branching MPC 环境一步,得到:
          backup: 备选策略/轨迹(可能用于环境模拟或安全 fallback)
          zPred: 预测树(多分支预测)
          obst_new_state: 更新后的障碍物状态(供下一步延续)
          branch_w: 每个分支/模式概率(belief)
          state_rec: 记录的状态序列(调试/可视化)

        NOTE:
        - 这里参数基本写死:N=8, dt=0.1, NB=2...
        - 每次 _step_planner 都会调用 sim_overtake,并且这里每次都重新 init MPC(很耗时)
          更合理做法:把 mpc 初始化放到 __init__,这里只做 step 更新。
        """
        N = 8     # 每个分支预测的步数
        n = 4     # 状态维度
        d = 2     # 控制输入维度
        
        # 仅用于 MPC 初始化的初始状态(不是仿真实际初始)
        x0 = np.array([0, 1.8, 0,0])  # Initial condition (only for initializing the MPC, not the actual initial state of the sim)
        
        am = 6.0   # 加速度上限
        rm = 0.3   # 可能是 steering rate / jerk 或风险参数,取决于 MPC 实现
        dt = 0.1
        NB = 2     # branching 因子:2 => 树形分支数量按层扩张

        N_lane = 4 # 车道数量(离散)

        # Initialize controller parameters
        # 参考状态:例如沿 s 方向 0.5,横向 1.8,速度 15,航向 0
        xRef = np.array([0.5, -1.8, 15, 0])  # MPC 期望状态(横向居中、纵向保持速度)

        # Branch_constants 参数说明(单位大多为 SI；未被当前链路直接读取的参数保留为兼容/调参位):
        # - s1: 分支概率 soft-saturation 的敏感度(越大越“激进”地区分安全分支)
        # - s2/c2/tran_diag/alpha/R/J_c/s_c/ylb/yub/col_alpha: 旧版或扩展模型预留参数(当前代码链路基本未直接使用)
        # - am/rm: 纵向加速度与横摆角速度(或转向角速度)输入上限
        # - L/W: 车辆长宽,用于碰撞与车道边界安全距离
        # - Kpsi: 备份控制器的航向角误差反馈增益
        cons = Branch_constants(
            s1=2, s2=3, c2=0.5, tran_diag=0.3, alpha=1, R=1.2,
            am=am, rm=rm, J_c=20, s_c=1, ylb=0., yub=7.2,
            L=4, W=2.5, col_alpha=5, Kpsi=0.1
        )
        
        # backup 控制律集合:保持、刹车、换道
        backupcons = [lambda x: backup_maintain(x, cons), 
                      lambda x: backup_brake(x, cons), 
                      lambda x: backup_lc(x, xRef)]
        
        # PredictiveModel:预测模型 + backup
        model = PredictiveModel(n, d, N, backupcons, dt, cons)

        # 初始化 BranchMPC 参数并构建 MPC
        mpcParam = initBranchMPC(n, d, N, NB, xRef, am, rm, N_lane, cons.W)
        mpc = BranchMPC(mpcParam, model)

        # 调用环境仿真一步:备份控制指令、障碍预测、障碍新状态、分支概率以及历史状态
        backup, zPred, self.obst_new_state, branch_w, state_rec = Highway_env_branch.sim_overtake(
            mpc, N_lane,
            self.time_step,
            self.ego_state,
            self.obst_new_state,
            self._trajectory)
        # 保存给后续绘图/调试
        self.N_lane = N_lane
        self.mpc = mpc
        return backup, zPred, state_rec, branch_w
    
    # ============================================================
    # 每个仿真步的主规划逻辑
    # ============================================================
    def _step_planner(self):
        """
        Frenet Planner step function.

        【核心流程】
        A) 更新 driven_traj(记录 ego 实际状态)
        B) 读取当前 Frenet 状态:c_s, c_d 及其导数
        C) 构造 v_list(末速度采样)
        D) 调用 calc_frenet_trajectories 生成候选轨迹 ft_list
        E) 获取预测(这里用 sim_overtake->zPred->predictions)
        F) 可达集(责任)计算(可选)
        G) sort_frenet_trajectories:validity + cost(对 shared trajectories)
        H) 对每条 shared plan 做 contingent planning:
           - 生成 contingent trajectories
           - 按每个 mode_num 选最优
           - 组装计划并计算加权总代价
           - 最终按总代价排序输出最优计划
        I) 可视化/风险图/仪表盘/保存记录
        J) 输出 best_trajectory 并更新 self._trajectory(供下一步使用)
        """
        self.exec_timer.start_timer("simulation/total")

        # =========================
        # A) 更新 driven trajectory(执行轨迹)
        # =========================
        with self.exec_timer.time_with_cm("simulation/update driven trajectory"):
            # 追加当前实车状态到驱动轨迹；时间步 0 只包含初始状态
            if self.ego_state.time_step != 0:
                current_state = State(
                    position=self.ego_state.position,
                    orientation=self.ego_state.orientation,
                    time_step=self.ego_state.time_step,
                    velocity=self.ego_state.velocity,
                    acceleration=self.ego_state.acceleration,
                )

                self.driven_traj.append(current_state)

        # =========================
        # B) 读取当前 Frenet 状态(来自 self.trajectory)
        # =========================
        # self.trajectory 通常是上一步 best_trajectory 生成的轨迹缓存
        # 这里取 index 1(而不是0)常见原因:
        # - index 0 是当前时刻(已执行),index 1 才是下一小步预测点用于更新初值
        c_s = self.trajectory["s_loc_m"][1]
        c_s_d = self.ego_state.velocity
        c_s_dd = self.ego_state.acceleration

        # 横向偏移 c_d:第0步做特殊处理(你这里强制置0)
        if self.time_step == 0:
            # c_d = -3.6  # 你注释里写过 -3.6,可能表示初始在右侧车道
            # 初始时刻与参考线对齐,避免横向抖动
            c_d = 0
        else:
            c_d = self.trajectory["d_loc_m"][1]

        # 横向速度/加速度来自上一轨迹缓存
        c_d_d = self.trajectory["d_d_loc_mps"][1]
        c_d_dd = self.trajectory["d_dd_loc_mps2"][1]

        # get the end velocities for the frenét paths
        # =========================
        # C) 构造末速度采样 v_list(基于加速度上限推 min/max)
        # =========================
        current_v = self.ego_state.velocity
        max_acceleration = self.p.longitudinal.a_max
        t_min = min(self.frenet_parameters["t_list"])
        t_max = max(self.frenet_parameters["t_list"])

        # 最大末速度:当前速度 + (a_max/2)*t_max(NOTE: 为啥 /2？可能是留余量或 jerk 约束的粗略近似)
        max_v = min(
            current_v + (max_acceleration / 2.0) * t_max, self.p.longitudinal.v_max
        )
        # 最小末速度:当前速度 - a_max*t_min(并且不能小于 0.01)
        min_v = max(0.01, current_v - max_acceleration * t_min)

        with self.exec_timer.time_with_cm("simulation/get v list"):
            # 依据当前速度与最大加速度约束,确定采样速度集合
            v_list = get_v_list(
                v_min=min_v,
                v_max=max_v,
                v_cur=current_v,
                v_goal_min=self.v_goal_min,
                v_goal_max=self.v_goal_max,
                mode=self.frenet_parameters["v_list_generation_mode"],
                n_samples=self.frenet_parameters["n_v_samples"],
            )

        # =========================
        # D) 生成候选 Frenet trajectories(shared horizon)
        # =========================
        with self.exec_timer.time_with_cm("simulation/calculate trajectories/total"):
            # 在纵向时间/速度、横向 offset 的组合上批量生成多项式轨迹
            # NOTE: 你这里覆盖了 frenet_parameters["d_list"],改成固定从 -3.6 到 0 的 10 点
            # 这相当于只在某一侧车道范围采样(比如只向左变道或只向右回正)
            # 若要更通用,应该用 self.frenet_parameters["d_list"]
            # d_list = self.frenet_parameters["d_list"]
            d_list = np.linspace(-3.6, 0, 10)
            t_list = self.frenet_parameters["t_list"]

            # if self.ego_state.time_step == 0 or self.open_loop is False:
            ft_list = calc_frenet_trajectories(
                c_s=c_s,
                c_s_d=c_s_d,
                c_s_dd=c_s_dd,
                c_d=c_d,
                c_d_d=c_d_d,
                c_d_dd=c_d_dd,
                d_list=d_list,
                t_list=t_list,
                v_list=v_list,
                dt=self.frenet_parameters["dt"],
                csp=self.reference_spline,
                v_thr=self.frenet_parameters["v_thr"],
                exec_timer=self.exec_timer,
                # NOTE: 这些参数在你 calc_frenet_trajectories 里目前没用来过滤(若你没改那文件)
                t_min=t_min,
                t_max=t_max,
                max_acceleration=max_acceleration,
                max_velocity=self.p.longitudinal.v_max,
                v_goal_min=self.v_goal_min,
                v_goal_max=self.v_goal_max,
                mode=self.frenet_parameters["v_list_generation_mode"],
                n_samples=self.frenet_parameters["n_v_samples"],
                contin=False
            )

        # =========================
        # E) 预测:用 Branching MPC 生成 scenario tree -> predictions
        # =========================
        with self.exec_timer.time_with_cm("simulation/prediction"):
            # Overwrite later
            visible_area = None  # NOTE: 这里没计算可见域,绘图时可能会用到
            # 调用分支 MPC 仿真获取多模式预测及对应权重
            backup, zPred, state_rec, branch_w = self.sim_overtake()
            # predictions = get_obstacles_prediction_overtake(zPred, backup)
            # 将 zPred 转换成 predictions(你的 risk/cost/validity 期望的格式)
            predictions = get_prediction_from_scenario_tree(zPred)
            # 用预测换算的状态覆盖 CommonRoad 障碍物,保证下游检测一致
            # NOTE: 下面这几行直接改 scenario.dynamic_obstacles[0].initial_state
            # 这在 CommonRoad 里可能有副作用:你在运行过程中“篡改”了场景初始状态
            # 建议:不要改 initial_state,而是改一个当前 state 或单独的预测结构
            self.scenario.dynamic_obstacles[0].initial_state.position[0] = state_rec[1][0][0]
            self.scenario.dynamic_obstacles[0].initial_state.position[1] = -state_rec[1][0][1]
            self.scenario.dynamic_obstacles[0].initial_state.orientation = state_rec[1][0][3]

        # =========================
        # F) 可达集计算(责任相关)
        # =========================
        if self.responsibility:
            with self.exec_timer.time_with_cm(
                    "simulation/calculate and check reachable sets"
            ):
                # reachable sets 对哪些障碍算:list(predictions.keys())
                self.reach_set.calc_reach_sets(self.ego_state, list(predictions.keys()))

        # =========================
        # G) shared trajectories:validity + cost,得到有效/无效列表
        # =========================
        with (self.exec_timer.time_with_cm("simulation/sort trajectories/total")):
            # ====== 计算轨迹可行性与代价 ======

            # 当前 belief 直接使用分支树概率,可依据需要替换为 Bayesian belief_updater
            # sorted list (increasing costs)

            # if self.ego_state.time_step == 0 or self.open_loop is False:
            # belief = [self.belief[self.ego_state.time_step], 1 - self.belief[self.ego_state.time_step]]
            # belief(分支概率)用于风险/代价权重
            # NOTE: 你这里 belief=branch_w(来自 MPC),而不是 belief_updater 的输出
            belief = branch_w
            # belief = [1] * 12
            # 基于碰撞/越界/舒适性指标筛选轨迹,返回按成本排序前的有效/无效集合
            ft_list_valid, ft_list_invalid, validity_dict = sort_frenet_trajectories(
                ego_state=self.ego_state,
                fp_list=ft_list,
                global_path=self.global_path,
                predictions=predictions,
                mode=self.mode,
                params=self.params_dict,
                planning_problem=self.planning_problem,
                scenario=self.scenario,
                vehicle_params=self.p,
                ego_id=self.ego_id,
                dt=self.frenet_parameters["dt"],
                sensor_radius=self.sensor_radius,
                collision_checker=self.collision_checker,
                exec_timer=self.exec_timer,
                start_idx=0,
                mode_num=100,
                belief=belief,
                reach_set=(self.reach_set if self.responsibility else None)
            )

            # =========================
            # H) 进一步按 cost 排序 valid trajectories + 做 contingent planning
            # =========================
            with self.exec_timer.time_with_cm(
                    "simulation/sort trajectories/sort list by costs"
            ):
                # 依据 cost 属性升序排序,cost 已综合舒适性/碰撞风险/责任等指标
                # 把 shared 轨迹按 cost 从小到大排序
                ft_list_valid.sort(key=lambda fp: fp.cost, reverse=False)

                # contingency 阶段的速度范围推算,使用 contingency_parameters 的 t_list
                max_acceleration = self.p.longitudinal.a_max
                t_min = min(self.contingency_parameters["t_list"])
                t_max = max(self.contingency_parameters["t_list"])

                # NOTE: 同样覆盖 contingency_parameters["d_list"],固定采样 -3.6..0
                # d_list = self.contingency_parameters["d_list"]
                d_list = np.linspace(-3.6, 0, 6)
                t_list = self.contingency_parameters["t_list"]

                ft_final_list = []       # 每个 shared 轨迹对应一个 final_plan(字典)
                ft_all_plans_list = []   # 用于绘图:保存 shared + 所有 contingent 候选

                # 遍历每条共享轨迹,生成不同预测模式下的备选方案
                for plan in ft_list_valid:
                    final_plan = {}   # 保存该 shared plan 对应的最佳 contingent plans(按 mode_num)
                    ft_all_plans = {} # 保存该 shared plan 及其所有 contingent candidates
                    
                    # 根据 shared plan 末速度推 contingency 速度范围
                    max_v = min(
                        plan.v[-1] + (max_acceleration / 2.0) * t_max, self.p.longitudinal.v_max
                    )
                    min_v = max(0.01, plan.v[-1] - max_acceleration * t_min)

                    # Plan contingent plans for only valid shared trajectories
                    # 生成 contingency 的末速度采样 v_list(基于 shared plan 末速度与加速度约束)
                    v_list = get_v_list(
                        v_min=min_v,
                        v_max=max_v,
                        v_cur=plan.v[-1],
                        v_goal_min=self.v_goal_min,
                        v_goal_max=self.v_goal_max,
                        mode=self.contingency_parameters["v_list_generation_mode"],
                        n_samples=self.contingency_parameters["n_v_samples"],
                    )

                    # shared_plan 永远存在
                    final_plan['shared_plan'] = plan
                    ft_all_plans['shared_plan'] = plan

                    # 如果 contingency t_list 第一项不是 0,则需要规划后半段 contingent
                    if t_list[0] != 0:
                        # 仅当备选规划需要延长时间(t_list[0] > 0)才计算
                        # 生成 contingent trajectories:以 shared plan 的末状态作为初值
                        ft_contingent_list = calc_frenet_trajectories(
                            c_s=plan.s[-1],
                            c_s_d=plan.s_d[-1],
                            c_s_dd=plan.s_dd[-1],
                            c_d=plan.d[-1],

                            # NOTE: 这里危险:plan.d_d[-1] 在你的 FrenetTrajectory 里存的是“时间域 d_dot”
                            # 但 calc_frenet_trajectories 里会根据 lat_mode 决定如何解释 c_d_d/c_d_dd
                            # 若 plan.s_d[-1] 很小,会导致转换问题；建议明确传 d'(s) 还是 d_dot(t)
                            c_d_d=plan.d_d[-1],
                            c_d_dd=plan.d_dd[-1],
                            
                            d_list=d_list,
                            t_list=t_list,
                            v_list=v_list,
                            dt=self.contingency_parameters["dt"],
                            csp=self.reference_spline,
                            v_thr=self.contingency_parameters["v_thr"],
                            exec_timer=self.exec_timer,
                            
                            t_min=t_min,
                            t_max=t_max,
                            max_acceleration=max_acceleration,
                            max_velocity=self.p.longitudinal.v_max,
                            v_goal_min=self.v_goal_min,
                            v_goal_max=self.v_goal_max,
                            mode=self.contingency_parameters["v_list_generation_mode"],
                            n_samples=self.contingency_parameters["n_v_samples"],
                            contin=True
                        )
                        # 保存所有 contingent candidates(用于绘图/调试)
                        for index in range(len(ft_contingent_list)):
                            ft_all_plans[index] = ft_contingent_list[index]

                        ft_all_plans_list.append(ft_all_plans)

                        # we want to calculate the best contingent plan per mode
                        # ========== 对每个“模式/分支”挑一个最优 contingent ==========
                        # predictions[1]['pos_list']:推测是 obstacle id=1 的多个模式位置序列
                        # mode_num 遍历每个模式,sort_frenet_trajectories 用 mode_num 选择对应预测分支
                        # 针对每一个预测模式(不同障碍路径),挑选最便宜的备选轨迹
                        for mode_num in range(len(predictions[1]['pos_list'])):
                            ft_conting_list_valid, ft_conting_list_invalid, validity_conting_dict = sort_frenet_trajectories(
                                ego_state=self.ego_state,
                                fp_list=ft_contingent_list,
                                global_path=self.global_path,
                                predictions=predictions,
                                mode=self.mode,
                                params=self.params_dict,
                                planning_problem=self.planning_problem,
                                scenario=self.scenario,
                                vehicle_params=self.p,
                                ego_id=self.ego_id,
                                dt=self.frenet_parameters["dt"],
                                sensor_radius=self.sensor_radius,
                                collision_checker=self.collision_checker,
                                exec_timer=self.exec_timer,
                                # start_idx:从 shared horizon 结束处开始评估(避免重复从0开始)
                                start_idx=int(max(self.frenet_parameters["t_list"]) / self.frenet_parameters["dt"]),
                                mode_num=mode_num,
                                reach_set=(self.reach_set if self.responsibility else None)
                            )
                            # 按 cost 排序,选最小 cost 的 contingent 作为该 mode 的最优
                            ft_conting_list_valid.sort(key=lambda fp: fp.cost, reverse=False)
                            # NOTE: 若 ft_conting_list_valid 为空会 IndexError
                            # 建议:做保护:if not ft_conting_list_valid: continue/用备选
                            final_plan[mode_num] = ft_conting_list_valid[0]

                    ft_final_list.append(final_plan)

                # we need to get the belief over the modes to use it as weights in the cost function
                '''
                self.belief = belief_updater(predictions, self.belief)
                self.belief_list.append(self.belief[0])
                '''
                # iterate over the final frenet list, and assign a cost to the entire traj
                # print('belief is: ', belief[0])
                # =========================
                # I) 组合 shared + contingent 的总代价(belief 加权)
                # =========================
                # NOTE: 你这里 belief_updater 被注释掉了,用的是 branch_w
                # 将共享轨迹与各模式应急轨迹组合,形成总成本；权重来源于分支概率
                            
                for plan in ft_final_list:
                    if len(plan) == 1:
                        # This means we have only a single plan along the horizon
                        # 只有 shared_plan,没有 contingent(例如 t_list[0]==0 或没算 contingent)
                        plan['cost'] = plan['shared_plan'].cost
                    else:
                        # 有 contingent:总代价 = shared_cost + Σ belief_i * contingent_cost_i
                        # BUG?: 你这里写法有明显换行拼接问题,会导致 Python 语法/结果错误:
                        # plan['cost'] = plan['shared_plan'].cost + branch_w[0] * plan[0].cost + branch_w[1] * plan[3].cost
                        # + branch_w[2] * plan[6].cost
                        #
                        # 正确写法应该用括号包起来:
                        # plan['cost'] = plan['shared_plan'].cost + (
                        #     branch_w[0]*plan[0].cost + branch_w[1]*plan[3].cost + branch_w[2]*plan[6].cost
                        # )
                        #
                        # 另外:你这里用 plan[0], plan[3], plan[6] 是硬编码索引,
                        # 但上面 final_plan[mode_num] 的 mode_num 是 range(len(predictions[1]['pos_list']))
                        # 不一定是 {0,3,6} 这种步长。应按实际 mode_num 集合遍历求和。
                        plan['cost'] = plan['shared_plan'].cost + branch_w[0] * plan[0].cost + branch_w[1] * plan[
                            3].cost + branch_w[2] * plan[6].cost

                # sort the final plan
                # 最终按总代价排序,ft_final_list[0] 就是全计划最优
                ft_final_list.sort(key=lambda fp: fp['cost'], reverse=False)

        # =========================
        # J) 可视化、风险图、记录输出
        # =========================
        with self.exec_timer.time_with_cm("plot trajectories"):
            # if self.ego_state.time_step == 0 or self.open_loop == False:
            # 生成 harm/risk 图(需要 risk 模式)
            if self.params_mode["figures"]["create_figures"] is True:
                if self.mode == "risk":
                    create_risk_files(
                        scenario=self.scenario,
                        time_step=self.ego_state.time_step,
                        destination=os.path.join(os.path.dirname(__file__), "results"),
                        risk_modes=self.params_mode,
                        weights=self.params_weights,
                        marked_vehicle=self.ego_id,
                        planning_problem=self.planning_problem,
                        traj=ft_list_valid,
                        global_path=self.global_path_to_goal,
                        global_path_after_goal=self.global_path_after_goal,
                        driven_traj=self.driven_traj,
                    )

                else:
                    warnings.warn(
                        "Harm diagrams could not be created."
                        "Please select mode risk.",
                        UserWarning,
                    )
            # 风险仪表盘
            if self.params_mode["risk_dashboard"] is True:
                if self.mode == "risk":
                    risk_dashboard(
                        scenario=self.scenario,
                        time_step=self.ego_state.time_step,
                        destination=os.path.join(
                            os.path.dirname(__file__), "results/risk_plots"
                        ),
                        risk_modes=self.params_mode,
                        weights=self.params_weights,
                        planning_problem=self.planning_problem,
                        traj=(ft_list_valid + ft_list_invalid),
                    )

                else:
                    warnings.warn(
                        "Risk dashboard could not be created."
                        "Please select mode risk.",
                        UserWarning,
                    )

            # print some information about the frenet trajectories
            # 终端打印与本地绘图开关
            if self.plot_frenet_trajectories:
                matplotlib.use("TKAgg")
                print(
                    "Time step: {} | Velocity: {:.2f} m/s | Acceleration: {:.2f} m/s2".format(
                        self.time_step, current_v, c_s_dd
                    )
                )
                '''
                Highway_env_branch.plot_scenario(self.mpc, self.N_lane, self.time_step, self.ego_state,
                                                 self.obst_new_state, ft_final_list[0],
                                                 state_rec, zPred)
                '''
                # 记录用于最终画图
                self.traj_rec.append(ft_final_list[0])
                self.state_rec.append(state_rec)
                self.zPred_rec.append(zPred)
                self.branch_w_rec.append(branch_w)
                
                # 在 time_step==100 时画整个回放
                if self.time_step == 100:
                    Highway_env_branch.plot_scenario(self.mpc, self.N_lane, self.time_step, self.ego_state,
                                                     self.obst_new_state, self.traj_rec,
                                                     self.state_rec, self.zPred_rec)

            try:
                '''
                draw_all_contingent_trajectories(
                    scenario=self.scenario,
                    time_step=self.ego_state.time_step,
                    marked_vehicle=self.ego_id,
                    planning_problem=self.planning_problem,
                    traj=None,
                    global_path=self.global_path_to_goal,
                    global_path_after_goal=self.global_path_after_goal,
                    driven_traj=self.driven_traj,
                    animation_area=50.0,
                    predictions=predictions,
                    visible_area=visible_area,
                    valid_traj=ft_final_list,
                    best_traj=self.contingency_trajectory,
                    open_loop=self.open_loop,
                )
                '''
                # 绘制所有计划(shared+contingent)与最优选择
                draw_all_plans(
                    scenario=self.scenario,
                    time_step=self.ego_state.time_step,
                    marked_vehicle=self.ego_id,
                    planning_problem=self.planning_problem,
                    traj=None,
                    global_path=self.global_path_to_goal,
                    global_path_after_goal=self.global_path_after_goal,
                    driven_traj=self.driven_traj,
                    animation_area=50.0,
                    predictions=predictions,
                    visible_area=visible_area,
                    valid_traj=ft_all_plans_list,  # 所有候选(按 shared 分组)
                    best_traj=ft_final_list,       # 最终计划列表(按总代价排序)
                    open_loop=self.open_loop,
                )

            except Exception as e:
                print(e)
            # 初始时刻保存 contingency_trajectory(用于 open loop 可能复用)
            if self.ego_state.time_step == 0:
                self.contingency_trajectory = ft_final_list  # 第一次迭代记录全套计划,供可视化或调试

            # best trajectory
            # 选择 best trajectory(用于更新 self._trajectory)
            # NOTE: 这里 best_trajectory 取的是 ft_list_valid[0],而不是 ft_final_list[0]['shared_plan']
            #      也就是说:你最终输出的“控制轨迹”是 shared 轨迹的最优,而不是“加权后最优计划”的 shared 部分
            #      如果你想执行的是“最优全计划”的 shared 段,应改为 ft_final_list[0]['shared_plan']
            if len(ft_list_valid) > 0:
                best_trajectory = ft_list_valid[0]
            elif len(ft_list_invalid) > 0:
                best_trajectory = ft_list_invalid[0]  # 若全不可行,仍返回成本最低者,供上层降级处理
                # raise NoLocalTrajectoryFoundError('Failed. No valid frenét path found')
            # else: 如果两者都空,需要抛错或 fallback

        self.exec_timer.stop_timer("simulation/total")

        # 只需要记录多项式离散出的第二个采样点,便可为下一次迭代提供初始条件
        # =========================
        # K) 更新 self._trajectory(供下一步用 index=1 取初值)
        # =========================
        # NOTE: 你这里把 v_mps 设置为 best_trajectory.s_d(纵向速度),而不是 best_trajectory.v(全局速度)
        #      如果后续模块期望全局速度,可能会有偏差        
        self._trajectory = {
            "s_loc_m": best_trajectory.s,
            "d_loc_m": best_trajectory.d,
            "d_d_loc_mps": best_trajectory.d_d,
            "d_dd_loc_mps2": best_trajectory.d_dd,
            "x_m": best_trajectory.x,
            "y_m": best_trajectory.y,
            "psi_rad": best_trajectory.yaw,
            "kappa_radpm": best_trajectory.curv,
            "v_mps": best_trajectory.s_d,
            "ax_mps2": best_trajectory.s_dd,
            "time_s": best_trajectory.t,
        }
        # 返回最佳计划、仿真状态记录、预测轨迹以及障碍更新信息,供上层决策使用
        return ft_final_list[0], state_rec, zPred, self.obst_new_state


if __name__ == "__main__":
    # ===== 命令行入口:加载配置并调用 ScenarioEvaluator =====
    # 说明:
    # - 该脚本作为主程序运行时进入此分支(而被 import 时不会执行)
    # - 主要流程:解析命令行参数 -> 处理场景路径 -> 读取各类配置 JSON -> 构建 FrenetCreator
    #           -> 构建 ScenarioEvaluator -> 执行评测(可选 cProfile 性能采样)
    import argparse
    # argparse:用于解析命令行参数,例如 --scenario 与 --time
    # print("(frenet_planner_main)project begin!!!")
    from planner.plannertools.evaluate import ScenarioEvaluator
    # ScenarioEvaluator:评测器,通常负责:
    # - 加载场景
    # - 调用规划器生成轨迹/决策
    # - 进行碰撞检测、风险评估、统计指标输出等
    # - 将评测结果写入日志/报告目录
    from planner.Frenet.plannertools.frenetcreator import FrenetCreator
    # FrenetCreator:规划器创建器/工厂类(planner_creator)
    # - 通常根据 settings_dict 配置创建具体的 Frenet 规划器实例
    # - 供 ScenarioEvaluator 在评测时按需调用

    parser = argparse.ArgumentParser()
    # 创建参数解析器
    parser.add_argument("--scenario", default="recorded/hand-crafted/DEU_Muc-4_2_T-1"
                                              ".xml")
    # --scenario:指定要评测的场景路径
    # 默认值被拆成两段字符串拼接(Python 会自动连接相邻字符串常量)
    parser.add_argument("--time", action="store_true")  # 若传入 --time,则启用 cProfile 输出性能
    # --time:布尔开关参数
    # - 不传入时 args.time == False
    # - 传入 --time 时 args.time == True,用于启用 cProfile 性能分析并输出报告
    args = parser.parse_args()
    # 解析命令行参数,生成 args 对象(包含 args.scenario 与 args.time)

    # 场景路径兼容处理:
    # - 如果传入的 --scenario 参数包含 "commonroad" 字样,说明可能是带有 "scenarios/" 前缀的完整路径
    # - 则通过 split("scenarios/") 仅保留其后相对场景路径部分
    if "commonroad" in args.scenario:
        scenario_path = args.scenario.split("scenarios/")[-1]
    else:
        scenario_path = args.scenario
    # scenario_path:最终用于 evaluator.eval_scenario(...) 的场景相对路径/标识
    # print("(frenet_planner_main)scenario_path:", scenario_path)
    # 载入规划、风险与应急配置,必要时启用可视化
    # settings_dict:统一配置字典,用于驱动 FrenetCreator 与 ScenarioEvaluator 的行为
    settings_dict = load_planning_json("planning_fast.json")
    # 读取规划相关配置(例如 Frenet 采样参数、代价权重、约束等)
    # 注意:load_planning_json 函数在此片段外定义/导入,这里假定它会返回 dict

    settings_dict["contingency_settings"] = load_contingency_json("contingency.json")
    # 加载应急/预案配置(例如 fallback 策略、紧急制动、应急轨迹等),并写入 settings_dict

    settings_dict["risk_dict"] = risk_dict = load_risk_json()
    # 加载风险配置,写入 settings_dict["risk_dict"]
    # 同时把同一对象绑定到 risk_dict(便于后续直接使用；此处片段中未继续使用 risk_dict)

    if not args.time:
        settings_dict["evaluation_settings"]["show_visualization"] = True
    # 当不启用性能分析时(正常评测模式),打开可视化
    # 原因:可视化通常会降低速度、影响 profiling 结果,因此 args.time=True 时不打开

    eval_directory = (
        pathlib.Path(__file__).resolve().parents[0].joinpath("results").joinpath("eval")
    )
    # print("(frenet_planner_main)eval_directory:", eval_directory)
    # eval_directory:评测输出目录(碰撞报告等)
    # 路径构造说明:
    # - pathlib.Path(__file__).resolve():当前脚本文件的绝对路径
    # - parents[0]:当前脚本所在目录
    # - joinpath("results").joinpath("eval"):拼接 results/eval
    # 最终 eval_directory 指向:<当前脚本目录>/results/eval

    # Create the frenet creator
    frenet_creator = FrenetCreator(settings_dict)
    # 使用配置字典创建 FrenetCreator
    # FrenetCreator 通常会根据 settings_dict 内的 frenet_settings / evaluation_settings 等
    # 在评测时创建并配置实际的 Frenet 规划器对象

    # Create the scenario evaluator
    evaluator = ScenarioEvaluator(
        planner_creator=frenet_creator,
        # planner_creator:传入规划器创建器(而不是直接传规划器),方便评测器统一构建/复用规划器实例

        vehicle_type=settings_dict["evaluation_settings"]["vehicle_type"],
        # vehicle_type:车辆类型(例如轿车/卡车),用于车辆动力学/几何尺寸/碰撞模型等

        path_to_scenarios=pathlib.Path(
            os.path.join(mopl_path, "beliefplanning/scenarios/")
        ).resolve(),
        # path_to_scenarios:场景根目录(绝对路径)
        # - os.path.join(mopl_path, "beliefplanning/scenarios/"):将 mopl_path 与子目录拼接
        # - pathlib.Path(...).resolve():转换为绝对路径并规范化
        # 注意:mopl_path 在此片段外定义/导入,这里假定是项目根路径或某个工作目录

        log_path=pathlib.Path("./log/example").resolve(),
        # log_path:日志输出目录(绝对路径)
        # 例如存放评测过程记录、规划器输出、调试信息等
        collision_report_path=eval_directory,
        # collision_report_path:碰撞报告/评测结果汇总输出目录
        timing_enabled=settings_dict["evaluation_settings"]["timing_enabled"],
        # timing_enabled:是否启用评测器内部的计时统计(与 cProfile 不同,通常是自定义计时点)
    )
    # print("(frenet_planner_main传入evaluator)path_to_scenarios:", evaluator.path_to_scenarios)
    # print("(frenet_planner_main传入evaluator)log_path:", evaluator.log_path)
    # print("(frenet_planner_main传入evaluator)collision_report_path:", evaluator.collision_report_path)
    def main():
        """主评测循环；在 `--time` 模式下供 cProfile 采样使用。"""
        # 这里将 eval_scenario 包装成函数 main(),是为了在 cProfile.run('main()') 中直接采样
        # evaluator.eval_scenario(...):执行单场景评测
        # print("(frenet_planner_main传入evaluator.eval_scenario):", scenario_path)
        _ = evaluator.eval_scenario(scenario_path)
        # 打印 return_dict 方便快速查看评估结果,避免再去查日志或生成文件
        # print("(frenet_planner_main)eval_scenario 返回:", json.dumps(return_dict, indent=2, ensure_ascii=False))


    if args.time:
        # ===== 性能分析模式(cProfile) =====
        import cProfile
        # cProfile:Python 内置性能分析器,输出函数级别耗时统计
        cProfile.run('main()', "output.dat")
        # 对 main() 执行过程进行 profiling,并将原始统计数据写入 output.dat

        # 计算采样轨迹数量,用于命名性能报告
        no_trajectores = settings_dict["frenet_settings"]["frenet_parameters"]["n_v_samples"] * len(
            settings_dict["frenet_settings"]["frenet_parameters"]["d_list"])
        import pstats

        sortby = pstats.SortKey.CUMULATIVE
        # 排序方式:按 cumulative time(累计耗时)排序
        # cumulative:某函数自身耗时 + 它调用的所有子函数耗时
        with open(f"cProfile/{scenario_path.split('/')[-1]}_{no_trajectores}.txt", "w") as f:
            # 输出报告文件路径说明:
            # - cProfile/:假设存在该目录,用于存放性能报告
            # - scenario_path.split('/')[-1]:取场景文件名(不含目录)
            # - 加上 no_trajectores:把采样规模编码进文件名,便于比较不同采样参数下性能
            p = pstats.Stats("output.dat", stream=f).sort_stats(sortby)
            # 从 output.dat 读取统计数据,将输出流定向到文件 f,并按 cumulative 排序
            p.sort_stats(sortby).print_stats()
            # 打印统计结果到文件(默认会输出较多函数条目)
    else:
        # ===== 正常运行模式:直接执行评测 =====
        main()
        # 不做 cProfile 性能采样,直接运行 main()
        # 此模式下上方已启用可视化 settings_dict["evaluation_settings"]["show_visualization"] = True

# EOF
