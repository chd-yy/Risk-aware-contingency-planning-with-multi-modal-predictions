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
from itertools import product, combinations

# Third party imports
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# CommonRoad 核心对象
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State
from commonroad.scenario.obstacle import ObstacleRole
# CommonRoad-DC 碰撞检测
# from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import (
#     create_collision_checker,
# )
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
    draw_contingent_trajectories, draw_all_contingent_trajectories, draw_all_plans, \
    clear_plot_snapshots, replay_plot_snapshots

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
    build_multimodal_gmm_predictions,
    get_rule_based_base_predictions,
    update_yield_challenge_belief,
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

CREDIBLE_SET_ALPHA = 0.05


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
            sensor_radius: float = 11155.0,
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
        self.obst_new_state = None

        # 记录各时刻的轨迹/状态/预测,便于复现或可视化
        self.traj_rec = []
        self.state_rec = []
        self.zPred_rec = []
        self.exec_time = []
        self.branch_w_rec = []
        self.obstacle_belief_history = {}
        self.obstacle_mode_belief = {}
        self.joint_belief_history = {
            "timesteps": [],
            "joint_weights": [],
            "joint_mode_selections": [],
            "joint_mode_labels": [],
        }
        self.credible_joint_history = {
            "timesteps": [],
            "credible_indices": [],
            "credible_weights": [],
            "credible_labels": [],
            "credible_cumulative_prob": [],
            "credible_set_sizes": [],
            "alpha": CREDIBLE_SET_ALPHA,
        }
        self.recoverability_history = {
            "timesteps": [],
            "shared_plan_count": [],
            "recoverable_shared_plan_count": [],
            "credible_set_size": [],
            "recoverability_indicator": [],
        }
        self.adaptive_branching_history = {
            "timesteps": [],
            "selected_branch_time": [],
            "selected_branch_step": [],
            "selected_separability": [],
            "separability_threshold": [],
            "candidate_times": [],
            "separability_series": [],
            "selection_reason": [],
        }

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
                self.frenet_parameters = frenet_parameters
                self.contingency_settings = contingency_parameters
                if isinstance(contingency_parameters, dict) and "frenet_parameters" in contingency_parameters:
                    self.contingency_parameters = contingency_parameters["frenet_parameters"]
                else:
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
                # with self.exec_timer.time_with_cm(
                #         "initialization/initialize collision checker"
                # ):
                #     # deep copy 场景,移除 ego obstacle,避免自碰撞
                #     cc_scenario = copy.deepcopy(self.scenario)
                #     cc_scenario.remove_obstacle(
                #         obstacle=[cc_scenario.obstacle_by_id(ego_id)]
                #     )
                #     try:
                #         self.collision_checker = create_collision_checker(cc_scenario)
                #     except Exception:
                #         raise BrokenPipeError("Collision Checker fails.") from None
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
        xRef = np.array([0.5, 1.8, 15, 0])  # MPC 期望状态(横向居中、纵向保持速度)

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
        # 在这里更新障碍物状态(obst_new_state)以供下一步使用(比如持续跟踪障碍物或模拟其行为)
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

        current_v = self.ego_state.velocity
        max_acceleration = self.p.longitudinal.a_max
        shared_t_list = list(self.frenet_parameters["t_list"])
        shared_branch_time = float(max(shared_t_list))
        shared_start_idx = int(round(shared_branch_time / self.frenet_parameters["dt"]))
        # =========================
        # E) 多模态预测:目标车道 -> 模式轨迹 -> GMM -> 初始模式概率
        # =========================
        state_rec = None
        zPred = None
        visible_area = None  # NOTE: 这里没计算可见域,绘图时可能会用到
        predictions = None
        base_predictions = None
        prediction_belief = None
        adaptive_branching_info = None
        joint_mode_selections = []
        credible_joint_mode_selections = []
        credible_joint_mode_weights = [1.0]

        with self.exec_timer.time_with_cm("simulation/prediction"):
            visible_obstacle_ids = get_obstacles_in_radius(
                scenario=self.scenario,
                ego_id=self.ego_id,
                ego_state=self.ego_state,
                radius=self.sensor_radius,
            )
            # breakpoint()
            pred_horizon = max(
                int(max(self.frenet_parameters["t_list"]) / self.frenet_parameters["dt"]),
                int(max(self.contingency_parameters["t_list"]) / self.contingency_parameters["dt"]),
                1,
            ) + 1
            visible_dynamic_obstacle_ids = [
                obstacle_id for obstacle_id in visible_obstacle_ids
                if obstacle_id != self.ego_id
                and self.scenario.obstacle_by_id(obstacle_id).obstacle_role == ObstacleRole.DYNAMIC
            ]

            if self.prediction is not None:
                # breakpoint()
                base_predictions = {
                    obstacle_id: self.prediction[obstacle_id]
                    for obstacle_id in visible_dynamic_obstacle_ids
                    if obstacle_id in self.prediction
                }
                missing_obstacle_ids = [
                    obstacle_id for obstacle_id in visible_dynamic_obstacle_ids
                    if obstacle_id not in base_predictions
                ]
                if len(missing_obstacle_ids) > 0:
                    fallback_base_predictions = get_rule_based_base_predictions(
                        scenario=self.scenario,
                        obstacle_id_list=missing_obstacle_ids,
                        horizon=pred_horizon,
                        timestep=self.ego_state.time_step,
                        dt=self.scenario.dt,
                    )
                    if base_predictions is None:
                        base_predictions = fallback_base_predictions
                    else:
                        base_predictions.update(fallback_base_predictions)
                if len(base_predictions) > 0:
                    predictions = build_multimodal_gmm_predictions(
                        scenario=self.scenario,
                        base_prediction=base_predictions,
                        obstacle_id_list=list(base_predictions.keys()),
                        horizon=pred_horizon,
                        timestep=self.ego_state.time_step,
                    )
            elif len(visible_obstacle_ids) > 0:
                predictions = get_ground_truth_prediction(
                    obstacle_ids=visible_obstacle_ids,
                    scenario=self.scenario,
                    time_step=self.ego_state.time_step,
                    pred_horizon=pred_horizon,
                )

            if predictions is not None and len(predictions) > 0:
                predictions = get_orientation_velocity_and_shape_of_prediction(
                    predictions,
                    self.scenario,
                )
                predictions, self.obstacle_mode_belief = update_yield_challenge_belief(
                    predictions=predictions,
                    scenario=self.scenario,
                    ego_state=self.ego_state,
                    time_step=self.ego_state.time_step,
                    prior_belief=self.obstacle_mode_belief,
                    dt=self.scenario.dt,
                )
                prediction_belief = {
                    obstacle_id: pred["mode_prob"]
                    for obstacle_id, pred in predictions.items()
                    if pred.get("mode_prob") is not None
                }
                multimodal_obstacle_ids = []
                multimodal_mode_ranges = []
                for obstacle_id, pred in predictions.items():
                    pos_list = pred.get("pos_list")
                    if isinstance(pos_list, list) and len(pos_list) > 1:
                        multimodal_obstacle_ids.append(obstacle_id)
                        multimodal_mode_ranges.append(range(len(pos_list)))

                if len(multimodal_obstacle_ids) > 0:
                    joint_mode_selections = [
                        dict(zip(multimodal_obstacle_ids, mode_indices))
                        for mode_indices in product(*multimodal_mode_ranges)
                    ]
                if prediction_belief is not None:
                    self._record_obstacle_belief(
                        time_step=self.ego_state.time_step,
                        predictions=predictions,
                    )
                    self._record_joint_belief(
                        time_step=self.ego_state.time_step,
                        prediction_belief=prediction_belief,
                        joint_mode_selections=joint_mode_selections,
                        predictions=predictions,
                    )
                    if len(joint_mode_selections) > 0:
                        joint_mode_weights_all = self._compute_joint_mode_weights(
                            mode_belief=prediction_belief,
                            mode_selections=joint_mode_selections,
                        )
                        credible_joint_set = self._compute_credible_joint_set(
                            joint_weights=joint_mode_weights_all,
                            joint_labels=[
                                self._format_joint_mode_label(
                                    mode_selection=mode_selection,
                                    predictions=predictions,
                                )
                                for mode_selection in joint_mode_selections
                            ],
                            alpha=CREDIBLE_SET_ALPHA,
                        )
                        credible_joint_mode_selections = [
                            joint_mode_selections[idx]
                            for idx in credible_joint_set["indices"]
                        ]
                        credible_joint_mode_weights = list(
                            credible_joint_set["weights"]
                        )
                    else:
                        credible_joint_mode_selections = []
                        credible_joint_mode_weights = [1.0]

            adaptive_branching_info = self._select_adaptive_branch_time(
                predictions=predictions,
                credible_joint_mode_selections=credible_joint_mode_selections,
            )
            shared_branch_time = float(adaptive_branching_info["selected_branch_time"])
            shared_t_list = [shared_branch_time]
            shared_start_idx = int(round(shared_branch_time / self.frenet_parameters["dt"]))
            self._record_adaptive_branching(
                time_step=self.ego_state.time_step,
                branching_info=adaptive_branching_info,
            )

        # =========================
        # D) 生成候选 Frenet trajectories(shared horizon)
        # =========================
        shared_t_min = min(shared_t_list)
        shared_t_max = max(shared_t_list)

        max_v = min(
            current_v + (max_acceleration / 2.0) * shared_t_max, self.p.longitudinal.v_max
        )
        min_v = max(0.01, current_v - max_acceleration * shared_t_min)

        with self.exec_timer.time_with_cm("simulation/get v list"):
            v_list = get_v_list(
                v_min=min_v,
                v_max=max_v,
                v_cur=current_v,
                v_goal_min=self.v_goal_min,
                v_goal_max=self.v_goal_max,
                mode=self.frenet_parameters["v_list_generation_mode"],
                n_samples=self.frenet_parameters["n_v_samples"],
            )

        with self.exec_timer.time_with_cm("simulation/calculate trajectories/total"):
            d_list = np.linspace(-1.75, 1.75, 7)
            ft_list = calc_frenet_trajectories(
                c_s=c_s,
                c_s_d=c_s_d,
                c_s_dd=c_s_dd,
                c_d=c_d,
                c_d_d=c_d_d,
                c_d_dd=c_d_dd,
                d_list=d_list,
                t_list=shared_t_list,
                v_list=v_list,
                dt=self.frenet_parameters["dt"],
                csp=self.reference_spline,
                v_thr=self.frenet_parameters["v_thr"],
                exec_timer=self.exec_timer,
                t_min=shared_t_min,
                t_max=shared_t_max,
                max_acceleration=max_acceleration,
                max_velocity=self.p.longitudinal.v_max,
                v_goal_min=self.v_goal_min,
                v_goal_max=self.v_goal_max,
                mode=self.frenet_parameters["v_list_generation_mode"],
                n_samples=self.frenet_parameters["n_v_samples"],
                contin=False
            )

        def _get_joint_mode_weights(mode_belief, mode_selections):
            if not mode_selections:
                return [1.0]
            if not isinstance(mode_belief, dict) or len(mode_belief) == 0:
                uniform_weight = 1.0 / len(mode_selections)
                return [uniform_weight] * len(mode_selections)

            joint_weights = []
            for mode_selection in mode_selections:
                joint_weight = 1.0
                for obstacle_id, mode_idx in mode_selection.items():
                    obstacle_belief = mode_belief.get(obstacle_id)
                    if obstacle_belief is None or mode_idx >= len(obstacle_belief):
                        joint_weight = 0.0
                        break
                    joint_weight *= obstacle_belief[mode_idx]
                joint_weights.append(joint_weight)

            weight_sum = sum(joint_weights)
            if weight_sum <= 0.0:
                uniform_weight = 1.0 / len(mode_selections)
                return [uniform_weight] * len(mode_selections)
            inv_weight_sum = 1.0 / weight_sum
            return [weight * inv_weight_sum for weight in joint_weights]

        def _check_recoverability(final_plan, credible_mode_selections, contingency_required):
            if not contingency_required or len(credible_mode_selections) == 0:
                return True, []

            missing_modes = [
                mode_num
                for mode_num in range(len(credible_mode_selections))
                if mode_num not in final_plan
            ]
            return len(missing_modes) == 0, missing_modes
        # =========================
        # F) 可达集计算(责任相关)
        # =========================
        if self.responsibility and predictions is not None:
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
            '''
            如果当前是第0步，或者系统不是open-loop，

            那么根据当前时间步的belief值
            构造一个两元素概率向量：

            belief = [p, 1-p]
            '''
            belief = prediction_belief
            # belief(分支概率)用于风险/代价权重
            # NOTE: 你这里 belief=branch_w(来自 MPC),而不是 belief_updater 的输出
            # belief = branch_w
            # belief = [1] * 12
            # 基于碰撞/越界/舒适性指标筛选轨迹,返回按成本排序前的有效/无效集合
            ft_list_valid = sort_frenet_trajectories(
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
                if len(ft_list_valid) > 0 and getattr(ft_list_valid[0], "used_risk_fallback", False):
                    print(
                        f"[FrenetPlanner] timestep={self.ego_state.time_step}: "
                        "using max-risk fallback trajectory"
                    )

                # contingency 阶段的速度范围推算,使用 contingency_parameters 的 t_list
                max_acceleration = self.p.longitudinal.a_max
                t_min = min(self.contingency_parameters["t_list"])
                t_max = max(self.contingency_parameters["t_list"])

                # NOTE: 同样覆盖 contingency_parameters["d_list"],固定采样 -3.6..0
                # d_list = self.contingency_parameters["d_list"]
                d_list = np.linspace(-1.75, 1.75, 6)
                t_list = self.contingency_parameters["t_list"]

                ft_final_list = []       # 每个 shared 轨迹对应一个 final_plan(字典)
                ft_all_plans_list = []   # 用于绘图:保存 shared + 所有 contingent 候选
                recoverable_shared_plan_count = 0
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
                        ft_contingent_list = calc_frenet_trajectories(
                            c_s=plan.s[-1],
                            c_s_d=plan.s_d[-1],
                            c_s_dd=plan.s_dd[-1],
                            c_d=plan.d[-1],

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
                        for index in range(len(ft_contingent_list)):
                            ft_all_plans[index] = ft_contingent_list[index]

                        ft_all_plans_list.append(ft_all_plans)

                        for mode_num, mode_selection in enumerate(credible_joint_mode_selections):
                            ft_conting_list_valid = sort_frenet_trajectories(
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
                                exec_timer=self.exec_timer,
                                start_idx=shared_start_idx,
                                mode_num=mode_selection,
                                reach_set=(self.reach_set if self.responsibility else None)
                            )
                            ft_conting_list_valid.sort(key=lambda fp: fp.cost, reverse=False)
                            if len(ft_conting_list_valid) > 0:
                                final_plan[mode_num] = ft_conting_list_valid[0]

                    recoverable, missing_credible_modes = _check_recoverability(
                        final_plan=final_plan,
                        credible_mode_selections=credible_joint_mode_selections,
                        contingency_required=(t_list[0] != 0),
                    )
                    final_plan["recoverable"] = recoverable
                    final_plan["missing_credible_modes"] = list(missing_credible_modes)
                    final_plan["credible_set_size"] = len(credible_joint_mode_selections)

                    if recoverable:
                        recoverable_shared_plan_count += 1
                        ft_final_list.append(final_plan)

                self.recoverability_history["timesteps"].append(int(self.ego_state.time_step))
                self.recoverability_history["shared_plan_count"].append(int(len(ft_list_valid)))
                self.recoverability_history["recoverable_shared_plan_count"].append(
                    int(recoverable_shared_plan_count)
                )
                self.recoverability_history["credible_set_size"].append(
                    int(len(credible_joint_mode_selections))
                )
                self.recoverability_history["recoverability_indicator"].append(
                    int(recoverable_shared_plan_count > 0)
                )

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
                # breakpoint()
                for plan in ft_final_list:
                    # if len(plan) == 1:
                    #     # This means we have only a single plan along the horizon
                    #     # 只有 shared_plan,没有 contingent(例如 t_list[0]==0 或没算 contingent)
                    #     plan['cost'] = plan['shared_plan'].cost
                    # else:
                    plan['cost'] = plan['shared_plan'].cost
                    for mode_num, mode_weight in enumerate(credible_joint_mode_weights):
                        if mode_num in plan:
                            plan['cost'] += mode_weight * plan[mode_num].cost

                # sort the final plan
                # 最终按总代价排序,ft_final_list[0] 就是全计划最优
                # print("Final plans sorted")
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
            print(
                "Time step: {} | Velocity: {:.2f} m/s | Acceleration: {:.2f} m/s2".format(
                    self.time_step, current_v, c_s_dd
                )
            )
            # print some information about the frenet trajectories
            # 终端打印与本地绘图开关
            # breakpoint()
            if self.plot_frenet_trajectories:
                matplotlib.use("TkAgg")
                '''
                Highway_env_branch.plot_scenario(self.mpc, self.N_lane, self.time_step, self.ego_state,
                                                 self.obst_new_state, ft_final_list[0],
                                                 state_rec, zPred)
                '''
                # 记录用于最终画图
                self.traj_rec.append(ft_final_list[0])
                self.state_rec.append(state_rec)
                self.zPred_rec.append(zPred)
                self.branch_w_rec.append(belief)
                
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
                        global_path=self.global_path,
                        global_path_after_goal=self.global_path_after_goal,
                        driven_traj=self.driven_traj,
                        animation_area=50.0,
                        predictions=predictions,
                        base_predictions=base_predictions,
                        visible_area=visible_area,
                        valid_traj=ft_all_plans_list,  # 所有候选(按 shared 分组)
                        best_traj=ft_final_list,
                        open_loop=self.open_loop,
                    )

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(e)
            # 初始时刻保存 contingency_trajectory(用于 open loop 可能复用)
            # if self.ego_state.time_step == 0:
            #     self.contingency_trajectory = ft_final_list  # 第一次迭代记录全套计划,供可视化或调试

            # best trajectory
            # 选择 best trajectory(用于更新 self._trajectory)
            # NOTE: 这里 best_trajectory 取的是 ft_list_valid[0],而不是 ft_final_list[0]['shared_plan']
            #      也就是说:你最终输出的“控制轨迹”是 shared 轨迹的最优,而不是“加权后最优计划”的 shared 部分
            #      如果你想执行的是“最优全计划”的 shared 段,应改为 ft_final_list[0]['shared_plan']
            if len(ft_final_list) > 0:
                best_trajectory = ft_final_list[0]
            # elif len(ft_list_invalid) > 0:
            #     best_trajectory = {'shared_plan': ft_list_invalid[0], 'cost': ft_list_invalid[0].cost}
            #     raise RuntimeError('Failed. No valid frenét path found')
            else:
                raise RuntimeError("No Frenet plan available for current step")

        self.exec_timer.stop_timer("simulation/total")

        # 只需要记录多项式离散出的第二个采样点,便可为下一次迭代提供初始条件
        # =========================
        # K) 更新 self._trajectory(供下一步用 index=1 取初值)
        # =========================
        # NOTE: 你这里把 v_mps 设置为 best_trajectory.s_d(纵向速度),而不是 best_trajectory.v(全局速度)
        #      如果后续模块期望全局速度,可能会有偏差        
        self._trajectory = {
            "s_loc_m": best_trajectory['shared_plan'].s,
            "d_loc_m": best_trajectory['shared_plan'].d,
            "d_d_loc_mps": best_trajectory['shared_plan'].d_d,
            "d_dd_loc_mps2": best_trajectory['shared_plan'].d_dd,
            "x_m": best_trajectory['shared_plan'].x,
            "y_m": best_trajectory['shared_plan'].y,
            "psi_rad": best_trajectory['shared_plan'].yaw,
            "kappa_radpm": best_trajectory['shared_plan'].curv,
            "v_mps": best_trajectory['shared_plan'].s_d,
            "ax_mps2": best_trajectory['shared_plan'].s_dd,
            "time_s": best_trajectory['shared_plan'].t,
        }
        # 返回最佳计划、仿真状态记录、预测轨迹以及障碍更新信息,供上层决策使用
        return best_trajectory, state_rec, zPred, self.obst_new_state

    def _record_obstacle_belief(self, time_step, predictions):
        for obstacle_id, pred in predictions.items():
            belief_values = pred.get("mode_prob")
            if belief_values is None:
                continue
            history = self.obstacle_belief_history.setdefault(
                obstacle_id,
                {"timesteps": [], "beliefs": [], "mode_behavior": []},
            )
            history["timesteps"].append(int(time_step))
            history["beliefs"].append(list(belief_values))
            mode_behavior = pred.get("mode_behavior")
            if mode_behavior is not None:
                history["mode_behavior"] = list(mode_behavior)

    def _compute_joint_mode_weights(self, mode_belief, mode_selections):
        if not mode_selections:
            return [1.0]
        if not isinstance(mode_belief, dict) or len(mode_belief) == 0:
            uniform_weight = 1.0 / len(mode_selections)
            return [uniform_weight] * len(mode_selections)

        joint_weights = []
        for mode_selection in mode_selections:
            joint_weight = 1.0
            for obstacle_id, mode_idx in mode_selection.items():
                obstacle_belief = mode_belief.get(obstacle_id)
                if obstacle_belief is None or mode_idx >= len(obstacle_belief):
                    joint_weight = 0.0
                    break
                joint_weight *= obstacle_belief[mode_idx]
            joint_weights.append(joint_weight)

        weight_sum = sum(joint_weights)
        if weight_sum <= 0.0:
            uniform_weight = 1.0 / len(mode_selections)
            return [uniform_weight] * len(mode_selections)
        inv_weight_sum = 1.0 / weight_sum
        return [weight * inv_weight_sum for weight in joint_weights]

    def _format_joint_mode_label(self, mode_selection, predictions):
        if not mode_selection:
            return "shared only"

        label_parts = []
        for obstacle_id in sorted(mode_selection.keys()):
            mode_idx = mode_selection[obstacle_id]
            pred = predictions.get(obstacle_id, {})
            mode_behavior = pred.get("mode_behavior", [])
            if mode_idx < len(mode_behavior):
                mode_name = mode_behavior[mode_idx]
            else:
                mode_name = f"mode{mode_idx}"
            label_parts.append(f"{obstacle_id}={mode_name}")
        return ", ".join(label_parts)

    def _record_joint_belief(
        self,
        time_step,
        prediction_belief,
        joint_mode_selections,
        predictions,
    ):
        joint_weights = self._compute_joint_mode_weights(
            mode_belief=prediction_belief,
            mode_selections=joint_mode_selections,
        )
        joint_labels = [
            self._format_joint_mode_label(
                mode_selection=mode_selection,
                predictions=predictions,
            )
            for mode_selection in joint_mode_selections
        ]

        self.joint_belief_history["timesteps"].append(int(time_step))
        self.joint_belief_history["joint_weights"].append(list(joint_weights))
        self.joint_belief_history["joint_mode_selections"] = [
            dict(selection) for selection in joint_mode_selections
        ]
        self.joint_belief_history["joint_mode_labels"] = list(joint_labels)
        credible_set = self._compute_credible_joint_set(
            joint_weights=joint_weights,
            joint_labels=joint_labels,
            alpha=CREDIBLE_SET_ALPHA,
        )
        self.credible_joint_history["timesteps"].append(int(time_step))
        self.credible_joint_history["credible_indices"].append(
            list(credible_set["indices"])
        )
        self.credible_joint_history["credible_weights"].append(
            list(credible_set["weights"])
        )
        self.credible_joint_history["credible_labels"].append(
            list(credible_set["labels"])
        )
        self.credible_joint_history["credible_cumulative_prob"].append(
            float(credible_set["cumulative_prob"])
        )
        self.credible_joint_history["credible_set_sizes"].append(
            int(len(credible_set["indices"]))
        )

    def _compute_credible_joint_set(self, joint_weights, joint_labels, alpha=0.05):
        if joint_weights is None or len(joint_weights) == 0:
            return {
                "indices": [],
                "weights": [],
                "labels": [],
                "cumulative_prob": 0.0,
            }

        target_mass = max(0.0, min(1.0, 1.0 - float(alpha)))
        ranked_indices = sorted(
            range(len(joint_weights)),
            key=lambda idx: joint_weights[idx],
            reverse=True,
        )

        cumulative_prob = 0.0
        credible_indices = []
        credible_weights = []
        credible_labels = []
        for idx in ranked_indices:
            weight = float(joint_weights[idx])
            credible_indices.append(idx)
            credible_weights.append(weight)
            credible_labels.append(
                joint_labels[idx] if idx < len(joint_labels) else f"joint {idx}"
            )
            cumulative_prob += weight
            if cumulative_prob >= target_mass:
                break

        return {
            "indices": credible_indices,
            "weights": credible_weights,
            "labels": credible_labels,
            "cumulative_prob": cumulative_prob,
        }

    def _get_adaptive_branching_config(self):
        adaptive_cfg = {}
        if hasattr(self, "contingency_settings") and isinstance(self.contingency_settings, dict):
            adaptive_cfg = self.contingency_settings.get("adaptive_branching", {})
        elif isinstance(self.contingency_parameters, dict):
            adaptive_cfg = self.contingency_parameters.get("adaptive_branching", {})
        if not isinstance(adaptive_cfg, dict):
            adaptive_cfg = {}

        max_branch_time = float(
            adaptive_cfg.get(
                "max_branch_time",
                max(self.frenet_parameters["t_list"]),
            )
        )
        min_branch_time = float(
            adaptive_cfg.get(
                "min_branch_time",
                self.frenet_parameters["dt"],
            )
        )
        candidate_dt = float(
            adaptive_cfg.get(
                "candidate_dt",
                self.frenet_parameters["dt"],
            )
        )
        candidate_dt = max(candidate_dt, 1e-3)
        min_branch_time = max(min_branch_time, candidate_dt)
        max_branch_time = max(max_branch_time, min_branch_time)

        explicit_candidate_times = adaptive_cfg.get("candidate_times")
        if explicit_candidate_times:
            candidate_times = sorted(
                {
                    round(float(value), 6)
                    for value in explicit_candidate_times
                    if min_branch_time <= float(value) <= max_branch_time
                }
            )
        elif len(self.frenet_parameters["t_list"]) > 1:
            candidate_times = sorted(
                {
                    round(float(value), 6)
                    for value in self.frenet_parameters["t_list"]
                    if min_branch_time <= float(value) <= max_branch_time
                }
            )
        else:
            candidate_times = [
                round(float(value), 6)
                for value in np.arange(
                    min_branch_time,
                    max_branch_time + 0.5 * candidate_dt,
                    candidate_dt,
                )
            ]

        if len(candidate_times) == 0:
            candidate_times = [round(max_branch_time, 6)]

        return {
            "enabled": bool(adaptive_cfg.get("enabled", False)),
            "candidate_times": candidate_times,
            "separability_threshold": float(
                adaptive_cfg.get("separability_threshold", 1.0)
            ),
        }

    @staticmethod
    def _bhattacharyya_distance(mu_a, cov_a, mu_b, cov_b):
        mu_a = np.asarray(mu_a, dtype=float).reshape(-1)
        mu_b = np.asarray(mu_b, dtype=float).reshape(-1)
        cov_a = np.asarray(cov_a, dtype=float)
        cov_b = np.asarray(cov_b, dtype=float)

        dim = mu_a.shape[0]
        regularizer = 1e-6 * np.eye(dim)
        cov_a_reg = cov_a + regularizer
        cov_b_reg = cov_b + regularizer
        cov_bar = 0.5 * (cov_a_reg + cov_b_reg)

        delta_mu = (mu_a - mu_b).reshape(-1, 1)
        try:
            inv_cov_bar = np.linalg.inv(cov_bar)
        except np.linalg.LinAlgError:
            inv_cov_bar = np.linalg.pinv(cov_bar)

        quad_term = 0.125 * float(delta_mu.T @ inv_cov_bar @ delta_mu)
        det_cov_a = max(float(np.linalg.det(cov_a_reg)), 1e-12)
        det_cov_b = max(float(np.linalg.det(cov_b_reg)), 1e-12)
        det_cov_bar = max(float(np.linalg.det(cov_bar)), 1e-12)
        log_term = 0.5 * math.log(
            det_cov_bar / math.sqrt(det_cov_a * det_cov_b)
        )
        return max(0.0, quad_term + log_term)

    def _compute_joint_separability_at_step(
        self,
        predictions,
        credible_joint_mode_selections,
        future_step_idx,
    ):
        if len(credible_joint_mode_selections) < 2:
            return math.inf

        min_pair_distance = math.inf
        for idx_a, idx_b in combinations(range(len(credible_joint_mode_selections)), 2):
            mode_selection_a = credible_joint_mode_selections[idx_a]
            mode_selection_b = credible_joint_mode_selections[idx_b]
            obstacle_ids = sorted(
                set(mode_selection_a.keys()) | set(mode_selection_b.keys())
            )

            pair_distance = 0.0
            valid_obstacle_count = 0
            for obstacle_id in obstacle_ids:
                pred = predictions.get(obstacle_id)
                if pred is None:
                    continue

                pos_list = pred.get("pos_list")
                cov_list = pred.get("cov_list")
                if not isinstance(pos_list, list) or not isinstance(cov_list, list):
                    continue

                mode_idx_a = int(mode_selection_a.get(obstacle_id, 0))
                mode_idx_b = int(mode_selection_b.get(obstacle_id, 0))
                if (
                    mode_idx_a >= len(pos_list)
                    or mode_idx_b >= len(pos_list)
                    or mode_idx_a >= len(cov_list)
                    or mode_idx_b >= len(cov_list)
                ):
                    continue

                mode_mean_a = np.asarray(pos_list[mode_idx_a], dtype=float)
                mode_mean_b = np.asarray(pos_list[mode_idx_b], dtype=float)
                mode_cov_a = np.asarray(cov_list[mode_idx_a], dtype=float)
                mode_cov_b = np.asarray(cov_list[mode_idx_b], dtype=float)
                if len(mode_mean_a) == 0 or len(mode_mean_b) == 0:
                    continue

                step_idx = min(
                    future_step_idx,
                    len(mode_mean_a) - 1,
                    len(mode_mean_b) - 1,
                    len(mode_cov_a) - 1,
                    len(mode_cov_b) - 1,
                )
                pair_distance += self._bhattacharyya_distance(
                    mu_a=mode_mean_a[step_idx],
                    cov_a=mode_cov_a[step_idx],
                    mu_b=mode_mean_b[step_idx],
                    cov_b=mode_cov_b[step_idx],
                )
                valid_obstacle_count += 1

            if valid_obstacle_count == 0:
                continue
            min_pair_distance = min(min_pair_distance, pair_distance)

        return min_pair_distance if np.isfinite(min_pair_distance) else 0.0

    def _select_adaptive_branch_time(
        self,
        predictions,
        credible_joint_mode_selections,
    ):
        config = self._get_adaptive_branching_config()
        candidate_times = list(config["candidate_times"])
        default_branch_time = float(max(candidate_times))
        default_result = {
            "selected_branch_time": default_branch_time,
            "selected_branch_step": int(round(default_branch_time / self.scenario.dt)),
            "selected_separability": 0.0,
            "separability_threshold": float(config["separability_threshold"]),
            "candidate_times": candidate_times,
            "separability_series": [],
            "selection_reason": "adaptive_branching_disabled",
        }
        if not config["enabled"]:
            return default_result

        if predictions is None or len(predictions) == 0:
            default_result["selection_reason"] = "no_predictions"
            return default_result

        if len(credible_joint_mode_selections) < 2:
            default_result["selection_reason"] = "insufficient_credible_joint_modes"
            return default_result

        separability_series = []
        for candidate_time in candidate_times:
            future_step_idx = max(0, int(round(float(candidate_time) / self.scenario.dt)))
            separability_value = self._compute_joint_separability_at_step(
                predictions=predictions,
                credible_joint_mode_selections=credible_joint_mode_selections,
                future_step_idx=future_step_idx,
            )
            separability_series.append(float(separability_value))

        threshold = float(config["separability_threshold"])
        selected_idx = len(candidate_times) - 1
        selection_reason = "fallback_latest_branch_time"
        for idx, separability_value in enumerate(separability_series):
            if separability_value >= threshold:
                selected_idx = idx
                selection_reason = "first_separable_candidate"
                break

        return {
            "selected_branch_time": float(candidate_times[selected_idx]),
            "selected_branch_step": int(
                round(float(candidate_times[selected_idx]) / self.scenario.dt)
            ),
            "selected_separability": float(separability_series[selected_idx]),
            "separability_threshold": threshold,
            "candidate_times": candidate_times,
            "separability_series": separability_series,
            "selection_reason": selection_reason,
        }

    def _record_adaptive_branching(self, time_step, branching_info):
        self.adaptive_branching_history["timesteps"].append(int(time_step))
        self.adaptive_branching_history["selected_branch_time"].append(
            float(branching_info.get("selected_branch_time", 0.0))
        )
        self.adaptive_branching_history["selected_branch_step"].append(
            int(branching_info.get("selected_branch_step", 0))
        )
        self.adaptive_branching_history["selected_separability"].append(
            float(branching_info.get("selected_separability", 0.0))
        )
        self.adaptive_branching_history["separability_threshold"].append(
            float(branching_info.get("separability_threshold", 0.0))
        )
        self.adaptive_branching_history["candidate_times"].append(
            list(branching_info.get("candidate_times", []))
        )
        self.adaptive_branching_history["separability_series"].append(
            list(branching_info.get("separability_series", []))
        )
        self.adaptive_branching_history["selection_reason"].append(
            str(branching_info.get("selection_reason", "unknown"))
        )

    def save_obstacle_belief_plots(self, output_dir, scenario_name=None):
        if (
            len(self.obstacle_belief_history) == 0
            and len(self.joint_belief_history.get("timesteps", [])) == 0
            and len(self.credible_joint_history.get("timesteps", [])) == 0
            and len(self.recoverability_history.get("timesteps", [])) == 0
            and len(self.adaptive_branching_history.get("timesteps", [])) == 0
        ):
            return

        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        scenario_prefix = "" if scenario_name is None else f"{scenario_name}_"

        for obstacle_id, history in self.obstacle_belief_history.items():
            timesteps = history["timesteps"]
            belief_series = history["beliefs"]
            mode_behavior = history.get("mode_behavior", [])
            if len(timesteps) == 0 or len(belief_series) == 0:
                continue

            mode_count = max(len(values) for values in belief_series)
            fig, ax = plt.subplots()

            for mode_idx in range(mode_count):
                mode_values = [
                    values[mode_idx] if mode_idx < len(values) else np.nan
                    for values in belief_series
                ]
                ax.plot(
                    timesteps,
                    mode_values,
                    marker="o",
                    linewidth=1.5,
                    label=(
                        f"mode {mode_idx} ({mode_behavior[mode_idx]})"
                        if mode_idx < len(mode_behavior)
                        else f"mode {mode_idx}"
                    ),
                )

            ax.set_xlabel("timestep")
            ax.set_ylabel("belief")
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"Obstacle {obstacle_id} belief evolution")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(
                output_path.joinpath(
                    f"{scenario_prefix}obstacle_{obstacle_id}_belief.png"
                )
            )
            plt.close(fig)

        joint_timesteps = self.joint_belief_history.get("timesteps", [])
        joint_weight_series = self.joint_belief_history.get("joint_weights", [])
        joint_labels = self.joint_belief_history.get("joint_mode_labels", [])
        if len(joint_timesteps) > 0 and len(joint_weight_series) > 0:
            joint_count = max(len(values) for values in joint_weight_series)
            if joint_count > 0:
                fig, ax = plt.subplots(figsize=(12, 6))
                for joint_idx in range(joint_count):
                    joint_values = [
                        values[joint_idx] if joint_idx < len(values) else np.nan
                        for values in joint_weight_series
                    ]
                    label = (
                        f"joint {joint_idx}: {joint_labels[joint_idx]}"
                        if joint_idx < len(joint_labels)
                        else f"joint {joint_idx}"
                    )
                    ax.plot(
                        joint_timesteps,
                        joint_values,
                        marker="o",
                        linewidth=1.5,
                        label=label,
                    )

                ax.set_xlabel("timestep")
                ax.set_ylabel("joint belief")
                ax.set_ylim(0.0, 1.0)
                ax.set_title("Joint scenario belief evolution")
                ax.grid(True, alpha=0.3)

                if len(joint_labels) > 0:
                    legend_anchor_x = 1.02
                    ax.legend(loc="upper left", bbox_to_anchor=(legend_anchor_x, 1.0), fontsize=8)
                else:
                    ax.legend()

                fig.tight_layout()
                fig.savefig(
                    output_path.joinpath(
                        f"{scenario_prefix}joint_belief.png"
                    ),
                    bbox_inches="tight",
                )
                plt.close(fig)

        credible_timesteps = self.credible_joint_history.get("timesteps", [])
        credible_sizes = self.credible_joint_history.get("credible_set_sizes", [])
        credible_probs = self.credible_joint_history.get("credible_cumulative_prob", [])
        if len(credible_timesteps) > 0 and len(credible_sizes) > 0:
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(
                credible_timesteps,
                credible_sizes,
                marker="o",
                linewidth=1.5,
                color="tab:blue",
                label="credible set size",
            )
            ax1.set_xlabel("timestep")
            ax1.set_ylabel("credible set size", color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            ax1.grid(True, alpha=0.3)

            ax2 = ax1.twinx()
            ax2.plot(
                credible_timesteps,
                credible_probs,
                marker="x",
                linewidth=1.2,
                color="tab:red",
                label="credible cumulative prob",
            )
            ax2.set_ylabel("cumulative prob", color="tab:red")
            ax2.tick_params(axis="y", labelcolor="tab:red")
            ax2.set_ylim(0.0, 1.05)

            fig.suptitle(
                f"Credible joint scenario set (1-a, a={self.credible_joint_history['alpha']:.2f})"
            )
            fig.tight_layout()
            fig.savefig(
                output_path.joinpath(
                    f"{scenario_prefix}credible_joint_set_summary.png"
                ),
                bbox_inches="tight",
            )
            plt.close(fig)

            credible_dump = {
                "alpha": self.credible_joint_history["alpha"],
                "timesteps": [],
            }
            for idx, timestep in enumerate(credible_timesteps):
                credible_dump["timesteps"].append(
                    {
                        "time_step": int(timestep),
                        "credible_indices": self.credible_joint_history["credible_indices"][idx],
                        "credible_weights": self.credible_joint_history["credible_weights"][idx],
                        "credible_labels": self.credible_joint_history["credible_labels"][idx],
                        "credible_cumulative_prob": self.credible_joint_history["credible_cumulative_prob"][idx],
                        "credible_set_size": self.credible_joint_history["credible_set_sizes"][idx],
                    }
                )
            with open(
                output_path.joinpath(
                    f"{scenario_prefix}credible_joint_set.json"
                ),
                "w",
                encoding="utf-8",
            ) as credible_file:
                json.dump(credible_dump, credible_file, indent=2, ensure_ascii=False)

        recoverability_timesteps = self.recoverability_history.get("timesteps", [])
        recoverable_counts = self.recoverability_history.get("recoverable_shared_plan_count", [])
        shared_counts = self.recoverability_history.get("shared_plan_count", [])
        recoverability_credible_sizes = self.recoverability_history.get("credible_set_size", [])
        recoverability_indicators = self.recoverability_history.get("recoverability_indicator", [])
        if (
            len(recoverability_timesteps) > 0
            and len(shared_counts) == len(recoverability_timesteps)
            and len(recoverable_counts) == len(recoverability_timesteps)
        ):
            recoverability_ratio = []
            for shared_count, recoverable_count in zip(shared_counts, recoverable_counts):
                if shared_count <= 0:
                    recoverability_ratio.append(0.0)
                else:
                    recoverability_ratio.append(float(recoverable_count) / float(shared_count))

            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(
                recoverability_timesteps,
                shared_counts,
                marker="o",
                linewidth=1.5,
                color="tab:blue",
                label="shared plan count",
            )
            ax1.plot(
                recoverability_timesteps,
                recoverable_counts,
                marker="s",
                linewidth=1.5,
                color="tab:green",
                label="recoverable shared count",
            )
            ax1.set_xlabel("timestep")
            ax1.set_ylabel("plan count")
            ax1.grid(True, alpha=0.3)

            ax2 = ax1.twinx()
            ax2.plot(
                recoverability_timesteps,
                recoverability_ratio,
                marker="x",
                linewidth=1.2,
                color="tab:red",
                label="recoverability ratio",
            )
            ax2.set_ylabel("recoverability ratio", color="tab:red")
            ax2.tick_params(axis="y", labelcolor="tab:red")
            ax2.set_ylim(0.0, 1.05)

            if len(recoverability_indicators) == len(recoverability_timesteps):
                ax2.step(
                    recoverability_timesteps,
                    recoverability_indicators,
                    where="post",
                    linewidth=1.2,
                    color="tab:orange",
                    linestyle="--",
                    label="recoverability indicator",
                )

            if len(recoverability_credible_sizes) == len(recoverability_timesteps):
                ax1.plot(
                    recoverability_timesteps,
                    recoverability_credible_sizes,
                    marker="^",
                    linewidth=1.2,
                    color="tab:purple",
                    label="credible set size",
                )

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

            fig.suptitle("Shared-plan recoverability summary")
            fig.tight_layout()
            fig.savefig(
                output_path.joinpath(
                    f"{scenario_prefix}recoverability_summary.png"
                ),
                bbox_inches="tight",
            )
            plt.close(fig)

            recoverability_dump = {"timesteps": []}
            for idx, timestep in enumerate(recoverability_timesteps):
                recoverability_dump["timesteps"].append(
                    {
                        "time_step": int(timestep),
                        "shared_plan_count": int(shared_counts[idx]),
                        "recoverable_shared_plan_count": int(recoverable_counts[idx]),
                        "recoverability_ratio": float(recoverability_ratio[idx]),
                        "recoverability_indicator": (
                            int(recoverability_indicators[idx])
                            if idx < len(recoverability_indicators)
                            else int(recoverable_counts[idx] > 0)
                        ),
                        "credible_set_size": (
                            int(recoverability_credible_sizes[idx])
                            if idx < len(recoverability_credible_sizes)
                            else None
                        ),
                    }
                )

            with open(
                output_path.joinpath(
                    f"{scenario_prefix}recoverability.json"
                ),
                "w",
                encoding="utf-8",
            ) as recoverability_file:
                json.dump(recoverability_dump, recoverability_file, indent=2, ensure_ascii=False)

        adaptive_timesteps = self.adaptive_branching_history.get("timesteps", [])
        adaptive_branch_times = self.adaptive_branching_history.get("selected_branch_time", [])
        adaptive_selected_sep = self.adaptive_branching_history.get("selected_separability", [])
        adaptive_thresholds = self.adaptive_branching_history.get("separability_threshold", [])
        adaptive_candidate_times = self.adaptive_branching_history.get("candidate_times", [])
        adaptive_sep_series = self.adaptive_branching_history.get("separability_series", [])
        if (
            len(adaptive_timesteps) > 0
            and len(adaptive_branch_times) == len(adaptive_timesteps)
            and len(adaptive_selected_sep) == len(adaptive_timesteps)
        ):
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(
                adaptive_timesteps,
                adaptive_branch_times,
                marker="o",
                linewidth=1.5,
                color="tab:blue",
                label="selected branch time",
            )
            ax1.set_xlabel("timestep")
            ax1.set_ylabel("branch time [s]", color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            ax1.grid(True, alpha=0.3)

            ax2 = ax1.twinx()
            ax2.plot(
                adaptive_timesteps,
                adaptive_selected_sep,
                marker="x",
                linewidth=1.2,
                color="tab:red",
                label="selected separability",
            )
            if len(adaptive_thresholds) == len(adaptive_timesteps):
                ax2.plot(
                    adaptive_timesteps,
                    adaptive_thresholds,
                    linewidth=1.2,
                    color="tab:orange",
                    linestyle="--",
                    label="separability threshold",
                )
            ax2.set_ylabel("separability", color="tab:red")
            ax2.tick_params(axis="y", labelcolor="tab:red")

            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

            fig.suptitle("Adaptive branching summary")
            fig.tight_layout()
            fig.savefig(
                output_path.joinpath(
                    f"{scenario_prefix}adaptive_branching_summary.png"
                ),
                bbox_inches="tight",
            )
            plt.close(fig)

            adaptive_dump = {"timesteps": []}
            for idx, timestep in enumerate(adaptive_timesteps):
                adaptive_dump["timesteps"].append(
                    {
                        "time_step": int(timestep),
                        "selected_branch_time": float(adaptive_branch_times[idx]),
                        "selected_branch_step": int(
                            self.adaptive_branching_history["selected_branch_step"][idx]
                        ),
                        "selected_separability": float(adaptive_selected_sep[idx]),
                        "separability_threshold": (
                            float(adaptive_thresholds[idx])
                            if idx < len(adaptive_thresholds)
                            else None
                        ),
                        "candidate_times": self.adaptive_branching_history["candidate_times"][idx],
                        "separability_series": self.adaptive_branching_history["separability_series"][idx],
                        "selection_reason": self.adaptive_branching_history["selection_reason"][idx],
                    }
                )

            with open(
                output_path.joinpath(
                    f"{scenario_prefix}adaptive_branching.json"
                ),
                "w",
                encoding="utf-8",
            ) as adaptive_file:
                json.dump(adaptive_dump, adaptive_file, indent=2, ensure_ascii=False)

            if (
                len(adaptive_candidate_times) == len(adaptive_timesteps)
                and len(adaptive_sep_series) == len(adaptive_timesteps)
                and len(adaptive_candidate_times) > 0
            ):
                unique_candidate_times = sorted(
                    {
                        round(float(candidate_time), 6)
                        for candidate_times in adaptive_candidate_times
                        for candidate_time in candidate_times
                    }
                )
                if len(unique_candidate_times) > 0:
                    sep_matrix = np.full(
                        (len(adaptive_timesteps), len(unique_candidate_times)),
                        np.nan,
                        dtype=float,
                    )
                    candidate_index = {
                        candidate_time: idx
                        for idx, candidate_time in enumerate(unique_candidate_times)
                    }
                    for row_idx, (candidate_times, sep_values) in enumerate(
                        zip(adaptive_candidate_times, adaptive_sep_series)
                    ):
                        for candidate_time, sep_value in zip(candidate_times, sep_values):
                            col_idx = candidate_index.get(round(float(candidate_time), 6))
                            if col_idx is not None:
                                sep_matrix[row_idx, col_idx] = float(sep_value)

                    fig, ax = plt.subplots(figsize=(10, 5))
                    image = ax.imshow(
                        sep_matrix.T,
                        aspect="auto",
                        origin="lower",
                        interpolation="nearest",
                        extent=[
                            adaptive_timesteps[0] - 0.5,
                            adaptive_timesteps[-1] + 0.5,
                            unique_candidate_times[0],
                            unique_candidate_times[-1],
                        ],
                        cmap="viridis",
                    )
                    ax.plot(
                        adaptive_timesteps,
                        adaptive_branch_times,
                        color="tab:red",
                        linewidth=1.5,
                        label="selected branch time",
                    )
                    ax.set_xlabel("timestep")
                    ax.set_ylabel("candidate branch time [s]")
                    ax.set_title("Adaptive branching separability evolution")
                    ax.legend(loc="upper right")
                    colorbar = fig.colorbar(image, ax=ax)
                    colorbar.set_label("Sep_k")
                    fig.tight_layout()
                    fig.savefig(
                        output_path.joinpath(
                            f"{scenario_prefix}adaptive_branching_separability_heatmap.png"
                        ),
                        bbox_inches="tight",
                    )
                    plt.close(fig)


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
    parser.add_argument("--scenario", default="recorded/hand-crafted/BRA_VilaVelha-92_1_T-10"
                                              ".xml")
    # --scenario:指定要评测的场景路径
    # 默认值被拆成两段字符串拼接(Python 会自动连接相邻字符串常量)
    parser.add_argument("--time", action="store_true", default=False)  # 若传入 --time,则启用 cProfile 输出性能
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

    def _write_single_exec_timing(return_dict, eval_dir, scenario_rel_path):
        if "exec_times_dict" not in return_dict:
            return

        exec_times_dict = return_dict["exec_times_dict"]
        if not exec_times_dict:
            return

        total_time = sum(exec_times_dict.get("total", []))
        if total_time <= 0.0:
            total_time = sum(sum(times) for times in exec_times_dict.values())
        if total_time <= 0.0:
            total_time = 1.0

        evaluated_dict = {}
        for key, item in exec_times_dict.items():
            if len(item) == 0:
                continue
            evaluated_dict[key] = " || ".join(
                [
                    f"Percentage from total: {100 * sum(item) / total_time:.3f} %",
                    f"Total time: {sum(item):.4f} s",
                    f"Number of calls: {len(item)}",
                    f"Avg exec time per call: {sum(item) / len(item):.6f}",
                ]
            )

        def _group_dict_recursive(input_rec_dict):
            working_dict = {}
            for key, item in input_rec_dict.items():
                split_key = key.split("/", 1)
                if len(split_key) == 1:
                    working_dict[split_key[0]] = item
                else:
                    if split_key[0] not in working_dict:
                        working_dict[split_key[0]] = {}
                    working_dict[split_key[0]][split_key[1]] = item
            for key, item in working_dict.items():
                if isinstance(item, dict):
                    working_dict[key] = _group_dict_recursive(item)
            return working_dict

        scenario_name = pathlib.Path(scenario_rel_path).stem
        file_path = eval_dir.joinpath(f"exec_timing_{scenario_name}.json")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as file_obj:
            json.dump(_group_dict_recursive(evaluated_dict), file_obj, indent=6)

    def main():
        """主评测循环；在 `--time` 模式下供 cProfile 采样使用。"""
        clear_plot_snapshots()
        # 这里将 eval_scenario 包装成函数 main(),是为了在 cProfile.run('main()') 中直接采样
        # evaluator.eval_scenario(...):执行单场景评测
        # print("(frenet_planner_main传入evaluator.eval_scenario):", scenario_path)
        return_dict = evaluator.eval_scenario(scenario_path)
        _write_single_exec_timing(return_dict, eval_directory, scenario_path)
        replay_plot_snapshots(
            fps=max(int(round(1.0 / evaluator.scenario.dt)), 1),
            save_path=str(
                eval_directory.joinpath(
                    f"plot_replay_{pathlib.Path(scenario_path).stem}.gif"
                )
            ),
            clear_after=True,
        )
        return return_dict

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
