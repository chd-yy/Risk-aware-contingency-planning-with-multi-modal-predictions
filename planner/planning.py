# ============================================
# 这段代码的用途(总体说明):
# 1) 定义一个 PlanningAgent(规划智能体/车辆),每个仿真步:
#    - 先用 predictor 预测其它车(可选)
#    - 再用 planner 规划一条新的轨迹
#    - 如果有控制器/动力学模型则用它跟踪轨迹,否则“理想跟踪”直接把状态设置成轨迹的下一点
#
# 2) 提供一个函数 add_ego_vehicles_to_scenario,把“自车(ego)”以 DynamicObstacle 的形式加到场景里
#
# 3) 定义 Planner 基类,包含全局路径规划(这里你写死了一条直线 global path)
#
# 4) check_curvature_of_global_path:检查全局路径曲率是否过大,过大则删除一些点来变平滑
# ============================================

# ===== CommonRoad 相关的类导入 =====
# Scenario:场景(道路、障碍物、车道线等都在里面)
from commonroad.scenario.scenario import Scenario

# PlanningProblem:规划问题(起点状态、目标区域等)
# PlanningProblemSet:一组规划问题(可能一个场景多个 ego)
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.geometry.shape import Rectangle

# State:状态(位置、速度、朝向等)
# Trajectory:轨迹(由一系列 State 组成)
from commonroad.scenario.trajectory import State, Trajectory
# TrajectoryPrediction:预测(用一条轨迹作为预测)
from commonroad.prediction.prediction import TrajectoryPrediction

from agent_sim.agent import Agent
import numpy as np
import os
import sys
# pickle:Python 序列化(把对象保存到文件里/从文件里读取)
import pickle

# --------------------------------------------
# 这段是为了把“项目根目录”加入 sys.path,方便 import 你自己写的模块
# __file__:当前文件的路径
# os.path.abspath(__file__):得到当前文件的绝对路径
# os.path.dirname(...):取上一级目录
# 两次 dirname 就相当于“当前文件的上上级目录”
# --------------------------------------------
module_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(module_path)

# ===== 你项目中的异常类型 =====
# GoalReachedNotification:到达目标时会抛出这个异常(通知上层结束)
# NoGlobalPathFoundError:找不到全局路径
# ScenarioCompatibilityError:场景与规划器不兼容(比如时间步长不一致)
from commonroad_helper_functions.exceptions import (
    GoalReachedNotification,
    NoGlobalPathFoundError,
    ScenarioCompatibilityError,
)

# GoalReachedChecker:检查是否到达目标
from planner.utils.goalcheck import GoalReachedChecker
# LaneletPathPlanner:基于 lanelet(车道网络)的全局路径规划器
from planner.GlobalPath.lanelet_based_planner import LaneletPathPlanner
from planner.utils.timers import ExecTimer  # 引入定时器类
from planner.Frenet.utils.helper_functions import get_max_curvature  # 引入计算最大曲率的函数
from planner.Frenet.utils.calc_trajectory_cost import distance # 引入计算路径距离的函数
# CubicSpline2D:二维三次样条曲线(用离散点拟合成平滑曲线)
from commonroad_helper_functions.utils.cubicspline import CubicSpline2D # 引入二维样条曲线工具类

# 规划代理类
class PlanningAgent(Agent):
    def __init__(
            self,
            scenario: Scenario,          # CommonRoad 场景
            agent_id: int,               # 这个智能体/车辆的 ID
            predictor=None,              # 预测器(可选),用于预测其它车辆未来轨迹
            planner=None,                # 规划器(必须有),用于生成自车轨迹
            control_dynamics=None,       # 控制器+动力学模型(可选)
            enable_logging: bool = True, # 是否记录日志
            log_path: str = "/log",      # 日志路径
            debug_step: bool = False,    # 是否一步一步调试
    ):

        # initialize parent class (Agent)
        super().__init__(
            scenario=scenario,
            agent_id=agent_id,
            enable_logging=enable_logging,
            log_path=log_path,
            debug_step=debug_step,
        )

        self.__control_dynamics = control_dynamics

        # predictor
        self.__predictor = predictor

        # planner
        self.__planner = planner

        # 初始化一些状态变量:s为弧长,d为横向偏移,d_d为横向速度,d_dd为横向加速度
        # additional state variables:
        # s: arc-length:
        # d: lateral offset from reference spline
        # d_d: lateral velocity
        # d_dd: lateral acceleration
        self._s = 0.0
        self._d = 0.0
        self._d_d = 0.0
        self._d_dd = 0.0

        # 如果状态没有初始化加速度,手动设为0
        # missing initial state variables
        if not hasattr(self.state, "acceleration"):
            self._state.acceleration = 0.0

    def _step_agent(self, delta_time):
        """Step Agent

        This is the step function for a planning agent.
        It first calls the predictor, then the planner and calculates the new state according to the specified controller and dynamic model.

        :param delta_time: time since previous step
        """
        """Step Agent(每个仿真时间步调用一次)

        这个函数是 planning agent 的“单步执行”:
        1) 如果有 predictor,就先做预测
        2) 调 planner 生成轨迹
        3) 如果有控制器/动力学,就用它“跟踪轨迹”得到新状态
           否则假设“理想跟踪”:直接把新状态设为轨迹的第 2 个点

        :param delta_time: 距离上一步过去了多少秒(这里你没用到它)
        """

        # ==================================================
        # 1) 预测其它车辆(如果 predictor 存在)
        # ==================================================
        if self.predictor is not None:
            # with prediction
            # predictor.step(...) 的含义:让预测器“在当前时刻”给出预测
            # obstacle_id_list:需要预测的动态障碍物 ID 列表
            # multiprocessing=False:不使用多进程
            self.predictor.step(
                scenario=self.scenario,
                time_step=self.time_step,
                obstacle_id_list=list(self.scenario.dynamic_obstacles.keys()),
                multiprocessing=False,
            )
            # 得到预测结果
            prediction = self.predictor.prediction_result
        else:
            # without prediction
            # 如果没有预测器,设为None
            prediction = None
        # ==================================================
        # 2) 调用 planner 生成轨迹(轨迹通常是未来几秒的采样点)
        # ==================================================
        best_traj, state_rec, zPred, obs_state = self.planner.step(
                                                                scenario=self.scenario,
                                                                current_lanelet_id=self.current_lanelet_id,
                                                                time_step=self.time_step,
                                                                ego_state=self.state,
                                                                prediction=prediction,
                                                                )
        # 如果有控制动态模型,执行状态更新, 否则假设理想跟踪,直接从轨迹中获取下一个状态
        if self.control_dynamics is not None:
            self._state = self.control_dynamics.step(self.planner.trajectory)

        else:
            # --------------------------------------------
            # 没有控制器/动力学:就假设车辆能“完美跟踪轨迹”
            # 所以直接把 ego 的状态设置为轨迹的下一个采样点
            # --------------------------------------------

            # 检查轨迹时间间隔是否等于场景 dt
            # self.planner.trajectory["time_s"][1] 是轨迹第二个时间点
            # self.dt 是场景时间步长(比如 0.1s)
            if self.planner.trajectory["time_s"][1] != self.dt:
                # current assumption: a trajectory is planned every time step
                raise ScenarioCompatibilityError(
                    # "场景的时间步长与轨迹的时间步长或重新规划的频率不匹配。"
                    "The scenario time step size does not match the time discretization of "
                    "the trajectory or the replanning frequency."
                )
            else:
                # assuming that the trajectory is discretized with the scenario time step size and replanned every
                # time step
                # 轨迹时间间隔正确
                # i=1 表示取轨迹的第 2 个点作为“下一时刻的状态”
                # (i=0 一般就是当前状态)
                i = 1

                # 更新位置 position(二维坐标)
                self._state.position = np.array(
                    [
                        self.planner.trajectory["x_m"][i],
                        self.planner.trajectory["y_m"][i],
                    ]
                )
                # 更新朝向 orientation(弧度)
                self._state.orientation = self.planner.trajectory["psi_rad"][i]
                self._state.velocity = self.planner.trajectory["v_mps"][i]
                self._state.acceleration = self.planner.trajectory["ax_mps2"][i]

                self._s = self.planner.trajectory["s_loc_m"][i]
                self._d = self.planner.trajectory["d_loc_m"][i]
                self._d_d = self.planner.trajectory["d_d_loc_mps"][i]
                self._d_dd = self.planner.trajectory["d_dd_loc_mps2"][i]

        return best_traj, state_rec, zPred, obs_state

    @property
    def predictor(self):
        """Predictor of the planning agent"""
        return self.__predictor

    @property
    def planner(self):
        """Planner of the planning agent"""
        return self.__planner

    @property
    def control_dynamics(self):
        """Controller and dynamic model of the planning agent"""
        return self.__control_dynamics

# 添加自车到场景
def add_ego_vehicles_to_scenario(
        scenario: Scenario, planning_problem_set: PlanningProblemSet, vehicle_params
):
    """Add Ego Vehicle to Scenario

    This function adds an ego vehicle represented by a dynamic obstacle for each planning problem specified in the planning problem set

    :param scenario: commonroad scenario
    :param planning_problem_set: commonroad planning problem set
    :return: new scenario, dictionary with agent IDs as key and planning problem ID as value
    """
    """Add Ego Vehicle to Scenario(把自车加入场景)

    对 planning_problem_set 里的每一个 planning problem:
    - 创建一个动态障碍物(DynamicObstacle)来表示这个 ego 车
    - 设置车辆形状(Rectangle)
    - 设置初始状态为 planning_problem.initial_state
    - 设置 prediction 为一个仅包含 initial_state 的 TrajectoryPrediction(很短)

    :param scenario: CommonRoad 场景
    :param planning_problem_set: 一组规划问题
    :param vehicle_params: 车辆参数(这里用到 l=长度, w=宽度)
    :return: (new_scenario, agent_planning_problem_id_assignment)
             agent_planning_problem_id_assignment 是一个 dict:
             key=ego车辆ID, value=对应的 planning_problem_id
    """
    # dictionary to gather all ego IDs and assign them to a planning problem
    # 用一个字典记录:每个 ego 的 agent_id 对应哪个 planning_problem_id
    agent_planning_problem_id_assignment = {}  # 存储每个代理的规划问题ID

    # planning_problem_set.planning_problem_dict 是一个字典:
    # key=planning_problem_id, value=planning_problem
    for (
            planning_problem_id,
            planning_problem,
    ) in planning_problem_set.planning_problem_dict.items():
        # Obstacle shape
        # TODO: Implement geometric parameters like length and width
        # 障碍物形状
        obstacle_shape = Rectangle(length=vehicle_params.l, width=vehicle_params.w)
        
        # --------------------------------------------
        # 生成一个新的障碍物 ID(保证不和场景里现有对象冲突)
        # --------------------------------------------
        agent_id = scenario.generate_object_id()
        # 记录这个 agent_id 对应哪个 planning_problem_id
        agent_planning_problem_id_assignment[agent_id] = planning_problem_id

        # --------------------------------------------
        # 创建一个轨迹 Trajectory:
        # initial_time_step=1 表示轨迹从 time step 1 开始
        # state_list=[planning_problem.initial_state]:轨迹只有一个点(初始状态)
        # 注意:这只是“占位”的预测
        # --------------------------------------------
        trajectory = Trajectory(
            initial_time_step=1, state_list=[planning_problem.initial_state]
        )
        # print(f"Created initial trajectory for planning_problem {planning_problem_id}: {trajectory}")

        # 用轨迹 + 形状,创建预测对象 TrajectoryPrediction
        prediction = TrajectoryPrediction(trajectory=trajectory, shape=obstacle_shape)

        # --------------------------------------------
        # 创建一个 DynamicObstacle(动态障碍物)表示 ego 车
        # obstacle_type=ObstacleType.CAR 表示是汽车
        # obstacle_shape:车辆几何形状
        # initial_state:初始状态
        # prediction:预测(这里只包含一个点)
        # --------------------------------------------
        ego_obstacle = DynamicObstacle(
            obstacle_id=agent_id,
            obstacle_type=ObstacleType.CAR,
            obstacle_shape=obstacle_shape,
            initial_state=planning_problem.initial_state,
            prediction=prediction,
        )

        # 将自车添加到场景中
        # add object to scenario
        scenario.add_objects(ego_obstacle)

    return scenario, agent_planning_problem_id_assignment

# ==========================================================
# Planner:主规划器基类(抽象类)
# ==========================================================
class Planner(object):
    """Main Planner Class(规划器基类)

    这是一个“基类”,它本身不会生成轨迹。
    你需要继承它并实现 _step_planner() 才能真正规划。
    """

    def __init__(
            self,
            scenario: Scenario,                 # 场景
            planning_problem: PlanningProblem,  # 当前要解决的规划问题
            ego_id: int,                        # ego 车辆 ID
            vehicle_params,                     # 车辆参数
            exec_timer=None,                    # 计时器(可选)
    ):
        # commonroad scenario
        self.__scenario = scenario

        # current lanelet the planning vehicle is moving on
        self.__current_lanelet_id = None

        # planning problem for the planner
        self.__planning_problem = planning_problem

        # initial time step
        self.__time_step = 0

        # ID of the planning vehicle
        self.__ego_id = ego_id

        # initial state of the planning vehicle
        # ego 当前状态:先用 planning problem 的 initial_state 初始化,后续每步更新
        self.__ego_state = planning_problem.initial_state

        # minimum trajectory length
        # 最小轨迹长度(离散点数量),这里是 50 个点
        self.__min_trajectory_length = 50

        # prediction
        self.__prediction = None

        # 你自定义的:障碍物新状态(看起来像 [x,y,v,psi] 的某种格式)
        # TODO(yanjun): hard coded for now, need to be updated according to the actual scenario and prediction
        self.obst_new_state = [0, 5.4, 20, 0]

        # --------------------------------------------
        # ExecTimer:用于统计某些代码块耗时
        # timing_enabled=False 表示默认不开启统计(或不打印)
        # --------------------------------------------
        # Timer for timing execution times.
        self.__exec_timer = (
            ExecTimer(timing_enabled=False) if exec_timer is None else exec_timer
        )

        # --------------------------------------------
        # 全局路径相关变量
        # global_path:一条离散点组成的路线
        # __reference_spline:把离散点拟合成平滑曲线,便于 Frenet 坐标计算
        # --------------------------------------------
        # Variables that contain the global path
        # TODO Remove everything except the reference spline.....
        # NOTE They are all referenced within the Frenet Planner -> tricky to remove....
        self.global_path_length = None
        self.global_path = None
        self.global_path_to_goal = None
        self.global_path_after_goal = None
        self.__reference_spline = None

        # 规划一条全局路径(你这里写死了一个直线)
        self.plan_global_path(scenario, planning_problem, vehicle_params)
        # 应急轨迹,先置空
        self.contingency_trajectory = None

        # start with equal belief over both mode
        # self.belief_1 = 0.5
        # self.belief_2 = 0.5
        '''
        self.belief = [0.5, 0.5]
        self.belief_list = []
        '''

        # File path of the pickle file
        '''
        file_path = 'belief_updater.pickle'

        # Load the vector from the pickle file
        with open(file_path, 'rb') as file:
            self.belief = pickle.load(file)

        self.belief = self.belief[7:]
        self.min_dist = 1000
        self.traveled_dist = 0
        '''
        # --------------------------------------------
        # belief:这里像是你做“模式/意图”概率的先验
        # 你把它设置成 3 个模式的概率:[0.5, 0.4, 0.1]
        # --------------------------------------------
        self.belief = [0.5, 0.4, 0.1]

        # trajectory
        dt = 0.1
        self._trajectory = {
            "s_loc_m": np.zeros(self.min_trajectory_length),        # Frenet s
            "d_loc_m": np.zeros(self.min_trajectory_length),        # Frenet d
            "d_d_loc_mps": np.zeros(self.min_trajectory_length),    # d 的一阶导(横向速度)
            "d_dd_loc_mps2": np.zeros(self.min_trajectory_length),  # d 的二阶导(横向加速度)
            "x_m": np.zeros(self.min_trajectory_length),            # 世界坐标 x
            "y_m": np.zeros(self.min_trajectory_length),            # 世界坐标 y
            "psi_rad": np.zeros(self.min_trajectory_length),        # 航向角/朝向(弧度)
            "kappa_radpm": np.zeros(self.min_trajectory_length),    # 曲率 kappa(rad/m)
            "v_mps": np.zeros(self.min_trajectory_length),          # 速度(m/s)
            "ax_mps2": np.zeros(self.min_trajectory_length),        # 纵向加速度(m/s^2)
            "time_s": np.arange(0, dt * self.min_trajectory_length, dt),  # 时间序列
        }

        # 用来记录速度/代价的数组(可能用于调试或画图)
        self.velocity_vec = []
        self.cost_vec = []

        # open_loop:是否开环(不根据反馈修正),这里默认 False,表示闭环(会根据反馈修正)
        self.open_loop = False

    def step(
            self,
            scenario: Scenario,
            current_lanelet_id: int,
            time_step: int,
            ego_state: State,
            prediction=None,
            v_max=50,
    ):
        """Main Step Function of the Planner(规划器每步调用一次)

        功能:
        - 把外部传入的场景、ego_state、预测结果等保存到内部变量
        - 然后调用 _step_planner() 生成新轨迹

        注意:
        - _step_planner() 是抽象方法,必须由子类实现
        """
        """Main Step Function of the Planner

        This method generates a new trajectory for the current scenario und prediction.
        It is a wrapper for the planner-type depending actual step method "_step_planner()" that updates the trajectory.
        "_step_planner()" must be overloaded by inheriting classes that implement planning methods.

        :param scenario: commonroad scenario
        :param current_lanelet_id: current lanelet id of the planning vehicle
        :param time_step: current time step
        :param ego_state: current state of the planning vehicle
        :param prediction: prediction
        :param v_max: maximum allowed velocity on the current lanelet in m/s
        """
        # 更新内部场景引用
        self.__scenario = scenario
        self.__current_lanelet_id = current_lanelet_id
        self.__time_step = time_step
        self.__ego_state = ego_state
        self.__prediction = prediction

        # TODO: Include maximum allowed speed
        self.__v_max = v_max

        # call the planner-type depending step function to generate a new trajectory
        # 生成轨迹:由子类实现
        best_traj, state_rec, zPred, obs_state = self._step_planner()
        return best_traj, state_rec, zPred, obs_state

    def _step_planner(self):
        """Planner step function(子类必须实现)

        这个方法必须由继承 Planner 的子类来实现,否则就报错。
        """
        """Planner step function

        This method directly changes the planner trajectory. It must be overloaded by an inheriting planner class.
        There is no basic trajectory planning implemented.
        """
        raise NotImplementedError(
            "No basic trajectory planning implemented. "
            "Overload the method _step_planner() to generate a trajectory."
        )

    def __check_goal_reached(self):
        """检查是否到达目标(内部函数)

        goal_checker 会把当前 ego 状态注册进去,然后判断是否到达目标。
        到达则抛出 GoalReachedNotification,通知上层可以结束仿真。
        """
        # Get the ego vehicle
        self.goal_checker.register_current_state(self.ego_state)

        # 如果到达目标且未超时
        if self.goal_checker.goal_reached_status():
            raise GoalReachedNotification("Goal reached in time!")
        # 如果到达目标但超时(ignore_exceeded_time=True 表示忽略超时条件判断)
        elif self.goal_checker.goal_reached_status(ignore_exceeded_time=True):
            raise GoalReachedNotification("Goal reached but time exceeded!")

    def plan_global_path(self, scenario, planning_problem, vehicle_params, initial_state=None):
        """Plan a global path(规划全局路径)

        说明:
        - “全局路径”一般是从起点到目标区域的一条可行路线(离散点或车道序列)
        - 你的实现里:直接手写了 100 个点,形成一条水平直线 y=5.4
        - 然后用这些点生成 CubicSpline2D 作为 reference_spline

        Args:
            scenario: 场景(你这里没用到)
            planning_problem: 规划问题(你这里也没用到)
            vehicle_params: 车辆参数(你这里也没用到)
            initial_state: 可选,初始状态(你这里没用到)
        """

        """Plan a global path to the planning's problem target area.

        Args:
            scenario (_type_): _description_
            planning_problem (_type_): _description_
            vehicle_params (_type_): _description_
            initial_state (_type_, optional): _description_. Defaults to None.

        Raises:
            NoGlobalPathFoundError: _description_
        """
        # 用计时器包裹这个代码块,统计“规划全局路径”的耗时
        with self.exec_timer.time_with_cm("initialization/plan global path"):
            x = []
            y = []
            # 生成 100 个点:
            # x 从 -8 开始,每次 +3
            # y 固定为 -5.4
            for i in range(100):
                x.append(-8 + i * 3)
                y.append(-5.4)
                # x = [-8, -6, -4, -2, 0, 2, 4, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 20]
                # y = [-1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8]
            self.global_path = np.transpose(np.array([x, y]))
            # create the reference spline from the global path
            # 通过离散点生成平滑的 2D 样条曲线(reference spline),方便后续 Frenet 坐标计算
            # TODO(yanjun): 增加可视化
            self.__reference_spline = CubicSpline2D(
                x=self.global_path[:, 0], y=self.global_path[:, 1]
            )

    # ========== 一堆只读属性(getter)==========
    @property
    def planning_problem(self):
        """Planning problem to be solved"""
        return self.__planning_problem

    @property
    def goal_checker(self):
        """Return the goal checker."""
        return self.__goal_checker

    @property
    def exec_timer(self):
        """Return the exec_timer object."""
        return self.__exec_timer

    @property
    def reference_spline(self):
        """Return the reference spline object."""
        return self.__reference_spline

    @property
    def scenario(self):
        """Commonroad scenario"""
        return self.__scenario

    @property
    def time_step(self):
        """Current time step"""
        return self.__time_step

    @property
    def ego_id(self):
        """ID of the planning vehicle"""
        return self.__ego_id

    @property
    def ego_state(self):
        """Current state of the planning vehicle"""
        return self.__ego_state

    @property
    def min_trajectory_length(self):
        """Minimum length of the planned trajectory"""
        return self.__min_trajectory_length

    @property
    def trajectory(self):
        """Planned trajectory"""
        return self._trajectory

    @property
    def prediction(self):
        """Prediction"""
        return self.__prediction

    @property
    def v_max(self):
        """maximum velocity"""
        return self.__v_max

    @property
    def current_lanelet_id(self):
        """Current lanelet"""
        return self.__current_lanelet_id

# ==========================================================
# check_curvature_of_global_path:检查全局路径曲率是否过大
# ==========================================================
# TODO move to separate file
def check_curvature_of_global_path(
        global_path: np.ndarray, planning_problem, vehicle_params, ego_state
):
    """
    检查全局路径的曲率(拐弯程度)是否过大。

    如果曲率过大:
    - 删除一些路径点让路径更平滑(减少急转弯)
    - 还会插入一个点来确保起始朝向(初始方向)一致

    参数解释:
    global_path:形状 (N,2) 的数组,每行是 [x,y]
    planning_problem:用于获取初始速度
    vehicle_params:用于计算车辆可承受的最大曲率(跟最小转弯半径有关)
    ego_state:用于获取初始位置和朝向
    """
    """
    Check the curvature of the global path.

    If the curvature is to high, points of the global path are removed to smooth the global path. In addition, a new point is added which ensures the initial orientation.

    Args:
        global_path (np.ndarray): Coordinates of the global path.

    Returns:
        np.ndarray: Coordinates of the new, smooth global path.

    """
    global_path_curvature_ok = False

    # get start velocity of the planning problem
    start_velocity = planning_problem.initial_state.velocity

    # 计算在初始速度下允许的最大曲率
    # get_max_curvature 返回 (max_curvature, 其它信息)
    max_initial_curvature, _ = get_max_curvature(
        vehicle_params=vehicle_params, v=start_velocity
    )

    # get x and y from the global path
    global_path_x = global_path[:, 0].tolist()
    global_path_y = global_path[:, 1].tolist()

    # add a point to the global path to ensure the initial orientation of the planning problem
    # never delete this point or the initial point
    # ---------------------------------------------------
    # 插入一个点:确保初始朝向(方向)在路径上体现
    # 为什么要插？
    # - 如果路径的前两个点方向和车辆初始朝向差很多,会导致样条拟合出来一开始就很别扭
    # 插入的位置 index=1(第二个点),保证不会删掉它
    # ---------------------------------------------------
    new_x = ego_state.position[0] + np.cos(ego_state.orientation) * 0.1
    new_y = ego_state.position[1] + np.sin(ego_state.orientation) * 0.1
    global_path_x.insert(1, new_x)
    global_path_y.insert(1, new_y)

    # ---------------------------------------------------
    # 不断检查曲率,如果不满足就删点,直到满足为止
    # ---------------------------------------------------
    # check if the curvature of the global path is ok
    while global_path_curvature_ok is False:
        # calc the already covered arc length for the points of global path
        # 先计算每个路径点对应的“累计弧长” s
        # global_path_s[i] 表示从起点到第 i 个点走过的距离
        global_path_s = [0.0]

        for i in range(len(global_path_x) - 1):
            p_start = np.array([global_path_x[i], global_path_y[i]])
            p_end = np.array([global_path_x[i + 1], global_path_y[i + 1]])

            # distance(p_start, p_end):两点距离
            # global_path_s[-1]:上一个累计弧长
            global_path_s.append(distance(p_start, p_end) + global_path_s[-1])
        # ---------------------------------------------------
        # 用数值微分(np.gradient)计算曲率
        #
        # 曲率公式(二维曲线):
        # kappa = |x' y'' - y' x''| / (x'^2 + y'^2)^(3/2)
        #
        # 这里 x' = dx/ds, y' = dy/ds
        #      x'' = d^2x/ds^2, y'' = d^2y/ds^2
        # ---------------------------------------------------
        # calculate the curvature of the global path
        dx = np.gradient(global_path_x, global_path_s)
        dy = np.gradient(global_path_y, global_path_s)

        ddx = np.gradient(dx, global_path_s)
        ddy = np.gradient(dy, global_path_s)

        curvature = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5

        # 先假设 OK,如果发现有一个点曲率太大就改成 False 并删点
        # loop through every curvature of the global path
        global_path_curvature_ok = True
        for i in range(len(curvature)):
            # check if the curvature of the global path is too big
            # be generous (* 2.) since the curvature might increase again when converting to a cubic spline
            # ---------------------------------------------------
            # 第一轮:用 “初始速度对应的最大曲率” 来检查
            # 乘以 2.0:你这里是“放宽判断”,因为后续样条拟合可能会让曲率稍微变大
            # ---------------------------------------------------
            if (curvature[i] * 2.0) > max_initial_curvature:
                # if the curvature is too big, then delete the global path point to smooth the global path
                # never remove the first (starting) point of the global path
                # and never remove the second point of the global path to keep the initial orientation
                index_closest_path_point = max(2, i)
                # only consider the first part of the global path, later on it gets smoothed by the frenét planner itself
                # 这里没有限制 10m,直接删
                if global_path_s[index_closest_path_point] <= 10.0:
                    global_path_x.pop(index_closest_path_point)
                    global_path_y.pop(index_closest_path_point)
                    global_path_curvature_ok = False
                    break
        # ---------------------------------------------------
        # 第二轮:还要检查曲率是否超过 “速度=0 时的最大曲率”
        # 这是更严格的限制(接近车辆最小转弯半径)
        # ---------------------------------------------------
        # also check if the curvature is smaller than the turning radius anywhere
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
                # only consider the first part of the global path, later on it gets smoothed by the frenét planner itself
                # 没有限制删除路径点时的距离,只要发现曲率过大,会直接删除路径中的点。
                global_path_x.pop(index_closest_path_point)
                global_path_y.pop(index_closest_path_point)
                global_path_curvature_ok = False
                break
    # ---------------------------------------------------
    # 根据最终的 global_path_x / global_path_y 重新拼回 (N,2) numpy 数组
    # ---------------------------------------------------
    # create the new global path
    new_global_path = np.array([np.array([global_path_x[0], global_path_y[0]])])
    for i in range(1, len(global_path_y)):
        new_global_path = np.concatenate(
            (
                new_global_path,
                np.array([np.array([global_path_x[i], global_path_y[i]])]),
            )
        )

    return new_global_path

# EOF
