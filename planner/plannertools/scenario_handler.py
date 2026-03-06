"""A module for executing a planner on a scenario."""
"""
这个模块的作用:在一个 CommonRoad 场景(scenario)上运行规划器(planner),
让“自车/智能体(ego agent)”每个时间步都规划一次轨迹并推进仿真。

你可以把它理解成:
1) 读入场景文件
2) 把自车放进去(或找到自车)
3) 给每辆自车创建一个 PlanningAgent(它里面有 planner)
4) 循环时间步:让每个 agent.step() 规划/行动,然后更新 scenario
5)(可选)检测是否发生碰撞,并计算伤害(harm)
"""

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader

# 计时工具:用于统计某些步骤耗时(可开可不开)
from beliefplanning.planner.utils.timers import ExecTimer
from beliefplanning.planner.utils.vehicleparams import VehicleParameters
# 超时工具:在某个代码块里执行超过 X 秒就中断并报错(防止卡死)
from beliefplanning.planner.utils.timeout import Timeout
# agent_sim 里的一些工具函数:清理/更新场景
from agent_sim.agent import clean_scenario, update_scenario

# 规划相关:PlanningAgent 是“智能体”,add_ego_vehicles_to_scenario 用于把自车加入场景
from beliefplanning.planner.planning import (
    PlanningAgent,
    add_ego_vehicles_to_scenario,
)

import copy

# 碰撞检测相关(CommonRoad DC)
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import (
    create_collision_checker,
    create_collision_object,
)

# 一个自定义异常:没有找到可行局部轨迹(这里也被用来作为“仿真终止原因”)
from commonroad_helper_functions.exceptions import NoLocalTrajectoryFoundError

# 伤害评估:harm_model 会根据碰撞信息计算 ego 和对方的伤害概率/值
from beliefplanning.risk_assessment.harm_estimation import harm_model

# 可视化:生成碰撞报告图片之类
from beliefplanning.risk_assessment.visualization.collision_visualization import (
    collision_vis,
)

# 一些数学/角度辅助函数
from beliefplanning.risk_assessment.helpers.collision_helper_function import (
    angle_range,
)

# 保护性伤害概率模型(逻辑回归),这里忽略角度
from beliefplanning.risk_assessment.utils.logistic_regression_symmetrical import \
    get_protected_inj_prob_log_reg_ignore_angle

# 将 ego 当前状态(位置、朝向等)转成碰撞检测需要的“TV obstacle”对象
from beliefplanning.planner.Frenet.utils.helper_functions import create_tvobstacle

# 读取 harm 模型参数(JSON)
from beliefplanning.planner.Frenet.configs.load_json import load_harm_parameter_json

# CommonRoad 中障碍物的角色/类型枚举
from commonroad.scenario.obstacle import (
    ObstacleRole,
    ObstacleType,
)


class ScenarioHandler:
    """Generic class for looping a scenario with a planner."""
    """
    这个类是“场景执行器/仿真器”的封装:
    - 负责读场景、初始化自车 agent、创建碰撞检测器
    - 然后按时间步循环调用 agent.step() 来推进仿真
    """
    def __init__(
            self,
            planner_creator,
            vehicle_type,
            path_to_scenarios,
            log_path,
            collision_report_path,
            timing_enabled=False,
    ):
        """
        构造函数:创建 ScenarioHandler 时会先把一些配置存起来。

        参数解释(小白版):
        - planner_creator:一个“工厂对象”,用于根据 ego_vehicle_id 创建 planner
        - vehicle_type:车辆类型,用来查车辆长宽等参数
        - path_to_scenarios:场景文件所在目录(或路径)
        - log_path:日志输出路径(如果 enable_logging=True)
        - collision_report_path:碰撞报告输出路径(如果你开启碰撞可视化)
        - timing_enabled:是否统计每一步耗时(True 会记录更多计时信息)
        """

        """
        Create scenario handler.

        Args:
            evaluation_settings (dict): Documentation in work ;)
            collision_report_path (Str): Path to save collision reports.
        """
        self.planner_creator = planner_creator
        self.vehicle_type = vehicle_type
        self.exec_timer = ExecTimer(timing_enabled=timing_enabled)
        self.timing_enabled = timing_enabled
        self.path_to_scenarios = path_to_scenarios
        self.log_path = log_path
        self.collision_report_path = collision_report_path

        # 下面这些属性先初始化为 None,之后在 _initialize() 里再赋值
        self.scenario_path = None
        self.scenario = None
        self.planning_problem_set = None
        self.vehicle_params = None
        # agent_list 存放所有 PlanningAgent(一般每辆 ego 车一个 agent)
        self.agent_list = None
        # vel_list 记录 ego 的速度历史(用于碰撞时的速度估计)
        self.vel_list = []
        # harm 用来统计伤害结果(你可以理解为一个“分类统计表”)
        self.harm = {
            "Ego": 0.0,
            "Unknown": 0.0,
            "Car": 0.0,
            "Truck": 0.0,
            "Bus": 0.0,
            "Bicycle": 0.0,
            "Pedestrian": 0.0,
            "Priority_vehicle": 0.0,
            "Parked_vehicle": 0.0,
            "Construction_zone": 0.0,
            "Train": 0.0,
            "Road_boundary": 0.0,
            "Motorcycle": 0.0,
            "Taxi": 0.0,
            "Building": 0.0,
            "Pillar": 0.0,
            "Median_strip": 0.0,
            "Total": 0.0,
        }

    def _initialize(self):
        """WIP."""
        """
        初始化阶段(非常关键):
        1) 读 scenario 文件(CommonRoad XML 等)
        2) 根据 vehicle_type 读车辆参数
        3) 把 ego 车辆加入场景,并记录“ego vehicle id”和“planning problem id”的对应关系
        4) 为每个 ego vehicle 创建对应的 PlanningAgent(里面有 planner)
        5) 创建碰撞检测器 collision_checker(可选功能,用于检测是否撞到别人/边界)
        """

        # 1) 读入场景文件:self.scenario_path 必须在调用 _initialize() 之前被设置好
        with self.exec_timer.time_with_cm("read scenario"):
            self.scenario, self.planning_problem_set = CommonRoadFileReader(
                self.scenario_path
            ).open()
        # print("Loaded scenario object:\n{}".format(self.scenario))
        # print("Loaded planning_problem_set:\n{}".format(self.planning_problem_set))
        # 2) 读车辆参数(车长、车宽等),用于规划和碰撞模型
        with self.exec_timer.time_with_cm("read vehicle parameters"):
            # get the parameters of the vehicle
            self.vehicle_params = VehicleParameters(self.vehicle_type)
        # print("Loaded vehicle parameters:\n{}".format(self.vehicle_params))

        # 3) 把 ego 车加到 scenario 里
        # add_ego_vehicles_to_scenario 返回:
        # - 更新后的 scenario(包含 ego 车)
        # - 一个映射:agent_id(ego obstacle_id) -> planning_problem_id
        with self.exec_timer.time_with_cm("add vehicle to scenario"):
            (
                self.scenario,
                self.agent_planning_problem_id_assignment,
            ) = add_ego_vehicles_to_scenario(
                scenario=self.scenario,
                planning_problem_set=self.planning_problem_set,
                vehicle_params=self.vehicle_params,
            )
        # Debug helper: print the agent<->planning problem pairing once per initialization
        print(
            f"agent_planning_problem_id_assignment: {self.agent_planning_problem_id_assignment}"
        )
        # 4) 遍历场景里的 dynamic_obstacles(动态障碍物),找到哪些是 ego,然后创建 agent
        self.agent_list = []
        for dynamic_obstacle in self.scenario.dynamic_obstacles:
            print(f"Processing dynamic obstacle id: {dynamic_obstacle.obstacle_id}")
            # 如果这个 obstacle_id 在映射里,说明它是一个 ego 车辆(需要规划     
            if (
                    dynamic_obstacle.obstacle_id
                    in self.agent_planning_problem_id_assignment
            ):
                self._create_planner_agent_for_ego_vehicle(dynamic_obstacle.obstacle_id)

        # create a collision checker
        # remove the ego vehicle from the scenario
        # 5) 创建碰撞检测器
        # 注意:碰撞检测器里我们不希望包含 ego 自己(否则会“撞到自己”)
        # 所以这里复制一份 scenario,并把 ego obstacles 移除,再创建 collision_checker
        with self.exec_timer.time_with_cm(
                "initialization/initialize collision checker"
        ):
            cc_scenario = copy.deepcopy(self.scenario)
            # 从 cc_scenario 里移除所有 ego agent 对应的 obstacle
            for agent in self.agent_list:
                cc_scenario.remove_obstacle(
                    obstacle=[cc_scenario.obstacle_by_id(agent.agent_id)]
                )

            try:
                self.collision_checker = create_collision_checker(cc_scenario)
            except Exception:
                # 这里捕获任何异常,然后统一转成 BrokenPipeError(你可以理解为初始化失败)
                raise BrokenPipeError("Collision Checker fails.") from None

    def _simulate(self):
        """
        执行仿真(核心循环):
        - 每个 time_step:对每个 agent 调用 agent.step()(让 planner 规划并更新 agent 状态)
        - 更新 scenario(把 agent 的新状态写回 scenario)
        - 如果时间太长还没达到目标,就报错停止

        你可以把它理解成“游戏主循环”。
        """
        """WIP.

        Raises:
            Exception: [description]
        """
        # Timeout(30, ...) 的意思:下面这段准备工作最多允许 30 秒,超时就抛异常
        # TODO(yanjun): 临时关闭超时限制,正常使用记得打开
        with Timeout(3000000, "Simulation preparation"):
            # clean_scenario 可能会做一些清理:例如移除无用障碍物、重置状态等(具体看实现)
            # 重置代理的轨迹:将代理的未来轨迹(trajectory.state_list)清空,仅保留初始状态。这样,代理不再拥有未来的运动预测。
            # 清空车道段分配:移除代理与车道段之间的所有分配信息。包括代理的中心点车道段分配(center_lanelet_assignment)和形状车道段分配(shape_lanelet_assignment)。
            # 从车道段移除代理:在场景中的每个车道段上,移除所有与该代理相关的障碍物信息,意味着该代理不再参与任何车道段的障碍物计算。
            self.scenario = clean_scenario(
                scenario=self.scenario, agent_list=self.agent_list
            )

            # get the max time for the simulation
            # max_time_steps 是“任务期望的最大时间步”(这里固定写 51)
            max_time_steps = 51

            # run the simulation no longer to avoid simulating forever
            # max_simulation_time_steps 是一个保险上限:避免陷入无限循环
            # 这里设为 max_time_steps 的 2 倍
            max_simulation_time_steps = int(max_time_steps * 2.0)
        # 进入主循环:time_step 从 0 到 max_simulation_time_steps - 1
        for time_step in range(max_simulation_time_steps):
            for agent in self.agent_list:
                # Also pass in timestep because it is needed in the recorder
                # That gets derived from the evaluator
                # self._check_collision(agent=agent, time_step=time_step)
                # 做一步规划:agent.step 会调用 planner 生成轨迹/控制
                self._do_simulation_step(agent=agent, time_step=time_step)
            # stop if the max_simulation_time is reached and no reason was found
            # 如果走到最后一步还没结束(比如没到达目标),就强制抛异常
            if time_step == (max_simulation_time_steps - 1):
                raise Exception("Goal was not reached in time")

            # update the scenario with new states of the agents
            # 把 agent 最新的状态写回 scenario,让下一步规划能看到“更新后的世界”
            self.scenario = update_scenario(
                scenario=self.scenario, agent_list=self.agent_list
            )

    def _do_simulation_step(self, **kwargs):
        """
        做单步规划(非常简单的封装):
        - kwargs["agent"] 是当前要执行的 agent
        - agent.step(scenario=...) 会让 planner 规划并更新自身状态
        """
        # This does the planning of the trajectory nothing else
        agent = kwargs["agent"]
        agent.step(scenario=self.scenario)

    def _create_planner_agent_for_ego_vehicle(self, ego_vehicle_id):
        """
        为某一辆 ego 车创建对应的 planner + agent。

        参数:
        - ego_vehicle_id:这辆车在 scenario 里的 obstacle_id
        """
        # TimeOut 10 seconds to create a planner
        # TODO(yanjun): 临时关闭超时限制,正常使用记得打开
        # with Timeout(10, "Creation of planner object"):
            # 先创建 planner(最多允许 10 秒)
        planner = self.planner_creator.get_planner(
            scenario_handler=self,
            ego_vehicle_id=ego_vehicle_id,
        )
        # 再创建 PlanningAgent(最多允许 10 秒)
        # TODO(yanjun): 临时关闭超时限制,正常使用记得打开            
        with Timeout(100000, "Creation of agent object"):
            self.agent_list.append(
                PlanningAgent(
                    scenario=self.scenario,
                    agent_id=ego_vehicle_id,
                    planner=planner,
                    control_dynamics=None,
                    enable_logging=False,
                    log_path=self.log_path,
                    debug_step=False,
                )
            )

    def _check_collision(self, agent, time_step):
        """
        碰撞检测 + 伤害计算(比较复杂的一块)。

        逻辑大致是:
        1) 取出 ego 当前的位置、速度、朝向
        2) 把 ego 当前状态转换成碰撞检测对象 current_state_collision_object
        3) 用 collision_checker 检测是否与其他障碍物/边界发生碰撞
        4) 如果碰撞:
           - 找出撞到的是谁(obs_id)
           - 计算双方速度、朝向、碰撞角度等
           - 调 harm_model 算伤害
           - (可选)生成碰撞报告图片
           - 最后抛出 NoLocalTrajectoryFoundError 来终止仿真,并给出原因
        """
        # check if the current state is collision-free
        # vis 是风险评估相关设置(例如是否输出报告)
        vis = self.planner_creator.settings["risk_dict"]
        # 读取伤害模型参数(例如逻辑回归系数)
        coeffs = load_harm_parameter_json()

        # get ego position and orientation
        # ---------- 1) 获取 ego 当前的位置 ----------
        try:
            # 这里用 prediction.trajectory.state_list[-1] 取“预测轨迹最后一个状态”的位置
            # 注意:这不是 occupancy_at_time 的中心点写法,而是直接取预测的 state            
            ego_pos = self.scenario.obstacle_by_id(obstacle_id=agent.agent_id).prediction.trajectory.state_list[
                -1].position
            # 下面注释掉的是另一种获取位置的方法:occupancy 的中心点
            # ego_pos = (
            #     self.scenario.obstacle_by_id(obstacle_id=agent.agent_id)
            #     .occupancy_at_time(time_step=time_step)
            #     .shape.center
            # )
            # print(ego_pos)

        except AttributeError:
            # 如果 prediction 或 trajectory 为空,就会进这里
            print("None-type error")
        # ---------- 2) 计算 ego 速度 ego_vel 和朝向 ego_yaw ----------
        # time_step == 0:没有上一帧,只能用 initial_state
        if time_step == 0:
            ego_vel = self.scenario.obstacle_by_id(
                obstacle_id=agent.agent_id
            ).initial_state.velocity
            ego_yaw = self.scenario.obstacle_by_id(
                obstacle_id=agent.agent_id
            ).initial_state.orientation

            self.vel_list.append(ego_vel)
        else:
            # 获取上一帧位置(occupancy 的中心点)
            ego_pos_last = (
                self.scenario.obstacle_by_id(obstacle_id=agent.agent_id)
                .occupancy_at_time(time_step=time_step - 1)
                .shape.center
            )
            # 位置差向量 = 当前 - 上一帧
            delta_ego_pos = ego_pos - ego_pos_last

            # 速度 = 位移长度 / 时间间隔
            # agent.dt 是每一步的时间间隔(秒)
            ego_vel = np.linalg.norm(delta_ego_pos) / agent.dt

            self.vel_list.append(ego_vel)
            # 朝向 yaw = atan2(dy, dx)
            ego_yaw = np.arctan2(delta_ego_pos[1], delta_ego_pos[0])
        # ---------- 3) 构造“ego 当前状态”的碰撞对象 ----------
        # create_tvobstacle 会创建一个矩形盒子的“轨迹障碍物”,这里仅给一帧
        current_state_collision_object = create_tvobstacle(
            traj_list=[
                [
                    ego_pos[0],
                    ego_pos[1],
                    ego_yaw,
                ]
            ],
            # 注意:这里传的是半长、半宽(box_length = l/2, box_width = w/2)
            box_length=self.vehicle_params.l / 2,
            box_width=self.vehicle_params.w / 2,
            start_time_step=time_step,
        )

        # Add road boundary to collision checker
        # self.collision_checker.add_collision_object(road_boundary)
        # ---------- 4) 用 collision_checker 检查碰撞 ----------
        # 如果没有碰撞,直接返回(这一时间步无事发生)
        if not self.collision_checker.collide(current_state_collision_object):
            return

        # get the colliding obstacle
        # ---------- 5) 如果发生碰撞:找到到底撞到哪个 obstacle ----------        
        obs_id = None
        for obs in self.scenario.obstacles:
            # 把场景里每个障碍物都变成碰撞对象
            co = create_collision_object(obs)
            # 检查 ego 当前状态是否与该障碍物碰撞
            if current_state_collision_object.collide(co):
                # 排除“自己撞自己”
                if obs.obstacle_id != agent.agent_id:
                    if obs_id is None:
                        obs_id = obs.obstacle_id
                    else:
                        # 如果发现同时撞到了多个障碍物,就抛异常(作者认为不应该发生)
                        raise Exception("More than one collision detected")

        # Collision with boundary
        # ---------- 6) 如果 obs_id 仍为 None:说明是撞到了“道路边界”等非 obstacle 对象 ----------
        if obs_id is None:
            # 用保护性伤害模型估计 ego 伤害(忽略角度)
            self.harm["Ego"] = get_protected_inj_prob_log_reg_ignore_angle(
                velocity=ego_vel, coeff=coeffs
            )
            self.harm["Total"] = self.harm["Ego"]
            # 抛出异常终止仿真,并附带 harm 信息
            raise NoLocalTrajectoryFoundError("Collision with road boundary. (Harm: {:.2f})".format(self.harm["Ego"]))

        # ---------- 7) 获取对方障碍物的基本信息(位置、速度、朝向、尺寸等) ----------
        # get information of colliding obstace
        obs_pos = (
            self.scenario.obstacle_by_id(obstacle_id=obs_id)
            .occupancy_at_time(time_step=time_step)
            .shape.center
        )
        obs_pos_last = (
            self.scenario.obstacle_by_id(obstacle_id=obs_id)
            .occupancy_at_time(time_step=time_step - 1)
            .shape.center
        )
        # 尺寸:这里用 length * width 表示面积(不是严格“尺寸”但可作输入特征)        
        obs_size = (
                self.scenario.obstacle_by_id(obstacle_id=obs_id).obstacle_shape.length
                * self.scenario.obstacle_by_id(obstacle_id=obs_id).obstacle_shape.width
        )
        # ---------- 8) 过滤“初始状态就发生碰撞”的情况 ----------
        # filter initial collisions
        if time_step < 1:
            raise ValueError("Collision at initial state")
        # ---------- 9) 计算对方速度 obs_vel 与朝向 obs_yaw ----------
        if (
                self.scenario.obstacle_by_id(obstacle_id=obs_id).obstacle_role
                == ObstacleRole.ENVIRONMENT
        ):
            # 如果是环境障碍物(比如静态建筑/路边),认为速度和朝向为 0            
            obs_vel = 0
            obs_yaw = 0
        else:
            # 对方位移            
            pos_delta = obs_pos - obs_pos_last

            obs_vel = np.linalg.norm(pos_delta) / agent.dt
            if (
                    self.scenario.obstacle_by_id(obstacle_id=obs_id).obstacle_role
                    == ObstacleRole.DYNAMIC
            ):
                # 动态障碍物:用位移方向作为朝向                
                obs_yaw = np.arctan2(pos_delta[1], pos_delta[0])
            else:
                # 其他情况:用 initial_state.orientation                
                obs_yaw = self.scenario.obstacle_by_id(
                    obstacle_id=obs_id
                ).initial_state.orientation
        # ---------- 10) 计算碰撞角度相关量 ----------
        # pdof:碰撞角(对方相对 ego 的角度差),再归一化到 [-pi, pi)(取决于 angle_range 实现)
        # calculate crash angle
        pdof = angle_range(obs_yaw - ego_yaw + np.pi)
        # rel_angle:对方上一帧位置相对 ego 上一帧位置的方向角        
        rel_angle = np.arctan2(
            obs_pos_last[1] - ego_pos_last[1], obs_pos_last[0] - ego_pos_last[0]
        )
        # ego_angle:相对角度(归一化)        
        ego_angle = angle_range(rel_angle - ego_yaw)
        # obs_angle:对方的相对角度(归一化)        
        obs_angle = angle_range(np.pi + rel_angle - obs_yaw)

        # ---------- 11) 调 harm_model 计算伤害 ----------
        # harm_model 返回:
        # - ego_harm:自车伤害值/概率
        # - obs_harm:对方伤害值/概率
        # - ego_obj:自车对象(包含 type, mass 等)
        # - obs_obj:对方对象(包含 type, mass 等)
        # calculate harm
        self.ego_harm, self.obs_harm, ego_obj, obs_obj = harm_model(
            scenario=self.scenario,
            ego_vehicle_id=agent.agent_id,
            vehicle_params=self.vehicle_params,
            ego_velocity=ego_vel,
            ego_yaw=ego_yaw,
            obstacle_id=obs_id,
            obstacle_size=obs_size,
            obstacle_velocity=obs_vel,
            obstacle_yaw=obs_yaw,
            pdof=pdof,
            ego_angle=ego_angle,
            obs_angle=obs_angle,
            modes=vis,
            coeffs=coeffs,
        )
        # ---------- 12) 可视化碰撞报告(如果开启) ----------
        # if collision report should be shown
        if vis["collision_report"]:
            collision_vis(
                scenario=self.scenario,
                destination=self.collision_report_path,
                ego_harm=self.ego_harm,
                ego_type=ego_obj.type,
                ego_v=ego_vel,
                ego_mass=ego_obj.mass,
                obs_harm=self.obs_harm,
                obs_type=obs_obj.type,
                obs_v=obs_vel,
                obs_mass=obs_obj.mass,
                pdof=pdof,
                ego_angle=ego_angle,
                obs_angle=obs_angle,
                time_step=time_step,
                modes=vis,
                marked_vehicle=agent.agent_id,
                planning_problem=agent.planner.planning_problem,
                global_path=None,
                driven_traj=None,
            )
        # ---------- 13) 把 harm 写入统计字典 ----------
        # add ego harm to dict
        self.harm["Ego"] = self.ego_harm
        self.harm["Total"] = self.ego_harm + self.obs_harm

        # 根据对方障碍物类型,把 obs_harm 计入对应分类
        # add obstacle harm of respective type to dict
        if obs_obj.type is ObstacleType.UNKNOWN:  # worst case assumption
            self.harm["Pedestrian"] = self.obs_harm
        elif obs_obj.type is ObstacleType.CAR:
            self.harm["Car"] = self.obs_harm
        elif obs_obj.type is ObstacleType.BUS:
            self.harm["Bus"] = self.obs_harm
        elif obs_obj.type is ObstacleType.TRUCK:
            self.harm["Truck"] = self.obs_harm
        elif obs_obj.type is ObstacleType.BICYCLE:
            self.harm["Bicycle"] = self.obs_harm
        elif obs_obj.type is ObstacleType.PEDESTRIAN:
            self.harm["Pedestrian"] = self.obs_harm
        elif obs_obj.type is ObstacleType.PRIORITY_VEHICLE:
            self.harm["Priority_vehicle"] = self.obs_harm
        elif obs_obj.type is ObstacleType.PARKED_VEHICLE:
            self.harm["Parked_vehicle"] = self.obs_harm
        elif obs_obj.type is ObstacleType.CONSTRUCTION_ZONE:
            self.harm["Construction_zone"] = self.obs_harm
        elif obs_obj.type is ObstacleType.TRAIN:
            self.harm["Train"] = self.obs_harm
        elif obs_obj.type is ObstacleType.ROAD_BOUNDARY:
            self.harm["Road_boundary"] = self.obs_harm
        elif obs_obj.type is ObstacleType.MOTORCYCLE:
            self.harm["Motorcycle"] = self.obs_harm
        elif obs_obj.type is ObstacleType.TAXI:
            self.harm["Taxi"] = self.obs_harm
        elif obs_obj.type is ObstacleType.BUILDING:
            self.harm["Building"] = self.obs_harm
        elif obs_obj.type is ObstacleType.PILLAR:
            self.harm["Pillar"] = self.obs_harm
        elif obs_obj.type is ObstacleType.MEDIAN_STRIP:
            self.harm["Median_strip"] = self.obs_harm
        else:
            raise AttributeError("Error in obstacle type")
        
        # ---------- 14) 最后抛异常终止仿真 ----------
        # 这里用 NoLocalTrajectoryFoundError 来告诉上层:
        # “当前路径发生碰撞,所以认为规划失败/仿真结束”
        raise NoLocalTrajectoryFoundError(
            "Collision in the driven path with {0}. Total harm: {1:.2f}".format(obs_obj.type,
                                                                                self.ego_harm + self.obs_harm))


class PlannerCreator:
    """Class for constructing a planner object from a Handler object."""

    def __init__(self):
        """__init__ function to constuct the object.

        This function is called from the user.
        """

    def get_planner(self, scenario_handler, ego_vehicle_id):
        """
        用 scenario_handler 和 ego_vehicle_id 来创建 planner。

        这里故意抛 NotImplementedError,表示你必须在子类中实现。

        参数:
        - scenario_handler:ScenarioHandler 实例(里面有 scenario, vehicle_params 等)
        - ego_vehicle_id:当前 ego 车辆的 obstacle_id
        """
        """Create the planner from the scenario handler object.

        Args:
            scenario_handler (obj): scenario handler object

        Raises:
            NotImplementedError: Abstract Method

        Returns:
            obj: a planner object.
        """
        raise NotImplementedError("Overwrite this method.")

    @staticmethod
    def get_blacklist():
        """Create the blacklist for the planner.

        Raises:
            NotImplementedError: Abstract Method

        Returns:
            list(str): scenario blacklist
        """
        """
        返回一个“场景黑名单”:
        - 也就是哪些 scenario 不要跑(可能因为不兼容、太难、会报错等)

        同样需要你在子类实现。
        """
        raise NotImplementedError("Overwrite this method.")
