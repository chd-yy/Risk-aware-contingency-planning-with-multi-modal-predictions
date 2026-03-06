"""Evaluate a frenet Planner."""
"""
FrenetCreator = 负责组装零件的人
FrenetPlanner = 真正执行规划算法的人(核心)
scenario_handler = 仓库(里面放着场景、车辆参数、计时器、规划问题等)
settings = 配置文件(告诉你用什么模式、参数怎么设、要不要画图等)
"""

# PlannerCreator 是一个“基类/父类”,规定了“创建 planner 的类应该长什么样”
# (比如必须实现 get_planner() 之类的方法)
from beliefplanning.planner.plannertools.scenario_handler import PlannerCreator

# FrenetPlanner 是最终要创建出来的“规划器对象”
from beliefplanning.planner.Frenet.frenet_planner import FrenetPlanner


class FrenetCreator(PlannerCreator):
    """
    FrenetCreator:用于“从 scenario_handler + settings 构造 FrenetPlanner”的类。

    继承(class FrenetCreator(PlannerCreator))是什么意思？
    - PlannerCreator 是父类,它可能定义了一些统一接口(例如 get_planner())。
    - FrenetCreator 继承它,就相当于“我也遵守这个接口规范”。
    - 这样外部系统可以用同一种方式调用不同的 Creator。
    """

    def __init__(self, settings, weights=None):
        """
        构造函数:创建 FrenetCreator 对象时会自动调用。

        参数:
        - settings: 一个字典(dict),通常从配置文件读进来,里面包含各种参数。
          例如:
          settings = {
              "evaluation_settings": {"show_visualization": True},
              "frenet_settings": {...},
              "contingency_settings": {...},
              ...
          }

        - weights: 权重(可选)。一般用于代价函数/评价指标里不同项的权重。
          如果不传入,就是 None；planner 内部可能会使用默认权重。
        """
        """__init__ function to construct the object.

        This function is called from the user.
        """
        # Settings specific for a frenet planner.
        self.show_visualization = settings["evaluation_settings"]["show_visualization"]
        self.frenet_settings = settings["frenet_settings"]
        self.contingency_settings = settings["contingency_settings"]
        self.settings = settings
        self.weights = weights

    def get_planner(self, scenario_handler, ego_vehicle_id):
        """Create the planner from the scenario handler object.

        Args:
            scenariWo_handler (obj): scenario handler object

        Raises:
            NotImplementedError: Abstract Method

        Returns:
            obj: a planner object.

        从 scenario_handler 创建一个 FrenetPlanner。

        你可以把 scenario_handler 理解成“场景管理器/打包对象”,里面包含:
        - scenario:场景本体(地图、道路、障碍车等)
        - planning_problem_set:规划问题集合(起点终点、目标、任务等)
        - agent_planning_problem_id_assignment:给每个车辆分配对应 planning problem 的映射
        - vehicle_params:车辆物理参数(长度、宽度、转向限制等)
        - exec_timer:计时器(用于统计运行时间/超时等)

        参数:
        - scenario_handler: 场景处理对象,提供 FrenetPlanner 需要的数据
        - ego_vehicle_id: ego 车(自车/主车)的 id,用来从 scenario_handler 找到对应 planning_problem
        """

        return FrenetPlanner(
            # -------- FrenetPlanner 需要场景信息 --------
            # scenario_handler.scenario:整个驾驶场景(地图+动态障碍物等)
            scenario=scenario_handler.scenario,
            # -------- FrenetPlanner 需要 planning_problem(规划任务) --------
            # 为什么要这么绕？
            # 1) scenario_handler.planning_problem_set 里可能有多个 planning problem
            #    (比如一个场景里有多个车辆/多个任务)
            # 2) scenario_handler.agent_planning_problem_id_assignment 是一个映射:
            #    用 ego_vehicle_id 找到该车辆对应的 planning_problem_id
            # 3) 然后 find_planning_problem_by_id(...) 才能取到真正的 planning_problem 对象
            planning_problem=scenario_handler.planning_problem_set.find_planning_problem_by_id(
                scenario_handler.agent_planning_problem_id_assignment[ego_vehicle_id]
            ),
            ego_id=ego_vehicle_id,
            vehicle_params=scenario_handler.vehicle_params,
            # -------- 计时器 --------
            # exec_timer:用于统计执行时间、控制超时等
            exec_timer=scenario_handler.exec_timer,
            # -------- planner 运行模式 --------
            mode=self.frenet_settings["mode"],
            plot_frenet_trajectories=self.show_visualization, # arg不加time的时候就是true
            frenet_parameters=self.frenet_settings["frenet_parameters"],
            contingency_parameters=self.contingency_settings["frenet_parameters"],
            weights=self.weights,
            settings=self.settings,
        )

    @staticmethod
    def get_blacklist():
        """
        返回“场景黑名单”。

        黑名单的意义(大白话):
        - 有些场景可能有 bug、数据不兼容、或对某种 planner 不适用。
        - 运行评测时,系统可能会跳过这些场景,避免报错或无意义对比。

        为什么用 staticmethod(静态方法)？
        - 你不需要创建 FrenetCreator 对象,也能直接调用它:
          FrenetCreator.get_blacklist()
        - 因为黑名单通常是“固定的”,不依赖 self.settings 等实例变量。
        """

        # -------- 1) bad_scenario_names:明确列出的坏场景 --------
        """Return the scenario blacklist for this planner."""
        bad_scenario_names = [
            "ARG",
            "Luckenwalde",
            "USA_US101-33_1_T-1",
            "USA_US101-22_2_T-1",
            "USA_US101-3_2_S-1",
            "USA_US101-9_4_T-1",
            "USA_US101-1_1_T-1",
            "USA_US101-1_1_S-1",
            "DEU_Hhr-1_1",
            "interactive",
        ]
        # -------- 2) set_based_scenarios:另一类需要排除的场景 --------
        # 注:这里命名叫 set_based_scenarios,暗示可能这些场景是“set-based”评估形式,
        # 或者与当前 FrenetPlanner 的 pipeline 不兼容,所以排除。
        set_based_scenarios = [
            "DEU_Ffb-1_2_S-1",
            "DEU_Ffb-2_2_S-1",
            "DEU_Muc-30_1_S-1",
            "USA_Lanker-1_1_S-1",
            "USA_US101-1_1_S-1",
            "USA_US101-1_2_S-1",
            "USA_US101-2_2_S-1",
            "USA_US101-2_3_S-1",
            "USA_US101-2_4_S-1",
            "USA_US101-3_2_S-1",
            "USA_US101-3_3_S-1",
            "USA_US101-7_1_S-1",
            "USA_US101-7_2_S-1",
            "ZAM_ACC-1_2_S-1",
            "ZAM_ACC-1_3_S-1",
            "ZAM_HW-1_1_S-1",
            "ZAM_Intersect-1_1_S-1",
            "ZAM_Intersect-1_2_S-1",
            "ZAM_Urban-1_1_S-1",
            "ZAM_Urban-4_1_S-1",
            "ZAM_Urban-5_1_S-1",
            "ZAM_Urban-6_1_S-1",
            "ZAM_Urban-7_1_S-1",
        ]

        # 把两类名单拼接成一个列表返回
        return bad_scenario_names + set_based_scenarios
