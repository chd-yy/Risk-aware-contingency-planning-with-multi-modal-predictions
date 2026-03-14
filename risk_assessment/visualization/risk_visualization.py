"""Function to create figures in "Bilder/results"."""

import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from commonroad.visualization.draw_dispatch_cr import draw_object

# ----------------------------
# 不同 Frenet 轨迹的配色
# ----------------------------
# 用于在场景图和 cost 图中区分“最优轨迹 / 次优轨迹 / 其他轨迹”
# 颜色顺序通常默认：
#   第1条（最优） -> green
#   第2条        -> greenyellow
#   第3条        -> yellow
#   第4条        -> orange
#   第5条        -> red
col = ['green', 'greenyellow', 'yellow', 'orange', 'red']


def create_risk_files(scenario,
                      time_step: int,
                      destination: str,
                      risk_modes,
                      weights,
                      marked_vehicle: [int] = None,
                      planning_problem=None,
                      traj=None,
                      fut_pos_list: np.ndarray = None,
                      visible_area=None,
                      global_path: np.ndarray = None,
                      global_path_after_goal: np.ndarray = None,
                      driven_traj=None):
    """
    Create plots to visualize the choosen Frenét traj and its risks.

    Args:
        scenario (Scenario): Considered Scenario.
        time_step (int): Current time step.
        destination (str) : Path to save output.
        risk_modes (Dict): Risk modes. Read from risk.json.
        weights (Dict): Read from weights.json.
        marked_vehicle ([int]): IDs of the marked vehicles.
            Defaults to None.
        planning_problem (PlanningProblem): Considered planning problem.
            Defaults to None.
        traj (FrenetTrajectory): List of valid frenét trajectories.
            Defaults to None.
        fut_pos_list (np.ndarray): Future positions of the vehicles.
            Defaults to None.
        visible_area (shapely.Polygon): Polygon of the visible area.
            Defaults to None.
        global_path (np.ndarray): Global path for the planning problem.
            Defaults to None.
        global_path_after_goal (np.ndarray): Global path for the planning
            problem after reaching the goal.
            Defaults to None.
        driven_traj ([States]): Already driven trajectory of the ego vehicle.
            Defaults to None.

    Returns:
        No return value.
    """

    # ----------------------------
    # 总入口：是否真的创建图片
    # ----------------------------
    # 只有当 risk_modes["figures"]["create_figures"] == True 时，才生成所有可视化文件。
    # 这是一个总开关，通常在 risk.json 里配置。
    if risk_modes["figures"]["create_figures"] is True:

        # 1) 创建场景图：道路 + 障碍物 + 全局路径 + 已行驶轨迹 + 候选 Frenet 轨迹
        create_scenario_figure(scenario=scenario,
                               time_step=time_step,
                               destination=destination,
                               risk_modes=risk_modes,
                               marked_vehicle=marked_vehicle,
                               planning_problem=planning_problem,
                               traj=traj,
                               fut_pos_list=fut_pos_list,
                               visible_area=visible_area,
                               global_path=global_path,
                               global_path_after_goal=global_path_after_goal,
                               driven_traj=driven_traj)

        # 2) 创建局部风险图：ego/obstacle 的 harm / collision prob / risk 分量
        create_partial_chart(scenario=scenario,
                             time_step=time_step,
                             destination=destination,
                             risk_modes=risk_modes,
                             traj=traj)

        # 3) 创建 cost 图：带权风险代价（bayes/equality/maximin/ego/总和）
        create_cost_chart(scenario=scenario,
                          time_step=time_step,
                          destination=destination,
                          weights=weights,
                          traj=traj)

        # 4) 创建多轨迹总 cost 对比图：展示前若干条轨迹的 bayes/equality/maximin/ego 曲线
        create_total_cost_chart(scenario=scenario,
                                time_step=time_step,
                                destination=destination,
                                risk_modes=risk_modes,
                                traj=traj)


def create_scenario_figure(scenario,
                           time_step: int,
                           destination: str,
                           risk_modes,
                           marked_vehicle: [int] = None,
                           planning_problem=None,
                           traj=None,
                           fut_pos_list: np.ndarray = None,
                           visible_area=None,
                           global_path: np.ndarray = None,
                           global_path_after_goal: np.ndarray = None,
                           driven_traj=None):
    """
    Create a figure with the most-costefficient Frenét trajectories.

    Args:
        scenario (Scenario): Considered Scenario.
        time_step (int): Current time step.
        destination (str) : Path to save output.
        risk_modes (Dict): Risk modes. Read from risk.json.
        marked_vehicle ([int]): IDs of the marked vehicles. Defaults to None.
        planning_problem (PlanningProblem): Considered planning problem.
            Defaults to None.
        traj (FrenetTrajectory): List of valid frenét trajectories.
            Defaults to None.
        fut_pos_list (np.ndarray): Future positions of the vehicles.
            Defaults to None.
        visible_area (shapely.Polygon): Polygon of the visible area.
            Defaults to None.
        global_path (np.ndarray): Global path for the planning problem.
            Defaults to None.
        global_path_after_goal (np.ndarray): Global path for the planning
            problem after reaching the goal. Defaults to None.
        driven_traj ([States]): Already driven trajectory of the ego vehicle.
            Defaults to None.

    Returns:
        No return value.
    """

    # ----------------------------
    # 确定要画多少条轨迹
    # ----------------------------
    # 如果传入了 traj（一般是“按 cost 排序后的 valid Frenet trajectories 列表”）：
    #   number = min(配置里要求画的条数, 实际可用条数)
    # 否则 number = 0
    if traj is not None:
        # check if enough trajectories are available to plot
        if risk_modes["figures"]["number_plotted_trajectories"] > len(traj):
            number = len(traj)
        else:
            number = risk_modes["figures"]["number_plotted_trajectories"]
    else:
        number = 0

    # ----------------------------
    # 清空当前 figure/axes 的内容
    # ----------------------------
    plt.cla()

    # ----------------------------
    # 设置绘图显示范围
    # ----------------------------
    # 以 ego 已行驶轨迹的最后一个点为中心，显示周围 40m × 40m 的区域
    # plot_limits 的格式是：
    #   [xmin, xmax, ymin, ymax]
    plot_limits = [driven_traj[-1].position[0] - 20,
                   driven_traj[-1].position[0] + 20,
                   driven_traj[-1].position[1] - 20,
                   driven_traj[-1].position[1] + 20]

    # ----------------------------
    # 画当前时刻的场景
    # ----------------------------
    # draw_object 是 CommonRoad 的通用绘图接口
    # draw_params 中 time_begin=time_step 表示画当前时间步的场景
    # dynamic_obstacle.show_label=False 表示不显示动态障碍物标签
    draw_object(scenario,
                draw_params={'time_begin': time_step, 'scenario':
                             {'dynamic_obstacle': {'show_label': False}}},
                plot_limits=plot_limits)

    # 保证 x/y 比例一致，不让车道和车辆变形
    plt.gca().set_aspect('equal')

    # ----------------------------
    # 画 planning problem（起点/终点/目标区域等）
    # ----------------------------
    if planning_problem is not None:
        draw_object(planning_problem)

    # ----------------------------
    # 高亮 ego vehicle
    # ----------------------------
    # marked_vehicle 一般传 ego_id
    # 用绿色填充当前时刻的 ego 车辆
    if marked_vehicle is not None:
        draw_object(obj=scenario.obstacle_by_id(marked_vehicle),
                    draw_params={'time_begin': time_step,
                                 'facecolor': 'g'})

    # ----------------------------
    # 画全局路径
    # ----------------------------
    if global_path is not None:
        # 主全局路径：蓝色实线
        plt.plot(global_path[:, 0], global_path[:, 1], color='blue',
                 zorder=20, label='global path')

        # 如果还有 goal 后面的延伸路径，用蓝色虚线画出来
        if global_path_after_goal is not None:
            plt.plot(global_path_after_goal[:, 0],
                     global_path_after_goal[:, 1], color='blue', zorder=20,
                     linestyle='--')

    # ----------------------------
    # 画 ego 已经行驶过的轨迹
    # ----------------------------
    if driven_traj is not None:
        x = [state.position[0] for state in driven_traj]
        y = [state.position[1] for state in driven_traj]
        plt.plot(x, y, color='green', zorder=25)

    # ----------------------------
    # 画当前规划出的 Frenet 轨迹
    # ----------------------------
    # traj 一般是“按 cost 从小到大排序”的轨迹列表
    # 第 0 条通常是最终选择的轨迹
    if traj is not None:
        for i in range(number):
            if i == 0:
                # 第一条轨迹：加粗 + 标签叫 Chosen trajectory
                plt.plot(traj[i].x, traj[i].y, alpha=1., color=col[i],
                         zorder=25 - i, lw=3.,
                         label='Chosen trajectory')
            else:
                # 后续轨迹：也画出来用于对比
                plt.plot(traj[i].x, traj[i].y, alpha=1., color=col[i],
                         zorder=25 - i, lw=3.,
                         label='Trajectory ' + str(i + 1))

    # ----------------------------
    # 画预测点（障碍物未来位置）
    # ----------------------------
    if fut_pos_list is not None:
        for fut_pos in fut_pos_list:
            # 用青色小点表示预测轨迹
            plt.plot(fut_pos[:, 0], fut_pos[:, 1], '.c', markersize=2,
                     alpha=0.8)

    # ----------------------------
    # 画可见区域 / 传感器视野
    # ----------------------------
    # visible_area 可能是：
    #   - Polygon
    #   - MultiPolygon
    #   - Polygon 列表
    #
    # 这里统一用绿色半透明区域填充
    if visible_area is not None:
        if visible_area.geom_type == 'MultiPolygon':
            for geom in visible_area.geoms:
                plt.fill(*geom.exterior.xy, 'g', alpha=0.2, zorder=10)
        elif visible_area.geom_type == 'Polygon':
            plt.fill(*visible_area.exterior.xy, 'g', alpha=0.2, zorder=10)
        else:
            for obj in visible_area:
                if obj.geom_type == 'Polygon':
                    plt.fill(*obj.exterior.xy, 'g', alpha=0.2, zorder=10)

    # ----------------------------
    # 生成标题中的目标时间字符串
    # ----------------------------
    # 如果 planning_problem.goal.state_list[0] 里定义了 time_step 区间，
    # 则显示“目标时间窗口”
    if hasattr(planning_problem.goal.state_list[0], 'time_step'):
        target_time_string = ('Target-time: %.1f s - %.1f s' %
                              (planning_problem.goal.state_list[0].
                               time_step.start * scenario.dt,
                               planning_problem.goal.state_list[0].
                               time_step.end * scenario.dt))
    else:
        target_time_string = 'No target-time'

    # ----------------------------
    # 图例和标题
    # ----------------------------
    plt.legend()
    plt.title('Time: {0:.1f} s'.format(time_step * scenario.dt) + '    ' +
              target_time_string)

    # ----------------------------
    # 创建保存目录
    # ----------------------------
    # 每个 benchmark_id 一个单独目录
    destination = os.path.join(destination, str(scenario.benchmark_id))
    if not os.path.exists(destination):
        os.makedirs(destination)

    # ----------------------------
    # 保存图片
    # ----------------------------
    # 默认文件名 Figure_<time_step>.png
    # 如果已存在同名文件，则尝试加后缀 -1 ... -9
    picture_path = destination + "/Figure_" + \
        str(time_step)
    if not os.path.exists(picture_path + ".png"):
        plt.savefig(picture_path)
    else:
        for i in range(1, 10):
            if not os.path.exists(picture_path + "-" + str(i) + ".png"):
                plt.savefig(picture_path + "-" + str(i) + ".png")
                break

    # 关闭当前 figure，避免内存累积
    plt.close()


def create_partial_chart(scenario,
                         time_step: int,
                         destination: str,
                         risk_modes,
                         traj=None):
    """
    Create a chart with partial harm, collision probability, and risks.

    Args:
        scenario (Scenario): Considered Scenario.
        time_step (int): Current time step.
        destination (str) : Path to save output.
        risk_modes (Dict): Risk modes. Read from risk.json.
        traj (FrenetTrajectory): List of valid frenét trajectories.
            Defaults to None.

    Returns:
        No return value.
    """

    # ----------------------------
    # 确定最多画多少条轨迹
    # ----------------------------
    if traj is not None:
        # check if enough trajectories are available to plot
        if risk_modes["figures"]["number_plotted_trajectories"] > len(traj):
            number = len(traj)
        else:
            number = risk_modes["figures"]["number_plotted_trajectories"]
    else:
        number = 0

    # ----------------------------
    # 只有至少有一条有效轨迹时才画
    # ----------------------------
    if number > 0:

        # ----------------------------
        # 创建 2×4 子图布局
        # ----------------------------
        # 第一行：ego harm / ego prob / ego risk / ego文字信息
        # 第二行：obst harm / obst prob / obst risk / obst文字信息
        fig, ((ax1, ax3, ax5, ax7), (ax2, ax4, ax6, ax8)) = \
            plt.subplots(nrows=2, ncols=4)

        # A4 横向的两倍大小，用于保证子图足够清晰
        fig.set_size_inches(11.69 * 2, 8.27 * 2)

        # ----------------------------
        # 处理 ego 侧的 harm/risk/prob
        # ----------------------------
        # ego_vehicle_data：一个文本字符串，用来在图上打印 ego 相关信息
        ego_vehicle_data = ""

        # traj[0] 默认是最优轨迹，因此这里只取 traj[0] 做详细分解图
        # ego_harm_dict 的结构通常类似：
        # { obstacle_id : [time_step_obj, time_step_obj, ...] }
        for obstacle_id, harm_dict in traj[0].ego_harm_dict.items():
            data_harm = []
            data_risk = []
            data_prob = []

            # 遍历每个时间步，把 harm/prob/risk 提出来
            for ts in harm_dict:
                if ts is not None:
                    data_harm.append(ts.harm)
                    data_risk.append(ts.risk)
                    data_prob.append(ts.prob)
                else:
                    # 若该时刻无数据，则补0，方便画图
                    data_harm.append(0)
                    data_risk.append(0)
                    data_prob.append(0)

            # ego 针对当前 obstacle 的 harm / probability / risk 曲线
            ax1.plot(data_harm, label="Obstacle " + obstacle_id)
            ax3.plot(data_prob, label="Obstacle " + obstacle_id)
            ax5.plot(data_risk, label="Obstacle " + obstacle_id)

            # 构造 ego 文本信息
            # 只取 harm_dict[0]（第一个时间步）中的基本参数进行展示
            # 包括质量、速度、yaw、size、harm、risk
            if ego_vehicle_data == "":
                ego_vehicle_data += "Mass: " + str(harm_dict[0].mass) + \
                    "\nVelocity: " + str(harm_dict[0].velocity) + \
                    "\nYaw: " + str(harm_dict[0].yaw) + \
                    "\nSize: " + str(harm_dict[0].size) + \
                    "\nHarm: " + str(harm_dict[0].harm) + \
                    "\nRisk: " + str(harm_dict[0].risk) + "\n\n"
            else:
                ego_vehicle_data += "Harm: " + str(harm_dict[0].harm) + \
                                    "\nRisk: " + str(harm_dict[0].risk) + \
                                    "\n\n"

        # ego harm 图说明
        ax1.legend(loc='upper right')
        ax1.set_ylabel("ego harm for different obstacles")

        # ego collision probability 图说明
        ax3.legend(loc='upper right')
        ax3.set_ylabel("collision probability for different obstacles")

        # ego risk 图说明
        ax5.legend(loc='upper right')
        ax5.set_ylabel("ego risk for different obstacles")

        # ----------------------------
        # 处理 obstacle 侧的 harm/risk/prob
        # ----------------------------
        obst_vehicle_data = ""

        # obst_harm_dict 的结构一般与 ego_harm_dict 类似：
        # { obstacle_id : [time_step_obj, time_step_obj, ...] }
        for obstacle_id, harm_dict in traj[0].obst_harm_dict.items():
            data_harm = []
            data_risk = []
            data_prob = []

            for ts in harm_dict:
                if ts is not None:
                    data_harm.append(ts.harm)
                    data_risk.append(ts.risk)
                    data_prob.append(ts.prob)
                else:
                    data_harm.append(0)
                    data_risk.append(0)
                    data_prob.append(0)

            # 障碍物损伤 / 概率 / 风险曲线
            ax2.plot(data_harm, label="Obstacle " + obstacle_id)
            ax4.plot(data_prob, label="Obstacle " + obstacle_id)
            ax6.plot(data_risk, label="Obstacle " + obstacle_id)

            # 构造 obstacle 文本信息
            obst_vehicle_data += str(harm_dict[0].type) + ", Mass: " + \
                str(harm_dict[0].mass) + "\nVelocity: " + \
                str(harm_dict[0].velocity) + "\nYaw: " + \
                str(harm_dict[0].yaw) + "\nSize: " + \
                str(harm_dict[0].size) + "\nHarm: " + \
                str(harm_dict[0].harm) + "\nRisk: " + \
                str(harm_dict[0].risk) + "\n\n"

        # obstacle harm 图说明
        ax2.legend(loc='upper right')
        ax2.set_ylabel("obstacle harm")

        # obstacle collision probability 图说明
        ax4.legend(loc='upper right')
        ax4.set_ylabel("collision probability for different obstacles")

        # obstacle risk 图说明
        ax6.legend(loc='upper right')
        ax6.set_ylabel("obstacle risk")

        # ----------------------------
        # 右侧两个子图专门写文本，不画坐标轴
        # ----------------------------
        ax7.axis('off')
        ax7.text(0, 1, ego_vehicle_data, verticalalignment='top',
                 fontsize=8)

        ax8.axis('off')
        ax8.text(0, 1, obst_vehicle_data, verticalalignment='top',
                 fontsize=8)

        fig.suptitle("Harm for ego vehicle and obstacles")

        # ----------------------------
        # 创建保存目录
        # ----------------------------
        destination = os.path.join(destination, str(scenario.benchmark_id))
        if not os.path.exists(destination):
            os.makedirs(destination)

        # ----------------------------
        # 保存图片：Partial_<time_step>.png
        # ----------------------------
        picture_path = destination + "/Partial_" + str(time_step)
        if not os.path.exists(picture_path + ".png"):
            plt.savefig(picture_path)
        else:
            for i in range(1, 10):
                if not os.path.exists(picture_path + "-" + str(i) + ".png"):
                    plt.savefig(picture_path + "-" + str(i) + ".png")
                    break
        plt.close(fig)


def create_cost_chart(scenario,
                      time_step: int,
                      destination: str,
                      weights,
                      traj=None):
    """
    Create a chart with costs according to the principles of ethics of risk.

    Args:
        scenario (Scenario): Considered Scenario.
        time_step (int): Current time step.
        destination (str) : Path to save output.
        weights (Dict): Read from weights.json. Defaults to None.
        traj (FrenetTrajectory): List of valid frenét trajectories.
            Defaults to None.

    Returns:
        No return value.
    """

    # 只要有至少一条轨迹就画（这里没有判断 traj is not None，默认调用者保证）
    if len(traj) > 0:

        # 单图单坐标轴
        fig, ax1 = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(11.69, 8.27)

        # ----------------------------
        # 读取并加权 risk_dict 中的各项 cost
        # ----------------------------
        # traj[0] 一般是最优轨迹
        #
        # risk_dict 里预期包含：
        #   "bayes"
        #   "equality"
        #   "maximin"
        #   "ego"
        #   "total_weighed"
        #
        # 这里把每个子 cost 再乘以对应的全局权重，得到“加权后的贡献曲线”
        bayes_weighed = [i * weights["bayes"]
                         for i in traj[0].risk_dict["bayes"]]
        equality_weighed = [i * weights["equality"]
                            for i in traj[0].risk_dict["equality"]]
        maximin_weighed = [i * weights["maximin"]
                           for i in traj[0].risk_dict["maximin"]]
        ego_weighed = [i * weights["ego"] for i in traj[0].risk_dict["ego"]]

        # 画各个 cost 分量曲线
        ax1.plot(bayes_weighed, label="Weighed Bayesian Costs",
                 color="green", lw=1)
        ax1.plot(equality_weighed, label="Weighed Equality Costs",
                 color="yellow", lw=1)
        ax1.plot(maximin_weighed, label="Weighed Maximin Costs",
                 color="red", lw=1)
        ax1.plot(ego_weighed, label="Weighed Ego Costs", color="orange", lw=1)

        # 再画总的加权 risk cost 曲线
        ax1.plot(traj[0].risk_dict["total_weighed"],
                 label="Total Weighed Risk Costs", color="blue", lw=2)

        # 图例和纵轴标签
        ax1.legend(loc='upper right')
        ax1.set_ylabel("risk cost (time adjusted and weighed)")

        fig.suptitle("Risk costs")

        # ----------------------------
        # 创建保存目录
        # ----------------------------
        destination = os.path.join(destination, str(scenario.benchmark_id))
        if not os.path.exists(destination):
            os.makedirs(destination)

        # ----------------------------
        # 保存图片：Costs_<time_step>.png
        # ----------------------------
        picture_path = destination + "/Costs_" + str(time_step)
        if not os.path.exists(picture_path + ".png"):
            plt.savefig(picture_path)
        else:
            for i in range(1, 10):
                if not os.path.exists(picture_path + "-" + str(i) + ".png"):
                    plt.savefig(picture_path + "-" + str(i) + ".png")
                    break
        plt.close(fig)


def create_total_cost_chart(scenario,
                            time_step: int,
                            destination: str,
                            risk_modes,
                            traj=None):
    """
    Create a chart with total risk costs for the most cost-efficient trajs.

    Args:
        scenario (Scenario): Considered Scenario.
        time_step (int): Current time step.
        destination (str) : Path to save output.
        risk_modes (Dict): Risk modes. Read from risk.json.
        traj (FrenetTrajectory): List of valid frenét trajectories.
            Defaults to None.

    Returns:
        No return value.
    """

    # ----------------------------
    # 确定最多画多少条轨迹
    # ----------------------------
    if traj is not None:
        # check if enough trajectories are available to plot
        if risk_modes["figures"]["number_plotted_trajectories"] > len(traj):
            number = len(traj)
        else:
            number = risk_modes["figures"]["number_plotted_trajectories"]
    else:
        number = 0

    # ----------------------------
    # 至少有一条有效轨迹才画
    # ----------------------------
    if number > 0:

        # ----------------------------
        # 创建 2×2 子图
        # ----------------------------
        # 分别画：
        #   ax1 -> bayes
        #   ax2 -> equality
        #   ax3 -> maximin
        #   ax4 -> ego
        fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(nrows=2, ncols=2)
        fig.set_size_inches(11.69, 8.27)

        i = 0

        # ----------------------------
        # 遍历前 number 条轨迹，分别画四种 risk cost 曲线
        # ----------------------------
        for ft in traj[0:number]:
            # Bayesian cost
            ax1.plot(ft.risk_dict["bayes"], label="Trajectory " + str(i + 1),
                     color=col[i], lw=1)

            # Equality cost
            ax2.plot(ft.risk_dict["equality"],
                     label="Trajectory " + str(i + 1), color=col[i], lw=1)

            # Maximin cost
            ax3.plot(ft.risk_dict["maximin"], label="Trajectory " + str(i + 1),
                     color=col[i], lw=1)

            # Ego cost
            ax4.plot(ft.risk_dict["ego"], label="Trajectory " + str(i + 1),
                     color=col[i], lw=1)

            i += 1

        # 总标题
        fig.suptitle("Risk costs")

        # 每个子图的纵轴标签
        ax1.set_ylabel("Bayesian Costs")
        ax2.set_ylabel("Equality Costs")
        ax3.set_ylabel("Maximin Costs")
        ax4.set_ylabel("Ego Costs")

        # 图例只放在一个子图里，避免太乱
        ax2.legend(loc='upper right')

        # ----------------------------
        # 创建保存目录
        # ----------------------------
        destination = os.path.join(destination, str(scenario.benchmark_id))
        if not os.path.exists(destination):
            os.makedirs(destination)

        # ----------------------------
        # 保存图片：Traj_<time_step>.png
        # ----------------------------
        picture_path = destination + "/Traj_" + str(time_step)
        if not os.path.exists(picture_path + ".png"):
            plt.savefig(picture_path)
        else:
            for i in range(1, 10):
                if not os.path.exists(picture_path + "-" + str(i) + ".png"):
                    plt.savefig(picture_path + "-" + str(i) + ".png")
                    break
        plt.close(fig)
