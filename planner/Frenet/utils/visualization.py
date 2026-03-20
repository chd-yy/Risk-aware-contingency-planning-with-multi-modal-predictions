#!/user/bin/env python

"""Visualization functions for the frenét planner."""
import warnings

# CommonRoad 场景对象类型
from commonroad.scenario.scenario import Scenario

# CommonRoad 通用绘图接口：可以画 scenario / planning_problem / obstacle 等对象
from commonroad.visualization.draw_dispatch_cr import draw_object

# 一些辅助可视化函数：
# - get_max_frames_from_scenario: 从 scenario 中估计最大帧数
# - get_plot_limits_from_scenario: 根据场景自动给出绘图边界
from commonroad_helper_functions.visualization import (
    get_max_frames_from_scenario,
    get_plot_limits_from_scenario,
)

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

# matplotlib.patches.Polygon 用于画可达集、多边形区域等
from matplotlib.patches import Polygon

import sys
import os
import pickle
from pathlib import Path

# Ignore Matplotlib DeprecationWarning
# 忽略 matplotlib 的弃用警告，避免运行时输出过多 warning
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# 全局默认 figure 尺寸
plt.rcParams["figure.figsize"] = (8, 8)

# 将项目根目录加入 sys.path，方便做绝对导入
module_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(module_path)

# helper_functions 中：
# - get_max_curvature: 给定车辆参数和速度，返回允许的最大曲率
# - green_to_red_colormap: 从绿色到红色的 colormap，常用来表示 cost 从低到高
from planner.Frenet.utils.helper_functions import (
    get_max_curvature,
    green_to_red_colormap,
)

# 执行计时器
from planner.utils.timers import ExecTimer

# 画不确定性预测结果
from prediction.utils.visualization import draw_uncertain_predictions

# 全局计数器（当前代码里几乎没真正用到，只在注释中出现）
i = 0
PLOT_SNAPSHOTS = []
_LAST_SNAPSHOT_ANIMATION = None


def capture_plot_snapshot(ax=None):
    """Capture the current matplotlib canvas as an RGB frame."""
    fig = ax.figure if ax is not None else plt.gcf()
    if fig is None or fig.canvas is None:
        return

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    PLOT_SNAPSHOTS.append(frame)


def clear_plot_snapshots():
    """Clear cached plot frames."""
    PLOT_SNAPSHOTS.clear()


def replay_plot_snapshots(fps: int = 8, save_path: str = None, clear_after: bool = False):
    """Replay captured plot snapshots and optionally save them as a gif."""
    global _LAST_SNAPSHOT_ANIMATION

    if len(PLOT_SNAPSHOTS) == 0:
        return None

    interval_ms = max(int(1000 / max(fps, 1)), 1)
    fig, ax = plt.subplots()
    ax.set_axis_off()
    image_artist = ax.imshow(PLOT_SNAPSHOTS[0])

    def _update(frame_idx):
        image_artist.set_data(PLOT_SNAPSHOTS[frame_idx])
        return [image_artist]

    _LAST_SNAPSHOT_ANIMATION = animation.FuncAnimation(
        fig,
        _update,
        frames=len(PLOT_SNAPSHOTS),
        interval=interval_ms,
        blit=True,
        repeat=False,
    )

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _LAST_SNAPSHOT_ANIMATION.save(
            str(output_path),
            writer=animation.PillowWriter(fps=max(fps, 1)),
        )

    plt.show()

    if clear_after:
        clear_plot_snapshots()

    return _LAST_SNAPSHOT_ANIMATION


def animate_scenario(
        scenario: Scenario,
        fps: int = 30,
        plot_limits: [float] = None,
        marked_vehicles: [int] = None,
        planning_problem=None,
        save_animation: bool = False,
        animation_directory: str = "./out/",
        animation_area: float = None,
        success: bool = None,
        failure_msg: str = None,
        exec_timer=None,
):
    """
    Animate a commonroad scenario.

    Args:
        scenario (Scenario): Scenario to be animated.
        fps (int): Frames per second. Defaults to 30.
        plot_limits ([float]): Plot limits for the scenario. Defaults to None.
        marked_vehicles ([int]): IDs of the vehicles that should be marked. Defaults to None.
        planning_problem (PlanningProblem): Considered planning problem. Defaults to None.
        save_animation (bool): True if the animation should be saved. Defaults to False.
        animation_directory (str): Directory to save the animation in. Defaults to './out/'.
        animation_area (float): Size of the animated area). Defaults to None.
        success (bool): True if it is a successfully solved scenario. Defaults to None.
        failure_msg (str): Failure-message of the scenario. Defaults to None.
        exec_times_dict (dict): Dictionary with the execution times. Defaults to None.

    Returns:
        animation: Animated scenario.
        dict: Dictionary with the execution times.
    """

    # 如果没有传入计时器，则创建一个“禁用 timing”的默认计时器
    if exec_timer is None:
        exec_timer = ExecTimer(False)

    # 如果没有手动指定 plot_limits，就根据整个场景自动推一个边界
    if plot_limits is None:
        plot_limits = get_plot_limits_from_scenario(scenario=scenario)

    # ----------------------------
    # 计算实际可用帧率 fps_available
    # ----------------------------
    # scenario.dt 是场景离散时间步长（例如 0.1s）
    # 如果用户要求的 fps 太高，超过了场景采样频率，就最多只能按场景实际 dt 播放
    #
    # 例如：
    #   scenario.dt = 0.1 => 最大 10 fps
    #   如果传 fps=30，则实际只能用 10 fps
    if 1 / fps < scenario.dt:
        fps_available = 1 / scenario.dt
    else:
        fps_available = fps

    # ----------------------------
    # 计算总帧数 frames
    # ----------------------------
    # 优先级：
    # 1) 若指定了 marked_vehicles，则用该车辆轨迹预测长度
    # 2) 否则若 planning_problem 定义了目标时间窗口，则用目标结束时间
    # 3) 否则从 scenario 中估计最大轨迹长度
    if marked_vehicles is not None:
        frames = (
                len(scenario.obstacle_by_id(marked_vehicles[0]).prediction.occupancy_set)
                + 1
        )
    elif planning_problem is not None and hasattr(
            planning_problem.goal.state_list[0], "time_step"
    ):
        frames = planning_problem.goal.state_list[0].time_step.end + 1
    else:
        trajectory_points = get_max_frames_from_scenario(scenario=scenario)
        frames = int(trajectory_points * scenario.dt * fps_available)

    # 至少保证 1 帧
    if frames == 0:
        frames = 1

    # ----------------------------
    # 预提取被标记车辆的状态序列，用于下面的速度/加速度/航向子图
    # ----------------------------
    t = []
    v = []
    a = []
    yaw = []

    with exec_timer.time_with_cm("animate create states"):

        if marked_vehicles is not None:
            # 这里只取第一个 marked vehicle 的完整轨迹信息
            trajectory = scenario.obstacle_by_id(
                marked_vehicles[0]
            ).prediction.trajectory

            # 先添加初始状态的信息
            if hasattr(
                    scenario.obstacle_by_id(marked_vehicles[0]).initial_state, "time_step"
            ):
                t.append(
                    scenario.obstacle_by_id(marked_vehicles[0]).initial_state.time_step
                )
            else:
                t.append(0)

            if hasattr(
                    scenario.obstacle_by_id(marked_vehicles[0]).initial_state, "velocity"
            ):
                v.append(
                    scenario.obstacle_by_id(marked_vehicles[0]).initial_state.velocity
                )
            else:
                v.append(0.0)

            if hasattr(
                    scenario.obstacle_by_id(marked_vehicles[0]).initial_state,
                    "acceleration",
            ):
                a.append(
                    scenario.obstacle_by_id(
                        marked_vehicles[0]
                    ).initial_state.acceleration
                )
            else:
                a.append(0.0)

            if hasattr(
                    scenario.obstacle_by_id(marked_vehicles[0]).initial_state, "orientation"
            ):
                yaw.append(
                    scenario.obstacle_by_id(
                        marked_vehicles[0]
                    ).initial_state.orientation
                )
            else:
                yaw.append(0.0)

            # 再把 trajectory.state_list 里的每个状态都追加进去
            for state in trajectory.state_list:
                if hasattr(state, "velocity"):
                    v.append(state.velocity)
                else:
                    v.append(0.0)

                if hasattr(state, "time_step"):
                    t.append(state.time_step)
                else:
                    t.append(0)

                if hasattr(state, "acceleration"):
                    a.append(state.acceleration)
                else:
                    a.append(0.0)

                if hasattr(state, "orientation"):
                    yaw.append(state.orientation)
                else:
                    yaw.append(0.0)

    # ----------------------------
    # 根据 success / failure_msg 设置输出目录与标题前缀
    # ----------------------------
    # get information about the success of the solved scenario
    # there are 2 directories, one for successful scenarios and one for failed ones
    # create these directories if they do not exist yet
    success_or_not = ""

    if success is not None:
        if success is True:
            success_or_not = "Succeeded!"
            if failure_msg is not None:
                success_or_not += "\n" + failure_msg
            animation_directory = animation_directory + "/succeeded/"
            if not os.path.exists(animation_directory):
                os.makedirs(animation_directory)
        else:
            success_or_not = "Failed!"
            if failure_msg is not None:
                success_or_not += "\n" + failure_msg
            animation_directory = animation_directory + "/failed/"
            if not os.path.exists(animation_directory):
                os.makedirs(animation_directory)

    def animate(j):
        """
        FuncAnimation 每一帧调用一次。
        j 是当前动画帧索引，不一定等于 scenario 的 time_step，
        需要根据 scenario.dt 和 fps_available 换算。
        """

        # ----------------------------
        # 子图1：场景图
        # ----------------------------
        ax1.cla()

        # 构造目标时间字符串，用于标题展示
        if hasattr(planning_problem.goal.state_list[0], "time_step"):
            target_time_string = "Target-time: %.1f s - %.1f s" % (
                planning_problem.goal.state_list[0].time_step.start * scenario.dt,
                planning_problem.goal.state_list[0].time_step.end * scenario.dt,
            )
        else:
            target_time_string = "No target-time"

        # 设置主标题
        ax1.set(
            title=(
                    str(scenario.benchmark_id)
                    + ": "
                    + success_or_not
                    + "\n\nTime: "
                    + str(round(j * scenario.dt, 1))
                    + " s\n"
                    + target_time_string
            )
        )

        ax1.set_aspect("equal")
        ax1.set_xlabel(r"$x$ in m")
        ax1.set_ylabel(r"$y$ in m")

        # 画 scenario 中所有对象
        # 注意：time_begin 这里用的是 int(j / (scenario.dt * fps_available))
        # 因为动画帧和 scenario 时间步不一定一一对应，需要映射回离散 time step
        draw_object(
            obj=scenario,
            ax=ax1,
            plot_limits=plot_limits,
            draw_params={"time_begin": int(j / (scenario.dt * fps_available))},
        )

        # 画 planning problem（目标区域等）
        if planning_problem is not None:
            draw_object(
                obj=planning_problem,
                ax=ax1,
                plot_limits=plot_limits,
                draw_params={"time_begin": int(j / (scenario.dt * fps_available))},
            )

        # 画被标记车辆（通常是 ego）
        if marked_vehicles is not None:
            for marked_vehicle in marked_vehicles:
                if marked_vehicle is not None:
                    draw_object(
                        obj=scenario.obstacle_by_id(marked_vehicle),
                        ax=ax1,
                        plot_limits=plot_limits,
                        draw_params={
                            "time_begin": int(j / (scenario.dt * fps_available)),
                            "facecolor": "g",
                        },
                    )

                    # 若指定了 animation_area，则把 ego 车放到视图中心附近
                    if animation_area is not None:
                        ego_vehicle = scenario.obstacle_by_id(marked_vehicle)

                        # j==0 时只能用 initial_state；后续用 occupancy_set 的中心
                        if j == 0:
                            ego_vehicle_pos = ego_vehicle.initial_state.position
                        else:
                            ego_vehicle_pos = ego_vehicle.prediction.occupancy_set[
                                j - 1
                                ].shape.center

                        ax1.set(
                            xlim=(
                                ego_vehicle_pos[0] - animation_area,
                                ego_vehicle_pos[0] + animation_area,
                            )
                        )
                        ax1.set(
                            ylim=(
                                ego_vehicle_pos[1] - animation_area,
                                ego_vehicle_pos[1] + animation_area,
                            )
                        )

                # ----------------------------
                # 子图2：速度曲线
                # ----------------------------
                ax2.cla()
                ax2.set(title="Velocity")
                ax2.set(ylabel=r"$v$ in m/s")
                ax2.set(xlabel=r"$t$ in s")

                # 若 planning_problem 定义了目标速度区间，则画出“目标区域框”
                if hasattr(planning_problem.goal.state_list[0], "velocity"):
                    v_min = planning_problem.goal.state_list[0].velocity.start
                    v_max = planning_problem.goal.state_list[0].velocity.end

                    if hasattr(planning_problem.goal.state_list[0], "time_step"):
                        ts_min = planning_problem.goal.state_list[0].time_step.start
                        ts_max = planning_problem.goal.state_list[0].time_step.end
                        ax2.plot(
                            [ts_min, ts_max, ts_max, ts_min, ts_min],
                            [v_min, v_min, v_max, v_max, v_min],
                            color="g",
                            label="goal area",
                        )
                    else:
                        ax2.plot([t[0], t[-1]], [v_min, v_min], color="g")
                        ax2.plot(
                            [t[0], t[-1]], [v_max, v_max], color="g", label="goal area"
                        )
                    ax2.legend()

                # 画速度曲线和当前时刻散点
                ax2.plot(t, v)
                ax2.scatter(j, v[j])

                # ----------------------------
                # 子图3：加速度曲线
                # ----------------------------
                ax3.cla()
                ax3.set(title="Acceleration")
                ax3.set(ylabel=r"$a$ in m/s²")
                ax3.set(xlabel=r"$t$ in s")

                # 若定义了目标加速度区间，则画绿色目标框
                if hasattr(planning_problem.goal.state_list[0], "acceleration"):
                    a_min = planning_problem.goal.state_list[0].acceleration.start
                    a_max = planning_problem.goal.state_list[0].acceleration.end
                    if hasattr(planning_problem.goal.state_list[0], "time_step"):
                        ts_min = planning_problem.goal.state_list[0].time_step.start
                        ts_max = planning_problem.goal.state_list[0].time_step.end
                        ax3.plot(
                            [ts_min, ts_max, ts_max, ts_min, ts_min],
                            [a_min, a_min, a_max, a_max, a_min],
                            color="g",
                            label="goal area",
                        )
                    else:
                        ax3.plot([t[0], t[-1]], [a_min, a_min], color="g")
                        ax3.plot(
                            [t[0], t[-1]], [a_max, a_max], color="g", label="goal area"
                        )
                    ax3.legend()

                ax3.plot(t, a)
                ax3.scatter(j, a[j])

                # ----------------------------
                # 子图4：航向角曲线
                # ----------------------------
                ax4.cla()
                ax4.set(title="Orientation")
                ax4.set(ylabel=r"$\psi$ in rad")
                ax4.set(xlabel=r"$t$ in s")

                # 若定义了目标航向区间，则画绿色目标框
                if hasattr(planning_problem.goal.state_list[0], "orientation"):
                    yaw_min = planning_problem.goal.state_list[0].orientation.start
                    yaw_max = planning_problem.goal.state_list[0].orientation.end
                    if hasattr(planning_problem.goal.state_list[0], "time_step"):
                        ts_min = planning_problem.goal.state_list[0].time_step.start
                        ts_max = planning_problem.goal.state_list[0].time_step.end
                        ax4.plot(
                            [ts_min, ts_max, ts_max, ts_min, ts_min],
                            [yaw_min, yaw_min, yaw_max, yaw_max, yaw_min],
                            color="g",
                            label="goal area",
                        )
                    else:
                        ax4.plot([t[0], t[-1]], [yaw_min, yaw_min], color="g")
                        ax4.plot(
                            [t[0], t[-1]],
                            [yaw_max, yaw_max],
                            color="g",
                            label="goal area",
                        )
                    ax4.legend()

                ax4.plot(t, yaw)
                ax4.scatter(j, yaw[j])

    # 先关闭可能已有的 figure，避免叠加
    plt.close()

    # 创建总 figure
    fig = plt.figure(constrained_layout=False, figsize=(22, 15))

    # 提高整体字体大小，便于演示和保存
    plt.rcParams.update({"font.size": 25})

    # gridspec 布局：
    # 上面两行给场景图 ax1
    # 下面一行分三块分别给 ax2/ax3/ax4
    gs = fig.add_gridspec(3, 11, left=0.05, top=0.9, right=0.95, wspace=0.3, hspace=0.5)
    ax1 = fig.add_subplot(gs[0:2, :])
    ax2 = fig.add_subplot(gs[2, 0:3])
    ax3 = fig.add_subplot(gs[2, 4:7])
    ax4 = fig.add_subplot(gs[2, 8:11])

    # 生成动画对象
    with exec_timer.time_with_cm("animate create animation"):
        anim = animation.FuncAnimation(
            fig=fig,
            func=animate,
            frames=frames,
            interval=1 / fps_available * 1000,  # 毫秒
            repeat=False,
            repeat_delay=1000,
            blit=False,
        )

    # 若 save_animation=True，则导出 gif
    with exec_timer.time_with_cm("animate save"):
        if save_animation:
            writergif = animation.PillowWriter(fps=fps_available)
            anim.save(
                animation_directory + scenario.benchmark_id + ".gif", writer=writergif
            )

    return anim


def draw_contingent_trajectories(
        scenario,
        time_step: int,
        marked_vehicle: [int] = None,
        planning_problem=None,
        traj=None,
        predictions: dict = None,
        visible_area=None,
        animation_area: float = 40.0,
        global_path: np.ndarray = None,
        global_path_after_goal: np.ndarray = None,
        driven_traj=None,
        ax=None,
        picker=False,
        show_label=False,
        live=True,
        valid_traj=None,
        best_traj=None,
        open_loop=False,
):
    """
    绘制 contingent planning 结果：
    - shared plan
    - 各 mode 下的 contingent trajectories
    - 预测轨迹

    注意：这段代码对 y 坐标做了负号变换（-y），说明当前绘图坐标系和内部状态坐标系方向相反。
    """

    # 如果 live=True，先把场景底图画出来
    if live:
        ax = draw_scenario(
            scenario,
            time_step,
            marked_vehicle,
            planning_problem,
            traj,
            visible_area,
            animation_area,
            global_path,
            global_path_after_goal,
            driven_traj,
            ax,
            picker,
            show_label,
        )

    # Draw all possible trajectories with their costs as colors
    if valid_traj is not None and len(valid_traj) != 0:
        ax.set_xlabel("x[m]")
        ax.set_ylabel("y[m]")

        # 把坐标轴范围固定到 shared_plan 起点附近
        ax.set_xlim(
            valid_traj[0]['shared_plan'].x[0] - animation_area / 6,
            valid_traj[0]['shared_plan'].x[0] + animation_area - 15
        )

        # y 轴范围被硬编码成 [-10.8, 3.6]，明显是特定车道环境下的设置
        '''
        ax.set_ylim(
            valid_traj[0]['shared_plan'].y[0] - animation_area / 2,
            valid_traj[0]['shared_plan'].y[0]
        )
        '''
        ax.set_ylim(
            -10.8, 3.6
        )

    # 选择“最佳 shared 轨迹”
    if len(valid_traj) > 0:
        best_shared_trajectory = valid_traj[0]['shared_plan']
    else:
        best_shared_trajectory = best_traj[1]['shared_plan']

    # 预设颜色集
    # if time_step == 0 or open_loop is False:
    # if open_loop is True:
    # x and y axis description
    colorset = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                'tab:olive', 'tab:cyan', 'y', 'm', 'c', 'g']

    # 画最佳 shared trajectory（蓝色）
    ax.plot(
        best_shared_trajectory.x,
        -best_shared_trajectory.y,
        alpha=1.0,
        color="blue",
        zorder=25,
        label="Best trajectory",
        picker=picker,
    )

    # keys_list 里通常包括：
    # ['shared_plan', 0, 1, 2, ..., 'cost'] 或类似结构
    # 这里跳过第一个和最后一个，画中间的 contingent trajectories
    keys_list = list(valid_traj[0])
    j = 1
    for key in keys_list[1:-1]:
        ax.plot(
            valid_traj[0][key].x,
            -valid_traj[0][key].y,
            alpha=1.0,
            color=colorset[j],
            zorder=25,
            label="Best trajectory",
            picker=picker,
        )
        j += 1
    '''
    ax.plot(
        best_traj_mode_2.x,
        best_traj_mode_2.y,
        alpha=1.0,
        color="yellow",
        zorder=25,
        lw=3.0,
        label="Best trajectory",
        picker=picker,
    )
    ax.plot(
        best_traj_mode_1.x,
        best_traj_mode_1.y,
        alpha=1.0,
        color="blue",
        zorder=25,
        lw=3.0,
        label="Best trajectory",
        picker=picker,
    )
    '''
    # draw predictions
    # 这里只取 predictions.values() 的前10个元素，再从中提 fut_pos_list
    prediction_plot_list = list(predictions.values())[:10]
    fut_pos_list = [
        prediction_plot_list[i]["pos_list"][:20][:]
        for i in range(len(prediction_plot_list))
    ]

    # 画预测轨迹点，y 同样取负
    for i in range(len(fut_pos_list[0])):
        ax.plot(fut_pos_list[0][i][:, 0], -fut_pos_list[0][i][:, 1], alpha=0.5,
                color='tab:gray',
                lw=0.5,
                zorder=25,
                marker='o',
                markersize=2,
                picker=picker, )

    # 如果想画不确定性预测云团，可以打开下面注释
    '''
    if predictions is not None:
        draw_uncertain_predictions(predictions, ax)
    '''

    # 短暂 pause，使 live 绘图能刷新

    plt.pause(0.000001)


def draw_all_plans(
        scenario,
        time_step: int,
        marked_vehicle: [int] = None,
        planning_problem=None,
        traj=None,
        predictions: dict = None,
        base_predictions: dict = None,
        visible_area=None,
        animation_area: float = 40.0,
        global_path: np.ndarray = None,
        global_path_after_goal: np.ndarray = None,
        driven_traj=None,
        ax=None,
        picker=False,
        show_label=False,
        live=True,
        valid_traj=None,
        best_traj=None,
        open_loop=False,
):
    """
    画出所有 shared plans + all contingent plans，并高亮 best plan。
    """
    if live:
        ax = draw_scenario(
            scenario,
            time_step,
            marked_vehicle,
            planning_problem,
            traj,
            visible_area,
            animation_area,
            global_path,
            global_path_after_goal,
            driven_traj,
            ax,
            picker,
            show_label,
        )

    if best_traj is not None and len(best_traj) != 0:
        # x and y axis description
        ax.set_xlabel("x[m]")
        ax.set_ylabel("y[m]")

        # align ego position to the center
        # ax.set_xlim(
        #     valid_traj[0]['shared_plan'].x[0] - animation_area / 6,
        #     valid_traj[0]['shared_plan'].x[0] + animation_area - 15
        # )

        # 同样硬编码 y 范围，确保画面居中
        ax.set_xlim(
            -45, 40
        )
        ax.set_ylim(
            -37, 24
        )
        # 第一套 colorset 被第二套覆盖
        colorset = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                    'tab:olive', 'tab:cyan', 'y', 'm', 'c', 'g']

        # 实际使用的是这一套
        colorset = ['tab:pink', 'tab:blue', 'tab:purple', 'tab:olive', 'm', 'tab:cyan']

        # TODO(yanjun): Draw all possible trajectories with their costs as colors
        # 先把所有 shared_plan 用淡黑色画出来
        # j = 0
        for p in reversed(best_traj):
            ax.plot(p['shared_plan'].x, p['shared_plan'].y, alpha=0.1, color='k', zorder=25, picker=picker)
            # j = j + 1
            # print(f"Loop count: {j}") 

        # 再把每个 plan 里的 contingent trajectories 画出来
        for i in range(len(best_traj)):
            keys_list = list(best_traj[i])
            j = 1
            for key in keys_list[1:-1]:
                if abs(best_traj[i][key].d[-1] + 3.6) < 0.1:
                    j = 0
                elif abs(best_traj[i][key].d[-1] + 2.88) < 0.1:
                    j = 1
                elif abs(best_traj[i][key].d[-1] + 2.16) < 0.1:
                    j = 2
                elif abs(best_traj[i][key].d[-1] + 1.44) < 0.1:
                    j = 3
                elif abs(best_traj[i][key].d[-1] + 0.72) < 0.1:
                    j = 4
                elif abs(best_traj[i][key].d[-1]) < 0.1:
                    j = 5

                ax.plot(
                    best_traj[i][key].x,
                    best_traj[i][key].y,
                    alpha=0.05,
                    color=colorset[j],
                    zorder=25,
                    label="Best trajectory",
                    picker=picker,
                )
                j += 1
                if j == len(colorset):
                    j = 0

    if base_predictions is not None:
        draw_base_predictions(
            base_predictions=base_predictions,
            ax=ax,
            picker=picker,
        )

    # plot best trajectory（最佳计划的 shared 部分）
    ax.plot(best_traj[0]['shared_plan'].x, best_traj[0]['shared_plan'].y, alpha=1.0, linewidth=3.0, color='tab:green',
            zorder=25, picker=picker)

    # 再画最佳计划的 contingent 部分
    keys_list = list(best_traj[0])
    for idx, key in enumerate(keys_list[1:-1]):
        ax.plot(
            best_traj[0][key].x,
            best_traj[0][key].y,
            alpha=1.0,
            color='tab:blue',
            linewidth=2.0,
            zorder=25,
            label="Best trajectory" if idx == 0 else None,
            picker=picker,
        )


    # draw predictions
    prediction_plot_list = list(predictions.values())[:10]

    for pred in prediction_plot_list:
        pos_list = pred["pos_list"][:20]

        if isinstance(pos_list, np.ndarray):
            pos_list = [pos_list]

        for traj_xy in pos_list:
            ax.plot(
                traj_xy[:, 0],
                traj_xy[:, 1],
                alpha=0.5,
                color='tab:gray',
                lw=0.5,
                zorder=25,
                marker='o',
                markersize=2,
                picker=picker,
            )


    # if predictions is not None:
        # breakpoint()
        # for prediction in predictions:
        # draw_uncertain_predictions(predictions, ax)

    # show the figure until the next one ins ready
    # plt.savefig(str(i).zfill(4) + ".png")
    # i += 1
    capture_plot_snapshot(ax=ax)
    plt.pause(0.000001)


def draw_base_predictions(
        base_predictions: dict,
        ax,
        picker=False,
):
    """
    Draw all single-modal base prediction trajectories.
    """
    if base_predictions is None or ax is None:
        return

    for pred in list(base_predictions.values()):
        pos_list = pred.get("pos_list")
        if pos_list is None:
            continue

        if isinstance(pos_list, np.ndarray):
            traj_list = [pos_list]
        elif isinstance(pos_list, list) and len(pos_list) > 0 and isinstance(pos_list[0], np.ndarray):
            traj_list = pos_list
        else:
            traj_array = np.asarray(pos_list, dtype=float)
            if traj_array.ndim != 2 or traj_array.shape[1] != 2:
                continue
            traj_list = [traj_array]

        for traj_xy in traj_list:
            if traj_xy.ndim != 2 or traj_xy.shape[1] != 2:
                continue
            ax.plot(
                traj_xy[:, 0],
                traj_xy[:, 1],
                alpha=0.9,
                color="tab:green",
                lw=1.2,
                linestyle="--",
                zorder=24,
                marker="x",
                markersize=3,
                picker=picker,
            )


def draw_all_contingent_trajectories(
        scenario,
        time_step: int,
        marked_vehicle: [int] = None,
        planning_problem=None,
        traj=None,
        predictions: dict = None,
        visible_area=None,
        animation_area: float = 40.0,
        global_path: np.ndarray = None,
        global_path_after_goal: np.ndarray = None,
        driven_traj=None,
        ax=None,
        picker=False,
        show_label=False,
        live=True,
        valid_traj=None,
        best_traj=None,
        open_loop=False,
):
    """
    画所有 contingent trajectories，并根据 cost 进行可视化。
    """

    if live:
        ax = draw_scenario(
            scenario,
            time_step,
            marked_vehicle,
            planning_problem,
            traj,
            visible_area,
            animation_area,
            global_path,
            global_path_after_goal,
            driven_traj,
            ax,
            picker,
            show_label,
        )

    # Draw all possible trajectories with their costs as colors
    if valid_traj is not None and len(valid_traj) != 0:
        # x and y axis description
        ax.set_xlabel("x[m]")
        ax.set_ylabel("y[m]")

        # align ego position to the center
        ax.set_xlim(
            valid_traj[0]['shared_plan'].x[0] - animation_area / 6,
            valid_traj[0]['shared_plan'].x[0] + animation_area - 15
        )

        ax.set_ylim(
            -10.8, 3.6
        )

        # 用 cost 做归一化，生成颜色映射器
        norm = matplotlib.colors.Normalize(
            vmin=min([valid_traj[i]['cost'] for i in range(len(valid_traj))]),
            vmax=max([valid_traj[i]['cost'] for i in range(len(valid_traj))]),
            clip=True,
        )
        mapper = cm.ScalarMappable(norm=norm, cmap=green_to_red_colormap())

    # best trajectory
    if len(valid_traj) > 0:
        best_shared_trajectory = valid_traj[0]['shared_plan']
    else:
        best_shared_trajectory = best_traj[1]['shared_plan']

    colorset = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                'tab:olive', 'tab:cyan', 'y', 'm', 'c', 'g']

    # 先画所有 shared_plan
    j = 0
    for p in reversed(valid_traj):
        color = mapper.to_rgba(p['cost'])  # 这里算了 color，但下面实际没用它，而是用了 colorset[j]
        ax.plot(p['shared_plan'].x, -p['shared_plan'].y, alpha=0.15, color=colorset[j], zorder=25, picker=picker)
        j = j + 1
        if j == len(colorset):
            j = 0

    # 再画每个 contingent trajectory
    for i in range(len(valid_traj)):
        keys_list = list(valid_traj[i])
        j = 1
        for key in keys_list[1:-1]:
            ax.plot(
                valid_traj[i][key].x,
                -valid_traj[i][key].y,
                alpha=0.15,
                color=colorset[j],
                zorder=25,
                label="Best trajectory",
                picker=picker,
            )
            j += 1

    # 最后高亮最优计划（shared + contingent）
    ax.plot(valid_traj[0]['shared_plan'].x, -valid_traj[0]['shared_plan'].y, alpha=1.0, linewidth=3.0, color='tab:blue',
            zorder=25, picker=picker)

    keys_list = list(valid_traj[0])
    for key in keys_list[1:-1]:
        ax.plot(
            valid_traj[0][key].x,
            -valid_traj[0][key].y,
            alpha=1.0,
            color='tab:blue',
            linewidth=1.5,
            zorder=25,
            label="Best trajectory",
            picker=picker,
        )
        j += 1

    # draw predictions
    prediction_plot_list = list(predictions.values())[:10]
    fut_pos_list = [
        prediction_plot_list[i]["pos_list"][:20][:]
        for i in range(len(prediction_plot_list))
    ]

    for i in range(len(fut_pos_list[0])):
        ax.plot(fut_pos_list[0][i][:, 0], -fut_pos_list[0][i][:, 1], alpha=0.5,
                color='tab:gray',
                lw=0.5,
                zorder=25,
                marker='o',
                markersize=2,
                picker=picker, )

    '''
    if predictions is not None:
        draw_uncertain_predictions(predictions, ax)
    '''
    # show the figure until the next one ins ready
    # plt.savefig(str(i).zfill(4) + ".png")
    # i += 1
    plt.pause(0.000001)


def draw_frenet_trajectories(
        scenario,
        time_step: int,
        marked_vehicle: [int] = None,
        planning_problem=None,
        traj=None,
        all_traj=None,
        predictions: dict = None,
        visible_area=None,
        animation_area: float = 40.0,
        global_path: np.ndarray = None,
        global_path_after_goal: np.ndarray = None,
        driven_traj=None,
        ax=None,
        picker=False,
        show_label=False,
        live=True,
        valid_traj=None,
        invalid_traj=None,
        best_traj=None,
        open_loop=False,
):
    """
    Plot all frenét trajectories.

    Args:
        scenario (Scenario): Considered Scenario.
        time_step (int): Current time step.
        marked_vehicle ([int]): IDs of the marked vehicles. Defaults to None.
        planning_problem (PlanningProblem): Considered planning problem. Defaults to None.
        traj (FrenetTrajectory): The best trajectory of all frenét trajectories. Defaults to None.
        all_traj ([FrenetTrajectory]): All frenét trajectories. Defaults to None.
        fut_pos_list (np.ndarray): Future positions of the vehicles. Defaults to None.
        visible_area (shapely.Polygon): Polygon of the visible area. Defaults to None.
        animation_area (float): Area that should be shown. Defaults to 40.0.
        global_path (np.ndarray): Global path for the planning problem. Defaults to None.
        global_path_after_goal (np.ndarray): Global path for the planning problem after reaching the goal. Defaults to None.
        driven_traj ([States]): Already driven trajectory of the ego vehicle. Defaults to None.
        save_fig (bool): True if the figure should be saved. Defaults to False.
        :param ax:
    """

    # 先画底图
    if live:
        ax = draw_scenario(
            scenario,
            time_step,
            marked_vehicle,
            planning_problem,
            traj,
            visible_area,
            animation_area,
            global_path,
            global_path_after_goal,
            driven_traj,
            ax,
            picker,
            show_label,
        )

    # Draw all possible trajectories with their costs as colors
    if all_traj is not None and len(all_traj) != 0:
        ax.set_xlabel("x in m")
        ax.set_ylabel("y in m")

        # 把画面中心对准 all_traj[0] 的起点附近
        ax.set_xlim(
            all_traj[0].x[0] - animation_area, all_traj[0].x[0] + animation_area
        )
        ax.set_ylim(
            all_traj[0].y[0] - animation_area / 2, all_traj[0].y[0] + animation_area / 2
        )

        # 将轨迹 cost 归一化，准备映射成颜色
        norm = matplotlib.colors.Normalize(
            vmin=min([all_traj[i].cost for i in range(len(all_traj))]),
            vmax=max([all_traj[i].cost for i in range(len(all_traj))]),
            clip=True,
        )
        mapper = cm.ScalarMappable(norm=norm, cmap=green_to_red_colormap())

        # 原本想画所有 valid/invalid trajectories，但这部分被注释掉了
        '''
        for p in all_traj:
            if p.valid_level < 1:
                ax.plot(
                    p.x,
                    p.y,
                    alpha=0.4,
                    color=(0.7, 0.7, 0.7),
                    zorder=19,
                    picker=picker,
                )
            elif p.valid_level < 10:
                ax.plot(
                    p.x,
                    p.y,
                    alpha=0.6,
                    color=(0.3, 0.3, 0.7),
                    zorder=20,
                    picker=picker,
                )
        
        # then plot all valid trajectories
        for p in reversed(all_traj):
            if p.valid_level >= 10:
                color = mapper.to_rgba(p.cost)
                ax.plot(p.x, p.y, alpha=1.0, color=color, zorder=20, picker=picker)
       '''

    # 选择 best trajectory：
    # 优先 valid_traj[0]，若没有 valid，则取 invalid_traj[0]
    if len(valid_traj) > 0:
        best_trajectory = valid_traj[0]
    elif len(invalid_traj) > 0:
        best_trajectory = invalid_traj[0]

    # 如果 time_step==0 或 open_loop=False，说明当前用的是在线规划结果
    if time_step == 0 or open_loop == False:
        ax.plot(
            best_trajectory.x,
            best_trajectory.y,
            alpha=1.0,
            color="green",
            zorder=25,
            lw=3.0,
            label="Best trajectory",
            picker=picker,
        )
    else:
        # 否则使用 best_traj 字典中缓存的 open-loop 轨迹
        ax.set_xlabel("x in m")
        ax.set_ylabel("y in m")

        ax.set_xlim(
            best_traj['x_m'][time_step] - animation_area, best_traj['x_m'][time_step] + animation_area
        )
        ax.set_ylim(
            best_traj['y_m'][time_step] - animation_area / 2, best_traj['y_m'][time_step] + animation_area / 2
        )

        ax.plot(
            best_traj['x_m'],
            best_traj['y_m'],
            alpha=1.0,
            color="green",
            zorder=25,
            lw=3.0,
            label="Best trajectory",
            picker=picker,
        )

    # 旧版本单独画 traj 的逻辑被注释掉了
    '''
    # draw planned trajectory
    if traj is not None:
        ax.plot(
            traj.x,
            traj.y,
            alpha=1.0,
            color="green",
            zorder=25,
            lw=3.0,
            label="Best trajectory",
            picker=picker,
        )
   '''

    # 若有 predictions，则画不确定性预测
    if predictions is not None:
        draw_uncertain_predictions(predictions, ax)
    # show the figure until the next one ins ready
    # plt.savefig(str(i).zfill(4) + ".png")
    # i += 1
    plt.pause(0.000001)

#TODO(yanjun)
def show_frenet_details(vehicle_params, fp_list, global_path: np.ndarray = None):
    """
    Plot details about the frenét trajectories.

    Args:
        vehicle_params (VehicleParameters): Parameters of the ego vehicle.
        fp_list ([FrenetTrajectory]): Considered frenét trajectories.
        global_path (np.ndarray): Global path of the planning problem. Defaults to None
    """

    # 创建 figure 和 4 个子图
    fig = plt.figure(constrained_layout=False, figsize=(17, 10))
    plt.rcParams.update({"font.size": 15})

    gs = fig.add_gridspec(3, 2, left=0.05, top=0.9, right=0.95, wspace=0.3, hspace=0.5)
    ax1 = fig.add_subplot(gs[:, 0])   # 左边整列：全局轨迹图
    ax2 = fig.add_subplot(gs[0, 1])   # 右上：曲率
    ax3 = fig.add_subplot(gs[1, 1])   # 右中：横向偏移
    ax4 = fig.add_subplot(gs[2, 1])   # 右下：弧长 s

    # 画所有 Frenet 轨迹
    for fp in fp_list:
        # 注意：这里用的是 fp.valid，而前面别处大多用 fp.valid_level
        # 若 fp.valid >= 10 视为有效，用绿色，否则红色
        if fp.valid >= 10:
            col = "g"
        else:
            col = "r"
        ax1.plot(fp.x, fp.y, color=col)

    ax1.set_aspect("equal")
    ax1.set_title("Global trajectory")
    ax1.set_xlabel(r"$x$ in m")
    ax1.set_ylabel(r"$y$ in m")

    # 若给了全局路径，则一并画出来
    if global_path is not None:
        ax1.plot(global_path[:, 0], global_path[:, 1], color="b")

    # ----------------------------
    # 曲率子图
    # ----------------------------
    ax2.set_title("Curvature")
    ax2.set_ylim([-0.5, 0.5])
    ax2.set_ylabel(r"$\kappa$ in rad/m")
    ax2.set_xlabel(r"$t$ in s")

    for fp in fp_list:
        ax2.plot(fp.t, fp.curv)

        # 同时计算并画出“当前速度下允许的最大曲率”
        max_curv = []
        for i in range(len(fp.t)):
            max_curv_i, _ = get_max_curvature(vehicle_params=vehicle_params, v=fp.v[i])
            max_curv.append(abs(max_curv_i))

        ax2.plot(fp.t, max_curv, color="r")
        ax2.plot(fp.t, np.multiply((-1), max_curv), color="r")

    # ----------------------------
    # 横向偏移 d 子图
    # ----------------------------
    ax3.set_title("Lateral offset")
    ax3.set_ylabel(r"$d$ in m")
    ax3.set_xlabel(r"$t$ in s")
    for fp in fp_list:
        ax3.plot(fp.t, fp.d)

    # ----------------------------
    # 覆盖弧长 s 子图
    # ----------------------------
    ax4.set_title("Covered arc length")
    ax4.set_ylabel(r"$s$ in m")
    ax4.set_xlabel(r"$t$ in s")
    for fp in fp_list:
        ax4.plot(fp.t, fp.s)

    plt.show()


def draw_reach_sets(
        traj=None,
        animation_area: float = 55.0,
        reach_set=None,
        ax=None,
):
    """
    Plot reachable sets.

    Plot reachable sets of all objects except ego.

    Args:
        traj (FrenetTrajectory): The best trajectory of all frenét trajectories. Defaults to None.
        animation_area (float): Area that should be shown. Defaults to 55.0.
        ax (Axes): Plot.
    """

    # draw reach sets
    if reach_set is not None:
        # reach_set 的结构一般像：
        # reach_set[obj_id] = [ set_1, set_2, ... ]
        # 每个 set_x 里又按 step 存 polygon 顶点
        for idx in reach_set:
            no_sets = len(reach_set[idx])
            set_nr = 0
            for reach_set_of_id in reach_set[idx]:
                set_nr += 1
                for reach_set_step in reach_set_of_id.keys():

                    # 把每个 reachable set polygon 画成半透明蓝色区域
                    polygon = Polygon(
                        reach_set_of_id[reach_set_step],
                        closed=True,
                        alpha=0.075,
                        color="blue",
                        fill=True,
                        label="reach_set "
                              + str(set_nr)
                              + "/"
                              + str(no_sets)
                              + " of ID "
                              + str(idx)
                              + " , step = "
                              + str(reach_set_step),
                        zorder=25,
                        lw=0,  # line width zero to hide seam from exterior to interior
                    )
                    ax.add_patch(polygon)

        # 把显示范围对齐到 traj 起点附近
        ax.set_xlim(traj.x[0] - animation_area, traj.x[0] + animation_area)
        ax.set_ylim(traj.y[0] - animation_area, traj.y[0] + animation_area)


def draw_scenario(
        scenario: Scenario = None,
        time_step: int = 0,
        marked_vehicle=None,
        planning_problem=None,
        traj=None,
        visible_area=None,
        animation_area: float = 55.0,
        global_path: np.ndarray = None,
        global_path_after_goal: np.ndarray = None,
        driven_traj=None,
        ax=None,
        picker=False,
        show_label=False,
):
    """
    Draw scenario.

    General drawing function for scenario.

    Args:
        scenario (Scenario): Considered Scenario.
        time_step (int): Current time step.
        marked_vehicle ([int]): IDs of the marked vehicles. Defaults to None.
        planning_problem (PlanningProblem): Considered planning problem. Defaults to None.
        traj (FrenetTrajectory): The best trajectory of all frenét trajectories. Defaults to None.
        visible_area (shapely.Polygon): Polygon of the visible area. Defaults to None.
        animation_area (float): Area that should be shown. Defaults to 40.0.
        global_path (np.ndarray): Global path for the planning problem. Defaults to None.
        global_path_after_goal (np.ndarray): Global path for the planning problem after reaching the goal. Defaults to None.
        driven_traj ([States]): Already driven trajectory of the ego vehicle. Defaults to None.
        ax (Axes): Plot.

    Returns:
        Axes: Plot with scenario.
    """

    global i

    # 若没传 ax，则新建一个 subplot
    if ax is None:
        ax = plt.subplot()

    # 清空当前 axes
    ax.cla()
    # 画场景
    draw_object(
        scenario,
        draw_params={
            "time_begin": time_step,
            "dynamic_obstacle": {
                "draw_shape": True,
                "draw_bounding_box": False,
                "draw_icon": False,
                # TODO
                "show_label": False,
            },
        },
        ax=ax,
    )
    ax.set_aspect("equal")

    # 画 planning problem（起点/终点/目标区域等）
    if planning_problem is not None:
        draw_object(planning_problem, ax=ax)

    # 高亮 scenario.dynamic_obstacles
    if marked_vehicle is not None:
        draw_object(
            # obj=scenario.obstacle_by_id(marked_vehicle),
            obj=scenario.dynamic_obstacles,
            draw_params={
                "time_begin": time_step,
                "facecolor": "g",
                "dynamic_obstacle": {
                    "draw_shape": False,
                    "draw_bounding_box": False,
                    "draw_icon": True,
                },
            },
        )

    # 画全局路径
    if global_path is not None:
        ax.plot(
            global_path[:, 0],
            global_path[:, 1],
            color="turquoise",   # 青绿色
            alpha=0.5,           # 半透明 (0~1，越小越透明)
            zorder=20,
            label="Global path",
        )
        if global_path_after_goal is not None:
            ax.plot(
                global_path_after_goal[:, 0],
                global_path_after_goal[:, 1],
                color="blue",
                zorder=20,
                linestyle="--",
            )

    # 画 ego 已走过的轨迹
    if driven_traj is not None:
        x = [state.position[0] for state in driven_traj]
        y = [state.position[1] for state in driven_traj]
        ax.plot(x, y, color="green", zorder=25, label="Driven trajectory")

    # 画可视区域
    if visible_area is not None:
        if visible_area.geom_type == "MultiPolygon":
            for geom in visible_area.geoms:
                ax.fill(*geom.exterior.xy, "g", alpha=0.2, zorder=10)
        elif visible_area.geom_type == "Polygon":
            ax.fill(*visible_area.exterior.xy, "g", alpha=0.2, zorder=10)
        else:
            for obj in visible_area:
                if obj.geom_type == "Polygon":
                    ax.fill(*obj.exterior.xy, "g", alpha=0.2, zorder=10)

    # 标题中显示目标时间范围
    if planning_problem is not None and hasattr(planning_problem.goal.state_list[0], "time_step"):
        target_time_string = "Target-time: %.1f s - %.1f s" % (
            planning_problem.goal.state_list[0].time_step.start * scenario.dt,
            planning_problem.goal.state_list[0].time_step.end * scenario.dt,
        )
    else:
        target_time_string = "No target-time"


    # ax.legend() 当前被注释
    # TODO(yanjun) : 暂时注释掉，正常需要打开
    ax.set_title(
        "Time: {0:.1f} s".format(time_step * scenario.dt) + "    " + target_time_string
    )

    # 在目标开始前一帧，把 driven_traj pickle 到文件 "test"
    # 这是一个非常特定的调试/缓存逻辑
    # TODO(yanjun) : 暂时注释掉，正常需要打开
    if (time_step == planning_problem.goal.state_list[0].time_step.start - 1):
        with open("test", "wb") as fp:  # Pickling
            pickle.dump(driven_traj, fp)

    return ax

# EOF
