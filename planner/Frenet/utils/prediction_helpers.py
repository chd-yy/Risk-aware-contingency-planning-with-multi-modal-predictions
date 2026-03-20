#!/user/bin/env python

"""Helper functions to adjust the prediction to the needs of the frenét planner."""

# 数值计算库，主要用于数组、梯度、拼接、数学运算等
import numpy as np
# 操作系统相关接口，用于处理路径
import os
# Python 运行环境相关接口，用于修改模块搜索路径
import sys
from shapely.geometry import LineString, Point
# CommonRoad 中用于区分障碍物角色（动态 / 静态）的枚举
from commonroad.scenario.obstacle import ObstacleRole
# CommonRoad Drivability Checker 中的轨迹碰撞查询工具
from commonroad_dc.collision.trajectory_queries import trajectory_queries
# PyTorch，用于张量计算
import torch
# 正态分布，用于 belief_updater 中根据观测对模式概率进行更新
from torch.distributions.normal import Normal
from collections import deque

try:
    from beliefplanning.risk_assessment.collision_probability import (
        get_collision_probability_fast,
    )
except ModuleNotFoundError:
    from risk_assessment.collision_probability import get_collision_probability_fast

# ---------------------------------------------------------------------------
# 计算项目根目录路径：
# 当前文件所在路径逐级向上回退四层，得到工程模块根路径
# 这样做通常是为了让 Python 可以找到项目中的自定义模块
# ---------------------------------------------------------------------------
module_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
# 将项目根目录加入 Python 的模块搜索路径
# 这样后面就可以正常 import planner.xxx 下的模块
sys.path.append(module_path)

# 从 Frenet planner 的工具函数中导入：
# create_tvobstacle: 根据轨迹创建时变碰撞对象
# distance: 计算两点之间距离
from planner.Frenet.utils.helper_functions import create_tvobstacle, distance
from planner.plannertools.obstacle_intent_updater import _build_straight_lanelet_path


def _get_obstacle_state_at_timestep(obstacle, timestep: int = None):
    """
    从 scenario / obstacle 中获取指定 timestep 的状态。

    约定：
    - timestep 为 None 时，退化为 initial_state
    - timestep <= initial_state.time_step 时，直接返回 initial_state
    - 若对应时刻状态不存在，也回退为 initial_state
    """
    if timestep is None or timestep <= obstacle.initial_state.time_step:
        return obstacle.initial_state

    state = obstacle.state_at_time(timestep)
    if state is None:
        return obstacle.initial_state
    return state


def belief_updater(predictions: dict, belief):
    """
    基于预测得到的障碍物朝向信息，对 belief（通常是两种模式的先验概率）进行更新。

    参数
    ----
    predictions : dict
        预测结果字典。预期包含每个障碍物的 orientation_list。
        其中 orientation_list 一般表示多个行为模式下的朝向预测序列。

    belief : list / array-like
        当前 belief，通常是长度为 2 的概率向量，例如：
        [mode_0_probability, mode_1_probability]

    返回
    ----
    belief : updated belief
        利用 Bayes 风格的更新方式，基于观测朝向对两种模式概率重新归一化后的结果。

    说明
    ----
    这里默认：
    - 使用第一个时间步的朝向作为“观测值”
    - 每个模式下的朝向预测均值 mu 来构造一个标准差固定为 1 的正态分布
    - 根据观测值在不同模式下的概率密度来更新 belief
    """
    # prob 用来记录每个 mode 对当前“观测朝向”的似然值
    prob = {}
    # 遍历所有被预测到的障碍物
    for obstacle_id in list(predictions.keys()):
        # 取该障碍物第 0 个模式第 0 个时刻的朝向，作为观测状态
        # 这里实际上相当于把某个模式的首时刻朝向拿来做“观测值”
        obstacle_orientation_state = torch.tensor(predictions[obstacle_id]['orientation_list'][0][0])

        # 遍历该障碍物的所有预测模式
        for mode in range(len(predictions[obstacle_id]['orientation_list'])):
            # 取当前模式下第 0 个时刻的朝向作为正态分布均值
            mu = torch.tensor(predictions[obstacle_id]['orientation_list'][mode][0])
            # 构造一个方差固定的高斯分布 N(mu, 1)
            dist = Normal(mu, 1.)
            # 计算“观测朝向”在该模式下的概率密度（通过 log_prob 再 exp）
            prob[mode] = torch.exp(dist.log_prob(obstacle_orientation_state))

        # 基于两个模式的似然和先验 belief 做归一化更新
        # 这是一个典型的二分类贝叶斯更新公式
        belief[0] = prob[0] * belief[0] / (prob[0] * belief[0] + prob[1] * belief[1])
        # 第二个模式概率由 1 - belief[0] 得到
        belief[1] = 1 - belief[0]
        # transform tensor to floating point.
        # 将 torch tensor 转为 Python float，避免后续使用时类型不兼容
        belief[0] = belief[0].item()
        belief[1] = belief[1].item()

    return belief


def get_obstacles_in_radius(scenario, ego_id: int, ego_state, radius: float):
    """
    Get all the obstacles that can be found in a given radius.

    Args:
        scenario (Scenario): Considered Scenario.
        ego_id (int): ID of the ego vehicle.
        ego_state (State): State of the ego vehicle.
        radius (float): Considered radius.

    Returns:
        [int]: List with the IDs of the obstacles that can be found in the ball with the given radius centering at the ego vehicles position.
    """
    # 保存落在指定半径范围内的障碍物 ID
    obstacles_within_radius = []
    # 遍历场景中的所有障碍物
    for obstacle in scenario.obstacles:
        # do not consider the ego vehicle
        # 跳过自车本身，只考虑其他障碍物
        if obstacle.obstacle_id != ego_id:
            # 获取该障碍物在 ego_state.time_step 时刻的占用信息 occupancy
            occ = obstacle.occupancy_at_time(ego_state.time_step)
            # if the obstacle is not in the lanelet network at the given time, its occupancy is None
            # 如果在该时刻该障碍物不存在于场景中，则 occ 为 None
            if occ is not None:
                # calculate the distance between the two obstacles
                # 计算自车与障碍物的距离
                # 注意：occupancy 可能有两种形式：
                # 1) list：多模态占用结果
                # 2) 单个 occupancy 对象：单一占用
                if type(occ) == list:
                    # 如果是 list，则取第一个 occupancy 的中心点
                    dist = distance(
                        pos1=ego_state.position,
                        pos2=obstacle.occupancy_at_time(ego_state.time_step)[0].shape.center,
                    )
                else:
                    # 如果是单个 occupancy，则直接取其几何中心
                    dist = distance(
                        pos1=ego_state.position,
                        pos2=obstacle.occupancy_at_time(ego_state.time_step).shape.center,
                    )

                # add obstacles that are close enough
                # 若距离小于给定半径，则纳入结果
                if dist < radius:
                    obstacles_within_radius.append(obstacle.obstacle_id)

    return obstacles_within_radius


def get_dyn_and_stat_obstacles(obstacle_ids: [int], scenario):
    """
    Split a set of obstacles in a set of dynamic obstacles and a set of static obstacles.

    Args:
        obstacle_ids ([int]): IDs of all considered obstacles.
        scenario: Considered scenario.

    Returns:
        [int]: List with the IDs of all dynamic obstacles.
        [int]: List with the IDs of all static obstacles.

    """
    # 保存动态障碍物 ID
    dyn_obstacles = []
    # 保存静态障碍物 ID
    stat_obstacles = []
    # 遍历输入的障碍物 ID 列表
    for obst_id in obstacle_ids:
        # 通过 scenario.obstacle_by_id 获取障碍物对象，并根据其 obstacle_role 判断是动态还是静态
        if scenario.obstacle_by_id(obst_id).obstacle_role == ObstacleRole.DYNAMIC:
            dyn_obstacles.append(obst_id)
        else:
            stat_obstacles.append(obst_id)

    return dyn_obstacles, stat_obstacles


def get_orientation_velocity_and_shape_of_prediction(
        predictions: dict, scenario, safety_margin_length=1.0, safety_margin_width=0.5
):
    """
    Extend the prediction by adding information about the orientation, velocity and the shape of the predicted obstacle.

    Args:
        predictions (dict): Prediction dictionary that should be extended.
        scenario (Scenario): Considered scenario.

    Returns:
        dict: Extended prediction dictionary.
        :param safety_margin_width:
        :param predictions:
        :param scenario:
        :param safety_margin_length:
    """
    # go through every predicted obstacle
    # 遍历 prediction 字典中每一个被预测的障碍物
    obstacle_ids = list(predictions.keys())
    for obstacle_id in obstacle_ids:
        # 从场景中取出该障碍物对象
        obstacle = scenario.obstacle_by_id(obstacle_id)
        # get x- and y-position of the predicted trajectory
        # 取得该障碍物的预测轨迹，通常是位置序列或多模态位置序列
        pred_traj = predictions[obstacle_id]['pos_list']

        # added by Khaled. Convert the 2D array into a list of 2D array, to use the same routine
        # for both uni-modal and multi-modal predictions.
        # not needed anymore, the if condition can be removed, and only the else part should be kept
        # 这里是兼容单模态 / 多模态预测结果的数据结构：
        # - 单模态时 pred_traj 可能是一个二维数组 shape=(T,2)
        # - 多模态时 pred_traj 可能是 list，其中每个元素是一个二维轨迹数组
        if type(pred_traj) != list:
            # 若不是 list，说明它可能是单条轨迹的二维数组
            pred_traj_list_tmp = list()
            # 将二维数组逐行转换成 list，再重新转回 np.array
            for entry in pred_traj:
                pred_traj_list_tmp.append(entry)
            pred_traj_list = list()
            pred_traj_list.append(np.array(pred_traj_list_tmp))
        else:
            # 如果已经是 list，直接作为多模态轨迹列表使用
            pred_traj_list = pred_traj

        # orientation/v 初始化大小应与“实际模式数”一致，而不是 obstacle.prediction 长度
        mode_count = len(pred_traj_list)
        pred_orientation_list = [[None]] * mode_count
        pred_v_list = [[None]] * mode_count

        # mode 索引计数器
        index = 0
        # 遍历每个模式对应的预测轨迹
        for pred_traj in pred_traj_list:
            # 当前模式轨迹长度（时间步数）
            pred_length = len(pred_traj)

            # there may be some predictions without any trajectory (when the obstacle disappears due to exceeding time)
            # 有些模式下轨迹可能为空，例如障碍物超出场景预测范围后消失
            if pred_length == 0:
                del predictions[obstacle_id]
                continue

            # for predictions with only one timestep, the gradient can not be derived --> use initial orientation
            # 如果轨迹只有一个点，无法通过数值微分求导得到朝向和速度
            # 因此直接退化使用初始状态中的朝向和速度
            if pred_length == 1:
                pred_orientation = [obstacle.initial_state.orientation]
                pred_v = [obstacle.initial_state.velocity]
                pred_orientation_list[index] = pred_orientation
                pred_v_list[index] = pred_v
                index = index + 1
            else:
                # 构造时间序列：0, dt, 2dt, ...
                t = [0.0 + i * scenario.dt for i in range(pred_length)]
                # 提取 x 坐标序列
                x = pred_traj[:, 0][0:pred_length]
                # 提取 y 坐标序列
                y = pred_traj[:, 1][0:pred_length]

                # calculate the yaw angle for the predicted trajectory
                # 对 x(t)、y(t) 求数值梯度，得到速度在 x/y 方向上的分量 dx/dt 与 dy/dt
                dx = np.gradient(x, t)
                dy = np.gradient(y, t)
                # if the vehicle does barely move, use the initial orientation
                # otherwise small uncertainties in the position can lead to great orientation uncertainties
                # 如果车辆几乎不动，则由位置微小扰动算出来的朝向会非常不稳定
                # 因此在“几乎静止”情况下，直接使用初始朝向
                if all(dxi < 0.0001 for dxi in dx) and all(dyi < 0.0001 for dyi in dy):
                    init_orientation = obstacle.initial_state.orientation
                    # 构造与轨迹长度一致的常值朝向数组
                    pred_orientation = np.full((1, pred_length), init_orientation)[0]
                # if the vehicle moves, calculate the orientation
                else:
                    # 利用 arctan2(dy, dx) 计算航向角
                    pred_orientation = np.arctan2(dy, dx)

                # get the velocity from the derivation of the position
                # 根据速度分量计算速度大小
                pred_v = np.sqrt((np.power(dx, 2) + np.power(dy, 2)))

                # add the new information to the prediction dictionary
                # 将该 mode 的朝向和速度写入临时列表
                pred_orientation_list[index] = pred_orientation
                pred_v_list[index] = pred_v
                index = index + 1
                '''
                    predictions[obstacle_id]['orientation_list'] = pred_orientation
                    predictions[obstacle_id]['v_list'] = pred_v
                    obstacle_shape = obstacle.obstacle_shape
                    predictions[obstacle_id]['shape'] = {
                        'length': obstacle_shape.length + safety_margin_length,
                        'width': obstacle_shape.width + safety_margin_width,
                    }
                '''
        # this to make the orientation_list as a list of different modes. If the prediction
        # has a single mode, the list will have a single entry.
        # 将所有模式的朝向结果写回 prediction 字典
        predictions[obstacle_id]['orientation_list'] = pred_orientation_list
        # 将所有模式的速度结果写回 prediction 字典
        predictions[obstacle_id]['v_list'] = pred_v_list
        # 取障碍物原始形状
        obstacle_shape = obstacle.obstacle_shape
        # 为预测障碍物附加形状尺寸，并额外增加安全冗余边界
        predictions[obstacle_id]['shape'] = {
            'length': obstacle_shape.length + safety_margin_length,
            'width': obstacle_shape.width + safety_margin_width,
        }

    # return the updated predictions dictionary
    # 返回扩展后的 predictions
    return predictions


def _wrap_to_pi(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _safe_softmax(logits, temperature=1.0):
    """Numerically stable softmax."""
    temperature = max(temperature, 1e-6)
    scaled = np.asarray(logits, dtype=float) / temperature
    scaled = scaled - np.max(scaled)
    exp_x = np.exp(scaled)
    denom = np.sum(exp_x)
    if denom <= 0:
        return np.ones_like(exp_x) / len(exp_x)
    return exp_x / denom


def _build_heading_line(start_position, heading, distance=80.0):
    start = np.asarray(start_position, dtype=float)
    end = start + distance * np.array([np.cos(heading), np.sin(heading)])
    return LineString([start, end])


def _extract_first_conflict_point(ego_line: LineString, mode_traj) -> np.ndarray:
    mode_points = np.asarray(mode_traj, dtype=float)
    if mode_points.ndim != 2 or mode_points.shape[0] < 2:
        return None

    mode_line = LineString(mode_points)
    intersection = ego_line.intersection(mode_line)
    if intersection.is_empty:
        return None
    if intersection.geom_type == "Point":
        return np.array([intersection.x, intersection.y], dtype=float)
    if intersection.geom_type == "MultiPoint":
        point = list(intersection.geoms)[0]
        return np.array([point.x, point.y], dtype=float)
    if intersection.geom_type in {"LineString", "LinearRing"}:
        coords = np.asarray(intersection.coords, dtype=float)
        if len(coords) > 0:
            return coords[len(coords) // 2]
    if hasattr(intersection, "geoms"):
        for geom in intersection.geoms:
            if geom.geom_type == "Point":
                return np.array([geom.x, geom.y], dtype=float)
            if geom.geom_type in {"LineString", "LinearRing"}:
                coords = np.asarray(geom.coords, dtype=float)
                if len(coords) > 0:
                    return coords[len(coords) // 2]
    return None


def _gaussian_likelihood(value, mean, sigma):
    sigma = max(float(sigma), 1e-3)
    error = (float(value) - float(mean)) / sigma
    return float(np.exp(-0.5 * error * error))


def update_yield_challenge_belief(
        predictions: dict,
        scenario,
        ego_state,
        time_step: int,
        prior_belief: dict = None,
        dt: float = None,
):
    if predictions is None or len(predictions) == 0:
        return {}, {} if prior_belief is None else prior_belief

    if dt is None:
        dt = scenario.dt
    updated_predictions = predictions
    updated_belief = {} if prior_belief is None else dict(prior_belief)

    ego_heading = float(getattr(ego_state, "orientation", 0.0))
    ego_speed = max(0.1, float(getattr(ego_state, "velocity", 0.0)))
    ego_line = _build_heading_line(ego_state.position, ego_heading)

    for obstacle_id, pred in updated_predictions.items():
        pos_list = pred.get("pos_list")
        if not isinstance(pos_list, list) or len(pos_list) < 2:
            continue

        prior = np.asarray(updated_belief.get(obstacle_id, pred.get("mode_prob", [0.5, 0.5])), dtype=float)
        if prior.shape[0] != 2:
            prior = np.array([0.5, 0.5], dtype=float)
        prior = np.clip(prior, 1e-6, None)
        prior = prior / np.sum(prior)

        obstacle = scenario.obstacle_by_id(obstacle_id)
        obstacle_state = _get_obstacle_state_at_timestep(obstacle, time_step)
        observed_speed = float(max(0.0, getattr(obstacle_state, "velocity", 0.0)))
        observed_acc = float(getattr(obstacle_state, "acceleration", 0.0))

        yield_traj = np.asarray(pos_list[0], dtype=float)
        challenge_traj = np.asarray(pos_list[1], dtype=float)
        if len(yield_traj) < 2 or len(challenge_traj) < 2:
            pred["mode_prob"] = prior.tolist()
            updated_belief[obstacle_id] = prior.tolist()
            continue

        yield_ref_speed = np.linalg.norm(yield_traj[1] - yield_traj[0]) / max(dt, 1e-3)
        challenge_ref_speed = np.linalg.norm(challenge_traj[1] - challenge_traj[0]) / max(dt, 1e-3)

        yield_likelihood = _gaussian_likelihood(observed_speed, yield_ref_speed, sigma=max(0.5, 0.35 * challenge_ref_speed))
        challenge_likelihood = _gaussian_likelihood(observed_speed, challenge_ref_speed, sigma=max(0.5, 0.35 * challenge_ref_speed))

        conflict_point = _extract_first_conflict_point(ego_line, challenge_traj)
        if conflict_point is not None:
            challenge_line = LineString(challenge_traj)
            obstacle_progress = challenge_line.project(
                Point(float(obstacle_state.position[0]), float(obstacle_state.position[1]))
            )
            conflict_progress = challenge_line.project(
                Point(float(conflict_point[0]), float(conflict_point[1]))
            )
            obstacle_distance_to_conflict = max(0.0, float(conflict_progress - obstacle_progress))
            obstacle_ttc = obstacle_distance_to_conflict / max(observed_speed, 0.1)

            ego_distance_to_conflict = max(
                0.0,
                float(ego_line.project(Point(float(conflict_point[0]), float(conflict_point[1])))),
            )
            ego_ttc = ego_distance_to_conflict / ego_speed

            ttc_margin = 1.0
            if ego_ttc + ttc_margin < obstacle_ttc:
                yield_likelihood *= 2.5
                challenge_likelihood *= 0.6
            elif obstacle_ttc + 0.5 < ego_ttc:
                challenge_likelihood *= 2.2
                yield_likelihood *= 0.7

            if obstacle_distance_to_conflict < 25.0:
                if observed_acc < -0.2:
                    yield_likelihood *= (1.0 + min(2.0, -observed_acc))
                elif observed_acc > 0.2:
                    challenge_likelihood *= (1.0 + min(2.0, observed_acc))

        likelihood = np.array([yield_likelihood, challenge_likelihood], dtype=float)
        posterior = prior * np.clip(likelihood, 1e-6, None)
        posterior_sum = np.sum(posterior)
        if posterior_sum <= 0.0:
            posterior = np.array([0.5, 0.5], dtype=float)
        else:
            posterior = posterior / posterior_sum

        pred["mode_prob"] = posterior.tolist()
        updated_belief[obstacle_id] = posterior.tolist()

    return updated_predictions, updated_belief


def _lanelet_heading(lanelet):
    """Estimate lanelet heading using center vertices."""
    center = np.asarray(lanelet.center_vertices)
    if center.shape[0] < 2:
        return 0.0
    vec = center[-1] - center[0]
    return float(np.arctan2(vec[1], vec[0]))


def _polyline_arc_lengths(polyline):
    """Return cumulative arc-lengths for a polyline."""
    points = np.asarray(polyline, dtype=float)
    if len(points) == 0:
        return np.array([0.0])
    if len(points) == 1:
        return np.array([0.0])
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(seg)))


def _project_point_to_polyline(point, polyline):
    """Project a point to a polyline and return arc-length coordinate."""
    points = np.asarray(polyline, dtype=float)
    p = np.asarray(point, dtype=float)
    if len(points) <= 1:
        return 0.0

    s_vals = _polyline_arc_lengths(points)
    best_s = 0.0
    best_dist = float("inf")

    for i in range(len(points) - 1):
        a = points[i]
        b = points[i + 1]
        ab = b - a
        norm_sq = float(np.dot(ab, ab))
        if norm_sq < 1e-9:
            continue
        t = np.dot(p - a, ab) / norm_sq
        t = np.clip(t, 0.0, 1.0)
        proj = a + t * ab
        dist = float(np.linalg.norm(p - proj))
        if dist < best_dist:
            best_dist = dist
            best_s = float(s_vals[i] + t * np.linalg.norm(ab))
    return best_s


def _sample_polyline(polyline, start_s, ds, num_steps):
    """Sample points along a polyline with fixed arc-length increment."""
    points = np.asarray(polyline, dtype=float)
    if len(points) == 0:
        return np.zeros((num_steps, 2))
    if len(points) == 1:
        return np.repeat(points, num_steps, axis=0)

    s_vals = _polyline_arc_lengths(points)
    total = float(s_vals[-1])
    query_s = np.clip(start_s + ds * np.arange(num_steps), 0.0, total)
    out = np.zeros((num_steps, 2), dtype=float)

    for idx, s_query in enumerate(query_s):
        seg_idx = np.searchsorted(s_vals, s_query, side="right") - 1
        seg_idx = int(np.clip(seg_idx, 0, len(points) - 2))
        s0, s1 = s_vals[seg_idx], s_vals[seg_idx + 1]
        if s1 - s0 < 1e-9:
            out[idx] = points[seg_idx]
            continue
        ratio = (s_query - s0) / (s1 - s0)
        out[idx] = points[seg_idx] * (1.0 - ratio) + points[seg_idx + 1] * ratio
    return out


def _find_lanelet_path(lanelet_network, start_id, target_id, max_depth=3):
    """BFS shortest lanelet path from start to target via successors."""
    if start_id == target_id:
        return [start_id]

    queue = deque([(start_id, [start_id], 0)])
    visited = {start_id}
    while queue:
        node, path, depth = queue.popleft()
        if depth >= max_depth:
            continue
        lanelet = lanelet_network.find_lanelet_by_id(node)
        for succ in lanelet.successor:
            if succ == target_id:
                return path + [succ]
            if succ not in visited:
                visited.add(succ)
                queue.append((succ, path + [succ], depth + 1))
    return [start_id, target_id]


def get_reachable_lanelets_from_obstacle_position(
        scenario,
        obstacle_position=None,
        max_depth: int = 3,
        include_start_lanelet: bool = True,
):
    """
    根据 obstacle 的位置，返回沿 successor 可达的 lanelet 序列。

    Returns:
        list[list[int]]: 每个 start_lanelet_id 对应若干条可达 lanelet 序列。
    """
    lanelet_network = scenario.lanelet_network
    start_lanelet_ids = lanelet_network.find_lanelet_by_position([obstacle_position])[0]
    if len(start_lanelet_ids) == 0:
        return []

    reachable_lanelet_sequences = []

    for start_lanelet_id in start_lanelet_ids:
        frontier = deque([([start_lanelet_id], 0)])

        while frontier:
            path_lanelet_ids, depth = frontier.popleft()
            current_lanelet_id = path_lanelet_ids[-1]
            current_lanelet = lanelet_network.find_lanelet_by_id(current_lanelet_id)

            if current_lanelet is None:
                continue

            if depth >= max_depth or len(current_lanelet.successor) == 0:
                if include_start_lanelet:
                    reachable_lanelet_sequences.append(path_lanelet_ids)
                else:
                    reachable_lanelet_sequences.append(path_lanelet_ids[1:])
                continue

            extended = False
            for succ_id in current_lanelet.successor:
                if succ_id in path_lanelet_ids:
                    continue
                frontier.append((path_lanelet_ids + [succ_id], depth + 1))
                extended = True

            if not extended:
                if include_start_lanelet:
                    reachable_lanelet_sequences.append(path_lanelet_ids)
                else:
                    reachable_lanelet_sequences.append(path_lanelet_ids[1:])

    reachable_lanelet_sequences = [seq for seq in reachable_lanelet_sequences if len(seq) > 0]

    unique_reachable_lanelet_sequences = []
    seen_sequences = set()
    for reachable_lanelet_sequence in reachable_lanelet_sequences:
        sequence_key = tuple(reachable_lanelet_sequence)
        if sequence_key in seen_sequences:
            continue
        seen_sequences.add(sequence_key)
        unique_reachable_lanelet_sequences.append(reachable_lanelet_sequence)

    return unique_reachable_lanelet_sequences


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _build_yield_challenge_mode_trajectories(obstacle_state, base_mean, horizon, dt):
    base_points = np.asarray(base_mean, dtype=float)
    current_pos = np.asarray(obstacle_state.position, dtype=float)

    if base_points.ndim != 2 or base_points.shape[1] != 2 or len(base_points) == 0:
        heading = float(getattr(obstacle_state, "orientation", 0.0))
        fallback_end = current_pos + 30.0 * np.array([np.cos(heading), np.sin(heading)])
        path_points = np.vstack([current_pos, fallback_end])
    else:
        if np.linalg.norm(base_points[0] - current_pos) < 1e-6:
            path_points = base_points
        else:
            path_points = np.vstack([current_pos, base_points])

    if len(path_points) == 1:
        path_points = np.vstack([path_points, path_points])

    path_lengths = _polyline_arc_lengths(path_points)
    path_total_length = float(path_lengths[-1])

    if horizon <= 1 or path_total_length < 1e-6:
        repeated = np.repeat(path_points[:1], max(horizon, 1), axis=0)
        return [repeated, repeated.copy()]

    base_ds = path_total_length / max(horizon - 1, 1)
    base_speed = max(float(getattr(obstacle_state, "velocity", 0.0)), base_ds / max(dt, 1e-3), 0.2)

    challenge_ds = base_speed * dt
    yield_ds = min(challenge_ds * 0.35, 1.0)

    challenge_traj = _sample_polyline(path_points, 0.0, challenge_ds, horizon)
    yield_traj = _sample_polyline(path_points, 0.0, yield_ds, horizon)
    return [yield_traj, challenge_traj]


def get_rule_based_base_predictions(
        scenario,
        obstacle_id_list,
        horizon: int,
        timestep: int = None,
        dt: float = None,
):
    if dt is None:
        dt = scenario.dt

    prediction_result = {}
    for obstacle_id in obstacle_id_list:
        obstacle = scenario.obstacle_by_id(obstacle_id)
        if obstacle is None or obstacle.obstacle_role != ObstacleRole.DYNAMIC:
            continue

        obstacle_state = _get_obstacle_state_at_timestep(obstacle, timestep)
        path_points = _build_straight_lanelet_path(
            scenario=scenario,
            initial_state=obstacle.initial_state,
        )
        start_s = _project_point_to_polyline(
            point=np.asarray(obstacle_state.position, dtype=float),
            polyline=path_points,
        )
        speed = max(float(getattr(obstacle_state, "velocity", 0.0)), 0.2)
        base_mean = _sample_polyline(path_points, start_s, speed * dt, horizon)

        base_cov = []
        for step_idx in range(horizon):
            variance = 0.2 + 0.03 * step_idx
            base_cov.append([[variance, 0.0], [0.0, variance]])

        prediction_result[obstacle_id] = {
            "pos_list": np.asarray(base_mean, dtype=float),
            "cov_list": np.asarray(base_cov, dtype=float),
        }

    return prediction_result


def generate_gt_mode_trajectories_to_lanelets(
        scenario,
        obstacle_state,
        target_lanelet_sequences,
        horizon: int,
        dt: float = None,
):
    """
    Step (ii): build per-mode "ground truth-like" trajectories to lanelet sequences.

    这里直接使用完整的 target_lanelet_sequences。
    默认约定：每条序列的首个 lanelet 就是障碍物当前所在 lanelet。

    The trajectory generator follows lanelet center-lines with constant speed.
    """
    # ----------------------------------------------------------------------
    # 函数目标
    # ----------------------------------------------------------------------
    # 这个函数的作用是：
    # 给定一个障碍物，以及若干条候选的 target lanelet sequence，
    # 为每一条 lanelet sequence 生成一条“ground truth-like”的未来轨迹。
    #
    # 所谓 ground truth-like，可以理解为：
    # 不是直接从真实数据中读取未来轨迹，
    # 而是按照道路几何（lanelet 中心线）和当前速度，
    # 人工构造一条“看起来像真实车会沿车道行驶的轨迹”。
    #
    # 每条 sequence 对应一个 mode，因此最终返回的是：
    #   mode_trajs = [traj_mode_0, traj_mode_1, ...]
    #
    # 其中每条 traj_mode_i 都是一条未来位置轨迹，通常形状为 [horizon, 2]
    # ----------------------------------------------------------------------
    lanelet_network = scenario.lanelet_network
    current_pos = np.asarray(obstacle_state.position, dtype=float)
    current_speed = float(getattr(obstacle_state, "velocity", 0.0))
    if dt is None:
        dt = scenario.dt
    # ----------------------------------------------------------------------
    # ds 表示每个时间步沿路径前进的弧长近似步长
    # ----------------------------------------------------------------------
    # 这里采用“恒速沿中心线前进”的简化模型：
    #
    #   ds = v * dt
    #
    # 但为了避免 current_speed = 0 时完全不动，
    # 这里给了一个最小速度下限 0.2
    # 即使速度是 0，也会按 0.2 m/s 的最小值推进一点点
    ds = max(current_speed, 0.2) * dt

    mode_trajs = []
    for target_lanelet_sequence in target_lanelet_sequences:
        # --------------------------------------------------------------
        # Step 1: 将输入统一整理成 lanelet_id 列表
        # --------------------------------------------------------------
        if isinstance(target_lanelet_sequence, (list, tuple, np.ndarray)):
            lanelet_path = [int(lanelet_id) for lanelet_id in target_lanelet_sequence]
        else:
            lanelet_path = [int(target_lanelet_sequence)]

        if len(lanelet_path) == 0:
            continue
        # --------------------------------------------------------------
        # Step 2: 构造该 mode 的“路径中心线点序列”
        # --------------------------------------------------------------
        # centerline_points 用来保存整个 mode 的参考路径点。
        #
        # 这里先把当前障碍物位置 current_pos 作为路径起点放进去，
        # 这样可以保证生成出来的轨迹从障碍物当前位置开始。
        centerline_points = []
        for idx, lanelet_id in enumerate(lanelet_path):
            lanelet = lanelet_network.find_lanelet_by_id(lanelet_id)
            if lanelet is None:
                continue
            center = np.asarray(lanelet.center_vertices, dtype=float)
            if idx > 0 and len(centerline_points) > 0 and len(center) > 0:
                if np.linalg.norm(centerline_points[-1] - center[0]) < 1e-6:
                    center = center[1:]
            centerline_points.extend(center.tolist())

        if len(centerline_points) < 2:
            centerline_points = [
                current_pos.tolist(),
                (current_pos + np.array([ds, 0.0])).tolist(),
            ]

        centerline_points = np.asarray(centerline_points, dtype=float)
        start_s = _project_point_to_polyline(current_pos, centerline_points)

        # --------------------------------------------------------------
        # Step 5: 按固定步长 ds 在 polyline 上向前采样 horizon 个点
        # --------------------------------------------------------------
        # _sample_polyline(...) 的作用通常是：
        # 从折线 centerline_points 的弧长 start_s 开始，
        # 每次向前走 ds，
        # 一共采样 horizon 个未来点。
        #
        # 因为 ds = v * dt，所以这相当于：
        # 用“恒定速度 current_speed”沿着 lanelet 中心线生成未来轨迹。
        mode_traj = _sample_polyline(centerline_points, start_s, ds, horizon)
        mode_trajs.append(mode_traj)

    return mode_trajs


def build_multimodal_gmm_predictions(
        scenario,
        base_prediction: dict,
        obstacle_id_list,
        horizon: int = 50,
        max_modes: int = 3,
        likelihood_temperature: float = 1.0,
        timestep: int = None,
):
    """
    Step (iii): turn single-modal predictor output into multi-modal GMM-like prediction.

    For each obstacle:
    - infer intent lanelets
    - generate mode trajectories towards each lanelet
    - build Gaussian trajectories (mean per mode + covariance from single-modal model)
    - compute mode likelihoods and normalize to mode probabilities
    """
    # ----------------------------------------------------------------------
    # 该函数的作用：
    # 将“单模态预测器”的输出，转换成“单个障碍物对应多个 mode 的 GMM 风格预测结果”。
    #
    # 输入的 base_prediction 通常来自单模态预测器，例如：
    #   base_prediction[obstacle_id] = {
    #       "pos_list": 单条未来均值轨迹,      shape ~ [T, 2]
    #       "cov_list": 每个时间步的协方差,    shape ~ [T, 2, 2]
    #   }
    #
    # 本函数会进一步做以下事情：
    # 1. 对每个障碍物推断可能的目标 lanelet（可理解为意图候选）
    # 2. 针对每个目标 lanelet 生成一条 mode 轨迹
    # 3. 使用单模态预测器的协方差，给每条 mode 轨迹配上 Gaussian 不确定性
    # 4. 计算每个 mode 的相对似然，并归一化成 mode 概率
    #
    # 最终输出 prediction_result 的形式大致为：
    #   prediction_result[obstacle_id] = {
    #       "pos_list": [mode_0_mean, mode_1_mean, ...],
    #       "cov_list": [mode_0_cov,  mode_1_cov,  ...],
    #       "mode_prob": [p0, p1, ...]
    #   }
    # ----------------------------------------------------------------------
    prediction_result = {}

    for obstacle_id in obstacle_id_list:
        # ------------------------------------------------------------------
        # Step 2: 取出单模态预测器给出的“基准未来轨迹均值”和“基准协方差”
        # ------------------------------------------------------------------
        # base_mean: 单模态预测均值轨迹，期望形状为 [T, 2]
        # 含义：T 个未来时刻，每个时刻一个 (x, y)
        obstacle = scenario.obstacle_by_id(obstacle_id)
        obstacle_state = _get_obstacle_state_at_timestep(obstacle, timestep)
        obstacle_position = obstacle_state.position

        base_mean = np.asarray(base_prediction[obstacle_id]["pos_list"], dtype=float)
        # base_cov: 单模态预测协方差，期望形状为 [T, 2, 2]
        # 含义：每个未来时刻一个 2x2 协方差矩阵
        base_cov = np.asarray(base_prediction[obstacle_id]["cov_list"], dtype=float)
        
        # 若 base_mean 不是二维数组，或者第二维不是 2（x, y），则说明格式不正确
        # 直接跳过该障碍物
        if base_mean.ndim != 2 or base_mean.shape[1] != 2:
            continue
        # 若未来轨迹长度为 0，说明没有可用预测点，也跳过
        if len(base_mean) == 0:
            continue

        # ------------------------------------------------------------------
        # Step 3: 截断 / 对齐预测时域长度
        # ------------------------------------------------------------------

        # 实际使用的时域长度取 horizon 与 base_mean 实际长度的较小值
        # 这样可以防止超出单模态预测器原本输出的轨迹长度
        horizon_len = min(horizon, len(base_mean))
        # 只保留前 horizon_len 个时间步
        base_mean = base_mean[:horizon_len]
        
        # 如果 base_cov 的长度足够，也截取到同样长度
        if len(base_cov) >= horizon_len:
            base_cov = base_cov[:horizon_len]
        else:
            # Fallback covariance if predictor horizon is shorter.
            # 如果协方差长度比 horizon_len 短，则采用一个默认的回退协方差
            # 这里假设每个时间步的协方差都相同，为 [[0.2, 0], [0, 0.2]]
            # Fallback covariance if predictor horizon is shorter.
            base_cov = np.array([[[0.2, 0.0], [0.0, 0.2]]] * horizon_len)
        mode_trajs = _build_yield_challenge_mode_trajectories(
            obstacle_state=obstacle_state,
            base_mean=base_mean,
            horizon=horizon_len,
            dt=scenario.dt,
        )
        # ------------------------------------------------------------------
        # Step 7: 初始化多模态结果容器
        # ------------------------------------------------------------------
        # 保存每个 mode 的未来位置均值轨迹
        mode_pos_list = []
        # 保存每个 mode 的未来协方差序列
        mode_cov_list = []
        mode_behavior_list = ["yield", "challenge"]
        # 保存每个 mode 的“对数似然”分数，后面用于 softmax 得到概率
        mode_log_likelihood = []
        eps = 1e-6
        # ------------------------------------------------------------------
        # Step 8: 遍历每一个候选 mode，构造 GMM 分量并计算相对似然
        # ------------------------------------------------------------------
        for mode_idx, mode_traj in enumerate(mode_trajs):
            # 当前 mode 的均值轨迹，转成 float ndarray
            mode_mean = np.asarray(mode_traj, dtype=float)
            # 当前实现中，直接复用单模态预测器输出的协方差作为该 mode 的协方差
            # 即：不同 mode 的 mean 不同，但 covariance 暂时共享 base_cov
            mode_cov = np.asarray(base_cov, dtype=float)

            # Likelihood of this intent mode under single-modal prediction residual.
            # Approximation: diagonal Mahalanobis over x/y at each step.
            #
            # 这里要给每个 mode 一个相对“可信度”分数。
            # 当前近似方法是：
            #   用该 mode 的均值轨迹 mode_mean 与单模态预测均值 base_mean 做差，
            #   然后用 base_cov 的对角方差构造一个简化的 Mahalanobis 距离。
            #
            # diff[t] = base_mean[t] - mode_mean[t]
            # 如果某个 mode 轨迹与 base_mean 很接近，那么这个距离会小，似然会高；
            # 如果偏差很大，那么距离会大，似然会低。
            diff = base_mean - mode_mean
            # 提取每个时间步在 x 方向上的方差，并做下限裁剪，避免接近 0
            var_x = np.clip(mode_cov[:, 0, 0], eps, None)
            # 提取每个时间步在 y 方向上的方差，并做下限裁剪
            var_y = np.clip(mode_cov[:, 1, 1], eps, None)
            # 构造一个简化的负对数似然（negative log-likelihood, NLL）
            # 这里只用了对角项，没有用完整协方差矩阵的逆
            #
            # 公式上类似：
            #   0.5 * mean( dx^2 / var_x + dy^2 / var_y )
            #
            # 数值越小，说明 mode_mean 越接近 base_mean
            nll = 0.5 * np.mean((diff[:, 0] ** 2) / var_x + (diff[:, 1] ** 2) / var_y)
            # 由于 softmax 通常对“值越大概率越高”的分数进行归一化，
            # 所以这里存储的是 -nll 作为 log-likelihood 风格分数
            mode_log_likelihood.append(-nll)
            # 保存该 mode 的均值轨迹
            mode_pos_list.append(mode_mean)
            # 保存该 mode 的协方差序列
            mode_cov_list.append(mode_cov)
        # ------------------------------------------------------------------
        # Step 9: 对所有 mode 的分数做 softmax，得到 mode 概率
        # ------------------------------------------------------------------

        # _safe_softmax 应该是一个数值稳定的 softmax 实现
        # temperature 用于控制分布尖锐程度：
        # - temperature 小：概率更尖锐，更偏向最大分数的 mode
        # - temperature 大：概率更平滑，更均匀
        mode_prob = np.array([0.5, 0.5], dtype=float)
        # ------------------------------------------------------------------
        # Step 10: 将当前障碍物的多模态结果写入 prediction_result
        # ------------------------------------------------------------------
        prediction_result[obstacle_id] = {
            # 每个元素是一条 mode 的未来位置轨迹
            "pos_list": mode_pos_list,
            # 每个元素是一条 mode 的未来协方差序列
            "cov_list": mode_cov_list,
            # 各 mode 的归一化概率
            "mode_prob": mode_prob.tolist(),
            "mode_behavior": mode_behavior_list,
        }
    # 返回所有障碍物的多模态 GMM 风格预测结果
    return prediction_result


class _TrajectoryAdapter:
    """Minimal adapter with x/y/yaw for collision-probability utility."""

    def __init__(self, x, y, yaw):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.yaw = np.asarray(yaw, dtype=float)


def compute_mode_weights_from_collision_softmax(
        predictions: dict,
        ego_traj,
        vehicle_params,
        obstacle_id: int = None,
        start_idx: int = 0,
        collision_gain: float = 8.0,
        softmax_temperature: float = 1.0,
):
    """
    Step (iv): collision-aware dynamic mode weights via softmax.

    Args:
        predictions: multimodal prediction dictionary.
        ego_traj: FrenetTrajectory-like object (needs x/y/yaw arrays).
        vehicle_params: ego vehicle parameters.
        obstacle_id: optional, which obstacle's mode weight to use.
        start_idx: prediction index offset.
    Returns:
        list[float]: normalized mode weights.
    """
    if not predictions:
        return [1.0]

    if obstacle_id is None:
        obstacle_id = sorted(list(predictions.keys()))[0]
    if obstacle_id not in predictions:
        return [1.0]

    mode_count = len(predictions[obstacle_id]["pos_list"])
    if mode_count == 0:
        return [1.0]

    prior = predictions[obstacle_id].get("mode_prob", None)
    if prior is None or len(prior) != mode_count:
        prior = np.ones(mode_count) / mode_count
    else:
        prior = np.asarray(prior, dtype=float)
        prior = np.clip(prior, 1e-8, None)
        prior = prior / np.sum(prior)

    coll_prob_dict = get_collision_probability_fast(
        traj=ego_traj,
        predictions=predictions,
        vehicle_params=vehicle_params,
        start_idx=start_idx,
        mode_num=100,
    )
    obstacle_coll = coll_prob_dict.get(obstacle_id, [])

    # Per-mode collision score: take max over time as risk indicator.
    mode_collision_scores = np.zeros(mode_count, dtype=float)
    for mode_idx in range(mode_count):
        if mode_idx >= len(obstacle_coll):
            continue
        series = np.asarray(obstacle_coll[mode_idx], dtype=float)
        mode_collision_scores[mode_idx] = float(np.max(series)) if series.size else 0.0

    logits = np.log(prior + 1e-12) - collision_gain * mode_collision_scores
    weights = _safe_softmax(logits, temperature=softmax_temperature)

    predictions[obstacle_id]["mode_prob"] = weights.tolist()
    return weights.tolist()


def collision_checker_prediction(
        predictions: dict, scenario, ego_co, frenet_traj, ego_state, start_idx, mode_num
):
    """
    Check predictions for collisions.

    Args:
        predictions (dict): Dictionary with the predictions of the obstacles.
        scenario (Scenario): Considered scenario.
        ego_co (TimeVariantCollisionObject): The collision object of the ego vehicles trajectory.
        frenet_traj (FrenetTrajectory): Considered trajectory.
        ego_state (State): Current state of the ego vehicle.

    Returns:
        bool: True if the trajectory collides with a prediction.
    """
    # check every obstacle in the predictions
    # 遍历所有被预测的障碍物
    for obstacle_id in list(predictions.keys()):
        if mode_num == 100:
            selected_mode = None
        elif isinstance(mode_num, dict):
            selected_mode = min(
                mode_num.get(obstacle_id, 0),
                len(predictions[obstacle_id]['pos_list']) - 1,
            )
        else:
            selected_mode = min(mode_num, len(predictions[obstacle_id]['pos_list']) - 1)

        # 遍历该障碍物的所有模式轨迹
        for mode in range(len(predictions[obstacle_id]['pos_list'])):
            # mode_num == 100: shared-plan 阶段，检查所有模式
            # mode_num != 100: contingent 阶段，仅检查指定模式
            if selected_mode is not None:
                mode = selected_mode
            # check if the obstacle is not a rectangle (only shape with attribute length)
            # 碰撞检测器当前仅支持矩形障碍物，因此要求 obstacle_shape 具有 length 属性
            if not hasattr(scenario.obstacle_by_id(obstacle_id).obstacle_shape, 'length'):
                raise Warning('Collision Checker can only handle rectangular obstacles.')
            else:

                # get dimensions of the obstacle
                # 取出预测中保存的带安全裕度的障碍物长度和宽度
                length = predictions[obstacle_id]['shape']['length']
                width = predictions[obstacle_id]['shape']['width']

                # only check for collision as long as both trajectories (frenét trajectory and prediction) are visible
                # 只检查 prediction 与自车轨迹都有定义的重叠区间
                pred_traj = predictions[obstacle_id]['pos_list'][mode][start_idx:]
                pred_length = min(len(frenet_traj.t), len(pred_traj))
                # 若重叠长度为 0，则跳过
                if pred_length == 0:
                    continue

                # get x, y and orientation of the prediction
                # 提取预测轨迹的 x/y 序列
                x = pred_traj[:, 0][0:pred_length]
                y = pred_traj[:, 1][0:pred_length]
                # 提取对应的朝向序列
                pred_orientation = predictions[obstacle_id]['orientation_list'][mode][start_idx:]

                # create a time variant collision object for the predicted vehicle
                # 将每个时刻的 [x, y, yaw] 组织成轨迹列表，供碰撞对象创建函数使用
                traj = [[x[i], y[i], pred_orientation[i]] for i in range(pred_length)]

                # 构造预测障碍物的时变碰撞对象
                # 注意：这里传入的是 half length / half width，因此外部函数可能使用的是半长半宽定义
                prediction_collision_object_raw = create_tvobstacle(
                    traj_list=traj,
                    box_length=length / 2,
                    box_width=width / 2,
                    start_time_step=ego_state.time_step + 1,
                )

            # preprocess the collision object
            # if the preprocessing fails, use the raw trajectory
            # 对碰撞对象进行预处理（通常是将 OBB 轨迹进行更高效的碰撞检测预处理）
            (
                prediction_collision_object,
                err,
            ) = trajectory_queries.trajectory_preprocess_obb_sum(
                prediction_collision_object_raw
            )
            # 若预处理失败，则退回到原始碰撞对象
            if err:
                prediction_collision_object = prediction_collision_object_raw

            # check for collision between the trajectory of the ego obstacle and the predicted obstacle
            # 执行自车轨迹与预测障碍物轨迹之间的动态碰撞检测
            collision_at = trajectory_queries.trajectories_collision_dynamic_obstacles(
                trajectories=[ego_co],
                dynamic_obstacles=[prediction_collision_object],
                method='grid',
                num_cells=32,
            )

            # if there is no collision (returns -1) return False, else True
            # 如果返回值不是 -1，说明存在碰撞，立刻返回 True
            if collision_at[0] != -1:
                return True

            # 如果指定只检查某一个模式，则该模式检查完后直接跳出 mode 循环
            if selected_mode is not None:
                break

    # 所有障碍物 / 模式检查完都没有碰撞，则返回 False
    return False


def add_static_obstacle_to_prediction(
        predictions: dict, obstacle_id_list: [int], scenario, pred_horizon: int = 50
):
    """
    Add static obstacles to the prediction since predictor can not handle static obstacles.

    Args:
        predictions (dict): Dictionary with the predictions.
        obstacle_id_list ([int]): List with the IDs of the static obstacles.
        scenario (Scenario): Considered scenario.
        pred_horizon (int): Considered prediction horizon. Defaults to 50.

    Returns:
        dict: Dictionary with the predictions.
    """
    # 遍历所有静态障碍物 ID
    for obstacle_id in obstacle_id_list:
        # 从场景中取出静态障碍物对象
        obstacle = scenario.obstacle_by_id(obstacle_id)
        # 未来位置列表
        fut_pos = []
        # 未来协方差列表
        fut_cov = []
        # create a mean and covariance matrix for every time step in the prediction horizon
        # 对于静态障碍物，在整个预测时域内位置保持不变
        for ts in range(int(pred_horizon)):
            # 每个时刻的位置都设置为初始位置
            fut_pos.append(obstacle.initial_state.position)
            # 给一个固定且很小的协方差，表示静态障碍物位置几乎确定
            fut_cov.append([[0.02, 0.0], [0.0, 0.02]])

        # 转换成 numpy 数组
        fut_pos = np.array(fut_pos)
        fut_cov = np.array(fut_cov)

        # add the prediction to the prediction dictionary
        # 写入 predictions 字典
        predictions[obstacle_id] = {'pos_list': fut_pos, 'cov_list': fut_cov}

    return predictions


def get_obstacles_prediction_overtake(zPred, backup):
    """
    针对超车场景，将外部预测结果 zPred / backup 整理成统一 prediction_result 格式。

    这个函数看起来是为一个特定的 overtaking 预测输出接口服务的，
    其中 backup[1] 中存放了多个 mode 的未来状态序列。
    """
    # 最终输出的 prediction 字典
    prediction_result = {}
    # 初始状态，取 zPred[0][0]
    init_state = zPred[0][0]
    # 固定障碍物长度
    obst_length = 6
    # 固定障碍物宽度
    obst_width = 2.5
    # 每个 mode 的状态维度块大小：看后续切片可知 [x, y, v, yaw] 共 4 维
    n = 4
    # number of policies
    # 模式数量，固定为 3
    m = 3
    # 未来位置列表：长度为 m，每个元素对应一个 mode
    fut_pos_list = [[None]] * m
    # 未来协方差列表：长度为 m
    fut_cov_list = [[None]] * m
    # 未来朝向列表：长度为 m
    pred_orientation_list = [[None]] * m
    # 未来速度列表：长度为 m
    pred_v_list = [[None]] * m
    # 最终转成 numpy 之后的位置列表
    fut_pos = []
    # 最终转成 numpy 之后的协方差列表
    fut_cov = []

    # 逐时刻构造每个 mode 的未来位置和协方差
    # backup[1].shape[0] 表示未来预测时域长度
    # +1 是因为要把初始状态作为第 0 个时刻插入
    for ts in range(backup[1].shape[0] + 1):
        for mode in range(3):
            if ts == 0:
                # 第一个时间步进行列表初始化
                fut_pos_list[mode] = list()
                fut_cov_list[mode] = list()
                # 初始位置来自 init_state[0:2]
                fut_pos_list[mode].append(init_state[0:2])
                # 初始协方差设为常数
                fut_cov_list[mode].append([[0.1, 0.0], [0.0, 0.1]])
            else:
                # 从 backup[1] 中取出对应 mode 的 [x, y]
                # 每个 mode 占 n=4 维，因此位置索引为 mode*n : mode*n+2
                fut_pos_list[mode].append(backup[1][ts - 1, mode * n: (mode * n + 2)])
                # 协方差固定常数
                fut_cov_list[mode].append([[0.1, 0.0], [0.0, 0.1]])

    # add orientation and velocity to the prediction dict
    # 从 backup[1] 中提取每个 mode 的 yaw 与 v
    for mode in range(m):
        # yaw 在每个 mode 块中的第 4 维（索引 3）
        pred_orientation_list[mode] = backup[1][:, n * mode + 3]
        # velocity 在每个 mode 块中的第 3 维（索引 2）
        pred_v_list[mode] = backup[1][:, n * mode + 2]

    # 在朝向和速度序列开头插入初始状态
    for mode in range(m):
        pred_orientation_list[mode] = np.insert(pred_orientation_list[mode], 0, init_state[3])
        pred_v_list[mode] = np.insert(pred_v_list[mode], 0, init_state[2])

    # 若 fut_pos 仍为空，则将临时 list 赋值给最终结构，并转为 numpy 数组
    if len(fut_pos) == 0:
        fut_pos = fut_pos_list
        fut_cov = fut_cov_list
        for i in range(len(fut_pos)):
            fut_pos[i] = np.array(fut_pos[i])
            fut_cov[i] = np.array(fut_cov[i])

    # 构造 obstacle_id=1 的预测结果
    prediction_result[1] = {'pos_list': fut_pos, 'cov_list': fut_cov, 'orientation_list': pred_orientation_list,
                            'v_list': pred_v_list}
    # 写入障碍物尺寸
    prediction_result[1]['shape'] = {'length': obst_length, 'width': obst_width}
    return prediction_result


def get_prediction_from_scenario_tree(zPred):
    """
    从 Branch MPC 生成的 scenario tree (zPred) 中提取未来预测轨迹，
    并整理成统一的 prediction_result 字典格式。

    输入
    ----
    zPred : scenario tree 预测结果
            结构通常是一个列表或数组，包含多条分支轨迹。
            每条轨迹一般是一个二维数组：
                shape = (T, state_dim)

            state 一般为：
                [x, y, v, yaw]

    输出
    ----
    prediction_result : dict
        包含所有未来场景的预测信息：
            - 每个场景的未来位置
            - 协方差
            - 朝向
            - 速度
            - 障碍物尺寸
    """
    # 最终返回的预测结果
    prediction_result = {}
    # 每层 scenario 数量（branching factor）
    # 这里写死为3，表示每个节点有3种行为模式
    m = 3
    # 存储最终组合后的所有场景轨迹
    scenarios_list = []

    # ------------------------------------------------------------
    # Step1：从 scenario tree 组合所有完整轨迹
    # ------------------------------------------------------------
    # zPred 实际是一个两层 scenario tree：
    #
    #      root
    #     /  |  \
    #    A   B   C
    #   /|\ /|\ /|\
    #  ... ... ...
    #
    # 第一层：3个行为模式
    # 第二层：每个模式再分3个
    #
    # 因此最终场景数：
    #   3 × 3 = 9
    #
    # 下面代码就是把两层轨迹拼接起来。
    #
    for mode_index in range(m):
        # 第二层节点索引
        for i in range(mode_index * m + m, mode_index * m + 2 * m):
            # 拼接轨迹
            #
            # zPred[mode_index][:-1]
            #    第一层轨迹（去掉最后一个点）
            #
            # zPred[i]
            #    第二层轨迹
            #
            # concat 后得到完整未来轨迹
            scenarios_list.append(np.concatenate((zPred[mode_index][:-1], zPred[i])))
    # ------------------------------------------------------------
    # Step2：初始化未来预测数据结构
    # ------------------------------------------------------------

    # 未来位置列表
    # 共9个场景
    fut_pos_list = [[None]] * 9
    # 协方差列表
    fut_cov_list = [[None]] * 9
    # 未来朝向
    pred_orientation_list = [[None]] * 9
    # 未来速度
    pred_v_list = [[None]] * 9
    # 最终返回结构
    fut_pos = []
    fut_cov = []
    # 障碍物尺寸（写死）
    obst_length = 6
    obst_width = 2.6

    # ------------------------------------------------------------
    # Step3：提取未来位置
    # ------------------------------------------------------------
    # scenarios_list[mode] 是一条完整未来轨迹：
    #
    # shape = (T, state_dim)
    #
    # 每个 state:
    #   [x, y, v, yaw]

    for ts in range(scenarios_list[0].shape[0]):
        # ts = time step
        for mode in range(m * m):  # 9个场景
            if ts == 0:
                # 第一次循环初始化列表
                fut_pos_list[mode] = list()
                fut_cov_list[mode] = list()
                # 提取位置信息
                fut_pos_list[mode].append(scenarios_list[mode][ts][0:2])
                # 设置协方差
                # 这里固定为常数,也可以根据时间步数增加（不确定性随时间增加）
                fut_cov_list[mode].append([[0.1, 0.0], [0.0, 0.1]])
            else:
                fut_pos_list[mode].append(scenarios_list[mode][ts][0:2])
                fut_cov_list[mode].append([[0.1, 0.0], [0.0, 0.1]])
    # ------------------------------------------------------------
    # Step4：提取朝向和速度
    # ------------------------------------------------------------
    # add orientation and velocity to the prediction dict
    for mode in range(m * m):
        pred_orientation_list[mode] = scenarios_list[mode][:, 3]
        pred_v_list[mode] = scenarios_list[mode][:, 2]
    '''
    # 这段代码被注释掉
    # 原本是把当前状态插入预测序列开头
    # TODO(yanjun): check if this is necessary, since scenario tree 里第一行就是从当前状态开始的
    for mode in range(m):
        pred_orientation_list[mode] = np.insert(pred_orientation_list[mode], 0, init_state[3])
        pred_v_list[mode] = np.insert(pred_v_list[mode], 0, init_state[2])
    '''

    # ------------------------------------------------------------
    # Step5：转换为 numpy array
    # ------------------------------------------------------------
    if len(fut_pos) == 0:
        fut_pos = fut_pos_list
        fut_cov = fut_cov_list
        for i in range(len(fut_pos)):
            fut_pos[i] = np.array(fut_pos[i])
            fut_cov[i] = np.array(fut_cov[i])
    # ------------------------------------------------------------
    # Step6：构造 prediction_result 字典
    # ------------------------------------------------------------
    prediction_result[1] = {
        'pos_list': fut_pos,                 # 每个场景未来位置
        'cov_list': fut_cov,                 # 位置协方差
        'orientation_list': pred_orientation_list,  # 朝向
        'v_list': pred_v_list                # 速度
    }
    # 障碍物尺寸
    prediction_result[1]['shape'] = {'length': obst_length, 'width': obst_width}
    return prediction_result


def get_ground_truth_prediction(
        obstacle_ids: [int], scenario, time_step: int, pred_horizon: int = 100
):
    """
    Transform the ground truth to a prediction. Use this if the prediction fails.

    Args:
        obstacle_ids ([int]): IDs of the visible obstacles.
        scenario (Scenario): considered scenario.
        time_step (int): Current time step.
        pred_horizon (int): Prediction horizon for the prediction.

    Returns:
        dict: Dictionary with the predictions.
    """
    # create a dictionary for the predictions
    # 创建最终的预测结果字典
    prediction_result = {}
    # 遍历所有需要生成 ground-truth prediction 的障碍物
    for obstacle_id in obstacle_ids:
        # 取出障碍物对象
        obstacle = scenario.obstacle_by_id(obstacle_id)
        # 用于单模态情况下的位置序列
        fut_pos = []
        # 用于单模态情况下的协方差序列
        fut_cov = []
        # predefine the length of the lists. It should be the same as number of prediction modes
        # 预分配多模态位置 / 协方差列表
        # 长度与 obstacle.prediction 的模式数量一致
        fut_pos_list = [[None]] * len(obstacle.prediction)
        fut_cov_list = [[None]] * len(obstacle.prediction)
        # predict dynamic obstacles as long as they are in the scenario
        if obstacle.obstacle_role == ObstacleRole.DYNAMIC:
            # The "list" check is added by Khaled
            # 动态障碍物的预测长度取决于它在 scenario 中可用的预测长度
            if type(obstacle.prediction) == list:
                len_pred = len(obstacle.prediction[0].occupancy_set)
            else:
                len_pred = len(obstacle.prediction.occupancy_set)

        # predict static obstacles for the length of the prediction horizon
        # 静态障碍物则直接预测整个 pred_horizon 长度
        else:
            len_pred = pred_horizon
        # create mean and the covariance matrix of the obstacles
        # for ts in range(time_step, min(pred_horizon, len_pred)):
        # 从当前 time_step 开始，最多预测 pred_horizon 步，但不能超过真实可用长度 len_pred
        for ts in range(time_step, min(time_step + pred_horizon, len_pred)):
            # get the occupancy of an obstacles (if it is not in the scenario at the given time step, the occupancy
            # is None)
            # 获取障碍物在时间 ts 的真实 occupancy
            occupancy = obstacle.occupancy_at_time(ts)
            if occupancy is not None:
                # create mean and covariance matrix
                # occupancy 可能是 list（多模态）也可能是单个 occupancy
                if type(occupancy) == list:
                    # 若是 list，则逐模式提取中心点
                    for index in range(len(obstacle.prediction)):
                        if ts == time_step:
                            # 第一个时间步，初始化对应 mode 的 list
                            fut_pos_list[index] = list()
                            fut_cov_list[index] = list()
                            # 记录当前位置中心
                            fut_pos_list[index].append(occupancy[index].shape.center)
                            # 协方差随时间步略有增长
                            fut_cov_list[index].append([[0.1 + (ts * 0.1) * 0.1, 0.0], [0.0, 0.1 + (ts * 0.1) * 0.1]])
                        else:
                            # 后续时间步继续追加
                            fut_pos_list[index].append(occupancy[index].shape.center)
                            fut_cov_list[index].append([[0.1 + (ts * 0.1) * 0.1, 0.0], [0.0, 0.1 + (ts * 0.1) * 0.1]])
                else:
                    # 若 occupancy 是单个对象，但 obstacle.prediction 本身是 list 且长度 > 1
                    # 说明这里是在兼容某种“多 prediction 但 occupancy 单值”的情况
                    if type(obstacle.prediction) == list and len(obstacle.prediction) > 1:
                        for index in range(len(obstacle.prediction)):
                            fut_pos_list[index] = list()
                            fut_cov_list[index] = list()
                            fut_pos_list[index].append(occupancy.shape.center)
                            fut_cov_list[index].append([[0.1, 0.0], [0.0, 0.1]])
                    else:
                        # 标准单模态情况：直接追加到 fut_pos / fut_cov
                        fut_pos.append(occupancy.shape.center)
                        fut_cov.append([[0.1, 0.0], [0.0, 0.1]])

        # 如果 fut_pos 为空，说明数据走的是多模态分支
        if len(fut_pos) == 0:
            fut_pos = fut_pos_list
            fut_cov = fut_cov_list
            # 每个 mode 转成 numpy array
            for i in range(len(fut_pos)):
                fut_pos[i] = np.array(fut_pos[i])
                fut_cov[i] = np.array(fut_cov[i])
        else:
            # 单模态情况：先转成 numpy array
            fut_pos = np.array(fut_pos)
            fut_cov = np.array(fut_cov)
            # 再包装成 list，使得输出格式与多模态一致：list[mode]
            fut_pos_list = list()
            fut_cov_list = list()
            fut_pos_list.append(fut_pos)
            fut_cov_list.append(fut_cov)
            fut_pos = fut_pos_list
            fut_cov = fut_cov_list

        # add the prediction for the considered obstacle
        # 将该障碍物的 ground-truth prediction 写入结果字典
        prediction_result[obstacle_id] = {'pos_list': fut_pos, 'cov_list': fut_cov}

    return prediction_result

# EOF
