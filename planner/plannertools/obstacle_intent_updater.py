from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from commonroad.prediction.prediction import Occupancy
from commonroad.scenario.trajectory import State
from shapely.geometry import LineString, Point

# emergency 触发阈值(默认):
# - distance_close: 调大 -> 更早进入近距离 emergency；调小 -> 更晚触发
# - min_ttc_close: 调大 -> TTC 稍微危险就触发；调小 -> 只有更危险才触发
# - distance_mid: 调大 -> 中距离阶段更早介入；调小 -> 更晚介入
# - min_ttc_mid: 调大 -> 中距离更容易触发；调小 -> 更难触发
# - ttc_gap_mid: 调大 -> 允许更大的 TTC 差也触发；调小 -> 只在双方更接近同时到达时触发
# - ego_ttc_critical: 调大 -> ego 更早被视为“马上到冲突区”；调小 -> 判定更保守
# - obstacle_ttc_critical: 调大 -> obstacle 还没特别近也可能触发；调小 -> 只有 obstacle 更接近才触发
DEFAULT_EMERGENCY_PROFILE = {
    "distance_close": 8.0,
    "min_ttc_close": 1.8,
    "distance_mid": 12.0,
    "min_ttc_mid": 2.4,
    "ttc_gap_mid": 1.2,
    "ego_ttc_critical": 0.9,
    "obstacle_ttc_critical": 2.0,
}

OBSTACLE_EMERGENCY_PROFILES = {
    # 20044 更激进、速度更高，所以阈值整体前移:
    # - 距离和 TTC 阈值更大，意味着会更早进入 emergency
    20044: {
        "distance_close": 15.0,
        "min_ttc_close": 2.2,
        "distance_mid": 18.0,
        "min_ttc_mid": 2.8,
        "ttc_gap_mid": 1.5,
        "ego_ttc_critical": 1.3,
        "obstacle_ttc_critical": 2.0,
    },
    # 20087 当前先保持默认风格，但单独列出来，后面可单独手调
    20087: {
        "distance_close": 8.0,
        "min_ttc_close": 1.8,
        "distance_mid": 12.0,
        "min_ttc_mid": 2.4,
        "ttc_gap_mid": 1.2,
        "ego_ttc_critical": 0.9,
        "obstacle_ttc_critical": 2.0,
    },
}

# 已经过掉冲突点的容差:
# - 调大 -> 更容易认为“冲突已过去”，更早解除 emergency
# - 调小 -> 更不容易放过旧冲突点，但也更容易 lingering
CONFLICT_POINT_PASSED_TOLERANCE_M = 1.0
NO_CONFLICT_RECOVERY_STEPS = 2


def _wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _segment_heading(p0: np.ndarray, p1: np.ndarray) -> float:
    delta = p1 - p0
    return float(np.arctan2(delta[1], delta[0]))


def _polyline_arc_lengths(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return np.array([0.0], dtype=float)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))


def _interpolate_on_polyline(
    points: np.ndarray, arc_lengths: np.ndarray, s_query: float
) -> Tuple[np.ndarray, float]:
    if len(points) == 1:
        return points[0], 0.0

    s_clamped = float(np.clip(s_query, 0.0, arc_lengths[-1]))
    idx = int(np.searchsorted(arc_lengths, s_clamped, side="right") - 1)
    idx = max(0, min(idx, len(points) - 2))

    s0 = arc_lengths[idx]
    s1 = arc_lengths[idx + 1]
    if s1 - s0 < 1e-9:
        ratio = 0.0
    else:
        ratio = (s_clamped - s0) / (s1 - s0)

    p0 = points[idx]
    p1 = points[idx + 1]
    position = p0 + ratio * (p1 - p0)
    heading = _segment_heading(p0, p1)
    return position, heading


def _build_fallback_path(initial_state, distance: float = 120.0) -> np.ndarray:
    start = np.asarray(initial_state.position, dtype=float)
    heading = float(getattr(initial_state, "orientation", 0.0))
    end = start + distance * np.array([np.cos(heading), np.sin(heading)])
    return np.vstack([start, end])


def _choose_straight_successor(scenario, lanelet) -> Optional[int]:
    successor_ids = list(lanelet.successor) if lanelet.successor is not None else []
    if len(successor_ids) == 0:
        return None
    if len(lanelet.center_vertices) < 2:
        return successor_ids[0]

    current_heading = _segment_heading(
        lanelet.center_vertices[-2], lanelet.center_vertices[-1]
    )

    best_successor_id = successor_ids[0]
    best_heading_delta = float("inf")
    for successor_id in successor_ids:
        successor = scenario.lanelet_network.find_lanelet_by_id(successor_id)
        if successor is None or len(successor.center_vertices) < 2:
            continue
        successor_heading = _segment_heading(
            successor.center_vertices[0], successor.center_vertices[1]
        )
        heading_delta = abs(_wrap_to_pi(successor_heading - current_heading))
        if heading_delta < best_heading_delta:
            best_heading_delta = heading_delta
            best_successor_id = successor_id
    return best_successor_id


def _build_straight_lanelet_path(
    scenario, initial_state, max_hops: int = 8
) -> np.ndarray:
    lanelet_candidates = scenario.lanelet_network.find_lanelet_by_position(
        [initial_state.position]
    )[0]
    if len(lanelet_candidates) == 0:
        return _build_fallback_path(initial_state)

    lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_candidates[0])
    if lanelet is None or len(lanelet.center_vertices) == 0:
        return _build_fallback_path(initial_state)

    vertices = [np.asarray(lanelet.center_vertices, dtype=float)]
    visited: Set[int] = {lanelet.lanelet_id}
    current_lanelet = lanelet

    for _ in range(max_hops):
        successor_id = _choose_straight_successor(scenario, current_lanelet)
        if successor_id is None or successor_id in visited:
            break
        successor = scenario.lanelet_network.find_lanelet_by_id(successor_id)
        if successor is None or len(successor.center_vertices) == 0:
            break
        successor_vertices = np.asarray(successor.center_vertices, dtype=float)
        if np.linalg.norm(vertices[-1][-1] - successor_vertices[0]) < 1e-6:
            successor_vertices = successor_vertices[1:]
        if len(successor_vertices) == 0:
            break
        vertices.append(successor_vertices)
        visited.add(successor_id)
        current_lanelet = successor

    path = np.vstack(vertices)
    if len(path) < 2:
        return _build_fallback_path(initial_state)
    return path


def _line_from_xy(x_values: Sequence[float], y_values: Sequence[float]) -> Optional[LineString]:
    if x_values is None or y_values is None or len(x_values) < 2 or len(y_values) < 2:
        return None
    coords = [(float(x), float(y)) for x, y in zip(x_values, y_values)]
    if len(coords) < 2:
        return None
    return LineString(coords)


def _shape_safety_radius(shape) -> float:
    if shape is None:
        return 2.5

    radius = getattr(shape, "radius", None)
    if radius is not None:
        return float(radius)

    length = getattr(shape, "length", None)
    width = getattr(shape, "width", None)
    if length is not None and width is not None:
        return 0.5 * float(np.hypot(length, width))

    vertices = getattr(shape, "vertices", None)
    if vertices is not None and len(vertices) > 0:
        vertices = np.asarray(vertices, dtype=float)
        center = np.mean(vertices, axis=0)
        return float(np.max(np.linalg.norm(vertices - center, axis=1)))

    return 2.5


def _candidate_points_from_geometry(geometry) -> List[np.ndarray]:
    if geometry.is_empty:
        return []

    if geometry.geom_type == "Point":
        return [np.array([geometry.x, geometry.y], dtype=float)]

    if geometry.geom_type == "MultiPoint":
        return [np.array([point.x, point.y], dtype=float) for point in geometry.geoms]

    if geometry.geom_type in {"LineString", "LinearRing"}:
        coords = np.asarray(geometry.coords, dtype=float)
        if len(coords) == 0:
            return []
        return [coords[0], coords[len(coords) // 2], coords[-1]]

    if hasattr(geometry, "geoms"):
        candidate_points: List[np.ndarray] = []
        for geom in geometry.geoms:
            candidate_points.extend(_candidate_points_from_geometry(geom))
        return candidate_points

    centroid = geometry.centroid
    return [np.array([centroid.x, centroid.y], dtype=float)]


def _extract_conflict_point(
    obstacle_line: LineString,
    ego_line: LineString,
    obstacle_radius: float = 0.0,
    ego_radius: float = 0.0,
    obstacle_current_s: float = 0.0,
) -> Optional[np.ndarray]:
    clearance = max(0.0, float(obstacle_radius) + float(ego_radius))
    if clearance > 1e-6:
        intersection = obstacle_line.intersection(
            ego_line.buffer(clearance, cap_style=2, join_style=2)
        )
    else:
        intersection = obstacle_line.intersection(ego_line)

    candidate_points = _candidate_points_from_geometry(intersection)
    if len(candidate_points) == 0:
        return None

    progress_tolerance = max(
        CONFLICT_POINT_PASSED_TOLERANCE_M,
        0.25 * max(float(obstacle_radius), 1.0),
    )
    filtered_points = []
    for point in candidate_points:
        obstacle_progress = obstacle_line.project(Point(float(point[0]), float(point[1])))
        ego_progress = ego_line.project(Point(float(point[0]), float(point[1])))
        if obstacle_progress <= obstacle_current_s + progress_tolerance:
            continue
        if ego_progress <= 0.5 * max(float(ego_radius), 1.0):
            continue
        filtered_points.append(point)

    if len(filtered_points) == 0:
        return None

    return min(
        filtered_points,
        key=lambda point: obstacle_line.project(Point(float(point[0]), float(point[1]))),
    )


@dataclass
class _ObstacleIntentState:
    obstacle_id: int
    intent: str
    last_intent_switch_time: int
    path_points: np.ndarray
    arc_lengths: np.ndarray
    path_line: LineString
    current_s: float
    current_speed: float
    cruise_speed: float
    challenge_speed: float
    last_state: State
    emergency_brake: bool = False
    no_conflict_steps: int = 0


class YieldChallengeUpdater:
    def __init__(self, scenario, ego_ids: Iterable[int], dt: float):
        self.ego_ids = set(ego_ids)
        self.dt = float(dt)
        self.obstacle_states: Dict[int, _ObstacleIntentState] = {}
        self.debug_obstacle_ids = {20044, 20087}

        for obstacle in scenario.dynamic_obstacles:
            if obstacle.obstacle_id in self.ego_ids:
                continue

            path_points = _build_straight_lanelet_path(
                scenario=scenario, initial_state=obstacle.initial_state
            )
            arc_lengths = _polyline_arc_lengths(path_points)
            path_line = LineString(path_points)
            current_s = float(
                path_line.project(
                    Point(
                        float(obstacle.initial_state.position[0]),
                        float(obstacle.initial_state.position[1]),
                    )
                )
            )
            initial_speed = float(max(0.0, getattr(obstacle.initial_state, "velocity", 0.0)))
            cruise_speed = max(2.0, initial_speed)
            self.obstacle_states[obstacle.obstacle_id] = _ObstacleIntentState(
                obstacle_id=obstacle.obstacle_id,
                intent="challenge",
                last_intent_switch_time=int(getattr(obstacle.initial_state, "time_step", 0)),
                path_points=path_points,
                arc_lengths=arc_lengths,
                path_line=path_line,
                current_s=current_s,
                current_speed=cruise_speed,
                cruise_speed=cruise_speed,
                challenge_speed=max(cruise_speed + 4.0, cruise_speed * 1.6),
                last_state=obstacle.initial_state,
            )

    def reset_scenario_obstacles(self, scenario):
        for obstacle in scenario.dynamic_obstacles:
            if obstacle.obstacle_id in self.ego_ids:
                continue

            self._clear_lanelet_occupancy(scenario=scenario, obstacle_id=obstacle.obstacle_id)
            prediction = obstacle.prediction
            if prediction is None or not hasattr(prediction, "trajectory"):
                continue

            initial_state = obstacle.initial_state
            prediction.trajectory.initial_time_step = initial_state.time_step
            prediction.trajectory.state_list = [initial_state]
            prediction.occupancy_set = [
                Occupancy(
                    initial_state.time_step,
                    obstacle.obstacle_shape.rotate_translate_local(
                        initial_state.position, initial_state.orientation
                    ),
                )
            ]
            prediction.center_lanelet_assignment = {}
            prediction.shape_lanelet_assignment = {}

            if obstacle.obstacle_id in self.obstacle_states:
                self.obstacle_states[obstacle.obstacle_id].intent = "challenge"
                self.obstacle_states[obstacle.obstacle_id].emergency_brake = False
                self.obstacle_states[obstacle.obstacle_id].last_intent_switch_time = int(initial_state.time_step)
                self.obstacle_states[obstacle.obstacle_id].current_s = float(
                    self.obstacle_states[obstacle.obstacle_id].path_line.project(
                        Point(
                            float(initial_state.position[0]),
                            float(initial_state.position[1]),
                        )
                    )
                )
                self.obstacle_states[obstacle.obstacle_id].current_speed = max(
                    2.0, float(getattr(initial_state, "velocity", 0.0))
                )
                self.obstacle_states[obstacle.obstacle_id].last_state = initial_state
        return scenario

    def step(self, scenario, ego_agents: List, next_time_step: int):
        if len(ego_agents) == 0:
            return scenario

        for obstacle in scenario.dynamic_obstacles:
            obstacle_id = obstacle.obstacle_id
            if obstacle_id in self.ego_ids or obstacle_id not in self.obstacle_states:
                continue

            obstacle_state = self.obstacle_states[obstacle_id]
            obstacle_state.emergency_brake = False
            ego_agent, ego_line, ego_speed = self._select_reference_ego_line(
                ego_agents=ego_agents, obstacle_state=obstacle_state
            )

            desired_speed = obstacle_state.cruise_speed
            conflict_point = None
            if ego_line is not None:
                ego_radius = _shape_safety_radius(
                    getattr(ego_agent, "agent_shape", None)
                )
                obstacle_radius = _shape_safety_radius(obstacle.obstacle_shape)
                conflict_point = _extract_conflict_point(
                    obstacle_line=obstacle_state.path_line,
                    ego_line=ego_line,
                    obstacle_radius=obstacle_radius,
                    ego_radius=ego_radius,
                    obstacle_current_s=obstacle_state.current_s,
                )
            else:
                obstacle_state.emergency_brake = False

            if conflict_point is not None:
                obstacle_state.no_conflict_steps = 0
                obstacle_ttc, distance_to_conflict = self._compute_obstacle_ttc(
                    obstacle_state=obstacle_state, conflict_point=conflict_point
                )
                ego_ttc = self._compute_ego_ttc(
                    ego_line=ego_line,
                    ego_agent_speed=ego_speed,
                    conflict_point=conflict_point,
                )
                self._update_intent_from_interaction(
                    obstacle_state=obstacle_state,
                    distance_to_conflict=distance_to_conflict,
                    obstacle_ttc=obstacle_ttc,
                    ego_ttc=ego_ttc,
                    next_time_step=next_time_step,
                )

                if obstacle_state.emergency_brake:
                    desired_speed = self._emergency_yield_speed(
                        obstacle_state=obstacle_state,
                        distance_to_conflict=distance_to_conflict,
                        obstacle_ttc=obstacle_ttc,
                        ego_ttc=ego_ttc,
                    )
                elif obstacle_state.intent == "yield":
                    desired_speed = self._yield_speed(
                        obstacle_state=obstacle_state,
                        distance_to_conflict=distance_to_conflict,
                        obstacle_ttc=obstacle_ttc,
                        ego_ttc=ego_ttc,
                    )
                else:
                    desired_speed = self._challenge_speed(
                        obstacle_state=obstacle_state,
                        distance_to_conflict=distance_to_conflict,
                        obstacle_ttc=obstacle_ttc,
                        ego_ttc=ego_ttc,
                    )

                # if obstacle_id in self.debug_obstacle_ids:
                #     print(
                #         "[ObstacleDebug] "
                #         f"t={next_time_step} id={obstacle_id} "
                #         f"intent={obstacle_state.intent} "
                #         f"emergency={obstacle_state.emergency_brake} "
                #         f"curr_v={obstacle_state.current_speed:.2f} "
                #         f"des_v={desired_speed:.2f} "
                #         f"dist_conf={distance_to_conflict:.2f} "
                #         f"obs_ttc={obstacle_ttc:.2f} "
                #         f"ego_ttc={ego_ttc:.2f} "
                #         f"conflict=({float(conflict_point[0]):.2f},{float(conflict_point[1]):.2f})"
                #     )
            elif obstacle_id in self.debug_obstacle_ids:
                obstacle_state.no_conflict_steps += 1
                if (
                    obstacle_state.intent == "yield"
                    and obstacle_state.no_conflict_steps >= NO_CONFLICT_RECOVERY_STEPS
                ):
                    obstacle_state.intent = "challenge"
                    obstacle_state.last_intent_switch_time = next_time_step
                # print(
                #     "[ObstacleDebug_no_conflict_point] "
                #     f"t={next_time_step} id={obstacle_id} "
                #     f"intent={obstacle_state.intent} "
                #     f"emergency={obstacle_state.emergency_brake} "
                #     f"curr_v={obstacle_state.current_speed:.2f} "
                #     f"des_v={desired_speed:.2f} "
                #     "conflict=None"
                # )
            else:
                obstacle_state.no_conflict_steps += 1
                if (
                    obstacle_state.intent == "yield"
                    and obstacle_state.no_conflict_steps >= NO_CONFLICT_RECOVERY_STEPS
                ):
                    obstacle_state.intent = "challenge"
                    obstacle_state.last_intent_switch_time = next_time_step

            updated_state = self._propagate_state(
                obstacle_state=obstacle_state,
                desired_speed=desired_speed,
                next_time_step=next_time_step,
            )
            self._write_state_to_scenario(
                scenario=scenario,
                obstacle=obstacle,
                state=updated_state,
            )
            obstacle_state.last_state = updated_state

        return scenario

    def _update_intent_from_interaction(
        self,
        obstacle_state: _ObstacleIntentState,
        distance_to_conflict: float,
        obstacle_ttc: float,
        ego_ttc: float,
        next_time_step: int,
    ):
        if self._should_emergency_yield(
            obstacle_state=obstacle_state,
            distance_to_conflict=distance_to_conflict,
            obstacle_ttc=obstacle_ttc,
            ego_ttc=ego_ttc,
        ):
            obstacle_state.emergency_brake = True
            return

        obstacle_state.emergency_brake = False
        hold_steps = 6
        if next_time_step - obstacle_state.last_intent_switch_time < hold_steps:
            return

        if not np.isfinite(ego_ttc):
            if obstacle_state.intent != "challenge":
                obstacle_state.intent = "challenge"
                obstacle_state.last_intent_switch_time = next_time_step
            return

        if distance_to_conflict > 65.0:
            if obstacle_state.intent != "challenge":
                obstacle_state.intent = "challenge"
                obstacle_state.last_intent_switch_time = next_time_step
            return

        ego_dominant = ego_ttc + 2.8 < obstacle_ttc
        obstacle_dominant = obstacle_ttc + 0.4 < ego_ttc

        if obstacle_state.intent == "challenge":
            if ego_dominant and (
                distance_to_conflict < 24.0 or ego_ttc < 3.2
            ):
                obstacle_state.intent = "yield"
                obstacle_state.last_intent_switch_time = next_time_step
        else:
            if obstacle_dominant or (
                distance_to_conflict > 14.0 and ego_ttc > obstacle_ttc + 0.4
            ):
                obstacle_state.intent = "challenge"
                obstacle_state.last_intent_switch_time = next_time_step

    def _should_emergency_yield(
        self,
        obstacle_state: _ObstacleIntentState,
        distance_to_conflict: float,
        obstacle_ttc: float,
        ego_ttc: float,
    ) -> bool:
        if not np.isfinite(ego_ttc):
            return False

        profile = dict(DEFAULT_EMERGENCY_PROFILE)
        profile.update(OBSTACLE_EMERGENCY_PROFILES.get(obstacle_state.obstacle_id, {}))
        distance_close = profile["distance_close"]
        min_ttc_close = profile["min_ttc_close"]
        distance_mid = profile["distance_mid"]
        min_ttc_mid = profile["min_ttc_mid"]
        ttc_gap_mid = profile["ttc_gap_mid"]
        ego_ttc_critical = profile["ego_ttc_critical"]
        obstacle_ttc_critical = profile["obstacle_ttc_critical"]

        min_ttc = min(obstacle_ttc, ego_ttc)
        ttc_gap = abs(obstacle_ttc - ego_ttc)

        if distance_to_conflict < distance_close and min_ttc < min_ttc_close:
            return True
        if distance_to_conflict < distance_mid and min_ttc < min_ttc_mid and ttc_gap < ttc_gap_mid:
            return True
        if ego_ttc < ego_ttc_critical and obstacle_ttc < obstacle_ttc_critical:
            return True
        return False

    def _select_reference_ego_line(self, ego_agents: List, obstacle_state: _ObstacleIntentState):
        best_agent = None
        best_distance = float("inf")
        obstacle_position = np.asarray(obstacle_state.last_state.position, dtype=float)

        for ego_agent in ego_agents:
            ego_position = np.asarray(ego_agent.state.position, dtype=float)
            distance = float(np.linalg.norm(obstacle_position - ego_position))
            if distance < best_distance:
                best_distance = distance
                best_agent = ego_agent

        if best_agent is None:
            return None, None, 0.0

        if getattr(best_agent.planner, "trajectory", None) is not None:
            ego_line = _line_from_xy(
                best_agent.planner.trajectory.get("x_m"),
                best_agent.planner.trajectory.get("y_m"),
            )
            if ego_line is not None:
                return best_agent, ego_line, float(max(0.1, best_agent.state.velocity))

        heading = float(getattr(best_agent.state, "orientation", 0.0))
        start = np.asarray(best_agent.state.position, dtype=float)
        end = start + 80.0 * np.array([np.cos(heading), np.sin(heading)])
        return best_agent, LineString([start, end]), float(max(0.1, best_agent.state.velocity))

    def _compute_obstacle_ttc(self, obstacle_state: _ObstacleIntentState, conflict_point: np.ndarray):
        conflict_s = float(
            obstacle_state.path_line.project(
                Point(float(conflict_point[0]), float(conflict_point[1]))
            )
        )
        distance_to_conflict = conflict_s - obstacle_state.current_s
        if distance_to_conflict <= CONFLICT_POINT_PASSED_TOLERANCE_M:
            return float("inf"), float("inf")
        obstacle_ttc = distance_to_conflict / max(0.1, obstacle_state.current_speed)
        return obstacle_ttc, distance_to_conflict

    def _compute_ego_ttc(self, ego_line: LineString, ego_agent_speed: float, conflict_point: np.ndarray):
        ego_coords = np.asarray(ego_line.coords, dtype=float)
        if len(ego_coords) < 2:
            return float("inf")

        ego_start = ego_coords[0]
        ego_heading_vec = ego_coords[min(1, len(ego_coords) - 1)] - ego_start
        if np.linalg.norm(ego_heading_vec) < 1e-9:
            ego_heading_vec = ego_coords[-1] - ego_start

        point_vec = np.asarray(conflict_point, dtype=float) - ego_start
        if np.dot(point_vec, ego_heading_vec) <= 0.5:
            return float("inf")

        ego_distance = float(
            max(
                0.0,
                ego_line.project(Point(float(conflict_point[0]), float(conflict_point[1]))),
            )
        )
        return ego_distance / max(0.1, ego_agent_speed)

    def _yield_speed(
        self,
        obstacle_state: _ObstacleIntentState,
        distance_to_conflict: float,
        obstacle_ttc: float,
        ego_ttc: float,
    ) -> float:
        if not np.isfinite(ego_ttc):
            return obstacle_state.cruise_speed
        if distance_to_conflict > 55.0:
            return obstacle_state.cruise_speed
        if ego_ttc <= obstacle_ttc + 2.8:
            stop_distance = max(0.0, distance_to_conflict - 10.0)
            target_speed = float(np.sqrt(max(0.0, 2.0 * 3.2 * stop_distance)))
            if distance_to_conflict < 18.0:
                target_speed = min(target_speed, 1.2)
            return min(
                obstacle_state.cruise_speed,
                target_speed,
            )
        return obstacle_state.cruise_speed

    def _emergency_yield_speed(
        self,
        obstacle_state: _ObstacleIntentState,
        distance_to_conflict: float,
        obstacle_ttc: float,
        ego_ttc: float,
    ) -> float:
        min_speed = 1.0
        if distance_to_conflict < 5.0 or min(obstacle_ttc, ego_ttc) < 0.8:
            return min_speed

        stop_distance = max(0.0, distance_to_conflict - 4.0)
        target_speed = float(np.sqrt(max(0.0, 2.0 * 3.0 * stop_distance)))
        return max(min_speed, min(obstacle_state.current_speed, target_speed, 3.0))

    def _challenge_speed(
        self,
        obstacle_state: _ObstacleIntentState,
        distance_to_conflict: float,
        obstacle_ttc: float,
        ego_ttc: float,
    ) -> float:
        if distance_to_conflict > 55.0:
            return obstacle_state.challenge_speed
        if obstacle_ttc >= ego_ttc - 1.2:
            return obstacle_state.challenge_speed
        if distance_to_conflict < 18.0:
            return max(obstacle_state.challenge_speed, obstacle_state.current_speed + 2.5)
        return obstacle_state.cruise_speed

    def _propagate_state(
        self,
        obstacle_state: _ObstacleIntentState,
        desired_speed: float,
        next_time_step: int,
    ) -> State:
        max_acc = 3.5 if obstacle_state.intent == "challenge" else 1.5
        max_dec = 4.0 if obstacle_state.emergency_brake else 3.5
        speed_error = desired_speed - obstacle_state.current_speed
        acceleration = float(np.clip(speed_error / self.dt, -max_dec, max_acc))
        min_speed = 1.0 if obstacle_state.emergency_brake else 0.0
        next_speed = float(max(min_speed, obstacle_state.current_speed + acceleration * self.dt))
        travelled_distance = max(
            0.0,
            obstacle_state.current_speed * self.dt + 0.5 * acceleration * self.dt * self.dt,
        )
        obstacle_state.current_s = float(
            min(obstacle_state.arc_lengths[-1], obstacle_state.current_s + travelled_distance)
        )
        position, heading = _interpolate_on_polyline(
            points=obstacle_state.path_points,
            arc_lengths=obstacle_state.arc_lengths,
            s_query=obstacle_state.current_s,
        )
        obstacle_state.current_speed = next_speed
        return State(
            position=np.asarray(position, dtype=float),
            orientation=heading,
            velocity=next_speed,
            acceleration=acceleration,
            time_step=next_time_step,
        )

    def _write_state_to_scenario(self, scenario, obstacle, state: State):
        prediction = obstacle.prediction
        occupied_region = obstacle.obstacle_shape.rotate_translate_local(
            state.position, state.orientation
        )
        occupancy = Occupancy(state.time_step, occupied_region)
        trajectory = prediction.trajectory

        if len(trajectory.state_list) == 0:
            trajectory.initial_time_step = state.time_step
            trajectory.state_list = [state]
            prediction.occupancy_set = [occupancy]
        else:
            last_state_time = trajectory.state_list[-1].time_step
            if state.time_step == last_state_time:
                trajectory.state_list[-1] = state
                if len(prediction.occupancy_set) > 0:
                    prediction.occupancy_set[-1] = occupancy
                else:
                    prediction.occupancy_set = [occupancy]
            elif state.time_step > last_state_time:
                trajectory.state_list.append(state)
                prediction.occupancy_set.append(occupancy)

        lanelet_candidates = scenario.lanelet_network.find_lanelet_by_position([state.position])[0]
        if len(lanelet_candidates) > 0:
            lanelet_id = lanelet_candidates[0]
            prediction.center_lanelet_assignment[state.time_step] = set(lanelet_candidates)
            prediction.shape_lanelet_assignment[state.time_step] = set(lanelet_candidates)
            lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
            if lanelet.dynamic_obstacles_on_lanelet.get(state.time_step) is None:
                lanelet.dynamic_obstacles_on_lanelet[state.time_step] = set()
            lanelet.dynamic_obstacles_on_lanelet[state.time_step].add(obstacle.obstacle_id)

    def _clear_lanelet_occupancy(self, scenario, obstacle_id: int):
        for lanelet in scenario.lanelet_network.lanelets:
            for time_step, obstacle_ids in list(lanelet.dynamic_obstacles_on_lanelet.items()):
                if time_step <= 0:
                    continue
                obstacle_ids.discard(obstacle_id)
