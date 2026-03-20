from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from commonroad.prediction.prediction import Occupancy
from commonroad.scenario.trajectory import State
from shapely.geometry import LineString, Point


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


def _extract_conflict_point(
    obstacle_line: LineString, ego_line: LineString
) -> Optional[np.ndarray]:
    intersection = obstacle_line.intersection(ego_line)
    if intersection.is_empty:
        return None

    if intersection.geom_type == "Point":
        return np.array([intersection.x, intersection.y], dtype=float)

    if intersection.geom_type == "MultiPoint":
        point = list(intersection.geoms)[0]
        return np.array([point.x, point.y], dtype=float)

    if intersection.geom_type in {"LineString", "LinearRing"}:
        coords = np.asarray(intersection.coords, dtype=float)
        return coords[len(coords) // 2]

    if hasattr(intersection, "geoms") and len(intersection.geoms) > 0:
        for geom in intersection.geoms:
            if geom.geom_type == "Point":
                return np.array([geom.x, geom.y], dtype=float)
            if geom.geom_type in {"LineString", "LinearRing"}:
                coords = np.asarray(geom.coords, dtype=float)
                return coords[len(coords) // 2]
    return None


@dataclass
class _ObstacleIntentState:
    obstacle_id: int
    intent: str
    path_points: np.ndarray
    arc_lengths: np.ndarray
    path_line: LineString
    current_s: float
    current_speed: float
    cruise_speed: float
    challenge_speed: float
    last_state: State


class YieldChallengeUpdater:
    def __init__(self, scenario, ego_ids: Iterable[int], dt: float):
        self.ego_ids = set(ego_ids)
        self.dt = float(dt)
        self.obstacle_states: Dict[int, _ObstacleIntentState] = {}

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
                intent="yield" if obstacle.obstacle_id % 2 == 0 else "challenge",
                path_points=path_points,
                arc_lengths=arc_lengths,
                path_line=path_line,
                current_s=current_s,
                current_speed=cruise_speed,
                cruise_speed=cruise_speed,
                challenge_speed=max(cruise_speed + 2.0, cruise_speed * 1.25),
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
            ego_line, ego_speed = self._select_reference_ego_line(
                ego_agents=ego_agents, obstacle_state=obstacle_state
            )

            desired_speed = obstacle_state.cruise_speed
            conflict_point = None
            if ego_line is not None:
                conflict_point = _extract_conflict_point(
                    obstacle_line=obstacle_state.path_line, ego_line=ego_line
                )

            if conflict_point is not None:
                obstacle_ttc, distance_to_conflict = self._compute_obstacle_ttc(
                    obstacle_state=obstacle_state, conflict_point=conflict_point
                )
                ego_ttc = self._compute_ego_ttc(
                    ego_line=ego_line,
                    ego_agent_speed=ego_speed,
                    conflict_point=conflict_point,
                )

                if obstacle_state.intent == "yield":
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
            return None, 0.0

        if getattr(best_agent.planner, "trajectory", None) is not None:
            ego_line = _line_from_xy(
                best_agent.planner.trajectory.get("x_m"),
                best_agent.planner.trajectory.get("y_m"),
            )
            if ego_line is not None:
                return ego_line, float(max(0.1, best_agent.state.velocity))

        heading = float(getattr(best_agent.state, "orientation", 0.0))
        start = np.asarray(best_agent.state.position, dtype=float)
        end = start + 80.0 * np.array([np.cos(heading), np.sin(heading)])
        return LineString([start, end]), float(max(0.1, best_agent.state.velocity))

    def _compute_obstacle_ttc(self, obstacle_state: _ObstacleIntentState, conflict_point: np.ndarray):
        conflict_s = float(
            obstacle_state.path_line.project(
                Point(float(conflict_point[0]), float(conflict_point[1]))
            )
        )
        distance_to_conflict = max(0.0, conflict_s - obstacle_state.current_s)
        obstacle_ttc = distance_to_conflict / max(0.1, obstacle_state.current_speed)
        return obstacle_ttc, distance_to_conflict

    def _compute_ego_ttc(self, ego_line: LineString, ego_agent_speed: float, conflict_point: np.ndarray):
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
        if distance_to_conflict > 35.0:
            return obstacle_state.cruise_speed
        if ego_ttc <= obstacle_ttc + 1.5:
            stop_distance = max(0.0, distance_to_conflict - 4.0)
            return min(
                obstacle_state.cruise_speed,
                float(np.sqrt(max(0.0, 2.0 * 2.5 * stop_distance))),
            )
        return obstacle_state.cruise_speed

    def _challenge_speed(
        self,
        obstacle_state: _ObstacleIntentState,
        distance_to_conflict: float,
        obstacle_ttc: float,
        ego_ttc: float,
    ) -> float:
        if distance_to_conflict > 40.0:
            return obstacle_state.cruise_speed
        if obstacle_ttc >= ego_ttc - 0.5:
            return obstacle_state.challenge_speed
        return obstacle_state.cruise_speed

    def _propagate_state(
        self,
        obstacle_state: _ObstacleIntentState,
        desired_speed: float,
        next_time_step: int,
    ) -> State:
        max_acc = 1.5
        max_dec = 3.5
        speed_error = desired_speed - obstacle_state.current_speed
        acceleration = float(np.clip(speed_error / self.dt, -max_dec, max_acc))
        next_speed = float(max(0.0, obstacle_state.current_speed + acceleration * self.dt))
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
