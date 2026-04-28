import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_VENV_PYTHON = REPO_ROOT / ".venv38/bin/python"

if (
    LOCAL_VENV_PYTHON.exists()
    and Path(sys.executable).resolve() != LOCAL_VENV_PYTHON.resolve()
):
    os.execv(
        str(LOCAL_VENV_PYTHON),
        [str(LOCAL_VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
    )

if str(REPO_ROOT.parent) not in sys.path:
    sys.path.append(str(REPO_ROOT.parent))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

for site_packages in REPO_ROOT.glob(".venv*/lib/python*/site-packages"):
    if str(site_packages) not in sys.path:
        sys.path.append(str(site_packages))

from commonroad_helper_functions.exceptions import (  # noqa: E402
    ExecutionTimeoutError,
    GoalReachedNotification,
)

from beliefplanning.planner.Frenet.configs.load_json import (  # noqa: E402
    load_contingency_json,
    load_planning_json,
    load_risk_json,
)
from beliefplanning.planner.Frenet.plannertools.frenetcreator import (  # noqa: E402
    FrenetCreator,
)
from beliefplanning.planner.Frenet.utils.visualization import (  # noqa: E402
    PLOT_SNAPSHOTS,
    clear_plot_snapshots,
)
from beliefplanning.planner.plannertools.evaluate import ScenarioEvaluator  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_BASE_SCENARIO = "recorded/hand-crafted/BRA_VilaVelha-92_1_T-10.xml"
DEFAULT_SAMPLE_DIR = "recorded/hand-crafted/vv_samples"
DEFAULT_OUTPUT_DIR = "planner/Frenet/results/vv_batch_eval"


class LightweightScenarioEvaluator(ScenarioEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intent_history = {}
        self.clearance_history = []
        self.global_min_clearance = float("inf")

    def eval_scenario(self, scenario_path):
        self.exec_timer.reset()
        self.scenario_path = self.path_to_scenarios.joinpath(scenario_path)
        self.intent_history = {}
        self.clearance_history = []
        self.global_min_clearance = float("inf")
        self.motion_history = {}
        self.vel_list = []
        start_time = time.time()

        with self.exec_timer.time_with_cm("total"):
            try:
                self._initialize()
                self._simulate()
                return_dict = {
                    "success": False,
                    "reason_for_failure": "Simulation ended without terminal event.",
                }
            except GoalReachedNotification as excp:
                reached_in_time = "exceeded" not in str(excp).lower()
                return_dict = {
                    "success": True,
                    "reason_for_failure": None,
                    "reached_in_time": reached_in_time,
                }
            except ExecutionTimeoutError as excp:
                return_dict = {"success": False, "reason_for_failure": str(excp)}
            except NotImplementedError as excp:
                raise excp
            except Exception as excp:
                import traceback

                traceback.print_exc()
                print(f"{scenario_path} >>> {str(excp)}")
                return_dict = {"success": False, "reason_for_failure": str(excp)}
                if "Simulation" in str(excp):
                    print(
                        "Stopping Evaluation, results not valid anymore due to simulation "
                        f"time out in {scenario_path}"
                    )
                    sys.exit()

        return_dict["scenario_path"] = scenario_path
        return_dict["exec_time"] = time.time() - start_time
        return_dict["harm"] = self.harm
        return_dict["velocities"] = (
            0.0 if not self.vel_list else float(np.mean(self.vel_list))
        )
        return_dict["timesteps_agent"] = len(self.vel_list)
        return_dict["intent_history"] = self.intent_history
        return_dict["clearance_history"] = self.clearance_history
        return_dict["global_min_clearance"] = (
            None
            if not np.isfinite(self.global_min_clearance)
            else float(self.global_min_clearance)
        )
        if self.timing_enabled:
            return_dict["exec_times_dict"] = self.exec_timer.get_timing_dict()
        return return_dict

    def _record_motion_snapshot(self, time_step: int):
        super()._record_motion_snapshot(time_step=time_step)

        if self.obstacle_updater is not None and hasattr(self.obstacle_updater, "obstacle_states"):
            self.intent_history[int(time_step)] = {
                int(obstacle_id): str(obstacle_state.intent)
                for obstacle_id, obstacle_state in self.obstacle_updater.obstacle_states.items()
            }

        min_clearance = self._compute_min_clearance_at_timestep(time_step=time_step)
        self.clearance_history.append(
            {
                "timestep": int(time_step),
                "min_clearance": (
                    None if not np.isfinite(min_clearance) else float(min_clearance)
                ),
            }
        )
        if np.isfinite(min_clearance):
            self.global_min_clearance = min(self.global_min_clearance, float(min_clearance))

    def _compute_min_clearance_at_timestep(self, time_step: int) -> float:
        if self.scenario is None or self.agent_list is None:
            return float("inf")

        ego_ids = {agent.agent_id for agent in self.agent_list}
        ego_infos = []
        for ego_id in ego_ids:
            obstacle = self.scenario.obstacle_by_id(ego_id)
            state = obstacle.state_at_time(time_step)
            if state is None and int(time_step) == int(getattr(obstacle.initial_state, "time_step", 0)):
                state = obstacle.initial_state
            if state is None:
                continue
            ego_infos.append(np.asarray(state.position, dtype=float))

        min_clearance = float("inf")
        for ego_position in ego_infos:
            for obstacle in self.scenario.obstacles:
                if obstacle.obstacle_id in ego_ids:
                    continue
                state = obstacle.state_at_time(time_step)
                if state is None and int(time_step) == int(getattr(obstacle.initial_state, "time_step", 0)):
                    state = obstacle.initial_state
                if state is None:
                    continue
                obstacle_position = np.asarray(state.position, dtype=float)
                center_distance = float(np.linalg.norm(ego_position - obstacle_position))
                min_clearance = min(min_clearance, center_distance)
        return min_clearance


def _canonicalize_joint_label(label: str) -> str:
    parts = [part.strip() for part in str(label).split(",") if part.strip()]
    return ", ".join(sorted(parts))


def _extract_obstacle_ids_from_labels(labels: List[str]) -> List[int]:
    obstacle_ids = set()
    for label in labels:
        for part in str(label).split(","):
            part = part.strip()
            if "=" not in part:
                continue
            obstacle_id_str = part.split("=", 1)[0].strip()
            try:
                obstacle_ids.add(int(obstacle_id_str))
            except ValueError:
                continue
    return sorted(obstacle_ids)


def _actual_joint_label(intent_map: Dict[int, str], obstacle_ids: List[int]) -> str:
    parts = []
    for obstacle_id in obstacle_ids:
        if obstacle_id not in intent_map:
            continue
        parts.append(f"{int(obstacle_id)}={intent_map[obstacle_id]}")
    return _canonicalize_joint_label(", ".join(parts))


def _save_snapshots_to_gif(output_path: Path, fps: int) -> bool:
    if len(PLOT_SNAPSHOTS) == 0:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [Image.fromarray(frame) for frame in PLOT_SNAPSHOTS]
    duration_ms = max(int(round(1000.0 / max(int(fps), 1))), 1)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return True


def _build_scenario_list(path_to_scenarios: Path, base_scenario: str, sample_dir: str) -> List[str]:
    scenarios = [base_scenario]
    sample_root = path_to_scenarios.joinpath(sample_dir)
    sample_scenarios = sorted(
        str(path.relative_to(path_to_scenarios))
        for path in sample_root.glob("*.xml")
    )
    scenarios.extend(sample_scenarios)
    return scenarios


def _resolve_task_time_s(evaluator: LightweightScenarioEvaluator) -> float:
    dt = float(getattr(evaluator.scenario, "dt", 0.1)) if evaluator.scenario is not None else 0.1
    if evaluator.agent_list:
        max_timestep = max(int(getattr(agent.state, "time_step", 0)) for agent in evaluator.agent_list)
        return float(max_timestep * dt)

    ego_timesteps = []
    for history in evaluator.motion_history.values():
        if history.get("is_ego", False) and history.get("timesteps"):
            ego_timesteps.append(max(int(step) for step in history["timesteps"]))
    if ego_timesteps:
        return float(max(ego_timesteps) * dt)
    return 0.0


def _resolve_ego_avg_speed(evaluator: LightweightScenarioEvaluator) -> float:
    ego_speeds = []
    for history in evaluator.motion_history.values():
        if history.get("is_ego", False) and len(history.get("velocity", [])) > 0:
            ego_speeds.extend(float(v) for v in history["velocity"])
    if len(ego_speeds) == 0:
        return 0.0
    return float(np.mean(ego_speeds))


def _extract_planning_cycle_times(return_dict: Dict) -> List[float]:
    exec_times_dict = return_dict.get("exec_times_dict", {})
    raw_times = exec_times_dict.get("simulation/total", [])
    return [float(value) for value in raw_times]


def _scenario_metrics(
    evaluator: LightweightScenarioEvaluator,
    return_dict: Dict,
    gif_path,
) -> Dict:
    planner = None
    if evaluator.agent_list:
        planner = getattr(evaluator.agent_list[0], "planner", None)

    planning_cycle_times = _extract_planning_cycle_times(return_dict)
    credible_sizes = []
    credible_timesteps = []
    credible_labels = []
    recoverability_indicator = []
    recoverability_activation_indicator = []
    shared_plan_count = []
    recoverable_shared_plan_count = []
    selected_plan_recoverable_indicator = []
    recoverability_enforced = []
    if planner is not None:
        credible_sizes = [
            int(value)
            for value in planner.credible_joint_history.get("credible_set_sizes", [])
        ]
        credible_timesteps = [
            int(value)
            for value in planner.credible_joint_history.get("timesteps", [])
        ]
        credible_labels = [
            list(labels)
            for labels in planner.credible_joint_history.get("credible_labels", [])
        ]
        recoverability_indicator = [
            int(value)
            for value in planner.recoverability_history.get("recoverability_indicator", [])
        ]
        recoverability_activation_indicator = [
            int(value)
            for value in planner.recoverability_history.get(
                "recoverability_activation_indicator", []
            )
        ]
        shared_plan_count = [
            int(value)
            for value in planner.recoverability_history.get("shared_plan_count", [])
        ]
        recoverable_shared_plan_count = [
            int(value)
            for value in planner.recoverability_history.get("recoverable_shared_plan_count", [])
        ]
        selected_plan_recoverable_indicator = [
            int(value)
            for value in planner.recoverability_history.get(
                "selected_plan_recoverable_indicator", []
            )
        ]
        recoverability_enforced = [
            int(value)
            for value in planner.recoverability_history.get("recoverability_enforced", [])
        ]

    unrecoverable_ratio_series = []
    for shared_count, recoverable_count in zip(
        shared_plan_count, recoverable_shared_plan_count
    ):
        if shared_count <= 0:
            unrecoverable_ratio_series.append(1.0)
        else:
            unrecoverable_ratio_series.append(
                max(
                    0.0,
                    1.0 - float(recoverable_count) / float(shared_count),
                )
            )

    coverage_hits = []
    for timestep, label_list in zip(credible_timesteps, credible_labels):
        actual_intents = evaluator.intent_history.get(int(timestep), {})
        obstacle_ids = _extract_obstacle_ids_from_labels(label_list)
        actual_label = _actual_joint_label(actual_intents, obstacle_ids)
        canonical_credible = {_canonicalize_joint_label(label) for label in label_list}
        if actual_label:
            coverage_hits.append(1 if actual_label in canonical_credible else 0)

    reason = return_dict.get("reason_for_failure")
    collision = (not bool(return_dict.get("success", False))) and (
        isinstance(reason, str) and "collision" in reason.lower()
    )

    scenario_metric = {
        "scenario": str(return_dict["scenario_path"]),
        "scenario_name": Path(str(return_dict["scenario_path"])).stem,
        "success": bool(return_dict.get("success", False)),
        "collision": bool(collision),
        "reason_for_failure": reason,
        "gif_path": str(gif_path) if gif_path else "",
        "task_time_s": _resolve_task_time_s(evaluator),
        "avg_speed_mps": _resolve_ego_avg_speed(evaluator),
        "min_clearance_m": (
            float(return_dict["global_min_clearance"])
            if return_dict.get("global_min_clearance") is not None
            else None
        ),
        "t_c_s": (
            float(np.mean(planning_cycle_times)) if planning_cycle_times else None
        ),
        "t95_s": (
            float(np.percentile(planning_cycle_times, 95)) if planning_cycle_times else None
        ),
        "Omega_bar": float(np.mean(credible_sizes)) if credible_sizes else None,
        "C_Omega": float(np.mean(coverage_hits)) if coverage_hits else None,
        "URR": (
            float(np.mean(unrecoverable_ratio_series))
            if unrecoverable_ratio_series
            else None
        ),
        "recoverability_activation_ratio": (
            float(np.mean(recoverability_activation_indicator))
            if recoverability_activation_indicator
            else None
        ),
        "selected_plan_unrecoverable_ratio": (
            float(1.0 - np.mean(selected_plan_recoverable_indicator))
            if selected_plan_recoverable_indicator
            else None
        ),
        "credible_set_sizes": credible_sizes,
        "recoverability_indicator": recoverability_indicator,
        "recoverability_activation_indicator": recoverability_activation_indicator,
        "shared_plan_count": shared_plan_count,
        "recoverable_shared_plan_count": recoverable_shared_plan_count,
        "selected_plan_recoverable_indicator": selected_plan_recoverable_indicator,
        "recoverability_enforced": recoverability_enforced,
        "unrecoverable_ratio_series": unrecoverable_ratio_series,
        "planning_cycle_times": planning_cycle_times,
    }
    return scenario_metric


def _aggregate_summary(per_scenario_metrics: List[Dict]) -> Dict:
    scenario_count = len(per_scenario_metrics)
    success_values = [1.0 if item["success"] else 0.0 for item in per_scenario_metrics]
    collision_values = [1.0 if item["collision"] else 0.0 for item in per_scenario_metrics]

    successful_task_times = [
        float(item["task_time_s"])
        for item in per_scenario_metrics
        if item["success"]
    ]
    avg_speeds = [
        float(item["avg_speed_mps"])
        for item in per_scenario_metrics
        if item["avg_speed_mps"] is not None
    ]
    min_clearances = [
        float(item["min_clearance_m"])
        for item in per_scenario_metrics
        if item["min_clearance_m"] is not None
    ]
    all_cycle_times = [
        float(value)
        for item in per_scenario_metrics
        for value in item.get("planning_cycle_times", [])
    ]
    all_credible_sizes = [
        float(value)
        for item in per_scenario_metrics
        for value in item.get("credible_set_sizes", [])
    ]
    all_unrecoverable_ratio = [
        float(value)
        for item in per_scenario_metrics
        for value in item.get("unrecoverable_ratio_series", [])
    ]
    coverage_values = [
        float(item["C_Omega"])
        for item in per_scenario_metrics
        if item["C_Omega"] is not None
    ]
    activation_values = [
        float(item["recoverability_activation_ratio"])
        for item in per_scenario_metrics
        if item["recoverability_activation_ratio"] is not None
    ]
    selected_unrecoverable_values = [
        float(item["selected_plan_unrecoverable_ratio"])
        for item in per_scenario_metrics
        if item["selected_plan_unrecoverable_ratio"] is not None
    ]

    return {
        "scenario_count": scenario_count,
        "SR": float(np.mean(success_values)) if success_values else None,
        "CR": float(np.mean(collision_values)) if collision_values else None,
        "T_task": float(np.mean(successful_task_times)) if successful_task_times else None,
        "v_bar": float(np.mean(avg_speeds)) if avg_speeds else None,
        "d_min": float(np.mean(min_clearances)) if min_clearances else None,
        "d_min_global": float(np.min(min_clearances)) if min_clearances else None,
        "t_c": float(np.mean(all_cycle_times)) if all_cycle_times else None,
        "t95": float(np.percentile(all_cycle_times, 95)) if all_cycle_times else None,
        "Omega_bar": float(np.mean(all_credible_sizes)) if all_credible_sizes else None,
        "C_Omega": float(np.mean(coverage_values)) if coverage_values else None,
        "URR": (
            float(np.mean(all_unrecoverable_ratio))
            if all_unrecoverable_ratio
            else None
        ),
        "recoverability_activation_ratio": (
            float(np.mean(activation_values)) if activation_values else None
        ),
        "selected_plan_unrecoverable_ratio": (
            float(np.mean(selected_unrecoverable_values))
            if selected_unrecoverable_values
            else None
        ),
        "aggregation_notes": {
            "T_task": "Mean task completion time over successful scenarios only.",
            "d_min": "Mean of per-scenario minimum clearances.",
            "d_min_global": "Global minimum clearance over all evaluated scenarios.",
            "t_c": "Mean planning cycle time over all planning cycles.",
            "t95": "95th percentile of planning cycle time over all planning cycles.",
            "Omega_bar": "Mean credible joint scenario set size over all planning cycles.",
            "C_Omega": "Mean per-scenario true joint intent coverage by credible joint sets.",
            "URR": "Mean unrecoverable shared-plan ratio over all planning cycles: 1 - recoverable_shared_plan_count / shared_plan_count.",
            "recoverability_activation_ratio": "Mean fraction of planning cycles where recoverability filtered at least one shared plan.",
            "selected_plan_unrecoverable_ratio": "Mean fraction of planning cycles where the finally selected best plan is actually unrecoverable.",
        },
    }


def _cleanup_unwanted_outputs(scenario_name: str):
    for candidate in [
        REPO_ROOT / "planner/Frenet/results/logs" / f"{scenario_name}.csv",
        REPO_ROOT / "planner/Frenet/results/eval" / f"exec_timing_{scenario_name}.json",
        REPO_ROOT / "planner/Frenet/results/eval" / f"plot_replay_{scenario_name}.gif",
    ]:
        if candidate.exists():
            candidate.unlink()

    for directory in [
        REPO_ROOT / "planner/Frenet/results/eval/belief_plots",
        REPO_ROOT / "planner/Frenet/results/eval/motion_plots",
    ]:
        if not directory.exists():
            continue
        for path in directory.glob(f"{scenario_name}*"):
            if path.is_file():
                path.unlink()


def _write_metrics_csv(output_path: Path, per_scenario_metrics: List[Dict]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in per_scenario_metrics:
        row = dict(item)
        row.pop("credible_set_sizes", None)
        row.pop("recoverability_indicator", None)
        row.pop("recoverability_activation_indicator", None)
        row.pop("shared_plan_count", None)
        row.pop("recoverable_shared_plan_count", None)
        row.pop("selected_plan_recoverable_indicator", None)
        row.pop("recoverability_enforced", None)
        row.pop("unrecoverable_ratio_series", None)
        row.pop("planning_cycle_times", None)
        rows.append(row)

    fieldnames = [
        "scenario",
        "scenario_name",
        "success",
        "collision",
        "reason_for_failure",
        "gif_path",
        "task_time_s",
        "avg_speed_mps",
        "min_clearance_m",
        "t_c_s",
        "t95_s",
        "Omega_bar",
        "C_Omega",
        "URR",
        "recoverability_activation_ratio",
        "selected_plan_unrecoverable_ratio",
    ]
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-scenario", default=DEFAULT_BASE_SCENARIO)
    parser.add_argument("--sample-dir", default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--planning-config", default="planning_fast.json")
    parser.add_argument("--contingency-config", default="contingency.json")
    parser.add_argument("--risk-config", default="risk.json")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--recoverability-enabled", choices=["true", "false"], default=None)
    parser.add_argument("--longitudinal-a-max-scale", type=float, default=None)
    parser.add_argument("--lateral-a-max-scale", type=float, default=None)
    parser.add_argument("--longitudinal-v-max-scale", type=float, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N scenarios after sorting; base scenario is always first.",
    )
    args = parser.parse_args()

    plt.ioff()

    settings = load_planning_json(args.planning_config)
    settings["contingency_settings"] = load_contingency_json(args.contingency_config)
    settings["risk_dict"] = load_risk_json(args.risk_config)
    settings["evaluation_settings"]["show_visualization"] = True
    settings["evaluation_settings"]["timing_enabled"] = True
    settings["risk_dict"]["figures"]["create_figures"] = False
    settings["risk_dict"]["risk_dashboard"] = False
    settings["risk_dict"]["collision_report"] = False

    vehicle_param_overrides = {}
    if args.longitudinal_a_max_scale is not None:
        vehicle_param_overrides["longitudinal_a_max_scale"] = float(
            args.longitudinal_a_max_scale
        )
    if args.lateral_a_max_scale is not None:
        vehicle_param_overrides["lateral_a_max_scale"] = float(
            args.lateral_a_max_scale
        )
    if args.longitudinal_v_max_scale is not None:
        vehicle_param_overrides["longitudinal_v_max_scale"] = float(
            args.longitudinal_v_max_scale
        )
    settings["evaluation_settings"]["vehicle_param_overrides"] = vehicle_param_overrides
    if args.recoverability_enabled is not None:
        settings["contingency_settings"].setdefault("recoverability", {})
        settings["contingency_settings"]["recoverability"]["enabled"] = (
            args.recoverability_enabled.lower() == "true"
        )

    path_to_scenarios = (REPO_ROOT / "scenarios").resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    gif_dir = output_dir / "gifs"
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)

    frenet_creator = FrenetCreator(settings)
    evaluator = LightweightScenarioEvaluator(
        planner_creator=frenet_creator,
        vehicle_type=settings["evaluation_settings"]["vehicle_type"],
        path_to_scenarios=path_to_scenarios,
        log_path=(REPO_ROOT / "log/example").resolve(),
        collision_report_path=output_dir,
        timing_enabled=settings["evaluation_settings"]["timing_enabled"],
    )

    scenario_list = _build_scenario_list(
        path_to_scenarios=path_to_scenarios,
        base_scenario=args.base_scenario,
        sample_dir=args.sample_dir,
    )
    if len(scenario_list) < 2:
        raise FileNotFoundError(
            f"No sampled XML files found under {path_to_scenarios.joinpath(args.sample_dir)}. "
            "Please generate the `vv_samples` scenarios first."
        )
    if args.limit is not None:
        scenario_list = scenario_list[: max(1, int(args.limit))]
    if len(scenario_list) != 100:
        print(
            f"Warning: expected 100 scenarios (1 base + 99 samples), "
            f"but found {len(scenario_list)}."
        )

    per_scenario_metrics = []
    for scenario_idx, scenario_rel_path in enumerate(scenario_list, start=1):
        scenario_name = Path(scenario_rel_path).stem
        print(f"[{scenario_idx:03d}/{len(scenario_list):03d}] evaluating {scenario_name}")
        clear_plot_snapshots()
        plt.close("all")

        return_dict = evaluator.eval_scenario(scenario_rel_path)
        gif_path = gif_dir / f"{scenario_name}.gif"
        gif_saved = _save_snapshots_to_gif(output_path=gif_path, fps=args.fps)
        if not gif_saved:
            gif_path = ""

        metrics = _scenario_metrics(
            evaluator=evaluator,
            return_dict=return_dict,
            gif_path=gif_path,
        )
        per_scenario_metrics.append(metrics)
        _cleanup_unwanted_outputs(scenario_name=scenario_name)
        clear_plot_snapshots()
        plt.close("all")

    summary = _aggregate_summary(per_scenario_metrics)
    summary["evaluated_scenarios"] = len(scenario_list)
    summary["experiment_tag"] = str(args.experiment_tag)
    summary["recoverability_enabled"] = settings["contingency_settings"].get(
        "recoverability", {}
    ).get("enabled", True)
    summary["vehicle_param_overrides"] = vehicle_param_overrides

    metrics_json_path = output_dir / "metrics_summary.json"
    metrics_csv_path = output_dir / "metrics_per_scenario.csv"
    with open(metrics_json_path, "w") as json_file:
        json.dump(
            {
                "summary": summary,
                "per_scenario": per_scenario_metrics,
            },
            json_file,
            indent=2,
        )
    _write_metrics_csv(metrics_csv_path, per_scenario_metrics)

    print(f"Saved summary to {metrics_json_path}")
    print(f"Saved per-scenario metrics to {metrics_csv_path}")
    print(f"Saved GIFs to {gif_dir}")


if __name__ == "__main__":
    main()
