import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "test_fot_junction" / "evaluate_vv_batch.py"
DEFAULT_OUTPUT_DIR = "planner/Frenet/results/recoverability_stress_eval"
DEFAULT_BASE_SCENARIO = "recorded/hand-crafted/BRA_VilaVelha-92_1_T-10.xml"
DEFAULT_SAMPLE_DIR = "recorded/hand-crafted/vv_samples"

REGIMES = [
    {
        "name": "nominal",
        "vehicle_overrides": {},
    },
]

METHODS = [
    {
        "name": "ours",
        "recoverability_enabled": True,
    },
    {
        "name": "ours_wo_Rec",
        "recoverability_enabled": False,
    },
]

SUMMARY_KEYS = [
    "SR",
    "CR",
    "T_task",
    "v_bar",
    "d_min",
    "t_c",
    "t95",
    "Omega_bar",
    "C_Omega",
    "URR",
    "recoverability_activation_ratio",
    "selected_plan_unrecoverable_ratio",
]


def _run_one_eval(
    output_dir: Path,
    base_scenario: str,
    sample_dir: str,
    planning_config: str,
    contingency_config: str,
    risk_config: str,
    fps: int,
    limit: int,
    experiment_tag: str,
    recoverability_enabled: bool,
    vehicle_overrides: Dict,
):
    command = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--base-scenario",
        base_scenario,
        "--sample-dir",
        sample_dir,
        "--output-dir",
        str(output_dir.relative_to(REPO_ROOT)),
        "--planning-config",
        planning_config,
        "--contingency-config",
        contingency_config,
        "--risk-config",
        risk_config,
        "--fps",
        str(int(fps)),
        "--experiment-tag",
        experiment_tag,
        "--recoverability-enabled",
        "true" if recoverability_enabled else "false",
    ]
    if limit is not None:
        command.extend(["--limit", str(int(limit))])
    for key, value in vehicle_overrides.items():
        cli_name = "--" + key.replace("_", "-")
        command.extend([cli_name, str(value)])

    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def _load_summary(metrics_summary_path: Path) -> Dict:
    with open(metrics_summary_path, "r", encoding="utf-8") as summary_file:
        payload = json.load(summary_file)
    summary = payload.get("summary", {})
    per_scenario = payload.get("per_scenario", [])
    return {
        "summary": summary,
        "per_scenario": per_scenario,
    }


def _build_comparison_rows(regime_results: List[Dict]) -> List[Dict]:
    rows = []
    for result in regime_results:
        regime_name = result["regime"]
        method_name = result["method"]
        summary = result["summary"]
        row = {
            "regime": regime_name,
            "method": method_name,
        }
        for key in SUMMARY_KEYS:
            row[key] = summary.get(key)
        row["evaluated_scenarios"] = summary.get("evaluated_scenarios")
        rows.append(row)
    return rows


def _write_csv(output_path: Path, rows: List[Dict]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["regime", "method", "evaluated_scenarios"] + SUMMARY_KEYS
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_delta_rows(comparison_rows: List[Dict]) -> List[Dict]:
    by_regime = {}
    for row in comparison_rows:
        by_regime.setdefault(row["regime"], {})[row["method"]] = row

    delta_rows = []
    for regime_name, regime_dict in by_regime.items():
        if "ours" not in regime_dict or "ours_wo_Rec" not in regime_dict:
            continue
        ours = regime_dict["ours"]
        ablation = regime_dict["ours_wo_Rec"]
        delta_row = {
            "regime": regime_name,
            "delta_definition": "ours - ours_wo_Rec",
        }
        for key in SUMMARY_KEYS:
            ours_value = ours.get(key)
            ablation_value = ablation.get(key)
            if ours_value is None or ablation_value is None:
                delta_row[key] = None
            else:
                delta_row[key] = float(ours_value) - float(ablation_value)
        delta_rows.append(delta_row)
    return delta_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-scenario", default=DEFAULT_BASE_SCENARIO)
    parser.add_argument("--sample-dir", default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--planning-config", default="planning_fast.json")
    parser.add_argument("--contingency-config", default="contingency.json")
    parser.add_argument("--risk-config", default="risk.json")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    output_root = (REPO_ROOT / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    regime_results = []
    for regime in REGIMES:
        regime_name = regime["name"]
        vehicle_overrides = regime["vehicle_overrides"]
        for method in METHODS:
            method_name = method["name"]
            recoverability_enabled = method["recoverability_enabled"]
            run_output_dir = output_root / regime_name / method_name
            metrics_summary_path = run_output_dir / "metrics_summary.json"

            if not (args.skip_existing and metrics_summary_path.exists()):
                print(f"[run] regime={regime_name} method={method_name}")
                _run_one_eval(
                    output_dir=run_output_dir,
                    base_scenario=args.base_scenario,
                    sample_dir=args.sample_dir,
                    planning_config=args.planning_config,
                    contingency_config=args.contingency_config,
                    risk_config=args.risk_config,
                    fps=args.fps,
                    limit=args.limit,
                    experiment_tag=f"{regime_name}:{method_name}",
                    recoverability_enabled=recoverability_enabled,
                    vehicle_overrides=vehicle_overrides,
                )

            payload = _load_summary(metrics_summary_path)
            regime_results.append(
                {
                    "regime": regime_name,
                    "method": method_name,
                    "vehicle_overrides": vehicle_overrides,
                    "summary": payload["summary"],
                }
            )

    comparison_rows = _build_comparison_rows(regime_results)
    delta_rows = _build_delta_rows(comparison_rows)

    comparison_json_path = output_root / "comparison_summary.json"
    comparison_csv_path = output_root / "comparison_summary.csv"
    delta_csv_path = output_root / "comparison_delta_ours_minus_ablation.csv"

    with open(comparison_json_path, "w", encoding="utf-8") as output_file:
        json.dump(
            {
                "regimes": REGIMES,
                "methods": METHODS,
                "results": regime_results,
                "delta_rows": delta_rows,
            },
            output_file,
            indent=2,
            ensure_ascii=False,
        )
    _write_csv(comparison_csv_path, comparison_rows)

    with open(delta_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["regime", "delta_definition"] + SUMMARY_KEYS
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in delta_rows:
            writer.writerow(row)

    print(f"Saved comparison summary to {comparison_json_path}")
    print(f"Saved comparison table to {comparison_csv_path}")
    print(f"Saved delta table to {delta_csv_path}")


if __name__ == "__main__":
    main()
