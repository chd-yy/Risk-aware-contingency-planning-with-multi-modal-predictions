import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "test_fot_junction" / "evaluate_vv_batch.py"
DEFAULT_OUTPUT_DIR = "planner/Frenet/results/intent_mode_granularity_100_20260429"
DEFAULT_BASE_SCENARIO = "recorded/hand-crafted/BRA_VilaVelha-92_1_T-10.xml"
DEFAULT_SAMPLE_DIR = "recorded/hand-crafted/vv_samples"

MODE_SETTINGS = [
    {"name": "mode2", "intent_mode_count": 2},
    {"name": "mode3", "intent_mode_count": 3},
    {"name": "mode4", "intent_mode_count": 4},
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
    intent_mode_count: int,
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
        "--intent-mode-count",
        str(int(intent_mode_count)),
    ]
    if limit is not None:
        command.extend(["--limit", str(int(limit))])

    subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def _load_summary(metrics_summary_path: Path) -> Dict:
    with open(metrics_summary_path, "r", encoding="utf-8") as summary_file:
        payload = json.load(summary_file)
    return payload.get("summary", {})


def _write_csv(output_path: Path, rows: List[Dict]):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["setting", "intent_mode_count", "evaluated_scenarios"] + SUMMARY_KEYS
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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

    rows = []
    for mode_setting in MODE_SETTINGS:
        setting_name = mode_setting["name"]
        intent_mode_count = int(mode_setting["intent_mode_count"])
        run_output_dir = output_root / setting_name
        metrics_summary_path = run_output_dir / "metrics_summary.json"

        if not (args.skip_existing and metrics_summary_path.exists()):
            print(f"[run] setting={setting_name} intent_mode_count={intent_mode_count}")
            _run_one_eval(
                output_dir=run_output_dir,
                base_scenario=args.base_scenario,
                sample_dir=args.sample_dir,
                planning_config=args.planning_config,
                contingency_config=args.contingency_config,
                risk_config=args.risk_config,
                fps=args.fps,
                limit=args.limit,
                experiment_tag=f"{setting_name}",
                intent_mode_count=intent_mode_count,
            )

        summary = _load_summary(metrics_summary_path)
        row = {
            "setting": setting_name,
            "intent_mode_count": intent_mode_count,
            "evaluated_scenarios": summary.get("evaluated_scenarios"),
        }
        for key in SUMMARY_KEYS:
            row[key] = summary.get(key)
        rows.append(row)

    comparison_json_path = output_root / "comparison_summary.json"
    comparison_csv_path = output_root / "comparison_summary.csv"
    comparison_md_path = output_root / "comparison_summary.md"

    with open(comparison_json_path, "w", encoding="utf-8") as output_file:
        json.dump(
            {"mode_settings": MODE_SETTINGS, "results": rows},
            output_file,
            indent=2,
            ensure_ascii=False,
        )
    _write_csv(comparison_csv_path, rows)

    lines = [
        "# 意图模式粒度敏感性实验结果",
        "",
        "| 设置 | 模式数 | SR | CR | v_bar | d_min | t_c | t95 | Omega_bar | C_Omega | URR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {setting} | {intent_mode_count} | {SR:.4f} | {CR:.4f} | {v_bar:.4f} | {d_min:.4f} | {t_c:.4f} | {t95:.4f} | {Omega_bar:.4f} | {C_Omega:.4f} | {URR:.4f} |".format(
                setting=row["setting"],
                intent_mode_count=int(row["intent_mode_count"]),
                SR=float(row["SR"]) if row["SR"] is not None else float("nan"),
                CR=float(row["CR"]) if row["CR"] is not None else float("nan"),
                v_bar=float(row["v_bar"]) if row["v_bar"] is not None else float("nan"),
                d_min=float(row["d_min"]) if row["d_min"] is not None else float("nan"),
                t_c=float(row["t_c"]) if row["t_c"] is not None else float("nan"),
                t95=float(row["t95"]) if row["t95"] is not None else float("nan"),
                Omega_bar=float(row["Omega_bar"]) if row["Omega_bar"] is not None else float("nan"),
                C_Omega=float(row["C_Omega"]) if row["C_Omega"] is not None else float("nan"),
                URR=float(row["URR"]) if row["URR"] is not None else float("nan"),
            )
        )
    comparison_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved comparison summary to {comparison_json_path}")
    print(f"Saved comparison table to {comparison_csv_path}")
    print(f"Saved comparison markdown to {comparison_md_path}")


if __name__ == "__main__":
    main()
