import argparse
import copy
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from commonroad_helper_functions.utils.cubicspline import CubicSpline2D


SCENARIO_PATH = Path("scenarios/recorded/hand-crafted/BRA_VilaVelha-92_1_T-10.xml")
OUTPUT_DIR = Path("scenarios/recorded/hand-crafted/vv_samples")
NUM_SAMPLES = 99
DEFAULT_SEED = 42

EGO_POSITION_PERTURB_M = 2.0
OBSTACLE_POSITION_PERTURB_M = 3.0
EGO_SPEED_PERTURB_MPS = 0.5
OBSTACLE_SPEED_PERTURB_MPS = 0.7

EGO_GLOBAL_PATH_POINTS = np.array(
    [
        [15.9923642, -21.86624793],
        [14.925595841823453, -20.174501710854436],
        [13.858827483646904, -18.482755491708872],
        [12.792059125470356, -16.791009272563308],
        [11.725290767293806, -15.099263053417744],
        [10.658522409117259, -13.407516834272178],
        [9.591754050940711, -11.715770615126614],
        [8.524985692764162, -10.02402439598105],
        [7.458217334587614, -8.332278176835485],
        [6.391448976411066, -6.640531957689921],
        [5.259854326040924, -4.992258159902549],
        [3.960180094825631, -3.4778369140472356],
        [2.4574801087051568, -2.166923137888149],
        [0.6842910555594486, -1.2603370256293334],
        [-1.2587043808147595, -0.8246394091618383],
        [-3.2458598590390544, -0.9222052878272513],
        [-5.178082830793184, -1.4273801320466577],
        [-7.003871392029162, -2.242158366358507],
        [-8.724970240905629, -3.2602085478706564],
        [-10.420849103520283, -4.32039479723292],
        [-12.116727966134938, -5.380581046595183],
        [-13.812606828749594, -6.440767295957446],
        [-15.508485691364248, -7.50095354531971],
        [-17.204364553978905, -8.561139794681973],
        [-18.900243416593558, -9.621326044044237],
        [-20.596122279208213, -10.6815122934065],
        [-22.29200114182287, -11.741698542768763],
        [-23.98788000443752, -12.801884792131027],
        [-25.683758867052177, -13.862071041493289],
        [-27.379637729666833, -14.922257290855553],
        [-29.075516592281488, -15.982443540217815],
        [-30.771395454896144, -17.04262978958008],
        [-32.4672743175108, -18.102816038942343],
        [-34.163153180125455, -19.163002288304607],
        [-35.85903204274011, -20.223188537666868],
        [-37.55491090535476, -21.283374787029132],
        [-39.25078976796942, -22.343561036391396],
        [-40.94666863058407, -23.40374728575366],
        [-42.642547493198734, -24.463933535115924],
        [-44.33842635581338, -25.524119784478188],
        [-46.034305218428045, -26.58430603384045],
        [-47.730184081042694, -27.644492283202712],
        [-49.42606294365734, -28.704678532564976],
        [-51.121941806272005, -29.76486478192724],
        [-52.817820668886654, -30.825051031289505],
        [-54.513696611256165, -31.88524195183504],
        [-56.20956435864864, -32.94544598095111],
        [-57.90543210604112, -34.005650010067185],
        [-59.6012998534336, -35.065854039183264],
        [-61.29716760082608, -36.126058068299336],
        [-62.993035348218555, -37.18626209741541],
        [-64.68890309561104, -38.24646612653149],
        [-66.38477084300352, -39.30667015564756],
        [-68.08063859039599, -40.36687418476363],
        [-69.77650633778848, -41.42707821387971],
        [-71.47237408518096, -42.48728224299578],
        [-73.16824183257343, -43.547486272111854],
        [-74.86410957996591, -44.60769030122793],
        [-76.5599773273584, -45.667894330344005],
        [-78.25584507475088, -46.72809835946008],
        [-79.95171282214335, -47.788302388576156],
        [-81.64758056953583, -48.84850641769223],
        [-83.34344831692832, -49.9087104468083],
        [-85.03931606432079, -50.96891447592438],
        [-86.73518381171327, -52.029118505040444],
        [-88.43105155910575, -53.08932253415652],
        [-90.12691930649822, -54.1495265632726],
        [-91.8227870538907, -55.20973059238867],
        [-93.51865480128319, -56.269934621504746],
        [-95.21452254867566, -57.330138650620825],
        [-96.91039029606814, -58.39034267973689],
        [-98.60625804346063, -59.45054670885297],
        [-100.0234, -60.3365],
    ],
    dtype=float,
)


def _read_exact(node, path):
    exact_node = node.find(path)
    if exact_node is None or exact_node.text is None:
        raise ValueError(f"Missing XML node: {path}")
    return float(exact_node.text)


def _write_exact(node, path, value, digits=6):
    target = node.find(path)
    if target is None:
        raise ValueError(f"Missing XML node: {path}")
    target.text = f"{float(value):.{digits}f}"


def _collect_obstacle_path_points(dynamic_obstacle):
    initial_state = dynamic_obstacle.find("./initialState")
    trajectory_node = dynamic_obstacle.find("./trajectory")
    if initial_state is None or trajectory_node is None:
        raise ValueError("dynamicObstacle missing initialState or trajectory")

    path_points = [
        [
            _read_exact(initial_state, "./position/point/x"),
            _read_exact(initial_state, "./position/point/y"),
        ]
    ]
    for state_node in trajectory_node.findall("./state"):
        path_points.append(
            [
                _read_exact(state_node, "./position/point/x"),
                _read_exact(state_node, "./position/point/y"),
            ]
        )

    filtered_points = [path_points[0]]
    for point in path_points[1:]:
        if np.linalg.norm(np.asarray(point) - np.asarray(filtered_points[-1])) > 1e-6:
            filtered_points.append(point)
    if len(filtered_points) < 2:
        raise ValueError("Need at least two distinct path points to build spline")
    return np.asarray(filtered_points, dtype=float)


def _extended_spline_sample(spline, s_query, start_tangent, end_tangent):
    s_max = float(spline.s[-1])
    if s_query < 0.0:
        start_point = np.asarray(spline.calc_position(0.0), dtype=float)
        return start_point + float(s_query) * start_tangent
    if s_query > s_max:
        end_point = np.asarray(spline.calc_position(s_max), dtype=float)
        return end_point + float(s_query - s_max) * end_tangent
    return np.asarray(spline.calc_position(float(s_query)), dtype=float)


def _build_constant_speed_positions_from_path(path_points, start_s, speed, dt, num_states):
    spline = CubicSpline2D(x=path_points[:, 0], y=path_points[:, 1])
    if not hasattr(spline, "s") or len(spline.s) == 0:
        raise ValueError("Spline does not contain arc-length values")

    speed = max(0.0, float(speed))
    start_tangent = path_points[1] - path_points[0]
    end_tangent = path_points[-1] - path_points[-2]
    if np.linalg.norm(start_tangent) < 1e-9:
        start_tangent = np.array([1.0, 0.0], dtype=float)
    else:
        start_tangent = start_tangent / np.linalg.norm(start_tangent)
    if np.linalg.norm(end_tangent) < 1e-9:
        end_tangent = np.array([1.0, 0.0], dtype=float)
    else:
        end_tangent = end_tangent / np.linalg.norm(end_tangent)

    positions = []
    for idx in range(num_states):
        s_k = float(start_s + speed * idx * dt)
        positions.append(
            _extended_spline_sample(
                spline=spline,
                s_query=s_k,
                start_tangent=start_tangent,
                end_tangent=end_tangent,
            )
        )
    return np.asarray(positions, dtype=float)


def _sample_point_on_path(path_points, start_s):
    return _build_constant_speed_positions_from_path(
        path_points=path_points,
        start_s=start_s,
        speed=0.0,
        dt=1.0,
        num_states=1,
    )[0]


def _replace_trajectory_states(trajectory_node, positions, velocity, orientation, start_time_step):
    for child in list(trajectory_node):
        trajectory_node.remove(child)

    for state_idx, position in enumerate(positions):
        state = ET.SubElement(trajectory_node, "state")

        time_node = ET.SubElement(state, "time")
        time_exact = ET.SubElement(time_node, "exact")
        time_exact.text = str(int(start_time_step + state_idx))

        position_node = ET.SubElement(state, "position")
        point_node = ET.SubElement(position_node, "point")
        x_node = ET.SubElement(point_node, "x")
        x_node.text = f"{float(position[0]):.6f}"
        y_node = ET.SubElement(point_node, "y")
        y_node.text = f"{float(position[1]):.6f}"

        velocity_node = ET.SubElement(state, "velocity")
        velocity_exact = ET.SubElement(velocity_node, "exact")
        velocity_exact.text = f"{float(velocity):.6f}"

        orientation_node = ET.SubElement(state, "orientation")
        orientation_exact = ET.SubElement(orientation_node, "exact")
        orientation_exact.text = f"{float(orientation):.6f}"

        acceleration_node = ET.SubElement(state, "acceleration")
        acceleration_exact = ET.SubElement(acceleration_node, "exact")
        acceleration_exact.text = "0.000000"


def _update_planning_problem_initial_state(root, rng):
    planning_problem = root.find(".//planningProblem")
    if planning_problem is None:
        raise ValueError("No planningProblem found in scenario XML")

    initial_state = planning_problem.find("./initialState")
    if initial_state is None:
        raise ValueError("planningProblem initialState missing")

    v0 = _read_exact(initial_state, "./velocity/exact")
    yaw0 = _read_exact(initial_state, "./orientation/exact")
    x0 = _read_exact(initial_state, "./position/point/x")
    y0 = _read_exact(initial_state, "./position/point/y")

    delta_s = rng.uniform(-EGO_POSITION_PERTURB_M, EGO_POSITION_PERTURB_M)
    delta_v = rng.uniform(-EGO_SPEED_PERTURB_MPS, EGO_SPEED_PERTURB_MPS)
    new_position = _sample_point_on_path(
        path_points=EGO_GLOBAL_PATH_POINTS,
        start_s=delta_s,
    )
    new_v = max(0.0, v0 + delta_v)

    _write_exact(initial_state, "./position/point/x", new_position[0])
    _write_exact(initial_state, "./position/point/y", new_position[1])
    _write_exact(initial_state, "./velocity/exact", new_v)

    return {
        "entity": "ego",
        "x_nominal": float(x0),
        "y_nominal": float(y0),
        "orientation_nominal": float(yaw0),
        "velocity_nominal": float(v0),
        "ds": float(delta_s),
        "dx": float(new_position[0] - x0),
        "dy": float(new_position[1] - y0),
        "dv": float(delta_v),
        "x_new": float(new_position[0]),
        "y_new": float(new_position[1]),
        "orientation_new": float(yaw0),
        "velocity_new": float(new_v),
    }


def _update_dynamic_obstacle(root, obstacle_id, rng, dt):
    dynamic_obstacle = root.find(f".//dynamicObstacle[@id='{obstacle_id}']")
    if dynamic_obstacle is None:
        raise ValueError(f"dynamicObstacle {obstacle_id} not found")

    initial_state = dynamic_obstacle.find("./initialState")
    trajectory_node = dynamic_obstacle.find("./trajectory")
    if initial_state is None or trajectory_node is None:
        raise ValueError(f"dynamicObstacle {obstacle_id} missing initialState or trajectory")

    x0 = _read_exact(initial_state, "./position/point/x")
    y0 = _read_exact(initial_state, "./position/point/y")
    v0 = _read_exact(initial_state, "./velocity/exact")
    yaw0 = _read_exact(initial_state, "./orientation/exact")
    path_points = _collect_obstacle_path_points(dynamic_obstacle)

    state_nodes = trajectory_node.findall("./state")
    num_states = len(state_nodes)
    if num_states == 0:
        raise ValueError(f"dynamicObstacle {obstacle_id} has empty trajectory")

    first_time = int(_read_exact(state_nodes[0], "./time/exact"))

    delta_s = rng.uniform(-OBSTACLE_POSITION_PERTURB_M, OBSTACLE_POSITION_PERTURB_M)
    delta_v = rng.uniform(-OBSTACLE_SPEED_PERTURB_MPS, OBSTACLE_SPEED_PERTURB_MPS)
    new_speed = max(0.0, v0 + delta_v)
    positions = _build_constant_speed_positions_from_path(
        path_points=path_points,
        start_s=delta_s,
        speed=new_speed,
        dt=dt,
        num_states=num_states,
    )
    new_position = positions[0]

    _write_exact(initial_state, "./position/point/x", new_position[0])
    _write_exact(initial_state, "./position/point/y", new_position[1])
    _write_exact(initial_state, "./velocity/exact", new_speed)
    _replace_trajectory_states(
        trajectory_node=trajectory_node,
        positions=positions,
        velocity=new_speed,
        orientation=yaw0,
        start_time_step=first_time,
    )

    return {
        "entity": f"obstacle_{obstacle_id}",
        "obstacle_id": obstacle_id,
        "x_nominal": float(x0),
        "y_nominal": float(y0),
        "orientation_nominal": float(yaw0),
        "velocity_nominal": float(v0),
        "dx": float(new_position[0] - x0),
        "dy": float(new_position[1] - y0),
        "ds": float(delta_s),
        "dv": float(delta_v),
        "x_new": float(new_position[0]),
        "y_new": float(new_position[1]),
        "orientation_new": float(yaw0),
        "velocity_new": float(new_speed),
        "trajectory_num_states": int(num_states),
        "trajectory_start_time_step": int(first_time),
    }


def build_sample_scenario_tree(template_tree, sample_index, rng):
    tree = copy.deepcopy(template_tree)
    root = tree.getroot()
    dt = float(root.attrib.get("timeStepSize", "0.1"))

    scenario_name = f"vv_{sample_index:03d}"
    root.attrib["benchmarkID"] = scenario_name

    manifest_entry = {
        "scenario_name": scenario_name,
        "time_step_size": dt,
        "entities": [],
    }

    manifest_entry["entities"].append(
        _update_planning_problem_initial_state(root=root, rng=rng)
    )
    for obstacle_id in ["20044", "20087"]:
        manifest_entry["entities"].append(
            _update_dynamic_obstacle(root=root, obstacle_id=obstacle_id, rng=rng, dt=dt)
        )

    return tree, scenario_name, manifest_entry


def _write_manifest(output_dir, manifest_entries, seed):
    json_path = output_dir / "vv_manifest.json"
    csv_path = output_dir / "vv_manifest.csv"

    json_payload = {
        "seed": int(seed),
        "num_samples": int(len(manifest_entries)),
        "samples": manifest_entries,
    }
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(json_payload, json_file, indent=2, ensure_ascii=False)

    csv_rows = []
    for sample in manifest_entries:
        for entity in sample["entities"]:
            csv_rows.append(
                {
                    "scenario_name": sample["scenario_name"],
                    "entity": entity["entity"],
                    "obstacle_id": entity.get("obstacle_id", ""),
                    "x_nominal": entity["x_nominal"],
                    "y_nominal": entity["y_nominal"],
                    "velocity_nominal": entity["velocity_nominal"],
                    "orientation_nominal": entity.get("orientation_nominal", ""),
                    "ds": entity.get("ds", ""),
                    "dx": entity["dx"],
                    "dy": entity["dy"],
                    "dv": entity["dv"],
                    "x_new": entity["x_new"],
                    "y_new": entity["y_new"],
                    "orientation_new": entity.get("orientation_new", ""),
                    "velocity_new": entity["velocity_new"],
                    "trajectory_num_states": entity.get("trajectory_num_states", ""),
                    "trajectory_start_time_step": entity.get("trajectory_start_time_step", ""),
                }
            )

    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "scenario_name",
                "entity",
                "obstacle_id",
                "x_nominal",
                "y_nominal",
                "velocity_nominal",
                "orientation_nominal",
                "ds",
                "dx",
                "dy",
                "dv",
                "x_new",
                "y_new",
                "orientation_new",
                "velocity_new",
                "trajectory_num_states",
                "trajectory_start_time_step",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description="Generate perturbed BRA Vila Velha sample scenarios.")
    parser.add_argument("--input", type=Path, default=SCENARIO_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    template_tree = ET.parse(str(args.input))
    rng = np.random.default_rng(args.seed)
    manifest_entries = []

    for sample_index in range(1, args.num_samples + 1):
        sample_tree, scenario_name, manifest_entry = build_sample_scenario_tree(
            template_tree=template_tree,
            sample_index=sample_index,
            rng=rng,
        )
        output_path = args.output_dir / f"{scenario_name}.xml"
        sample_tree.write(output_path, encoding="utf-8", xml_declaration=True)
        manifest_entries.append(manifest_entry)
        print(f"[OK] wrote {output_path}")

    json_path, csv_path = _write_manifest(
        output_dir=args.output_dir,
        manifest_entries=manifest_entries,
        seed=args.seed,
    )
    print(f"[OK] wrote {json_path}")
    print(f"[OK] wrote {csv_path}")


if __name__ == "__main__":
    main()
