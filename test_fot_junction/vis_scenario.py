import pathlib
import sys
from pathlib import Path

import matplotlib
# Make sure the project root is on PYTHONPATH when running this file directly.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from planner.Frenet.utils.visualization import draw_scenario
import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader


if __name__ == "__main__":
    matplotlib.use("TKAgg")
    scenario_path = Path('/home/yanjun/NewDisk/beliefplanning/scenarios/recorded/hand-crafted/BRA_VilaVelha-92_1_T-10.xml')
    scenario, planning_problem_set = CommonRoadFileReader(scenario_path
    ).open()
  
    ax = draw_scenario(scenario)

