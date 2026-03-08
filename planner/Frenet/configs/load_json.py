"""Helper functions to load jsons."""

import json
import numpy as np
import os


def load_risk_json(filename="risk.json"):
    """
    Load the risk.json with harm weights.

    Returns:
        Dict: weights and modes form risk.json
    """
    risk_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        filename,
    )
    with open(risk_config_path, "r") as f:
        jsondata = json.load(f)

    # print(f"Loaded settings from {risk_config_path}")

    return jsondata


def load_harm_parameter_json():
    """
    Load the harm_parameters.json with model parameters.

    Returns:
        Dict: model parameters from parameter.json
    """
    parameter_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "harm_parameters.json",
    )
    with open(parameter_config_path, "r") as f:
        jsondata = json.load(f)
    return jsondata


def load_weight_json(filename="weights.json"):
    """
    Load the weights.json with cost weights for risk.

    Returns:
        Dict: model parameters from weights.json
    """
    weight_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        filename,
    )
    with open(weight_config_path, "r") as f:
        jsondata = json.load(f)

    # print(f"Loaded weights from {weight_config_path}")

    return jsondata


def load_planning_json(filename="planning.json"):
    """
    Load the planning.json with modes.

    Returns:
        Dict: parameters from planning.json
    """
    """
    作用:读取 planning.json 配置文件,并把里面某个“离散列表配置”转换成真正的数值数组。

    大白话解释:
    - 你的 planning.json 里通常会写很多参数(比如 frenet planner 的 mode、采样范围等)。
    - 其中 d_list 这个参数在 json 里可能不是一个真正的列表,而是用这种方式描述:
        {
          "d_list": {"d_max_abs": 1.5, "n": 31}
        }
      意思是:从 -1.5 到 +1.5 之间,均匀取 31 个采样点。
    - 但 planner 真正运行时更想要一个“真正的数组/列表”,例如:
        [-1.5, -1.4, ..., 0.0, ..., 1.4, 1.5]
    - 所以这个函数读取 json 后,会用 np.linspace 把 d_list 的配置“展开”为一个 numpy 数组。

    参数:
        filename (str):
            配置文件名,默认是 "planning.json"。
            你也可以传别的文件名,比如 "planning_fast.json"。

    返回:
        Dict:
            读取并处理后的配置字典(python dict)。
            注意:返回的 dict 已经被修改过,其中
            jsondata["frenet_settings"]["frenet_parameters"]["d_list"]
            会从“配置 dict”变成 “numpy array”。
    """
    planning_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        filename,
    )
    # ---- 2) 打开 json 文件并读取成 Python dict ----
    # with open(...): 会在读取完之后自动关闭文件(安全、推荐写法)
    # json.load(f): 读取 JSON 文件内容,并转成 Python 数据结构(通常是 dict + list)
    with open(planning_config_path, "r") as f:
        jsondata = json.load(f)

    # Create the d_list with linspace
    jsondata["frenet_settings"]["frenet_parameters"]["d_list"] = np.linspace(
        -jsondata["frenet_settings"]["frenet_parameters"]["d_list"]["d_max_abs"],
        jsondata["frenet_settings"]["frenet_parameters"]["d_list"]["d_max_abs"],
        jsondata["frenet_settings"]["frenet_parameters"]["d_list"]["n"],
    )

    return jsondata


def load_contingency_json(filename="contingency.json"):
    """
    Load the planning.json with modes.

    Returns:
        Dict: parameters from planning.json
    """
    planning_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        filename,
    )
    with open(planning_config_path, "r") as f:
        jsondata = json.load(f)

    # Create the d_list with linspace
    jsondata["frenet_parameters"]["d_list"] = np.linspace(
        -jsondata["frenet_parameters"]["d_list"]["d_max_abs"],
        jsondata["frenet_parameters"]["d_list"]["d_max_abs"],
        jsondata["frenet_parameters"]["d_list"]["n"],
    )

    return jsondata

# EOF
