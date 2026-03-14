import math
from commonroad_helper_functions.utils.cubicspline import CubicSpline2D # 引入二维样条曲线工具类

x_20044 = [
    -13.80150327,
    -5.7196,
    -1.31025,
    3.10105,
    15.99805,
    28.8915,
]

y_20044 = [
    18.76672966,
    5.9996,
    -0.9907,
    -7.9814,
    -28.57775,
    -49.16435,
]

reference_spline_20044 = CubicSpline2D(x=x_20044, y=y_20044)


def build_constant_speed_trajectory_xml(
    spline,
    v0: float,
    yaw0: float,
    dt: float = 0.1,
    start_time_step: int = 1,
    num_states: int = 20,
):
    """
    沿 spline 以恒定速度 v0 前进，航向角固定为 yaw0，加速度恒为 0。
    返回 CommonRoad 风格的 trajectory XML 字符串。
    """

    # 假设初始点就是 global_path 第一个点，因此起始弧长取 0
    s0 = 0.0

    # 尝试获取 spline 最大弧长
    # 很多 CubicSpline2D 实现里会有 self.s 存累计弧长数组
    if hasattr(spline, "s"):
        s_max = spline.s[-1]
    else:
        # 如果没有公开属性，可以退化处理：不断增长时由 calc_position 控制
        s_max = None

    xml_lines = []
    xml_lines.append("<trajectory>")

    for k in range(num_states):
        t = start_time_step + k

        # 恒速、零加速度下的弧长推进
        s_k = s0 + v0 * k * dt

        # 防止超过路径末端
        if s_max is not None:
            s_k = min(s_k, s_max)
        if (s_k > s_max):
            break
        x_k, y_k = spline.calc_position(s_k)

        xml_lines.append("  <state>")
        xml_lines.append("    <time>")
        xml_lines.append(f"      <exact>{t}</exact>")
        xml_lines.append("    </time>")
        xml_lines.append("    <position>")
        xml_lines.append("      <point>")
        xml_lines.append(f"        <x>{x_k:.6f}</x>")
        xml_lines.append(f"        <y>{y_k:.6f}</y>")
        xml_lines.append("      </point>")
        xml_lines.append("    </position>")
        xml_lines.append("    <velocity>")
        xml_lines.append(f"      <exact>{v0:.6f}</exact>")
        xml_lines.append("    </velocity>")
        xml_lines.append("    <orientation>")
        xml_lines.append(f"      <exact>{yaw0:.6f}</exact>")
        xml_lines.append("    </orientation>")
        xml_lines.append("    <acceleration>")
        xml_lines.append("      <exact>0.0</exact>")
        xml_lines.append("    </acceleration>")
        xml_lines.append("  </state>")

    xml_lines.append("</trajectory>")
    return "\n".join(xml_lines)

xml_text = build_constant_speed_trajectory_xml(
    spline=reference_spline_20044,
    v0=3.0,
    yaw0=-1.0058463890667786,
    dt=0.1,
    start_time_step=0,
    num_states=151,
)

# 保存到 xml 文件
output_file = "trajectory_20044.xml"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(xml_text)

print(f"已保存到 {output_file}")
