import xml.etree.ElementTree as ET



xml_path = "/home/yanjun/NewDisk/beliefplanning/test_fot_junction/your_scenario.xml"
target_obstacle_id = "2"
old_speed = 0.36489538
new_speed = 3.5
scale = new_speed / old_speed

tree = ET.parse(xml_path)
root = tree.getroot()

for dyn in root.iter("dynamicObstacle"):
    if dyn.get("id") != target_obstacle_id:
        continue

    # initialState
    init_state = dyn.find("initialState")
    x0 = float(init_state.find("./position/point/x").text)
    y0 = float(init_state.find("./position/point/y").text)

    init_vel = init_state.find("./velocity/exact")
    if init_vel is not None:
        init_vel.text = f"{new_speed:.8f}"

    # trajectory states
    traj = dyn.find("trajectory")
    for state in traj.findall("state"):
        x_node = state.find("./position/point/x")
        y_node = state.find("./position/point/y")
        v_node = state.find("./velocity/exact")

        old_x = float(x_node.text)
        old_y = float(y_node.text)

        new_x = x0 + scale * (old_x - x0)
        new_y = y0 + scale * (old_y - y0)

        x_node.text = f"{new_x:.8f}"
        y_node.text = f"{new_y:.8f}"

        if v_node is not None:
            v_node.text = f"{new_speed:.8f}"

tree.write("updated_scenario.xml", encoding="utf-8", xml_declaration=True)
print("done -> updated_scenario.xml")
