"""Simple reachable sets for objects in scenario."""
# -------------------------------------------------------------------------
# 这个文件的功能：
# 为场景中的交通参与者（尤其是动态障碍物）计算“简化可达集（reachable set）”。
#
# 可达集可以理解为：
# 在给定当前状态（位置、朝向、速度）和一定运动学约束（例如最大加速度）的情况下，
# 未来若干时间步内，这个物体“可能到达的空间区域”。
#
# 本文件中的 ReachSet 类主要做两件事：
# 1. 计算场景中各动态障碍物的可达集
# 2. 计算 ego 车辆的可达集，并根据安全距离规则进行扩张，
#    再用它去裁剪其他障碍物的可达集
# -------------------------------------------------------------------------

import os
import json
import numpy as np

# CommonRoad 场景对象
from commonroad.scenario.scenario import Scenario

# 根据位置和朝向查找最匹配 lanelet 的辅助函数
from beliefplanning.planner.GlobalPath.lanelet_based_planner import (
    find_lanelet_by_position_and_orientation,
)

# 可达集相关工具模块
from beliefplanning.planner.utils import reachable_set_simple

# 简单可达集生成函数
from beliefplanning.planner.utils.reachable_set_simple import simple_reachable_set

# 多边形 padding 工具，用于将不同点数的 polygon 补齐后批量送入 pygeos
from beliefplanning.planner.utils.responsibility import polygon_padding

# 几何计算库，用于 polygon、buffer、difference、intersection 等几何运算
import pygeos


class ReachSet(object):
    """
    Wrapper for simple reachable sets.

    Wrapper for simple reachable sets of
    all dynamic obstacles in scenario except ego.
    Currently the only supported obstacle type is car.

    """
    # ---------------------------------------------------------------------
    # ReachSet 类是一个“可达集封装器”。
    #
    # 它封装了：
    # - ego 车辆的可达集计算
    # - 其他动态障碍物的可达集计算
    # - 车道边界裁剪
    # - 与 ego 安全距离区域做差集
    #
    # 当前默认主要支持“汽车”类型障碍物。
    # ---------------------------------------------------------------------

    def __init__(
        self,
        scenario: Scenario,
        ego_id: int,
        ego_length: float,
        ego_width: float,
    ) -> None:
        """
        Initialize reachable set wrapper.

        Args:
        scenario (Scenario): Scenario.
        ego_id (int): ID of the ego vehicle.
        ego_length (float): length of ego vehicle.
        ego_width (float): width of ego vehicle.
        """
        # -----------------------------------------------------------------
        # 初始化函数
        #
        # 输入参数：
        # scenario    : CommonRoad 场景对象
        # ego_id      : ego 车辆的 obstacle id
        # ego_length  : ego 车辆长度
        # ego_width   : ego 车辆宽度
        #
        # 这里会加载 reachable set 参数配置，如果配置文件缺失，则使用默认值。
        # -----------------------------------------------------------------

        # reach_set_params (dict): Dictionary containing reachable set params:
        # dt (float): desired temporal resolution of the reachable set
        # t_max (float): maximum temporal horizon for the reachable set
        # a_max (float): assumed maximum acceleration of obstacle
        # rules (dict): Dictionary containing rules and their parameters

        # 从 reachable_set.json 中读取可达集参数
        self.reach_set_params = load_reach_set_json()

        # 如果没有读取到配置，则使用默认参数
        if self.reach_set_params is None:
            self.reach_set_params = {
                "dt": 0.2,
                "t_max": 2,
                "a_max": 8,
                "depth": 3,
                "rules": {"safe_distance": {"safe_distance_frac": 1.0}},
            }

        # 保存场景对象
        self.scenario = scenario

        # ego 车辆 id
        self.ego_id = ego_id

        # ego 车辆几何尺寸
        self.ego_length = ego_length
        self.ego_width = ego_width

        # simple reachable sets of all obstacles in scenario except ego
        # 保存各个时刻、各个障碍物的可达集结果
        #
        # 结构一般类似：
        # self.reach_sets[time_step][obstacle_id] = [reachable_set_variant_1, ...]
        self.reach_sets = {}

        # dictionary containing ReachSetSimple objects for all
        # lanelets that any obstacle has been on
        # key: lanelet id (int)
        #
        # 对每个出现过的 lanelet，缓存对应的 ReachSetSimple 对象列表
        # 注意同一个 lanelet 可能对应多个 bound 分支（例如多个 successor 组合）
        self.reach_set_objs = {}

        # dictionary containing ReachSetSimple objects for
        # lanelets, ignoring laterally adjacent lanelets
        # key: lanelet id (int)
        #
        # 这个字典主要给 ego 使用：
        # 只沿当前 lanelet 的前向结构考虑，不并入横向相邻车道
        self.reach_set_objs_single = {}

        # reachable set of ego (extended to account for safe distance)
        # does not consider laterally adjacent lanelets
        #
        # ego 的可达集，并且已经根据安全距离规则做了扩张
        self.ego_reach_set = {}

    def calc_reach_sets(self, ego_state, obstacle_list=None):
        """
        Calculate reachable sets.

        Calculate reachable sets of all dynamic obstacles in scenario except ego.

        Args:
        ego_state (Commonroad State object): state of ego vehicle at time_step
        """
        # -----------------------------------------------------------------
        # 计算某个时间步的可达集
        #
        # 主要流程：
        # 1. 若启用了 safe_distance 规则，先计算 ego 的可达集并扩张
        # 2. 遍历所有目标障碍物
        # 3. 为每个障碍物找到当前所在 lanelet
        # 4. 为对应 lanelet 构造/复用 ReachSetSimple 对象
        # 5. 生成障碍物的简单可达集
        # 6. 若启用 safe_distance，则从障碍物可达集中减去 ego 扩张可达集区域
        # -----------------------------------------------------------------

        # 如果规则中包含 safe_distance，先计算 ego 的安全距离可达集
        if "safe_distance" in self.reach_set_params["rules"]:
            self._ego_reach_set(ego_state)

        # 如果调用时指定了 obstacle_list，则只对这些障碍物计算
        if obstacle_list is not None:
            obstacles = [self.scenario.obstacle_by_id(obst_id) for obst_id in obstacle_list]
        else:
            # 否则默认对场景中所有障碍物计算
            obstacles = self.scenario.obstacles

        # 为当前 ego_state.time_step 初始化 reach_sets 容器
        self.reach_sets[ego_state.time_step] = {}

        # calculate polygon array for self._reach_set_difference(), avoid repeating
        # -----------------------------------------------------------------
        # b_dict 用于缓存 ego 扩张可达集在每个 step 对应的 union polygon，
        # 这样后面每个 obstacle 做差集时可以重复使用，不必重复 union。
        # -----------------------------------------------------------------
        b_dict = {}

        # self.ego_reach_set[ego_state.time_step] 是一个列表，
        # 列表中每个元素是一个 reachable set 字典，key 为预测 step。
        #
        # 这里用 [0] 取第一个 reachable set 的 key 集合，遍历所有未来 step。
        for step in self.ego_reach_set[ego_state.time_step][0]:
            # 取出 ego 在该 step 下的所有扩张 reachable set polygon
            b_poly_pygeos = [b_set[step] for b_set in self.ego_reach_set[ego_state.time_step]]

            # 由于不同 polygon 顶点数量不同，先做 padding
            len_max = max(len(b_set[step]) for b_set in self.ego_reach_set[ego_state.time_step])
            b_poly_pygeos = polygon_padding(len_max, b_poly_pygeos)

            # 转成 pygeos polygon
            b_poly_pygeos = pygeos.polygons(b_poly_pygeos)

            # b_poly_pygeos_test = [pygeos.polygons(b_set[step]) for b_set in self.ego_reach_set[ego_state.time_step]]

            # 对该 step 下所有 ego polygon 做 union，得到一个整体禁止区域
            b_poly_pygeos = pygeos.union_all(b_poly_pygeos)

            # b[step] = b_poly
            # 缓存到字典中，key 是未来 step
            b_dict[step] = b_poly_pygeos

        # uncomment to log ego
        # self.reach_sets[ego_state.time_step][self.ego_id] = self.ego_reach_set[ego_state.time_step]

        # 遍历所有障碍物
        for obstacle in obstacles:
            o_id = obstacle.obstacle_id

            # 跳过 ego 自己
            if o_id != self.ego_id:
                # get all lanelet ids of obstacle
                # ---------------------------------------------------------
                # 根据障碍物当前时刻的位置和朝向，查找其所在的 lanelet id 列表
                # 注意这里使用的是 ego 当前时间步对应的障碍物预测/轨迹状态
                # ---------------------------------------------------------
                l_ids = find_lanelet_by_position_and_orientation(
                    self.scenario.lanelet_network,
                    obstacle.prediction.trajectory.state_list[ego_state.time_step].position,
                    obstacle.prediction.trajectory.state_list[ego_state.time_step].orientation,
                )

                # 当前已经缓存过 ReachSetSimple 对象的 lanelet id
                all_ids = [int(i) for i in self.reach_set_objs.keys()]

                # 找出当前障碍物所在 lanelet 中，之前还没有建立 ReachSetSimple 对象的部分
                new_ids = set(l_ids) - set(all_ids)

                # 对所有新 lanelet 进行 ReachSetSimple 对象初始化
                for l_id in new_ids:
                    all_ids = [int(i) for i in self.reach_set_objs.keys()]
                    if l_id not in all_ids:
                        # if new lanelet, create new ReachSetSimple objects
                        # -------------------------------------------------
                        # 如果是新 lanelet：
                        # 1. 找出与其同方向横向相邻的 lanelet（parallel lanelets）
                        # 2. 计算该 lanelet 及其若干 successor 的边界组合
                        # 3. 为每一种边界组合生成一个 ReachSetSimple 对象
                        # 4. 同方向横向相邻的 lanelet 共用这组 ReachSetSimple 对象
                        # -------------------------------------------------
                        (parallel_lanelets, _, _) = self._get_parallel_lanelets(l_id)

                        bounds = self._calc_bounds_rec(
                            lanelet_id=l_id,
                            depth=self.reach_set_params["parameters"]["depth"],
                        )

                        # 先给 parallel lanelets 分配空列表
                        for lnlet_id in parallel_lanelets:
                            self.reach_set_objs[lnlet_id] = []

                        # create a new object for each boundry
                        # 对每一对左右边界，创建一个 ReachSetSimple 对象
                        for (l, r) in bounds:
                            # reach set object trimmed to lanelet bounds
                            obj = reachable_set_simple.ReachSetSimple(
                                bound_l=l, bound_r=r
                            )
                            # same bounds for laterally adjacent lanes in same direction
                            # 同方向横向相邻车道共用相同边界对象
                            for lnlet_id in parallel_lanelets:
                                self.reach_set_objs[lnlet_id].append(obj)

                # 为当前时刻、当前障碍物初始化可达集列表
                self.reach_sets[ego_state.time_step][o_id] = []

                # call simple_reachable_set() before for loop, avoid unnecessary repeating
                # ---------------------------------------------------------
                # 先基于障碍物当前状态生成“简单可达集” srs
                # 这是一个没有考虑车道边界裁剪的基础可达集
                # ---------------------------------------------------------
                srs = simple_reachable_set(
                    obj_pos=obstacle.prediction.trajectory.state_list[
                        ego_state.time_step
                    ].position,
                    obj_heading=obstacle.prediction.trajectory.state_list[
                        ego_state.time_step
                    ].orientation,
                    obj_vel=obstacle.prediction.trajectory.state_list[
                        ego_state.time_step
                    ].velocity,
                    obj_length=obstacle.obstacle_shape.length,
                    obj_width=obstacle.obstacle_shape.width,
                    dt=self.reach_set_params["parameters"]["dt"],
                    t_max=self.reach_set_params["parameters"]["t_max"],
                    a_max=self.reach_set_params["parameters"]["a_max"]
                )

                # 将 srs 每个 step 的 polygon 批量转为 pygeos polygon
                srs_t = pygeos.polygons([srs[t_key] for t_key in srs.keys()])

                # 对障碍物所在的每个 lanelet id 分别计算裁剪后的可达集
                for l_id in l_ids:
                    for reach_set_obj in self.reach_set_objs[l_id]:
                        # adjust call of calc_reach_set()
                        # -------------------------------------------------
                        # 原本这里可能是直接把状态传给 ReachSetSimple 再内部生成可达集，
                        # 现在改成：
                        # 直接把预先算好的 srs / srs_t 传进去，避免重复生成
                        # -------------------------------------------------
                        # rs = reach_set_obj.calc_reach_set(
                        #     obj_pos=obstacle.prediction.trajectory.state_list[
                        #         ego_state.time_step
                        #     ].position,
                        #     obj_heading=obstacle.prediction.trajectory.state_list[
                        #         ego_state.time_step
                        #     ].orientation,
                        #     obj_vel=obstacle.prediction.trajectory.state_list[
                        #         ego_state.time_step
                        #     ].velocity,
                        #     obj_length=obstacle.obstacle_shape.length,
                        #     obj_width=obstacle.obstacle_shape.width,
                        #     dt=self.reach_set_params["parameters"]["dt"],
                        #     t_max=self.reach_set_params["parameters"]["t_max"],
                        #     a_max=self.reach_set_params["parameters"]["a_max"],
                        # )
                        rs = reach_set_obj.calc_reach_set(srs, srs_t)

                        # 如果启用了 safe_distance，则从障碍物可达集 rs 中减去 ego 的安全距离区域
                        if "safe_distance" in self.reach_set_params["rules"]:

                            # adjust call of self._reach_set_difference()
                            # subtract safe distance polygon
                            reach_set_diffs = self._reach_set_difference(
                                rs, b_dict)

                            # 差集结果可能有多个不连通 polygon，因此 += 一个列表
                            self.reach_sets[ego_state.time_step][o_id] += reach_set_diffs
                        else:
                            # 若没有 safe_distance 规则，则直接保存 rs
                            self.reach_sets[ego_state.time_step][o_id].append(rs)

    def _calc_bounds_rec(self, lanelet_id, depth, lateral=True):
        """
        Bounds considering current and possible successor lanelets.

        Bounds of lanelet with id lanelet_id and all possible successor
        lanelets until maximum depth is reached.

        Args:
        lanelet_id (int): id of starting lanelet.
        depth (int): maximum depth of considered successor lanelets.
        lateral (bool): true indicates that laterally adjacent lanes are considered.

        Returns:
        list((np.ndarray, np.ndarray)): List of boundaries of possible lanes, which are
        tuples of left and right boundaries.
        """
        # -----------------------------------------------------------------
        # 递归计算 lanelet 的边界组合
        #
        # 给定一个起始 lanelet_id，向前递归展开 successor，直到 depth 用尽。
        # 返回值是一个列表，列表中的每个元素是：
        #   (左边界点集, 右边界点集)
        #
        # 如果 lateral=True，则会将当前 lanelet 同方向横向相邻 lanelet 一起并入考虑
        # 也就是将多条并行车道的最外侧左右边界作为整体边界。
        # -----------------------------------------------------------------

        # 深度小于 0 时停止递归
        if depth < 0:
            return []

        # 保存所有边界组合
        bound_list = []

        # 如果不考虑横向相邻 lanelet
        if not lateral:
            lnlet = self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
            successors = lnlet.successor
            bound_l = lnlet.left_vertices
            bound_r = lnlet.right_vertices
        else:
            # get lanelet bounds
            # -------------------------------------------------------------
            # 若考虑横向相邻 lanelet：
            # 1. 获取该 lanelet 所在同方向并行车道集合
            # 2. 用最左车道的 left boundary 和最右车道的 right boundary
            #    作为整体边界
            # -------------------------------------------------------------
            (lanelets, bound_l, bound_r) = self._get_parallel_lanelets(lanelet_id)

            # non-parallel successor lanelets
            # -------------------------------------------------------------
            # 收集所有并行 lanelet 的 successor，
            # 然后去掉其中彼此平行同向的重复 lanelet，只保留代表项
            # -------------------------------------------------------------
            successors = set()
            for lnlet in lanelets:
                suc = self.scenario.lanelet_network.find_lanelet_by_id(lnlet).successor
                if suc is not None:
                    successors = successors.union(set(suc))
            successors = self._get_non_parallel_lanelets(successors)

        # 如果已经到底（depth == 0）或者没有 successor，当前边界作为终止结果返回
        if depth == 0 or len(successors) == 0:
            bound_list.append((bound_l, bound_r))
            return bound_list

        # append bounds of successor lanelets
        # -------------------------------------------------------------
        # 对每个 successor 递归展开，
        # 并将当前 lanelet 边界与 successor 的未来边界首尾拼接
        # -------------------------------------------------------------
        for successor in successors:
            bounds = self._calc_bounds_rec(successor, depth - 1, lateral)
            for (l, r) in bounds:
                bound_list.append((np.append(bound_l, l, 0), np.append(bound_r, r, 0)))

        return bound_list

    def _get_parallel_lanelets(self, lanelet_id):
        """
        Get all laterally adjacent lanelets in same direction.

        Returns:
        List of ids of laterally adjacent lanelets in same direction.
        Outmost left boundary of all adjacent lanelets.
        Outmost right boundary of all adjacent lanelets.
        """
        # -----------------------------------------------------------------
        # 获取与给定 lanelet 横向相邻且方向相同的所有 lanelet
        #
        # 返回：
        # 1. parallels: 所有同方向横向相邻 lanelet 的 id 列表
        # 2. bound_l : 最左侧整体边界
        # 3. bound_r : 最右侧整体边界
        #
        # 例如三车道同向行驶时，如果给的是中间车道，
        # 那么会返回 [左车道, 中间车道, 右车道] 以及最外侧边界。
        # -----------------------------------------------------------------

        adj_left = []
        adj_right = []
        curr = lanelet_id
        left_most = curr

        # find leftmost lanelet in same direction
        # 沿着 _adj_left_same_direction 一直往左找，直到找到最左的同向 lanelet
        while self.scenario.lanelet_network.find_lanelet_by_id(
            curr
        )._adj_left_same_direction:
            lnlet = self.scenario.lanelet_network.find_lanelet_by_id(curr)
            curr = lnlet._adj_left
            left_most = curr
            adj_left.append(curr)

        curr = lanelet_id
        right_most = curr

        # find rightmost lanelet in same direction
        # 沿着 _adj_right_same_direction 一直往右找，直到找到最右的同向 lanelet
        while self.scenario.lanelet_network.find_lanelet_by_id(
            curr
        )._adj_right_same_direction:
            lnlet = self.scenario.lanelet_network.find_lanelet_by_id(curr)
            curr = lnlet._adj_right
            right_most = curr
            adj_right.append(curr)

        # 整体顺序：左侧相邻 + 当前 lanelet + 右侧相邻
        parallels = adj_left + [lanelet_id] + adj_right

        # 最左外边界取最左 lanelet 的 left_vertices
        lnlet = self.scenario.lanelet_network.find_lanelet_by_id(left_most)
        bound_l = lnlet.left_vertices

        # 最右外边界取最右 lanelet 的 right_vertices
        lnlet = self.scenario.lanelet_network.find_lanelet_by_id(right_most)
        bound_r = lnlet.right_vertices

        return parallels, bound_l, bound_r

    def _get_non_parallel_lanelets(self, lanelets):
        """
        Get lanelets which aren't laterally adjacent and in same direction.

        Get lanelets such that they are pairwise
        not laterally adjacent or not in the same direction.

        Returns:
        List of lanelet ids.
        """
        # -----------------------------------------------------------------
        # 从一组 lanelet 中去掉“彼此横向平行且同方向”的重复项
        #
        # 举例：
        # 如果 successor 集合里同时有三条并行同向车道，
        # 那么这里只保留其中一个代表 lanelet。
        #
        # 这样做是为了减少递归边界展开时的重复计算。
        # -----------------------------------------------------------------
        final = list(lanelets)

        for lnlet in lanelets:
            if lnlet in final:
                (parallel, _, _) = self._get_parallel_lanelets(lnlet)
                final = [l for l in final if l not in parallel or l == lnlet]

        return set(final)

    def _reach_set_difference(self, a, b_dict):
        """
        Calculate the difference between two reachable sets.

        Subtracts b from a.
        """
        # -----------------------------------------------------------------
        # 计算 reachable set 差集：a - b
        #
        # 其中：
        # a      : 某个障碍物的 reachable set（按 step 存字典）
        # b_dict : ego 的安全距离区域（按 step 存 pygeos polygon）
        #
        # 返回值是一个 reachable set 列表，
        # 因为差集后可能会被切成多个不连通 polygon。
        # -----------------------------------------------------------------
        rs_list_pygeos = []

        # 将 a 中每个 step 的 polygon 提取出来
        a_poly_pygeos = [a[step] for step in a]

        # 做 padding 以便批量构造 pygeos polygon
        len_max = max(len(a[step]) for step in a)
        a_poly_pygeos = polygon_padding(len_max, a_poly_pygeos)

        # 转成 pygeos polygon 数组
        a_poly_pygeos = pygeos.polygons(a_poly_pygeos)

        # a_poly_pygeos_test = [pygeos.polygons(a[step]) for step in a]

        # 对每个 step 做 difference：障碍物 reachable set 减去 ego 安全距离区域
        diff_pygeos = pygeos.difference(a_poly_pygeos, [b_dict[step] for step in a])

        # 保存原 step 顺序
        key_list = list(a.keys())

        # 将 difference 结果逐个转换回 reachable set 格式
        for i in range(len(key_list)):
            rs_list_pygeos += self._geom_to_reach_set(diff_pygeos[i], key_list[i])

        return rs_list_pygeos

    def _add_safe_distance(self, rs, rs_obj, safe_distance):
        """
        Extend a reachable set in longitudinal direction.

        Extend a reachable set in longitudinal direction.
        Applys the two-second heuristic for safe distances.
        """
        # -----------------------------------------------------------------
        # 对一个 reachable set 做“安全距离扩张”
        #
        # 输入：
        # rs            : 原 reachable set（按 step 存 polygon）
        # rs_obj        : ReachSetSimple 对象，用于获取车道裁剪 patch
        # safe_distance : 需要扩张的安全距离
        #
        # 扩张逻辑：
        # 1. 对 reachable set polygon 做 buffer(safe_distance)
        # 2. 如果有 patch（车道边界裁剪区域），则再与 patch 求交，防止越界
        # 3. 返回扩张后的 reachable set
        # -----------------------------------------------------------------

        # replace shapely operations by pygeos
        # Lane bounds used for reach set
        patch = rs_obj.intersection_patch_pygeos
        extended = {}

        # 提取每个 step 的 polygon 点
        poly_pygeos = [rs[step] for step in rs]

        # 做 padding 后统一构造 polygon
        len_max = max(len(rs[step]) for step in rs)
        poly_pygeos = polygon_padding(len_max, poly_pygeos)
        poly_pygeos = pygeos.polygons(poly_pygeos)

        # poly_pygeos = [pygeos.polygons(rs[step]) for step in rs]

        # 对每个 polygon 做 buffer，并取其 exterior ring 再还原成 polygon
        # safe_distance 表示扩张半径
        buffer_pygeos = pygeos.polygons(pygeos.get_exterior_ring(pygeos.buffer(poly_pygeos, safe_distance, quadsegs=16)))

        # 保存 step 顺序
        key_list = list(rs.keys())

        # 如果有车道裁剪 patch，就把 buffer 后的 polygon 再和 patch 做 intersection
        if patch is not None:
            intersection_pygeos = pygeos.intersection(patch, buffer_pygeos)
            for i in range(len(key_list)):
                # trim extended with patch
                # if intersection.geom_type == "Polygon" and not intersection.is_empty:
                if pygeos.get_type_id(intersection_pygeos[i]) == 3 and not pygeos.is_empty(intersection_pygeos[i]):
                    # convert intersection poly to points
                    outline = pygeos.get_coordinates(pygeos.get_exterior_ring(intersection_pygeos[i]))
                    extended[key_list[i]] = outline
                else:
                    # raise ValueError(
                    #     "Unhandled geometry type: " + repr(intersection.geom_type)
                    # )
                    raise ValueError(
                        "Unhandled geometry type: " + str(pygeos.get_type_id(intersection_pygeos[i]))
                    )
        else:
            # 如果没有 patch，则直接使用 buffer 结果
            for i in range(len(key_list)):
                if pygeos.get_type_id(buffer_pygeos[i]) == 3 and not pygeos.is_empty(buffer_pygeos[i]):
                    outline = pygeos.get_coordinates(pygeos.get_exterior_ring(buffer_pygeos[i]))
                    extended[key_list[i]] = outline
                else:
                    extended[key_list[i]] = rs[key_list[i]]

        return extended

    def _geom_to_reach_set(self, geometry, step):
        """
        Convert shapely geometry to reachable set.

        Convert shapely geometry to reachable set with only evaluation at step.
        """
        # -----------------------------------------------------------------
        # 将几何对象（Polygon / MultiPolygon）转换成 reachable set 格式
        #
        # 输入：
        # geometry : pygeos 几何对象
        # step     : 当前对应的预测时间步
        #
        # 返回：
        # rs_list  : reachable set 列表
        #
        # 因为 difference 后的几何对象可能是：
        # - Polygon
        # - MultiPolygon
        #
        # 所以这里统一转成 list[dict] 的 reachable set 形式。
        # -----------------------------------------------------------------
        rs_list = []

        # if polygons don't intersect, the result is a MultiPolygon
        # if geometry.geom_type == "Polygon":
        if pygeos.get_type_id(geometry) == 3:
            # geometry 是 Polygon
            if pygeos.is_empty(geometry):
                return rs_list

            # convert difference to points
            rs = {}
            rs[step] = self._get_points_of_polygon(geometry)
            rs_list.append(rs)

        # elif geometry.geom_type == "MultiPolygon":
        elif pygeos.get_type_id(geometry) == 6:
            # geometry 是 MultiPolygon，递归拆成多个 Polygon
            for i in range(pygeos.get_num_geometries(geometry)):
                rs_list += self._geom_to_reach_set(pygeos.get_geometry(geometry, i), step)
        else:
            # 其他几何类型目前不支持
            # raise ValueError("Unhandled geometry type: " + repr(geometry.geom_type))
            raise ValueError("Unhandled geometry type: " + str(pygeos.get_type_id(geometry)))

        return rs_list

    def _get_points_of_polygon(self, polygon):
        """
        Convert shapely polygon to numpy array.

        Convert a shapely polygon to a numpy array
        with columns [x,y].
        """
        # -----------------------------------------------------------------
        # 将 pygeos polygon 转为 numpy 点数组
        #
        # 返回值格式：
        # np.array([...])，每一行是一个 [x, y]
        #
        # 这里会同时提取：
        # - exterior ring（外轮廓）
        # - interior rings（内孔洞，如果有）
        # -----------------------------------------------------------------

        # replace polygon coordinates extraction for shapely by pygeos
        interior_line = []
        for i in range(pygeos.get_num_interior_rings(polygon)):
            interior_line = pygeos.get_coordinates(pygeos.get_interior_ring(polygon, i)).tolist()

        outline = pygeos.get_coordinates(pygeos.get_exterior_ring(polygon)).tolist()

        return np.array(outline + interior_line)

    def _ego_reach_set(self, ego_state):
        """
        Compute the safe distance polygons for safe distance rule.

        Compute the reachable sets of the ego vehicle and extend it
        to account for the safe distance, resulting in the safe distance polygons.

        """
        # -----------------------------------------------------------------
        # 计算 ego 车辆的扩张可达集（安全距离区域）
        #
        # 主要步骤：
        # 1. 确定 ego 当前所在 lanelet
        # 2. 递归构造 lanelet 边界（不考虑横向相邻车道）
        # 3. 生成 ego 的简单可达集
        # 4. 根据速度计算安全距离
        # 5. 对 ego 可达集做 buffer 扩张并裁剪到车道 patch 内
        # -----------------------------------------------------------------

        # 通过 ego 当前位置和朝向找到当前所在 lanelet，取第一个匹配项
        l_id = find_lanelet_by_position_and_orientation(
            self.scenario.lanelet_network, ego_state.position, ego_state.orientation
        )[0]

        # bounds, ignoring laterally adjacent lanes
        # -------------------------------------------------------------
        # 计算 ego 所在 lanelet 的边界展开，但 lateral=False，
        # 即不把横向并行车道并入，只沿当前车道前向递归 successor。
        # -------------------------------------------------------------
        bounds = self._calc_bounds_rec(
            lanelet_id=l_id,
            depth=self.reach_set_params["parameters"]["depth"],
            lateral=False,
        )

        # 初始化 ego 的单 lanelet ReachSetSimple 容器
        self.reach_set_objs_single[l_id] = []

        # 对每一种边界组合，构造 ReachSetSimple 对象
        for (l, r) in bounds:
            # reach set object trimmed to lanelet bounds
            obj = reachable_set_simple.ReachSetSimple(bound_l=l, bound_r=r)
            self.reach_set_objs_single[l_id].append(obj)

        # 初始化当前时刻 ego_reach_set 容器
        self.ego_reach_set[ego_state.time_step] = []

        # call simple_reachable_set() before for loop, avoid unnecessary repeating
        # calculate reachable set
        # -------------------------------------------------------------
        # 先生成 ego 的基础可达集 srs
        # 注意这里 a_max=0.01，几乎等价于近似匀速
        # -------------------------------------------------------------
        srs = simple_reachable_set(
            obj_pos=ego_state.position,
            obj_heading=ego_state.orientation,
            obj_vel=ego_state.velocity,
            obj_length=self.ego_length,
            obj_width=self.ego_width,
            dt=self.reach_set_params["parameters"]["dt"],
            t_max=self.reach_set_params["parameters"]["t_max"],
            a_max=0.01,
        )

        # 批量转为 pygeos polygon
        srs_t = pygeos.polygons([srs[t_key] for t_key in srs.keys()])

        # calculate safe_distance before for loop
        # 2-second safe distance heuristic
        # -------------------------------------------------------------
        # 根据 ego 当前速度，按经验规则确定安全距离因子：
        # - 城市低速：0.75 * v
        # - 一般道路：1.0 * v
        # - 高速道路：2.0 * v
        #
        # 最终 safe_distance 还会乘以配置中的 safe_distance_frac。
        # -------------------------------------------------------------
        if ego_state.velocity <= 8:
            safe_distance_factor = 0.75
        elif ego_state.velocity <= 15:
            safe_distance_factor = 1.0
        else:
            safe_distance_factor = 2.0

        min_safe_distance = safe_distance_factor * ego_state.velocity
        safe_distance = min_safe_distance * self.reach_set_params["rules"]["safe_distance"]["safe_distance_frac"]

        # 对每一个边界分支下的 ego ReachSetSimple 做可达集和扩张
        for reach_set_obj in self.reach_set_objs_single[l_id]:
            # reach set for ego assumes constant acceleration
            # essentially, this is the center of the vehicle,
            # given its current velocity
            rs = reach_set_obj.calc_reach_set(srs, srs_t)

            # 进行安全距离扩张
            extended_rs = self._add_safe_distance(
                rs,
                reach_set_obj,
                safe_distance
            )

            # 保存到 ego_reach_set
            self.ego_reach_set[ego_state.time_step].append(extended_rs)


def load_reach_set_json():
    """
    Load reachable_set.json with reach set parameters and rules.

    Returns:
        Dict: reach set parameters and rules from reachable_set.json
    """
    # ---------------------------------------------------------------------
    # 读取 reachable_set.json 配置文件
    #
    # 该配置文件通常包含：
    # - dt      : reachable set 时间分辨率
    # - t_max   : 最大预测时间范围
    # - a_max   : 默认最大加速度
    # - depth   : lanelet successor 展开深度
    # - rules   : 规则参数，例如 safe_distance
    # ---------------------------------------------------------------------
    reach_set_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "reachable_set.json",
    )

    with open(reach_set_config_path, "r") as f:
        jsondata = json.load(f)

    return jsondata
