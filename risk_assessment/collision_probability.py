#!/user/bin/env python

"""Calculate the collision probability of a trajectory and predictions."""

import os
import sys
import numpy as np
from scipy.stats import multivariate_normal, mvn
from scipy.spatial.distance import mahalanobis
import commonroad_dc.pycrcc as pycrcc
from beliefplanning.risk_assessment.helpers.coll_prob_helpers import (
    distance,
    get_unit_vector,
)

module_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(module_path)


def get_collision_probability(traj, predictions: dict, vehicle_params, safety_margin=1.0):
    """
    Calculate the collision probabilities of a trajectory and predictions.

    Args:
        traj (FrenetTrajectory): Considered trajectory.
        predictions (dict): Predictions of the visible obstacles.
        vehicle_params (VehicleParameters): Parameters of the considered
            vehicle.

    Returns:
        dict: Collision probability of the trajectory per time step with the
            prediction for every visible obstacle.
    """
    obstacle_ids = list(predictions.keys())
    collision_prob_dict = {}
    safety_length = vehicle_params.l
    safety_width = vehicle_params.w

    for obstacle_id in obstacle_ids:
        mean_list = predictions[obstacle_id]['pos_list']
        cov_list = predictions[obstacle_id]['cov_list']
        yaw_list = predictions[obstacle_id]['orientation_list']
        length = predictions[obstacle_id]['shape']['length']
        probs = []
        for i in range(1, len(traj.x)):

            # only calculate probability as the predicted obstacle is visible
            if i < len(mean_list):

                # get the current position of the ego vehicle
                ego_pos = [traj.x[i], traj.y[i]]

                # get the mean and the covariance of the prediction
                mean = mean_list[i - 1]

                # get the position of the front and the back of the vehicle
                mean_front = mean + get_unit_vector(yaw_list[i]) * length / 2
                mean_back = mean - get_unit_vector(yaw_list[i]) * length / 2

                # if the distance between the vehicles is bigger than 5 meters,
                # the collision probability is zero
                # avoids huge computation times
                if (
                        min(
                            distance(ego_pos, mean),
                            distance(ego_pos, mean_front),
                            distance(ego_pos, mean_back),
                        )
                        > 5.0
                ):
                    prob = 0.0
                else:
                    cov = cov_list[i - 1]

                    # if the covariance is a zero matrix, the prediction is
                    # derived from the ground truth
                    # a zero matrix is not singular and therefore no valid
                    # covariance matrix
                    allcovs = [cov[0][0], cov[0][1], cov[1][0], cov[1][1]]
                    if all(covi == 0 for covi in allcovs):
                        cov = [[0.1, 0.0], [0.0, 0.1]]

                    prob = 0.0
                    means = [mean, mean_front, mean_back]

                    # the occupancy of the ego vehicle is approximated by three
                    # axis aligned rectangles
                    # get the center points of these three rectangles
                    center_points = get_center_points_for_shape_estimation(
                        length=safety_length,
                        width=safety_width,
                        orientation=traj.yaw[i],
                        pos=[traj.x[i], traj.y[i]],
                    )

                    # in order to get the cdf, the upper right point and the
                    # lower left point of every rectangle is needed
                    urs = []
                    lls = []
                    for center_point in center_points:
                        ur, ll = get_upper_right_and_lower_left_point(
                            center_point,
                            length=safety_length / 3,
                            width=safety_width,
                        )
                        urs.append(ur)
                        lls.append(ll)

                    # the probability distribution consists of the partial
                    # multivariate normal distributions
                    # this allows to consider the length of the predicted
                    # obstacle
                    # consider every distribution
                    for mu in means:
                        multi_norm = multivariate_normal(mean=mu, cov=cov)
                        # add the probability of every rectangle
                        for center_point_index in range(len(center_points)):
                            prob += get_prob_via_cdf(
                                multi_norm=multi_norm,
                                upper_right_point=urs[center_point_index],
                                lower_left_point=lls[center_point_index],
                            )

            else:
                prob = 0.0
            # divide by 3 since 3 probability distributions are added up and
            # normalize the probability
            probs.append(prob / 3)
        collision_prob_dict[obstacle_id] = probs

    return collision_prob_dict


def _resolve_obstacle_mode(mode_num, obstacle_id, mode_count):
    if mode_num == 100:
        return None
    if isinstance(mode_num, dict):
        return min(mode_num.get(obstacle_id, 0), mode_count - 1)
    return min(mode_num, mode_count - 1)


def get_collision_probability_fast(traj, predictions: dict, vehicle_params, start_idx, mode_num, safety_margin=1.0):
    """
    Calculate the collision probabilities of a trajectory and predictions.

    Args:
        traj (FrenetTrajectory): Considered trajectory.
        predictions (dict): Predictions of the visible obstacles.
        vehicle_params (VehicleParameters): Parameters of the considered
            vehicle.

    Returns:
        dict: Collision probability of the trajectory per time step with the
            prediction for every visible obstacle.
    """
    # ----------------------------------------------------------------------
    # 函数目标
    # ----------------------------------------------------------------------
    # 该函数用于快速估计：
    # “ego 车辆沿给定 FrenetTrajectory 运动时，与各个障碍物预测轨迹在每个时间步发生碰撞的概率”
    #
    # 输入：
    # - traj: ego 车的候选轨迹（通常包含 x, y, yaw 等时间序列）
    # - predictions: 各障碍物的预测结果，通常是多模态格式
    # - vehicle_params: ego 车的几何参数（长度、宽度等）
    # - start_idx: 从预测序列的哪个下标开始计算
    # - mode_num: 指定使用哪个 mode；若为特殊值（这里是 100），则遍历所有 mode
    # - safety_margin: 安全裕度参数（当前函数中没有显式使用）
    #
    # 输出：
    # collision_prob_dict[obstacle_id] = probs_list
    # 其中 probs_list 是一个列表：
    # - 若遍历所有 mode，则每个元素对应一个 mode 的每时间步碰撞概率数组
    # - 若只用一个指定 mode，则列表里通常只有一个数组
    # ----------------------------------------------------------------------

    # 取出所有障碍物 ID
    obstacle_ids = list(predictions.keys())

    # 最终返回的碰撞概率字典
    collision_prob_dict = {}

    # get the current positions array of the ego vehicles
    # ----------------------------------------------------------------------
    # 将 ego 轨迹中的 x 和 y 堆叠成二维位置数组：
    # ego_pos.shape = [T, 2]
    # 每一行对应 ego 在某个时间步的位置 [x_t, y_t]
    # ----------------------------------------------------------------------
    ego_pos = np.stack((traj.x, traj.y), axis=-1)

    # offset between vehicle center point and corner point
    # ----------------------------------------------------------------------
    # 这里定义了一个用于近似 ego 占据区域的小矩形的半尺寸偏移量：
    # - 长度方向使用 vehicle_params.l / 6
    # - 宽度方向使用 vehicle_params.w / 2
    #
    # 后面 ego 车辆会被近似为 3 个小矩形，
    # 每个小矩形都用 (center_point +/- offset) 来得到包围框。
    # ----------------------------------------------------------------------
    offset = np.array([vehicle_params.l / 6, vehicle_params.w / 2])

    # iterate over obstacles
    # ----------------------------------------------------------------------
    # 逐个障碍物计算碰撞概率
    # ----------------------------------------------------------------------
    for obstacle_id in obstacle_ids:
        # probs_list 用于存该 obstacle 的所有 mode 对应的碰撞概率序列
        probs_list = []
        selected_mode = _resolve_obstacle_mode(
            mode_num=mode_num,
            obstacle_id=obstacle_id,
            mode_count=len(predictions[obstacle_id]['pos_list']),
        )

        # iterate over number of modes
        # ------------------------------------------------------------------
        # 遍历该障碍物预测中的所有 mode
        # predictions[obstacle_id]['pos_list'][mode]
        # 表示该障碍物在某个 mode 下的未来轨迹
        # ------------------------------------------------------------------
        for mode in range(len(predictions[obstacle_id]['pos_list'])):
            # This means, it is for a contingent plan.
            # ------------------------------------------------------------------
            # 如果 mode_num != 100，表示当前不是“遍历所有模式”的普通情形，
            # 而是 contingency plan 下只检查一个特定的 mode。
            # 于是直接把 mode 强制设为 mode_num。
            # ------------------------------------------------------------------
            if selected_mode is not None:
                mode = selected_mode

            # 取出该障碍物该 mode 下，从 start_idx 开始的未来均值位置序列
            mean_list = predictions[obstacle_id]['pos_list'][mode][start_idx:]

            # 取出对应的协方差序列
            cov_list = predictions[obstacle_id]['cov_list'][mode][start_idx:]

            # 取出对应的朝向序列
            yaw_list = predictions[obstacle_id]['orientation_list'][mode][start_idx:]

            # 取出障碍物长度（通常已经加过安全边界）
            length = predictions[obstacle_id]['shape']['length']

            # 用于保存该 mode 下每一个时间步的碰撞概率
            probs = []

            # mean distance calculation
            # determine the length of arrays
            # ------------------------------------------------------------------
            # ego 轨迹长度和障碍物预测长度可能不同，
            # 所以实际参与计算的时间长度取两者较小值。
            # ------------------------------------------------------------------
            min_len = min(len(traj.x), len(mean_list))

            # adjust array of the ego vehicles
            # ego_pos_array = np.stack((traj.x[1:min_len], traj.y[1:min_len]), axis=-1)
            # ------------------------------------------------------------------
            # 取 ego 位置数组的前 min_len 个点，与 obstacle 预测对齐
            # ego_pos_array.shape = [min_len, 2]
            # ------------------------------------------------------------------
            ego_pos_array = ego_pos[0:min_len]

            # get the positions array of the front and the back of the obstacle vehicle
            # ------------------------------------------------------------------
            # 为了考虑障碍物车辆的长度，这里不仅使用障碍物中心点 mean_array，
            # 还构造了障碍物前端点和后端点：
            #
            # mean_deviation_array: 根据障碍物朝向 yaw_list，计算沿车身纵向的偏移向量
            # 长度为 obstacle.length / 2
            #
            # mean_front_array = 中心点 + 纵向偏移
            # mean_back_array  = 中心点 - 纵向偏移
            #
            # 这样后面就用 3 个高斯分布（中心、前端、后端）来近似障碍物的占据概率。
            # ------------------------------------------------------------------
            mean_deviation_array = np.stack((np.cos(yaw_list[0:min_len]), np.sin(yaw_list[0:min_len])),
                                            axis=-1) * length / 2

            # 障碍物中心点序列
            mean_array = np.array(mean_list[:min_len])

            # 障碍物前端点序列
            mean_front_array = mean_array + mean_deviation_array

            # 障碍物后端点序列
            mean_back_array  = mean_array - mean_deviation_array

            # total_mean_array =  mean_front_array, mean_back_array))
            # ------------------------------------------------------------------
            # 将中心、前端、后端三个点堆成一个数组：
            # total_mean_array.shape = [3, min_len, 2]
            #
            # 第 0 维表示三个“代表性均值点”：
            #   0 -> 中心
            #   1 -> 前端
            #   2 -> 后端
            # ------------------------------------------------------------------
            total_mean_array = np.array([mean_array, mean_front_array, mean_back_array])

            # distance from ego vehicle
            # ------------------------------------------------------------------
            # 计算 ego 每个时间步位置到障碍物三个代表点（中心/前/后）的欧氏距离
            #
            # total_mean_array      : [3, min_len, 2]
            # ego_pos_array         : [min_len, 2]
            # 广播后 distance_array : [3, min_len]
            # ------------------------------------------------------------------
            distance_array = total_mean_array - ego_pos_array
            distance_array = np.sqrt(distance_array[:, :, 0] ** 2 + distance_array[:, :, 1] ** 2)

            # min distance of each column
            # ------------------------------------------------------------------
            # 对每个时间步，取 ego 到障碍物三个代表点中的最小距离
            # min_distance_array.shape = [min_len]
            # ------------------------------------------------------------------
            min_distance_array = distance_array.min(axis=0)

            # bool: whether min distance is larger than 5.0
            # ------------------------------------------------------------------
            # 将最小距离转为布尔值：
            # True  -> 距离大于 5 米
            # False -> 距离不大于 5 米
            #
            # 若距离较远，则直接认为碰撞概率为 0，避免做昂贵的概率积分计算。
            # ------------------------------------------------------------------
            min_distance_array = min_distance_array > 5.0

            # ------------------------------------------------------------------
            # 对 ego 轨迹逐时间步计算碰撞概率
            # 注意这里从 i=1 开始，不包含 traj 的第 0 个点
            # ------------------------------------------------------------------
            for i in range(1, len(traj.x)):
                # only calculate probability as the predicted obstacle is visible
                # ------------------------------------------------------------------
                # 只有当该时间步 i 仍在障碍物预测长度范围内时，才计算碰撞概率
                # 否则直接设为 0
                # ------------------------------------------------------------------
                if i < len(mean_list):
                    # if the distance between the vehicles is bigger than 5 meters,
                    # the collision probability is zero
                    # avoids huge computation times

                    # directly use previous bool result for the if statements
                    # ------------------------------------------------------------------
                    # 如果 ego 与障碍物的最小代表点距离大于 5m，
                    # 则认为碰撞概率足够小，直接近似为 0。
                    # ------------------------------------------------------------------
                    if min_distance_array[i]:
                        prob = 0.0
                    else:
                        # 当前时间步的障碍物位置协方差矩阵
                        cov = cov_list[i]

                        # if the covariance is a zero matrix, the prediction is
                        # derived from the ground truth
                        # a zero matrix is not singular and therefore no valid
                        # covariance matrix
                        # ------------------------------------------------------------------
                        # 如果协方差矩阵全是 0，说明这个 prediction 很可能来自 ground truth
                        # 但 0 矩阵并不是一个合法的用于高斯概率积分的有效协方差，
                        # 因此这里人为替换为一个小的对角协方差矩阵。
                        # ------------------------------------------------------------------
                        allcovs = [cov[0][0], cov[0][1], cov[1][0], cov[1][1]]
                        if all(covi == 0 for covi in allcovs):
                            cov = [[0.1, 0.0], [0.0, 0.1]]

                        # 初始化该时间步的总碰撞概率
                        prob = 0.0

                        # means = [mean, mean_front, mean_back]
                        # means = total_mean_array[:,i-1]

                        # the occupancy of the ego vehicle is approximated by three
                        # axis aligned rectangles
                        # get the center points of these three rectangles
                        # ------------------------------------------------------------------
                        # 用 3 个 axis-aligned 小矩形近似 ego 车辆的占据区域，
                        # 然后分别计算障碍物高斯分布落入这些矩形中的概率。
                        #
                        # get_center_points_for_shape_estimation(...) 会根据：
                        # - ego 车辆长度、宽度
                        # - ego 在该时刻的朝向 traj.yaw[i]
                        # - ego 中心位置 ego_pos_array[i]
                        #
                        # 返回 3 个小矩形的中心点。
                        # ------------------------------------------------------------------
                        center_points = get_center_points_for_shape_estimation(
                            length=vehicle_params.l,
                            width=vehicle_params.w,
                            orientation=traj.yaw[i],
                            pos=ego_pos_array[i],
                        )

                        # upper_right and lower_left points
                        # ------------------------------------------------------------------
                        # 转成 numpy 数组，方便后面加减 offset
                        # ------------------------------------------------------------------
                        center_points = np.array(center_points)

                        # in order to get the cdf, the upper right point and the lower left point of every rectangle
                        # is needed upper_right = ...
                        # ------------------------------------------------------------------
                        # 对每个小矩形，用 center_point +/- offset 得到：
                        # - upper_right: 右上角
                        # - lower_left : 左下角
                        #
                        # 后面会把每个小矩形看作二维高斯分布的积分区域。
                        # ------------------------------------------------------------------
                        upper_right = center_points + offset
                        lower_left = center_points - offset

                        # use mvn.mvnun to calculate multivariant cdf
                        # the probability distribution consists of the partial
                        # multivariate normal distributions
                        # this allows to consider the length of the predicted
                        # obstacle
                        # consider every distribution
                        # ------------------------------------------------------------------
                        # 这里核心思想：
                        #
                        # 障碍物并不是用一个单一高斯分布表示，而是用 3 个代表点
                        # （中心、前端、后端）对应的高斯分布来近似其长度影响。
                        #
                        # 对每个代表点 mu：
                        #   再对 ego 的每个小矩形做一次二维高斯积分，
                        #   得到该高斯落入该小矩形的概率。
                        #
                        # 最后把这些概率累加起来。
                        # ------------------------------------------------------------------
                        for mu in total_mean_array[:, i]:
                            for center_point_index in range(len(center_points)):
                                prob += mvn.mvnun(
                                    lower_left[center_point_index],
                                    upper_right[center_point_index],
                                    mu,
                                    cov
                                )[0]
                else:
                    # 如果 i 超出了障碍物预测长度，则该时刻碰撞概率记为 0
                    prob = 0.0

                # divide by 3 since 3 probability distributions are added up and
                # normalize the probability
                # ------------------------------------------------------------------
                # 因为上面把障碍物用 3 个分布（中心/前/后）近似，
                # 所以这里除以 3 进行归一化。
                #
                # 注意：
                # 这里并没有对 ego 的 3 个矩形再做额外平均，
                # 而是把 3 个 obstacle 分布对 3 个 ego 小矩形的概率直接累加后，再除以 3。
                # ------------------------------------------------------------------
                probs.append(prob / 3)

            # 保存该 mode 下的碰撞概率数组
            probs_list.append(np.array(probs))

            # we don't need to iterate over the mode, since this is a contingent plan
            # ------------------------------------------------------------------
            # 如果 mode_num != 100，说明只需要看指定 mode，
            # 那么这个 mode 计算完就直接 break，不必再遍历其他 mode。
            # ------------------------------------------------------------------
            if selected_mode is not None:
                break

        # 保存当前 obstacle 的所有 mode 碰撞概率
        collision_prob_dict[obstacle_id] = probs_list

    # 返回最终结果：
    # collision_prob_dict[obstacle_id] = [prob_array_mode0, prob_array_mode1, ...]
    return collision_prob_dict



def get_inv_mahalanobis_dist(traj, predictions: dict, vehicle_params, safety_margin=1.0):
    """
    Calculate the collision probabilities of a trajectory and predictions.

    Args:
        traj (FrenetTrajectory): Considered trajectory.
        predictions (dict): Predictions of the visible obstacles.
        vehicle_params (VehicleParameters): Parameters of the considered
            vehicle.

    Returns:
        dict: Collision probability of the trajectory per time step with the
            prediction for every visible obstacle.
    """
    obstacle_ids = list(predictions.keys())
    collision_prob_dict = {}

    for obstacle_id in obstacle_ids:
        mean_list = predictions[obstacle_id]['pos_list']
        cov_list = predictions[obstacle_id]['cov_list']
        inv_cov_list = np.linalg.inv(cov_list)
        inv_dist = []
        for i in range(1, len(traj.x)):
            if i < len(mean_list):
                u = [traj.x[i], traj.y[i]]
                v = mean_list[i - 1]
                iv = inv_cov_list[i - 1]
                # 1e-4 is regression param to be similar to collision probability
                inv_dist.append(1e-4 / mahalanobis(u, v, iv))
            else:
                inv_dist.append(0.0)
        collision_prob_dict[obstacle_id] = inv_dist

    return collision_prob_dict


def get_prob_via_cdf(
        multi_norm, upper_right_point: np.array, lower_left_point: np.array
):
    """
    Get CDF value.

    Get the CDF value for the rectangle defined by the upper right point and
    the lower left point.

    Args:
        multi_norm (multivariate_norm): Considered multivariate normal
            distribution.
        upper_right_point (np.array): Upper right point of the considered
            rectangle.
        lower_left_point (np.array): Lower left point of the considered
            rectangle.

    Returns:
        float: CDF value of the area defined by the upper right and the lower
            left point.
    """
    upp = upper_right_point
    low = lower_left_point
    # get the CDF for the four given areas
    cdf_upp = multi_norm.cdf(upp)
    cdf_low = multi_norm.cdf(low)
    cdf_comb_1 = multi_norm.cdf([low[0], upp[1]])
    cdf_comb_2 = multi_norm.cdf([upp[0], low[1]])
    # calculate the resulting CDF
    prob = cdf_upp - (cdf_comb_1 + cdf_comb_2 - cdf_low)

    return prob


def get_center_points_for_shape_estimation(
        length: float, width: float, orientation: float, pos: np.array
):
    """
    Get the 3 center points for axis aligned rectangles.

    Get the 3 center points for axis aligned rectangles that approximate an
    orientated rectangle.

    Args:
        length (float): Length of the oriented rectangle.
        width (float): Width of the oriented rectangle.
        orientation (float): Orientation of the oriented rectangle.
        pos (np.array): Center of the oriented rectangle.

    Returns:
        [np.array]: Array with 3 entries, every entry holds the center of one
            axis aligned rectangle.
    """
    # create the oriented rectangle
    obj = pycrcc.RectOBB(length / 2, width / 2, orientation, pos[0], pos[1])

    center_points = []
    obj_center = obj.center()
    # get the directional vector
    r_x = obj.r_x()
    # get the length
    a_x = obj.local_x_axis()
    # append three points (center point of the rectangle, center point of the
    # front third of the rectangle and center point of the back third of the
    # rectangle)
    center_points.append(obj_center)
    center_points.append(obj_center + r_x * (2 / 3) * a_x)
    center_points.append(obj_center - r_x * (2 / 3) * a_x)

    return center_points


def get_upper_right_and_lower_left_point(center: np.array, length: float, width: float):
    """
    Return upper right and lower left point of an axis aligned rectangle.

    Args:
        center (np.array): Center of the rectangle.
        length (float): Length of the rectangle.
        width (float): Width of the rectangle.

    Returns:
        np.array: Upper right point of the axis aligned rectangle.
        np.array: Lower left point of the axis aligned rectangle.
    """
    upper_right = [center[0] + length / 2, center[1] + width / 2]
    lower_left = [center[0] - length / 2, center[1] - width / 2]

    return upper_right, lower_left


def normalize_prob(prob: float):
    """
    Get a normalized value for the probability.

    Five partial linear equations are used to normalize the collision
    probability. This should avoid huge differences in the probabilities.
    Otherwise, low probabilities (e. g. 10⁻¹⁵⁰) would not be considered when
    other cost functions are used as well.
    This would result in a path planner, that does not consider risk at all if
    the risks appearing are pretty small.

    Args:
        prob (float): Initial probability.

    Returns:
        float: Resulting probability.
    """
    # dictionary with the factors of the linear equations
    factor_dict = {
        1: [0.6666666666666666, 0.33333333333333337],
        2: [1.1111111111111114, 0.28888888888888886],
        3: [10.101010101010099, 0.198989898989899],
        4: [1000.001000001, 0.0999998999999],
        5: [900000000.0000001, 0.01],
    }

    # normalize every probability with a suitable linear function
    if prob > 10 ** -1:
        return factor_dict[1][0] * prob + factor_dict[1][1]
    elif prob > 10 ** -2:
        return factor_dict[2][0] * prob + factor_dict[2][1]
    elif prob > 10 ** -4:
        return factor_dict[3][0] * prob + factor_dict[3][1]
    elif prob > 10 ** -10:
        return factor_dict[4][0] * prob + factor_dict[4][1]
    elif prob > 10 ** -70:
        return factor_dict[5][0] * prob + factor_dict[5][1]
    else:
        return 0.001

# EOF
