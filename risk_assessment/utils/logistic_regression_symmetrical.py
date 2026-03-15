
"""Logistic regression harm actual functions for symmetrical models."""

import numpy as np


def get_protected_inj_prob_log_reg_complete_sym(velocity,
                                                angle,
                                                coeff):
    """
    LR12S.

    Get the injury probability via logistic regression for 12 considered
    impact areas. Area coefficients are set symmetrically.

    Args:
        velocity (float): delta between pre-crash and post-crash velocity
            in m/s.
        angle (float): crash angle in rad.
        coeff (Dict): Risk parameters. Read from risk_parameters.json.

    Returns:
        float: MAIS 3+ probability
    """
    # get angle coefficient
    for i in range(len(angle)):
        if -15 / 180 * np.pi < angle[i] < 15 / 180 * np.pi:  # impact 12
            angle[i] = 0
        elif 15 / 180 * np.pi <= angle[i] < 45 / 180 * np.pi:  # impact 11
            angle[i] = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_1_11"]
        elif -15 / 180 * np.pi >= angle[i] > -45 / 180 * np.pi:  # impact 1
            angle[i] = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_1_11"]
        elif 45 / 180 * np.pi <= angle[i] < 75 / 180 * np.pi:  # impact 10
            angle[i] = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_2_10"]
        elif -45 / 180 * np.pi >= angle[i] > -75 / 180 * np.pi:  # impact 2
            angle[i] = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_2_10"]
        elif 75 / 180 * np.pi <= angle[i] < 105 / 180 * np.pi:  # impact 9
            angle[i] = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_3_9"]
        elif -75 / 180 * np.pi >= angle[i] > -105 / 180 * np.pi:  # impact 3
            angle[i] = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_3_9"]
        elif 105 / 180 * np.pi <= angle[i] < 135 / 180 * np.pi:  # impact 8
            angle[i] = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_4_8"]
        elif -105 / 180 * np.pi >= angle[i] > -135 / 180 * np.pi:  # impact 4
            angle[i] = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_4_8"]
        elif 135 / 180 * np.pi <= angle[i] < 165 / 180 * np.pi:  # impact 7
            angle[i] = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_5_7"]
        elif -135 / 180 * np.pi >= angle[i] > -165 / 180 * np.pi:  # impact 5
            angle[i] = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_5_7"]
        else:  # impact 6
            angle[i] = coeff["log_reg"]["complete_sym_angle_areas"]["Imp_6"]

    # logistic regression model
    p_mais = 1 / (1 + np.exp(- coeff["log_reg"]["complete_sym_angle_areas"]
                             ["const"] - coeff["log_reg"]
                             ["complete_sym_angle_areas"]["speed"] * velocity -
                             angle))

    return p_mais


def get_protected_inj_prob_log_reg_reduced_sym(velocity,
                                               angle,
                                               coeff):
    """
    LR4S.

    Get the injury probability via logistic regression for 4 considered
    impact areas. Area coefficients are set symmetrically.

    Args:
        velocity (float): delta between pre-crash and post-crash velocity
            in m/s.
        angle (float): crash angle in rad.
        coeff (Dict): Risk parameters. Read from risk_parameters.json.

    Returns:
        float: MAIS 3+ probability
    """
    # ----------------------------------------------------------------------
    # 函数目标
    # ----------------------------------------------------------------------
    # 该函数用于：
    # 使用一个“简化的对称四区域 logistic regression 模型”
    # 来估计受保护交通参与者（通常指车内乘员）的严重伤害概率。
    #
    # 这里的输出 p_mais 表示：
    #   MAIS 3+（Abbreviated Injury Scale 三级及以上）伤害概率
    #
    # 输入变量中最关键的有两个：
    # 1. velocity:
    #    碰撞前后速度差（delta-v），通常用来衡量碰撞强度
    #
    # 2. angle:
    #    碰撞角，用于判断碰撞发生在车辆的哪个区域：
    #    - 正前方
    #    - 左侧（驾驶员侧）
    #    - 右侧
    #    - 后方
    #
    # “reduced_sym” / “LR4S”的含义：
    # - reduced: 使用简化版本的区域划分
    # - sym: 左右侧使用相同系数（对称处理）
    # - 4: 共考虑 4 个碰撞区域
    #
    # 最终模型形式是一个 logistic regression：
    #
    #   p = 1 / (1 + exp(-(const + beta_speed * velocity + beta_angle)))
    #
    # 其中 beta_angle 不是连续角度项，而是先根据碰撞角所在区域，
    # 映射成对应的区域系数。
    # ----------------------------------------------------------------------

    # get angle coefficient
    # ----------------------------------------------------------------------
    # 定义碰撞区域划分的角度阈值
    #
    # t_a = 45°
    # t_b = 135°
    #
    # 也就是说，碰撞角会按以下方式划分：
    # - (-45°, 45°)                -> front crash（正面碰撞）
    # - [45°, 135°)                -> driver-side crash（左侧碰撞）
    # - (-135°, -45°]              -> right-side crash（右侧碰撞）
    # - 其余角度                   -> rear crash（追尾 / 后向碰撞）
    #
    # 注意这里的 angle 单位是弧度，因此先把角度转换成 rad。
    # ----------------------------------------------------------------------
    t_a = 45 / 180 * np.pi
    t_b = 3 * t_a

    # ----------------------------------------------------------------------
    # unpack 变量用于记录：
    # 输入的 angle 是否原本是单个 float。
    #
    # 这是因为后面的代码统一按“可迭代序列”处理 angle，
    # 所以如果 angle 只是一个 float，就临时把它包装成长度为 1 的列表。
    #
    # 最后计算结束后，再把结果拆回单个标量返回。
    # ----------------------------------------------------------------------
    unpack = False

    # 如果 angle 是单个 float，而不是数组 / 列表
    if isinstance(angle, float):
        # 把单个标量 angle 包装成列表，方便统一处理
        angle = [angle]
        # 记录“之后需要拆包回单值”
        unpack = True

    # ----------------------------------------------------------------------
    # 遍历每一个 angle，根据碰撞角所在区域，把“原始角度值”替换成“区域系数”
    #
    # 也就是说，从这里开始，angle[i] 不再表示碰撞角本身，
    # 而变成 logistic regression 里的区域特征值（区域对应的系数）。
    # ----------------------------------------------------------------------
    for i in range(len(angle)):

        # front crash
        # ------------------------------------------------------------------
        # 若碰撞角落在 (-45°, 45°) 之间，
        # 认为是正面碰撞。
        #
        # 这里把 angle[i] 直接置为 0，
        # 相当于把“front crash”作为基准区域（baseline category），
        # 不额外增加区域偏置项。
        # ------------------------------------------------------------------
        if -t_a < angle[i] < t_a:
            angle[i] = 0

        # driver-side crash
        # ------------------------------------------------------------------
        # 若碰撞角在 [45°, 135°) 之间，
        # 认为碰撞发生在驾驶员侧（左侧）。
        #
        # 将其映射为 coeff["log_reg"]["reduced_sym_angle_areas"]["side"]
        # 即 logistic regression 里“侧面碰撞”的区域系数。
        # ------------------------------------------------------------------
        elif t_a <= angle[i] < t_b:
            angle[i] = coeff["log_reg"]["reduced_sym_angle_areas"]["side"]

        # right-side crash
        # ------------------------------------------------------------------
        # 若碰撞角在 (-135°, -45°] 之间，
        # 认为碰撞发生在右侧。
        #
        # 这里使用和左侧相同的 side 系数，
        # 体现“sym（左右对称）”的建模假设。
        # ------------------------------------------------------------------
        elif -t_a >= angle[i] > -t_b:
            angle[i] = coeff["log_reg"]["reduced_sym_angle_areas"]["side"]

        # rear crash
        # ------------------------------------------------------------------
        # 剩余情况统统视为后向碰撞（rear crash）。
        #
        # 例如角度接近 ±180° 时，通常代表从车后方向发生碰撞。
        #
        # 对应的区域系数取：
        # coeff["log_reg"]["reduced_sym_angle_areas"]["rear"]
        # ------------------------------------------------------------------
        else:
            angle[i] = coeff["log_reg"]["reduced_sym_angle_areas"]["rear"]

    # logistic regression model
    # ----------------------------------------------------------------------
    # 使用 logistic regression 公式计算 MAIS 3+ 的概率：
    #
    #   p_mais = 1 / (1 + exp(-(const + beta_speed * velocity + beta_angle)))
    #
    # 这里代码写成：
    #
    #   p = 1 / (1 + exp(- const - beta_speed * velocity - angle))
    #
    # 注意这里 angle 此时已经不是几何角度，而是“区域系数”：
    # - front -> 0
    # - side  -> side coefficient
    # - rear  -> rear coefficient
    #
    # 因此这个模型可以理解为：
    # - baseline 为正面碰撞
    # - 侧面碰撞和后向碰撞通过额外系数调整伤害概率
    #
    # velocity 越大，通常碰撞更严重，伤害概率会升高；
    # 区域系数则体现不同撞击部位对严重伤害概率的影响。
    # ----------------------------------------------------------------------
    p_mais = 1 / (1 + np.exp(- coeff["log_reg"]["reduced_sym_angle_areas"]
                             ["const"] - coeff["log_reg"]
                             ["reduced_sym_angle_areas"]["speed"] * velocity -
                             angle))

    # ----------------------------------------------------------------------
    # 如果最开始输入的是单个 float，
    # 那么当前 p_mais 也是长度为 1 的列表 / 数组。
    #
    # 为了保持函数接口友好，这里把它拆回单个标量。
    # ----------------------------------------------------------------------
    if unpack:
        p_mais = p_mais[0]

    # 返回严重伤害概率（MAIS 3+ probability）
    return p_mais



# change add a parameter
def get_protected_inj_prob_log_reg_ignore_angle(velocity,
                                                coeff,
                                                angle=0):
    """
    LR1S.

    Get the injury probability via logistic regression. Impact areas are not
    considered.

    Args:
        velocity (float): delta between pre-crash and post-crash velocity
            in m/s.
        coeff (Dict): Risk parameters. Read from risk_parameters.json.

    Returns:
        float: MAIS 3+ probability
    """
    # logistic regression model
    p_mais = 1 / (1 + np.exp(- coeff["log_reg"]["ignore_angle"]["const"] -
                             coeff["log_reg"]["ignore_angle"]["speed"] *
                             velocity))

    return p_mais
