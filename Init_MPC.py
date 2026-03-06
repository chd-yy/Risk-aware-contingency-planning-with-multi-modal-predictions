import numpy as np
import pdb
from scipy import linalg
from PredictiveControllers import MPC, MPCParams
from MPC_branch import BranchMPC, BranchMPCParams


def initBranchMPC(n, d, N, NB, xRef, am, rm, N_lane, W):
    """
    初始化 Branch MPC（带分支的 MPC）参数集合。

    参数说明（按常见 MPC/车辆模型习惯推测）：
    - n: 状态维度（state dimension），例如 x = [vx, vy, psi, r] 或类似 4 维
    - d: 控制维度（input dimension），例如 u = [a, delta] 或 [delta, a]（看后面的 R 顺序）
    - N: MPC 预测步长（horizon length）
    - NB: 分支数（branch number），用于多情景/多轨迹分支 MPC
    - xRef: 参考轨迹/参考状态（可能是 time-varying reference）
    - am: acceleration bound（加速度幅值上限），用于约束 a ∈ [-am, am]
    - rm: steering bound（转向/转角幅值上限），用于约束 delta ∈ [-rm, rm]
    - N_lane: 车道数量（或道路宽度相关的倍数因子），这里用来算横向 y 的上界
    - W: 车宽（vehicle width），用于横向边界留安全余量（±W/2）

    返回：
    - mpcParameters: BranchMPCParams 对象，包含 Q/R/约束矩阵/参考/松弛变量设置等
    """
    # ==============================
    # 1) 状态约束：Fx * x <= bx
    # ==============================
    # Fx 的构造方式是典型的“上下界”线性不等式模板：
    # 若想约束某个量 z 满足 z <= z_max 与 z >= z_min
    # 可以写成：
    #   [ 1] z <= z_max
    #   [-1] z <= -z_min
    #
    # 这里 Fx 是 4x4，暗示状态向量 x 至少 4 维，并且对其中两个分量做上下界：
    # - 第2列(索引1)：看起来像是 y（横向位置）或某个横向量（因为它要被 N_lane、W 限制）
    # - 第4列(索引3)：看起来像是 psi（航向角）或某个角度误差（因为上下界 0.25 rad）
    #
    # Fx 每两行一组，分别是：+1 和 -1，对同一个状态分量做上/下界。
    Fx = np.array([[0., 1., 0., 0.],
                   [0., -1., 0., 0.],
                   [0., 0., 0., 1.],
                   [0., 0., 0., -1.]])

    # bx 是上面不等式的右侧边界（4x1）
    # 注意这里的注释写了 max y / min y / max psi / min psi
    # - y_max: N_lane * 3.6 - W/2
    # - y_min: W/2（因为 -x[1] <= -W/2 -> x[1] >= W/2）——不过这里写成 [-W/2] 表示 y_min = W/2？
    #
    # ⚠️ 这段特别容易“符号搞反”：
    # 第二行是 [-W/2]，对应不等式 -y <= -W/2 => y >= W/2
    # 也就是说最小 y 是 +W/2（而不是 -W/2）。
    #
    # 如果你的坐标系里 y=0 在道路左边界或右边界，这可能是刻意的；
    # 如果你以道路中心为 0，那通常 y_min 会是 -(...)，这里就可能写错了。
    bx = np.array([
        [N_lane * 3.6 - W / 2],  # y_max（给车宽留半个车宽的余量）
        [-W / 2],                # 这意味着 y >= W/2（注意符号等价关系）
        [0.25],                  # psi_max（单位 rad，约 14.3 度）
        [0.25]                   # 这里按上下界模板，理论应为 -psi_min 的形式
                                # 但写成 0.25 会导致 -psi <= 0.25 => psi >= -0.25
                                # 也就是 psi ∈ [-0.25, 0.25]（如果第三行也是 0.25）
    ])
    # 上面最后两行若配套正确，应实现：
    #   psi <= 0.25
    #   psi >= -0.25
    # 在你当前写法下：第三行 psi<=0.25，第四行 psi>=-0.25（是对的）

    # ==============================
    # 2) 控制约束：Fu * u <= bu
    # ==============================
    # 想对每个控制输入做上下界，例如：
    #   a ∈ [-am, am]
    #   delta ∈ [-rm, rm]
    # 同样可写成：
    #   +a <= am
    #   -a <= am
    #   +d <= rm
    #   -d <= rm
    #
    # 这里 Fu 用 Kronecker 积构造一个“对 2 个输入做上下界”的矩阵。
    #
    # np.array([1, -1]) 是 (2,) 向量
    # np.kron(np.eye(2), [1,-1]) 得到一个 2x4 的块结构：
    #   [[1, -1, 0,  0],
    #    [0,  0, 1, -1]]
    # 再 .T 变成 4x2：
    #   [[ 1,  0],
    #    [-1,  0],
    #    [ 0,  1],
    #    [ 0, -1]]
    # 这正对应：u0 上/下界 + u1 上/下界
    Fu = np.kron(np.eye(2), np.array([1, -1])).T

    # bu 是 4x1，对应上面 4 条不等式的右侧
    # ⚠️ 注释里写的 “-Min Acceleration / Max Acceleration” 有点迷惑，
    # 因为这种写法通常 bu 都是正的幅值上限：
    #   +a <= am
    #   -a <= am   => a >= -am
    #
    # 因此 am 应该是“加速度幅值上限”，不是 min acceleration。
    bu = np.array([
        [am],  # +a <= am   -> a 最大值
        [am],  # -a <= am   -> a 最小值为 -am
        [rm],  # +delta <= rm
        [rm]   # -delta <= rm -> delta 最小值为 -rm
    ])

    # Tuning Parameters
    # ==============================
    # 3) 代价函数权重（调参）
    # ==============================
    # Q：状态误差权重。这里是 4x4，对应 4 维状态。
    # 你注释里写了 "vx, vy, wz, epsi, s, ey"（6 个量），
    # 但 Q 实际是 4 维，这说明注释可能来自别的版本/更大状态。
    #
    # 这里的权重含义只能按你模型的状态定义来对齐：
    # - 第1项权重 0：x[0] 不惩罚（比如 vx 不惩罚，或已经靠约束/参考控制）
    # - x[1], x[2] 权重 3：中等惩罚
    # - x[3] 权重 10：更强惩罚（比如航向误差或偏航角速度）
    Q = np.diag([0., 3, 3, 10.])  # vx, vy, wz, epsi, s, ey

    # R：控制输入权重。2x2，说明控制是二维 u。
    # 注释写 "delta, a" 表明 u[0]=delta(转角), u[1]=a(加速度)
    # 但是 bu 的构造没有说明顺序；这里建议你确认系统里 u 的排列。
    #
    # R = diag([1, 100]) 表示：
    # - 对第一个控制（delta）惩罚较小
    # - 对第二个控制（a）惩罚非常大（更不愿意加减速，倾向平顺）
    R = np.diag([1, 100.0])  # delta, a

    # ==============================
    # 4) 松弛变量（slack）权重
    # ==============================
    # 这里 Qslack = [0, 300]
    # 常见含义：对某类约束允许软化（slack），并用代价惩罚 slack。
    # 数组长度 2 暗示：可能有两类 slack（例如状态约束 slack、输入约束 slack）
    # 或者对应某些特定软约束项。
    #
    # 第1项是 0：表示第一类 slack 不惩罚（很少见，可能意味着那类 slack 不启用或纯可行性）
    # 第2项 300：表示第二类 slack 惩罚很重（尽量别违反）
    Qslack = 1 * np.array([0, 300])
    
    # ==============================
    # 5) 打包成 MPC 参数对象
    # ==============================
    # slacks=True：启用软约束
    # timeVarying=True：参考 xRef 或系统矩阵/约束可能随时间变化（分支 MPC 常见）
    #
    # BranchMPCParams 是你工程里的配置类/数据结构：
    # - n,d,N,NB: 维度和预测设置
    # - Q,R: 代价权重
    # - Fx,bx,Fu,bu: 线性不等式约束
    # - xRef: 参考轨迹/参考状态
    # - slacks,Qslack: 软约束设置
    # - timeVarying: 是否时变
    mpcParameters = BranchMPCParams(n=n, d=d, N=N, NB=NB, Q=Q, R=R, Fx=Fx, bx=bx, Fu=Fu, bu=bu, xRef=xRef, slacks=True,
                                    Qslack=Qslack, timeVarying=True)
    return mpcParameters
