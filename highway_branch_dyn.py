from casadi import *
import pdb
import itertools
import numpy as np
from scipy import sparse
from itertools import product

'''
Use CasADi to calculate the dynamics, propagate trajectories under the backup policies, and calculate the branching probabilities
'''


def dubin(x, u):
    """
    Dubins / Unicycle 风格的“简化车辆运动学模型”（更准确：平面点质量 + 航向角）
    状态 x = [X, Y, v, psi]
      x[0] = X    : 全局坐标系下的 x 位置
      x[1] = Y    : 全局坐标系下的 y 位置
      x[2] = v    : 速度标量（沿航向方向的前进速度）
      x[3] = psi  : 航向角（yaw / heading），弧度

    输入 u = [a, r]
      u[0] = a : 纵向加速度（v 的变化率），即 dv/dt
      u[1] = r : 航向角速度 yaw rate（psi 的变化率），即 dpsi/dt

    连续时间动力学：
      dX/dt   = v * cos(psi)
      dY/dt   = v * sin(psi)
      dv/dt   = a
      dpsi/dt = r

    说明：
      - 该函数同时兼容 numpy 数值计算、CasADi 符号计算（SX/MX）
      - 在 MPC 中通常会用欧拉离散：x_{k+1} = x_k + dubin(x_k,u_k) * dt
    """
    if isinstance(x, numpy.ndarray):
        # ---- numpy 情况：直接返回一个 numpy array ----
        # xdot = [Xdot, Ydot, vdot, psidot]
        xdot = np.array([
            x[2] * cos(x[3]),  # Xdot = v*cos(psi)
            x[2] * sin(x[3]),  # Ydot = v*sin(psi)
            u[0],              # vdot = a
            u[1],              # psidot = r
        ])

    elif isinstance(x, casadi.SX):
        # ---- CasADi SX（符号标量/矩阵）情况 ----
        xdot = SX(4, 1)
        xdot[0] = x[2] * cos(x[3])
        xdot[1] = x[2] * sin(x[3])
        xdot[2] = u[0]
        xdot[3] = u[1]
    elif isinstance(x, casadi.MX):
        # ---- CasADi MX（更通用的符号图）情况 ----
        xdot = MX(4, 1)
        xdot[0] = x[2] * cos(x[3])
        xdot[1] = x[2] * sin(x[3])
        xdot[2] = u[0]
        xdot[3] = u[1]
    # 注意：这里默认 x 只会是 ndarray/SX/MX 三种之一
    return xdot


def softsat(x, s):
    """
    soft saturation / 平滑“压缩”函数：把任意实数输入 x 平滑映射到 (0, 1)

    定义：
      softsat(x, s) = (exp(s*x) - 1) / (exp(s*x) + 1) * 0.5 + 0.5

    关键性质（可化简）：
      (exp(sx)-1)/(exp(sx)+1) = tanh(sx/2)

    因此：
      softsat(x, s) = 0.5 * tanh(s*x/2) + 0.5

    直观解释：
      - 当 x -> +∞ 时，tanh -> 1，所以 softsat -> 1
      - 当 x -> -∞ 时，tanh -> -1，所以 softsat -> 0
      - 当 x = 0 时，tanh(0)=0，所以 softsat = 0.5
      - s 控制“陡峭程度”：s 越大，越像硬开关；s 越小，越平滑

    常见用途：
      - 在概率/权重计算前，把一个可能很大且正负都有的“安全函数 h”
        压缩到 (0,1)，避免后续 exp() 数值爆炸，并让梯度更平滑。
      - 作为“软阈值”或“平滑归一化”。
    """
    # 这里使用 numpy exp，如果你把 x 换成 casadi 符号类型，会报错；
    # 在你之前的代码中，softsat 常用于数值评估（np array），而不是 SX/MX。
    # 如果需要兼容 CasADi，通常要写成 casadi.exp 并判断类型。
    return (np.exp(s * x) - 1) / (np.exp(s * x) + 1) * 0.5 + 0.5


def backup_maintain(x, cons, psiref=None):
    """
    备用控制器：Maintain（保持）
    输出 u = [u_long, u_lat]^T，维度 2x1

    目标：
    - 纵向（u[0]）：不做动作（0），即保持当前速度/不加减速
    - 横向（u[1]）：让偏航角/航向误差 x[3] 收敛
      - 如果给了 psiref：跟踪参考 ψ_ref = psiref(x[0])
      - 否则：目标 ψ_ref=0

    参数：
    - x：状态向量（长度至少 4），可以是 CasADi MX/SX 或 numpy
    - cons：参数对象，需要包含 cons.Kpsi（偏航反馈增益）
    - psiref：可选函数，输入位置 x[0] 输出参考偏航/航向 ψ_ref
    """
    # -------- 情况 1：没有给参考航向 psiref，目标就是把 x[3] 拉回 0 --------
    if psiref is None:
        if isinstance(x, casadi.MX):
            u = MX(2, 1)       # 创建 2x1 的符号向量
            u[0] = 0           # 纵向控制：0（不加速不刹车）
            u[1] = -cons.Kpsi * x[3]   # 横向控制：比例反馈，抑制 x[3]
            return u
        elif isinstance(x, casadi.SX):
            u = SX(2, 1)
            u[0] = 0
            u[1] = -cons.Kpsi * x[3]
            return u
        # 数值类型（numpy 数组 / list / float）
        else:
            # 返回 numpy 数组（2维），用于仿真直接执行
            return np.array([0., -cons.Kpsi * x[3]])
    # -------- 情况 2：给了参考航向 psiref，目标是跟踪 ψ_ref --------
    else:
        if isinstance(x, casadi.MX):
            u = MX(2, 1)
            # 注意：这里没有显式设置 u[0]，默认是未初始化（可能是 0，也可能是符号垃圾）
            # 建议显式 u[0] = 0，保持和情况 1 一致
            u[0] = 0
            u[1] = psiref(x[0]) - cons.Kpsi * x[3]
            return u
        elif isinstance(x, casadi.SX):
            u = SX(2, 1)
            u[0] = 0
            u[1] = psiref(x[0]) - cons.Kpsi * x[3]
            return u
        else:
            return np.array([0., psiref(x[0]) - cons.Kpsi * x[3]])


def backup_brake(x, cons, psiref=None):
    """
    备用控制器：Brake（刹车）
    输出 u = [u_long, u_lat]^T

    目标：
    - 纵向 u[0]：产生一个“刹车”指令，幅值大致在 [-7, ...] 或 [-5, ...] 附近
      用 softmax/smoothmax 做平滑，避免 max() 不可导
    - 横向 u[1]：同 maintain，让偏航误差收敛或跟踪 ψ_ref

    备注：
    - 你这里 MX/SX 分支用了 softmax( [-7, -x[2]], 5 )
      但 numpy 分支用了 softmax( [-5, -x[2]], 3 )
      这不一致，可能是 bug 或不同模式下调参遗留。
    """
    if psiref is None:
        if isinstance(x, casadi.MX):
            u = MX(2, 1)
            u[0] = softmax(vertcat(-7, -x[2]), 5)
            u[1] = -cons.Kpsi * x[3]
            return u
        elif isinstance(x, casadi.SX):
            u = SX(2, 1)
            u[0] = softmax(vertcat(-7, -x[2]), 5)
            u[1] = -cons.Kpsi * x[3]
            return u
        else:
            return np.array([softmax(vertcat(-5, -x[2]), 3), -cons.Kpsi * x[3]])
    else:
        if isinstance(x, casadi.MX) or isinstance(x, casadi.SX):
            u = 0. * x[0:2]
            u[0] = softmax(vertcat(-5, -x[2]), 3)
            u[1] = psiref(x[0]) - cons.Kpsi * x[3]
            return u

        else:
            return np.array([softmax(vertcat(-5, -x[2]), 3), psiref(x[0]) - cons.Kpsi * x[3]])


def backup_lc(x, x0):
    """
    备用控制器：LC（可能是 lane change / linear control）
    输入：
    - x：当前状态
    - x0：参考状态/目标状态

    输出：
    - u[0]：根据速度误差 (x[2]-x0[2]) 做反馈
    - u[1]：根据横向误差 (x[1]-x0[1]) 和偏航误差 (x[3]-x0[3]) 做反馈

    形式是典型线性反馈：
        u = -K (x - x0)

    系数：
    -0.8558, -0.3162, -3.9889 看起来像 LQR/极点配置 得到的增益
    """
    if isinstance(x, casadi.MX):
        u = MX(2, 1)
        u[0] = -0.8558 * (x[2] - x0[2])
        u[1] = -0.3162 * (x[1] - x0[1]) - 3.9889 * (x[3] - x0[3])
        return u
    elif isinstance(x, casadi.SX):
        u = SX(2, 1)
        u[0] = -0.8558 * (x[2] - x0[2])
        u[1] = -0.3162 * (x[1] - x0[1]) - 3.9889 * (x[3] - x0[3])
        return u
    else:
        return np.array([-0.8558 * (x[2] - x0[2]), -0.3162 * (x[1] - x0[1]) - 3.9889 * (x[3] - x0[3])])


def softmin(x, gamma=1):
    """
    softmin：用“软最小值”平滑地近似 min(x)

    数学形式（本实现）：
        softmin(x) = Σ_i [ w_i * x_i ] / Σ_i [ w_i ]
        其中 w_i = exp(-gamma * x_i)

    直觉：
    - x_i 越小（越“危险/越接近最小”），exp(-gamma*x_i) 越大 → 权重越大
    - gamma 越大，权重越集中在最小的元素上 → 越接近真正的 min(x)
    - gamma 越小，权重更平均 → 更平滑但不够“像 min”

    注意：
    - 这不是常见的 log-sum-exp 形式 softmin（-1/gamma * log Σ exp(-gamma x)），
      而是“soft-argmin 加权平均”的版本：权重归一化后取加权均值。
    - 优点：输出仍在 x 的取值范围内（凸组合），平滑可导；
      缺点：数值上可能会溢出（gamma 大且 x 大/小），需注意稳定性。  
    """
    if isinstance(x, casadi.SX) or isinstance(x, casadi.MX):
        # CasADi 分支：用 CasADi 的 sum1/exp，保持符号表达式可导
        # sum1 是对向量/矩阵按列求和（CasADi 的约定）
        return sum1(exp(-gamma * x) * x) / sum1(exp(-gamma * x))
    else:
        return np.sum(np.exp(-gamma * x) * x) / np.sum(np.exp(-gamma * x))


def softmax(x, gamma=1):
    """
    softmax：用“软最大值”平滑近似 max(x)

    数学形式（本实现）：
        softmax(x) = Σ_i [ w_i * x_i ] / Σ_i [ w_i ]
        其中 w_i = exp(gamma * x_i)

    直觉：
    - x_i 越大，exp(gamma*x_i) 越大 → 权重越大
    - gamma 越大，越接近真正的 max(x)

    同样注意：
    - 这也不是经典的 log-sum-exp max（1/gamma * log Σ exp(gamma x)），
      而是“soft-argmax 加权平均”的版本。
    - 输出是 x 的加权平均，因此一定落在 [min(x), max(x)] 内。
    """
    if isinstance(x, casadi.SX) or isinstance(x, casadi.MX):
        return sum1(exp(gamma * x) * x) / sum1(exp(gamma * x))
    else:
        return np.sum(np.exp(gamma * x) * x) / np.sum(np.exp(gamma * x))


def propagate_backup(x, dyn, N, ts):
    '''
    Euler forward integration of the dynamics under the policy
    '''
    """
    在给定控制策略（policy）dyn(x) 下，用欧拉前向法（Euler forward）离散积分动力学，
    得到从当前状态 x 出发的 N 步预测轨迹。

    输入：
    - x  : 初始状态（可能是 numpy.ndarray 或 casadi.SX / casadi.MX）
           形状通常是 (n,) 或 (n,1) 之类，取决于你的系统定义
    - dyn: 一个函数 dyn(x) -> xdot
           返回状态导数（连续时间）
           例如 dubin(x,u) 或者 “把 u=policy(x) 代进去”的闭环动力学
    - N  : 预测步数（horizon length）
    - ts : 离散时间步长 Δt

    输出：
    - xs : 预测轨迹数组，shape = (N, n)
           第 i 行是第 i+1 步（做完一次更新后的）状态
           注意：这里没有把初始 x（t=0）存进去，而是从一步后的状态开始存。
                 如果你希望 xs[0] 是初始状态，需要改写循环/存储逻辑。

    数值说明：
    - 欧拉前向：x_{k+1} = x_k + dyn(x_k) * ts
      简单快速，但精度一般；ts 太大时误差会明显，甚至不稳定。
    """
    # -----------------------------
    # 根据输入类型分配轨迹容器 xs
    # -----------------------------
    if isinstance(x, numpy.ndarray):
        xs = np.empty([N, x.shape[0]])
    elif isinstance(x, casadi.SX):
        xs = SX(N, x.shape[0])
    elif isinstance(x, casadi.MX):
        xs = MX(N, x.shape[0])
    # xs = np.zeros(N,x.shape[0])
    # -----------------------------
    # 欧拉前向积分
    # -----------------------------
    for i in range(0, N):
        # 更新一步：x <- x + xdot * ts
        # dyn(x) 是闭环动力学（可能把 backup policy 的控制输入也包含进去了）
        x = x + dyn(x) * ts

        # 存储这一时刻的状态（更新后的 x）
        # xs[i, :] 代表第 i 步预测的状态向量
        xs[i, :] = x
    return xs


def lane_bdry_h(x, lb=0, ub=7.2):
    """
    车道边界“安全函数”h(x) 的构造。

    目标：
      - 返回一个标量/向量 h，使得：
          h >= 0  => 在车道内（满足边界约束）
          h <  0  => 越界（违反约束）
      - 但不能直接用 min(...)，因为 min 不光滑、不可导/不可微（对优化器不友好）
      - 所以用 softmin 来平滑近似 min

    输入：
      x : 车辆状态（或一段轨迹的状态序列）
          这里默认 x 的第二维索引 1 是 y 位置（横向位置）
          - 如果 x 是 1维: x = [X, Y, V, PSI] (或类似)
          - 如果 x 是 2维: x[i, :] 表示第 i 个时间步的状态
      lb : 车道左边界（lower bound），例如 0
      ub : 车道右边界（upper bound），例如 7.2

    输出：
      h :
        - 若输入是一段轨迹 (N×n)，输出是长度 N 的向量，每个时间步一个 h[i]
        - 若输入是单个状态 (n,)，输出是一个标量 h
    """
    if isinstance(x, casadi.SX):
        h = SX(x.shape[0], 1)
        for i in range(0, x.shape[0]):
            h[i] = softmin(vertcat(x[i, 1] - lb, ub - x[i, 1]), 5)
        return h
    elif isinstance(x, casadi.MX):
        h = MX(x.shape[0], 1)
        for i in range(0, x.shape[0]):
            h[i] = softmin(vertcat(x[i, 1] - lb, ub - x[i, 1]), 5)
        return h
    # ------------------------- 纯数值分支：numpy -------------------------
    else:
        # 如果 x 是 1 维数组，表示单个状态
        if x.ndim == 1:
            # 直接对两个 margin 做 softmin，返回一个标量
            # h >= 0 表示在边界内；h 越大表示越居中/离边界越远
            # h < 0 表示越界（负值的幅度表示越界程度）
            return softmin(np.array([x[1] - lb, ub - x[1]]), 5)
        # 如果 x 是 2 维数组，表示轨迹序列（N×n），对每个时间步计算 h[i]
        else:
            # 创建一个长度 N 的数组来存每个时间步的 h[i]
            h = np.zeros(x.shape[0])
            for i in range(0, x.shape[0]):
                h[i] = softmin(np.array([x[i, 1] - lb, ub - x[i, 1]]), 5)
            return h


def veh_col(x1, x2, size, alpha=1):
    '''
    vehicle collision constraints: h>=0 means no collision
    implemented via a softmax function
    '''
    """
    车辆碰撞约束的“安全函数”h(x1,x2)。

    设计目标（作者注释写得很关键）：
      - h >= 0 表示“无碰撞 / 足够安全”
      - h <  0 表示“发生重叠/碰撞风险”
    并且要尽量保持可导，方便在 MPC/QP/NLP 中做约束线性化或梯度计算。

    输入：
      x1, x2 : 两个车辆（或自车与障碍物）的位置/状态
          - 如果是单步状态：x1=[X,Y,...], x2=[X,Y,...]
          - 如果是轨迹：x1[i,0],x1[i,1] 表示第 i 步的位置
      size : [L_safe, W_safe]
          可以理解为在 x/y 方向要求的“安全间距”（半尺寸或等效尺寸）
          代码写法：abs(dx)-size[0] / abs(dy)-size[1]
          => 相当于用一个轴对齐矩形(AABB)的“分离距离”来近似碰撞边界
      alpha : softmax 的“尖锐度”参数，越大越接近 max(dx,dy)

    输出：
      h :
        - 若输入是轨迹，返回每步的 h[i]
        - 若输入是单点，返回标量 h

    关键思想：
      1) 先分别算 x 方向、y 方向的“分离裕度”：
         dx = |x1-x2| - size_x
         dy = |y1-y2| - size_y

         - 如果 dx >= 0：表示在 x 方向已经分离足够（间距大于阈值）
         - 如果 dx <  0：表示在 x 方向发生重叠（间距不足）
         y 方向同理。

      2) 对于轴对齐矩形碰撞，一个常见的保守安全函数是：
           h = max(dx, dy)
         因为：
           - 只要有一个方向分离足够（dx>=0 或 dy>=0），就可以“没有重叠”
           - 两个方向都负，才表示真正重叠（可能碰撞）
         但 max 也是不光滑，所以这里用 softmax 近似 max。

      3) 这个函数里实现的 softmax 形式是“加权平均”：
           h = (dx*e^{a dx} + dy*e^{a dy}) / (e^{a dx} + e^{a dy})
         当 alpha 大时，权重会集中到更大的那个（更接近 max(dx,dy)）。
    """
    if isinstance(x1, casadi.SX):
        h = SX(x1.shape[0])
        for i in range(0, x1.shape[0]):
            dx = (fabs(x1[i, 0] - x2[i, 0]) - size[0])
            dy = (fabs(x1[i, 1] - x2[i, 1]) - size[1])
            h[i] = (dx * np.exp(alpha * dx) + dy * np.exp(dy * alpha)) / (np.exp(alpha * dx) + np.exp(dy * alpha))
        return h
    elif isinstance(x1, casadi.MX):
        h = MX(x1.shape[0])
        for i in range(0, x1.shape[0]):
            dx = (fabs(x1[i, 0] - x2[i, 0]) - size[0])
            dy = (fabs(x1[i, 1] - x2[i, 1]) - size[1])
            h[i] = (dx * np.exp(alpha * dx) + dy * np.exp(dy * alpha)) / (np.exp(alpha * dx) + np.exp(dy * alpha))
        return h
    # ------------------------- 纯数值分支：numpy -------------------------
    else:
        # 单点情况：x1,x2 都是一维状态向量
        if x1.ndim == 1:
            # 这里加了 clip，把 dx,dy 限制在 [-5,5]
            # 目的：避免 exp(alpha*dx) 溢出（dx 很大时 e^{dx} 爆炸）
            # 这是非常常见的数值稳定技巧
            dx = np.clip((abs(x1[0] - x2[0]) - size[0]), -5, 5)
            dy = np.clip((abs(x1[1] - x2[1]) - size[1]), -5, 5)
            # 返回 softmax 近似 max(dx,dy)
            return (dx * np.exp(alpha * dx) + dy * np.exp(dy * alpha)) / (np.exp(alpha * dx) + np.exp(dy * alpha))
        # 轨迹情况：x1 是 N×n, x2 是 N×n，逐步计算每个时间步的 h[i]
        else:
            h = np.zeros(x1.shape[0])
            for i in range(0, x1.shape[0]):
                dx = np.clip((abs(x1[i][0] - x2[i][0]) - size[0]), -5, 5)
                dy = np.clip((abs(x1[i][1] - x2[i][1]) - size[1]), -5, 5)
                h[i] = (dx * np.exp(alpha * dx) + dy * np.exp(dy * alpha)) / (np.exp(alpha * dx) + np.exp(dy * alpha))
            return h


class PredictiveModel:
    def __init__(self, n, d, N, backupcons, dt, cons, N_lane=3):
        # -----------------------------
        # 基本维度
        # -----------------------------
        self.n = n  # state dimension
        # 例如 dubin 模型: [x, y, v, psi]
        self.d = d  # input dimension
        # 例如: [a, r] (加速度，yaw rate)
        self.N = N  # number of prediction steps
        # backup policy 数量（即 branch 数量）
        self.m = len(backupcons)  # number of policies
        self.dt = dt  # 离散步长
        # 车辆参数
        self.cons = cons  # parameters
        # -----------------------------
        # CasADi 函数占位符
        # 后面 calc_xp_expr() 会生成
        # -----------------------------
        # 动力学线性化矩阵 A, B
        self.Asym = None
        self.Bsym = None
        # 离散动力学
        self.xpsym = None
        # 安全函数
        self.hsym = None
        # 对手车辆预测轨迹
        self.zpred = None
        # ego backup 轨迹
        self.xpred = None
        # 默认 backup 控制
        self.u0sym = None
        self.Jh = None
        # -----------------------------
        # 车道边界
        # -----------------------------
        # 假设车道宽3.6m
        # y ∈ [W/2 , N_lane*3.6 - W/2]
        # 给车辆中心预留车宽
        self.LB = [cons.W / 2, N_lane * 3.6 - cons.W / 2]  # lane boundary
        # backup controllers
        # 每个 policy 对应一种行为
        self.backupcons = backupcons
        self.calc_xp_expr()

    # -------------------------------------------------
    # 动力学线性化
    # -------------------------------------------------
    def dyn_linearization(self, x, u):
        # linearizing the dynamics x^+=Ax+Bu+C
        A = self.Asym(x, u)  # ∂f/∂x
        B = self.Bsym(x, u)  # ∂f/∂u

        # 离散动力学
        xp = self.xpsym(x, u)

        # 构造 affine dynamics
        # x+ = Ax + Bu + C
        C = xp - A @ x - B @ u

        return np.array(A), np.array(B), np.squeeze(np.array(C)), np.squeeze(np.array(xp))
    
    # -------------------------------------------------
    # 计算 branch probability
    # -------------------------------------------------
    def branch_eval(self, x, z):
        # p_i = branch probability
        p = self.psym(x, z)
        # dp/dx
        dp = self.dpsym(x, z)
        return np.array(p).flatten(), np.array(dp)
    # -------------------------------------------------
    # 对手车辆轨迹预测
    # -------------------------------------------------
    def zpred_eval(self, z):
        # 输出 shape
        # (N , n*m)
        # 每个 policy 一段轨迹
        return np.array(self.zpred(z))
    
    # -------------------------------------------------
    # ego backup 轨迹
    # -------------------------------------------------
    def xpred_eval(self, x):
        return self.xpred(x), self.u0sym(x)

    # -------------------------------------------------
    # 碰撞约束线性化
    # -------------------------------------------------
    def col_eval(self, x, z):
        # dh = ∂h/∂x
        dh = np.squeeze(np.array(self.dhsym(x, z)))
        # h(x,z) = h(x)  (这里 h 只依赖 x，不依赖 z，因为 z 是对手的状态，不是决策变量)
        h = np.squeeze(np.array(self.hsym(x, z)))
        # 线性化:
        # h(x) ≈ h(x0) + dh (x-x0)
        #
        # QP约束形式
        #
        # dh * x >= dh*x0 - h(x0)
        return h - np.dot(dh, x), dh
    # -------------------------------------------------
    # 更新 backup policies
    # -------------------------------------------------
    def update_backup(self, backupcons):
        self.backupcons = backupcons
        self.m = len(backupcons)
        # 重新构建预测表达式
        self.calc_xp_expr()

    # -------------------------------------------------
    # trajectory safety function
    # -------------------------------------------------
    def BF_traj(self, x1, x2):
        # x1: 对手轨迹
        # x2: ego轨迹
        if isinstance(x1, casadi.SX):
            h = SX(x1.shape[0] * 2, 1)
        elif isinstance(x1, casadi.MX):
            h = MX(x1.shape[0] * 2, 1)
        for i in range(0, x1.shape[0]):
            # 碰撞约束
            h[i] = veh_col(x1[i, :], x2[i, :], [self.cons.L + 2, self.cons.W + 0.2])
            # 车道约束
            h[i + x1.shape[0]] = lane_bdry_h(x1[i, :], self.LB[0], self.LB[1])
        # 取整条轨迹最危险点
        return softmin(h, 5)

    def branch_prob(self, h):
        # branching probability as a function of the safety function
        h = softsat(h, 1)
        m = exp(self.cons.s1 * h)
        return m / sum1(m)

    # -------------------------------------------------
    # 构建所有 CasADi 表达式
    # -------------------------------------------------
    def calc_xp_expr(self):
        # -----------------------------
        # symbolic variables
        # -----------------------------
        u = SX.sym('u', self.d)
        x = SX.sym('x', self.n)
        z = SX.sym('z', self.n)
        zdot = SX.sym('zdot', self.m, self.n)
        # -----------------------------
        # ego dynamics
        # -----------------------------
        xp = x + dubin(x, u) * self.dt
        # -----------------------------
        # ego backup trajectory
        # -----------------------------
        dyn = lambda x: dubin(x, self.backupcons[0](x))
        x1 = propagate_backup(x, dyn, self.N, self.dt)
        # -----------------------------
        # opponent trajectories
        # -----------------------------
        x2 = SX(self.N, self.n * self.m)
        for i in range(0, self.m):
            dyn = lambda x: dubin(x, self.backupcons[i](x))
            x2[:, i * self.n:(i + 1) * self.n] = propagate_backup(z, dyn, self.N, self.dt)
        # -----------------------------
        # safety score per branch
        # -----------------------------
        hi = SX(self.m, 1)
        for i in range(0, self.m):
            hi[i] = self.BF_traj(x2[:, self.n * i:self.n * (i + 1)], x1)
        # -----------------------------
        # branch probability
        # -----------------------------
        p = self.branch_prob(hi)
        # -----------------------------
        # current collision constraint
        # -----------------------------        # -----------------------------
        # safety score per branch
        # -----------------------------
        h = veh_col(x.T, z.T, [self.cons.L + 1, self.cons.W + 0.2], 1)
        # -----------------------------
        # CasADi functions
        # -----------------------------
        self.xpsym = Function('xp', [x, u], [xp])
        self.Asym = Function('A', [x, u], [jacobian(xp, x)])
        self.Bsym = Function('B', [x, u], [jacobian(xp, u)])
        self.dhsym = Function('dh', [x, z], [jacobian(h, x)])
        self.hsym = Function('h', [x, z], [h])
        self.zpred = Function('zpred', [z], [x2])
        self.xpred = Function('xpred', [x], [x1])
        self.psym = Function('p', [x, z], [p])
        self.dpsym = Function('dp', [x, z], [jacobian(p, x)])
        self.u0sym = Function('u0', [x], [self.backupcons[0](x)])
