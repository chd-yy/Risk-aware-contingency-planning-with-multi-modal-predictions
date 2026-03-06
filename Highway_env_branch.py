import pdb
import osqp
import argparse
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
import numpy as np
from scipy.io import loadmat
from scipy import interpolate, sparse
import random
import math
from numpy import linalg as LA
from numpy.linalg import norm
from highway_branch_dyn import *
import pickle

# ----------------------------
# 全局参数：初速度/初始参考状态/车道几何
# ----------------------------

v0 = 15
# f0 看起来是一个“参考初始状态/默认状态”的向量
# 常见 state = [x, y, v, psi]（位置x、位置y、速度v、航向角psi）
# 这里 f0 = [v0, 0, 0, 0] 并不完全匹配上面的常见顺序，可能只是在别处作为初值/参数用
f0 = np.array([v0, 0, 0, 0])

lane_width = 3.6
# lm：车道中心线索引（0~6）乘以车道宽度，得到每条车道中心线的 y 偏移（或标记）
# 例如 lane 0 的中心 y≈0，lane 1 的中心 y≈3.6 ...
lm = np.arange(0, 7) * lane_width


def with_probability(P=1):
    """
    以概率 P 返回 True，否则 False。
    用于随机事件（例如随机换道/随机出现某种行为）——虽然这段代码里暂时没看到使用。
    """
    return np.random.uniform() <= P


class vehicle():
    def __init__(self, state=[0, 0, v0, 0], v_length=4, v_width=2.4, dt=0.05, backupidx=0, laneidx=0):
        """
        一个非常简化的车辆对象。

        state: 车辆状态向量（这里从 step() 看，使用了：
               state[0] = x 位置
               state[1] = y 位置
               state[2] = v 速度
               state[3] = psi 航向角
               即 state = [x, y, v, psi]
        v_length, v_width: 车辆外形尺寸（用于碰撞/安全距离计算）
        dt: 离散时间步长
        backupidx: 车辆当前选择的“备份策略”索引（用于 branching/backup policy）
        laneidx: 当前车道索引（用于估计车辆处在哪条车道）
        """
        self.state = np.array(state)
        self.dt = dt
        self.v_length = v_length
        self.v_width = v_width

        # x_pred/y_pred：用于存储预测轨迹（可能用于可视化或调试）
        # 但在这段代码里暂时没看到填充
        self.x_pred = []
        self.y_pred = []

        # xbackup：备份轨迹（backup trajectory），可能用于外部存储/调试
        self.xbackup = None

        self.backupidx = backupidx
        self.laneidx = laneidx

    def step(self, u):  # controlled vehicle
        """
        用非常简单的 Dubins-like 模型（或“点质量+航向”模型）做一步前向欧拉积分。

        u: 控制输入，长度为2：
           u[0] = 加速度 a （直接作为 v_dot）
           u[1] = 航向角速度 r （直接作为 psi_dot）

        动力学：
          x_dot   = v cos(psi)
          y_dot   = v sin(psi)
          v_dot   = a
          psi_dot = r

        离散更新：
          state_{k+1} = state_k + dxdt * dt
        """
        dxdt = np.array([self.state[2] * np.cos(self.state[3]),   # x_dot
                         self.state[2] * np.sin(self.state[3]),   # y_dot
                         u[0],                                    # v_dot
                         u[1]])                                   # psi_dot
        self.state = self.state + dxdt * self.dt


class Highway_env():
    def __init__(self, NV, mpc, N_lane=6, timestep=0, ego_state=[], obst_new_state=[]):
        '''
        高速路环境封装（用于一次 MPC 控制 + 其他车辆按备份策略推进）。

        Input:
          NV: number of vehicles（车辆数量，例如 2：自车+障碍车）
          mpc: mpc controller for the controlled vehicle（给自车用的 MPC 控制器对象）
          N_lane: number of lanes（车道数）
          timestep: 当前仿真/决策步编号（t=0 表示初始化；非0表示从外部状态初始化）
          ego_state: 外部传入的自车状态对象（timestep!=0 时使用）
          obst_new_state: 外部传入的障碍车状态（timestep!=0 时使用）
        '''

        # 从 MPC 的 predictiveModel 里取 dt（保证环境仿真与 MPC 离散步长一致）
        self.dt = mpc.predictiveModel.dt

        # veh_set：车辆对象列表
        self.veh_set = []

        self.NV = NV
        self.N_lane = N_lane

        # desired_x：每辆车的“期望状态”（这里只用到了期望 y 和期望 v）
        # 用于生成/更新备份控制策略（比如保持车道/换道目标等）
        self.desired_x = [None] * NV

        self.mpc = mpc
        self.predictiveModel = mpc.predictiveModel

        # backupcons：备份控制策略集合（列表），每个策略是一个函数 u=pi(x)
        self.backupcons = mpc.predictiveModel.backupcons

        # 备份策略数量 m
        self.m = len(self.backupcons)

        # cons：控制/车辆参数（如最大加速度 am、最大转向/角速度 rm、车长宽 L/W 等）
        self.cons = mpc.predictiveModel.cons

        # 车道边界 LB（lower/upper bound）
        # self.cons.W/2：通常是车辆宽度的一半，确保车辆“外轮廓”不越界
        # N_lane*3.6 - self.cons.W/2：道路最右边界（或最上边界）
        self.LB = [self.cons.W / 2, N_lane * 3.6 - self.cons.W / 2]

        # ----------------------------
        # 初始化车辆初始状态 x0
        # ----------------------------
        # x0 形状：NV × 4，每行是 [x, y, v, psi]
        # x0 = np.array([[0,1.8,v0,0],[5,5.4,v0,0]])
        if timestep == 0:
            # 初始场景：两辆车
            # 车0（自车）：x=-8, y=-5.4（车道中心附近）, v=v0, psi=0
            # 车1（障碍车）：x=9.5844, y=-3.1541（另一车道中心）, v=v0, psi=0
            x0 = np.array([[-8, -5.4, v0, 0],
                           [9.5844, -3.1541, v0, 0]])
        else:
            # 非初始化步：从外部输入状态构造
            # ego_state 似乎是一个对象，包含 position, velocity, orientation 等字段
            x0 = np.array([[ego_state.position[0], ego_state.position[1], ego_state.velocity,
                            ego_state.orientation],
                           obst_new_state])

        # ----------------------------
        # 创建 NV 个 vehicle 对象，并初始化 desired_x
        # ----------------------------
        for i in range(0, self.NV):
            # 每辆车都用相同 dt
            # backupidx 初始化为 0（默认使用第0个备份策略）
            self.veh_set.append(vehicle(x0[i], dt=self.dt, backupidx=0))

            # desired_x：期望 x 位置不重要（这里设为0），期望 y=当前 y，期望 v=v0，期望 psi=0
            # 主要用于后续更新目标车道/换道参考
            self.desired_x[i] = np.array([0, x0[i, 1], v0, 0])

    def step(self, t_):
        """
        环境推进一步（核心逻辑）：
          1) 计算每辆车在各备份策略下的预测轨迹（xx_set）
          2) 根据“与自车当前备份轨迹的安全度”，为障碍车选择一个最安全/最优的备份策略（backupidx）
          3) 为自车构造 xRef（期望 y 和期望 v），调用 MPC 求解控制
          4) 用障碍车的备份控制推进障碍车一步
          5) 返回控制/状态/预测轨迹等信息
        """

        # u_set：每辆车这一步的控制输入（自车由 MPC 给；障碍车用备份策略）
        u_set = [None] * self.NV

        # xx_set：每辆车的“备份预测轨迹集合”（在每个备份策略下的预测轨迹）
        # 具体结构看 zpred_eval 返回：通常是 N × (n*m) （把 m 条轨迹横向拼起来）
        xx_set = [None] * self.NV

        # u0_set：每辆车在其“当前选中的备份策略”下的控制输入（用于障碍车推进）
        u0_set = [None] * self.NV

        # x_set：保存每辆车更新后的状态（这里只对障碍车更新）
        x_set = [None] * self.NV

        # umax：控制输入上界（加速度上限 am，角速度上限 rm）
        # 这里定义了但在这段 step() 里没有直接用到
        umax = np.array([self.cons.am, self.cons.rm])

        # ----------------------------
        # 生成备份轨迹（backup trajectories）
        # ----------------------------

        # self.xbackup：看起来是用来记录/缓存所有车辆的备份轨迹
        # 形状预期：(?, (N+1)*4) 之类；这里初始化为空数组
        self.xbackup = np.empty([0, (self.mpc.N + 1) * 4])

        for i in range(0, self.NV):
            # z：车辆 i 的当前状态
            z = self.veh_set[i].state

            # xx_set[i]：该车辆在所有备份策略下的预测轨迹（由 predictiveModel 生成）
            # 注意：这里用的是 zpred_eval(z)，而 predictiveModel 里 zpred(z) 会传播“备份策略”轨迹
            xx_set[i] = self.predictiveModel.zpred_eval(z)

            # newlaneidx：根据 y 位置估计当前车道索引
            # 假设车道中心线在 y=1.8 + 3.6*k
            # 所以 (y-1.8)/3.6 ≈ k
            newlaneidx = round((z[1] - 1.8) / 3.6)

            # 如果 t_=0（第一步）或者车道索引发生变化且满足一些几何条件，则更新“目标车道”
            # abs(z[1] + 1.8 + 3.6 * newlaneidx) < 1.4 这句看起来有点怪（可能是某种“接近中心线”的判断）
            # 总体意图：车辆已经接近/进入新车道，更新 laneidx 与 desired_x 的 y 目标
            if t_ == 0 or (newlaneidx != self.veh_set[i].laneidx and abs(z[1] + 1.8 + 3.6 * newlaneidx) < 1.4):
                # 更新车辆 i 的车道索引
                # update the desired lane
                self.veh_set[i].laneidx = newlaneidx

                # 更新车辆 i 的期望 y（即该车道中心线 y）
                self.desired_x[i][1] = 1.8 + newlaneidx * 3.6

                # 这里对 i==1（障碍车）做特殊处理：根据自车/障碍车的相对车道，设置换道目标 xRef
                # 并基于这个 xRef 构造一个“换道备份策略 backup_lc”
                if i == 1:
                    # 如果自车车道在障碍车的左侧（laneidx更小），障碍车的换道目标是往左一条车道
                    if self.veh_set[0].laneidx < self.veh_set[1].laneidx:
                        xRef = np.array([0, 1.8 + 3.6 * (self.veh_set[1].laneidx - 1), v0, 0])

                    # 如果自车车道在障碍车的右侧（laneidx更大），障碍车换道目标往右一条车道
                    elif self.veh_set[0].laneidx > self.veh_set[1].laneidx:
                        xRef = np.array([0, 1.8 + 3.6 * (self.veh_set[1].laneidx + 1), v0, 0])

                    # 如果两者在同一车道，障碍车尝试换到相邻车道（优先往左，若不能则往右）
                    else:
                        if self.veh_set[1].laneidx > 0:
                            xRef = np.array([0, 1.8 + 3.6 * (self.veh_set[1].laneidx - 1), v0, 0])
                        else:
                            xRef = np.array([0, 1.8 + 3.6 * (self.veh_set[1].laneidx + 1), v0, 0])

                    # 构造三种备份策略：
                    #  1) backup_maintain：保持/维持（如保持航向稳定、速度维持等）
                    #  2) backup_brake：制动/减速（更保守）
                    #  3) backup_lc：换道控制（目标是 xRef 的 y 位置等）
                    # 注意：lambda x: backup_maintain(x, self.cons) 这样的写法把 cons 固定住
                    backupcons = [
                        lambda x: backup_maintain(x, self.cons),
                        lambda x: backup_brake(x, self.cons),
                        lambda x: backup_lc(x, xRef)
                    ]

                    # 更新 predictiveModel 的备份策略集（这会影响后续 zpred_eval 生成的备份轨迹）
                    self.predictiveModel.update_backup(backupcons)

        # ----------------------------
        # 为障碍车选择“最合适的备份策略”
        # ----------------------------

        # 自车当前选择的备份策略索引（idx0）
        idx0 = self.veh_set[0].backupidx

        # n：状态维度（此处应为 4）
        n = self.predictiveModel.n

        # x1：自车在当前备份策略 idx0 下的预测轨迹
        # xx_set[0] 是 N×(n*m)，取出第 idx0 条轨迹 => N×n
        x1 = xx_set[0][:, idx0 * n:(idx0 + 1) * n]

        for i in range(0, self.NV):
            if i != 0:
                # hi[j]：障碍车采用第 j 个备份策略时，
                # 与自车预测轨迹 x1 的“最小安全裕度”（越大越安全）
                hi = np.zeros(self.m)

                for j in range(0, self.m):
                    # veh_col(x1, x2traj, size) 返回每个时间步的碰撞安全函数 h(t)
                    # lane_bdry_h(x1, ...) 返回每个时间步的车道边界安全函数 h_lane(t)
                    #
                    # np.append(...)：把两个向量拼起来，然后取 min
                    # 意味着：把“碰撞安全”和“车道边界安全”都当作约束，
                    # 取所有约束在所有时刻中的最小值作为“最坏情况安全裕度”
                    #
                    # 直觉：只要有某个时刻某个约束特别危险（负得多），min 就会很小
                    hi[j] = min(
                        np.append(
                            veh_col(
                                x1,
                                xx_set[i][:, j * n:(j + 1) * n],
                                [self.cons.L + 1, self.cons.W + 0.2]
                            ),
                            lane_bdry_h(x1, self.LB[0], self.LB[1])
                        )
                    )

                # 对障碍车：选择 hi 最大的备份策略（即“最安全”的那条）
                self.veh_set[i].backupidx = np.argmax(hi)

            # 为每辆车取其当前选择的备份策略的控制输入 u0
            # 对障碍车，这个 u0 会用于实际 step 推进
            # 对自车，这里也算了，但后面自车实际控制由 MPC.solve() 决定
            u0_set[i] = self.backupcons[self.veh_set[i].backupidx](self.veh_set[i].state)

        # ----------------------------
        # 设置自车的参考状态 xRef 并调用 MPC 做“超车/跟车”决策
        # ----------------------------

        # 如果自车还没超过障碍车（自车 x < 障碍车 x）
        if self.veh_set[0].state[0] < self.veh_set[1].state[0]:
            # 自车期望保持在自己的车道中心（Ydes 为自车车道中心）
            Ydes = 1.8 + self.veh_set[0].laneidx * 3.6
        else:
            # 如果自车已经在前方，则 Ydes 设为障碍车当前 y（可能意味着“回到/保持某条轨迹”或“贴合障碍车所在车道”）
            Ydes = self.veh_set[1].state[1]

        # 如果自车已经接近目标横向位置（|y-Ydes|<1），并且自车已经明显超过障碍车（x > x_obs + 3）
        if abs(self.veh_set[0].state[1] - Ydes) < 1 and self.veh_set[0].state[0] > self.veh_set[1].state[0] + 3:
            # 速度目标恢复到 v0（正常巡航）
            vdes = v0
        else:
            # 否则做一个“基于相对距离的速度参考”（类似追踪/超车的简单策略）：
            # vdes = v_obs + gain*(x_obs + 1.5 - x_ego)
            # 若自车落后，(x_obs+1.5-x_ego)>0 => vdes > v_obs 促使追上
            # 若自车领先，可能降低 vdes
            vdes = self.veh_set[1].state[2] + 1 * (self.veh_set[1].state[0] + 1.5 - self.veh_set[0].state[0])

        # xRef：MPC 跟踪/优化的参考状态（这里只用 y 和 v，x/psi 设为 0）
        xRef = np.array([0, Ydes, vdes, 0])

        # 调用 MPC 求解：输入自车状态、障碍车状态、参考 xRef
        # mpc.solve 内部会建立/更新 QP 并求解得到自车控制序列
        self.mpc.solve(self.veh_set[0].state, self.veh_set[1].state, xRef)

        # 把 MPC 内部的树/分支结果转成数组（便于后续使用或可视化）
        # xPred：自车预测
        # zPred：障碍车（或分支）预测
        # utraj：自车控制轨迹
        # branch_w：分支权重/概率
        xPred, zPred, utraj, branch_w = self.mpc.BT2array()

        # ----------------------------
        # 推进障碍车（以及其他非自车车辆）
        # ----------------------------
        for i in range(1, self.NV):
            # 障碍车控制输入采用其备份策略 u0
            u_set[i] = u0_set[i]

            # 用 vehicle.step 推进一步（前向欧拉）
            self.veh_set[i].step(u_set[i])

            # 记录更新后的状态
            x_set[i] = self.veh_set[i].state

        # 打印障碍车速度（调试信息）
        print('obstacle vehicle velocity: ', x_set[1][2])

        # 返回：
        #  u_set: 控制输入集合（这里只对障碍车有填，若你想要自车 u 可能要从 mpc 里取）
        #  x_set: 更新后的状态集合（这里只对障碍车有填）
        #  xx_set: 所有车辆的备份预测轨迹集合
        #  xPred/zPred/branch_w: MPC 输出（自车预测、分支预测、分支权重）
        return u_set, x_set, xx_set, xPred, zPred, branch_w


def Highway_sim(env, T):
    # simulate the scenario
    """
    运行一次“高速路场景”仿真（但注意：这段实现只推进了一次 env.step(t)）
    T: 仿真时长（秒），但当前代码只拿它来算 N，并未循环推进多步
    """
    collision = False
    dt = env.dt
    t = 0

    # Ts_update：可能用于“隔多少步更新一次 MPC/策略”，但这里没有用到
    Ts_update = 4

    # N：按时长 T 和 dt 计算应有多少离散步
    N = int(round(T / dt))

    # state_rec：记录状态轨迹的数组
    # 维度：NV × N × 4（每辆车每一步记录 [x,y,v,psi]）
    # 但因为此函数目前只执行一步 env.step，不会真正填满 N 步
    state_rec = np.zeros([env.NV, N, 4])

    # backup_rec：记录每辆车每一步的备份轨迹（或备份策略相关信息），这里先占位
    backup_rec = [None] * env.NV

    # backup_choice_rec：记录每辆车每一步选择了哪个备份策略
    backup_choice_rec = [None] * env.NV

    for i in range(0, env.NV):
        backup_rec[i] = [None] * N
        backup_choice_rec[i] = [None] * N

    # 记录初始状态（t=0）
    for i in range(0, len(env.veh_set)):
        state_rec[i][t] = env.veh_set[i].state

    # dis：用于碰撞检测的“分离距离”（越大越安全；<0 认为碰撞）
    dis = 100

    # ----------------------------
    # 碰撞检测（只在 t=0 做一次）
    # ----------------------------
    if not collision:
        for i in range(0, env.NV):
            for j in range(0, env.NV):
                if i != j:
                    # 这里用一个非常简单的轴对齐矩形(AABB)方式计算“分离裕度”
                    # x方向裕度：|dx| - 0.5*(L_i+L_j)
                    # y方向裕度：|dy| - 0.5*(W_i+W_j)
                    #
                    # 取 max(x裕度, y裕度) 作为“近似分离程度”
                    # 若两方向都重叠，则两裕度都为负，max 仍为负 => 判定碰撞
                    dis = max(
                        abs(env.veh_set[i].state[0] - env.veh_set[j].state[0]) - 0.5 * (
                            env.veh_set[i].v_length + env.veh_set[j].v_length
                        ),
                        abs(env.veh_set[i].state[1] - env.veh_set[j].state[1]) - 0.5 * (
                            env.veh_set[i].v_width + env.veh_set[j].v_width
                        )
                    )
            if dis < 0:
                collision = True

    # 打印当前仿真时间
    print("t=", t * env.dt)

    # 调用环境 step：生成备份预测、调用 MPC、推进障碍车一步
    u_set, x_set, xx_set, xPred, zPred, branch_w = env.step(t)
    # x_set[1] includes the updated state of the obstacle vehicle
    # 返回结果说明：
    #  xx_set：两辆车在各备份策略下的预测轨迹集合
    #  zPred：MPC 输出的分支预测（具体含义取决于 BT2array 实现）
    #  x_set[1]：障碍车更新后的状态
    #  state_rec：状态记录（只记录了初始状态）
    #  branch_w：分支权重/概率
    return xx_set, zPred, x_set[1], state_rec, branch_w


def sim_overtake(mpc, N_lane, timestep, ego_state, obst_new_state, trajectory):
    """
    对外的封装函数：构造环境 -> 运行一次很短的仿真 -> 返回结果。

    输入：
      mpc: MPC 控制器
      N_lane: 车道数
      timestep: 是否为初始化步（0 初始化，否则用外部状态）
      ego_state: 外部自车状态对象
      obst_new_state: 外部障碍车状态（numpy 数组 [x,y,v,psi]）
      trajectory: 传进来了但当前函数里没用到（可能预留接口）

    输出：
      backup: env.step 内计算得到的备份预测轨迹集合（这里命名为 backup）
      zPred: MPC 分支预测
      obst_new_state: 更新后的障碍车状态
      branch_w: 分支权重
      state_rec: 状态记录（只含初始）
    """
    env = Highway_env(
        NV=2,
        mpc=mpc,
        N_lane=N_lane,
        timestep=timestep,
        ego_state=ego_state,
        obst_new_state=obst_new_state
    )

    # 这里 T=0.1 秒，但 Highway_sim 实际只执行一步 step
    backup, zPred, obst_new_state, state_rec, branch_w = Highway_sim(env, 0.1)

    return backup, zPred, obst_new_state, branch_w, state_rec
