# -*- coding: utf-8 -*-
"""分支模型预测控制(Branch MPC)求解器的主要实现。

本文件聚焦于多场景分支树的构建、对应的线性化动力学、约束与代价矩阵
的搭建,以及求解二次规划(QP)问题的全过程。注释使用中文详细说明
每一步骤,方便久未编写 Python 的读者快速理解代码结构。
"""

import pdb
import numpy as np
from cvxopt import spmatrix, matrix, solvers
from numpy import linalg as la
from scipy import linalg
from scipy import sparse
from cvxopt.solvers import qp
import datetime
from numpy import hstack, inf, ones
from scipy.sparse import vstack
from osqp import OSQP
from dataclasses import dataclass, field
import ecos

solvers.options['show_progress'] = False


@dataclass
class PythonMsg:
    """模仿 ROS/Fast RTPS 中“消息”的不可扩展数据类基础实现。"""

    def __setattr__(self, key, value):
        """阻止外部在运行期动态添加字段,避免参数拼写错误。"""
        if not hasattr(self, key):
            raise TypeError('Cannot add new field "%s" to frozen class %s' % (key, self))
        else:
            object.__setattr__(self, key, value)


@dataclass
class BranchMPCParams(PythonMsg):
    """封装 MPC 所需的全部静态配置参数,方便一次性传入控制器。"""

    n: int = field(default=None)  # 状态维度 size(x)
    d: int = field(default=None)  # 控制输入维度 size(u)
    NB: int = field(default=None)  # 分支深度(branch 数量)
    N: int = field(default=None)  # 同一分支内部的预测长度

    A: np.array = field(default=None)  # 预测模型矩阵(LTI 为单矩阵,LTV 为矩阵列表)
    B: np.array = field(default=None)  # 同上

    Q: np.array = field(default=np.array((n, n)))  # 状态二次型权重
    R: np.array = field(default=None)  # 控制量二次型权重
    Qf: np.array = field(default=None)  # 终端状态权重
    dR: np.array = field(default=None)  # 控制增量/平滑项的权重

    Qslack: float = field(default=None)  # 松弛变量权重向量 [线性权重, 二次权重]
    Fx: np.array = field(default=None)  # 状态约束 Fx * x <= bx
    bx: np.array = field(default=None)
    Fu: np.array = field(default=None)  # 控制约束 Fu * u <= bu
    bu: np.array = field(default=None)
    xRef: np.array = field(default=None)

    slacks: bool = field(default=False)  # 是否启用松弛变量
    timeVarying: bool = field(default=False)  # 是否使用时变线性化模型

    def __post_init__(self):
        """初始化缺省参数,确保后续矩阵运算不会因为 None 触发异常。"""
        if self.Qf is None:
            self.Qf = self.Q
        if self.dR is None:
            self.dR = np.zeros(self.d)
        if self.xRef is None:
            self.xRef = np.zeros(self.n)


############################################################################################
####################################### MPC CLASS ##########################################
############################################################################################
class BranchTree:
    def __init__(self, xtraj, ztraj, utraj, w, depth=0):
        """分支树节点:包含该场景下的状态/控制轨迹、概率与线性化信息。"""

        self.xtraj = xtraj  # 当前节点下的状态预测轨迹
        self.ztraj = ztraj  # 障碍物或其他车辆的预测轨迹
        self.utraj = utraj  # 控制输入预测轨迹
        self.dynmatr = [None] * xtraj.shape[0]  # 存储每一步线性化得到的 (A,B,C)
        self.w = w  # 当前节点的累积概率(根节点为 1)
        self.children = []  # 子节点列表
        self.depth = depth  # 当前节点的深度(根节点 0)
        self.p = None  # 子分支概率
        self.dp = None  # 子分支概率对状态的导数
        self.J = 0  # 节点代价,用于回溯累积

    def addchild(self, BT):
        """向节点添加子分支。"""
        self.children.append(BT)


class BranchMPC:
    """负责构建分支场景树、搭建 QP 并调用求解器的主控制器。"""

    def __init__(self, mpcParameters, predictiveModel):
        """初始化控制器核心数据结构."""
        self.N = mpcParameters.N  # 单个分支内的预测步数
        self.NB = mpcParameters.NB  # 分支深度
        self.Qslack = mpcParameters.Qslack  # 松弛成本参数
        self.Q = mpcParameters.Q  # 状态成本
        self.Qf = mpcParameters.Qf  # 终端状态成本
        self.R = mpcParameters.R  # 控制成本
        self.dR = mpcParameters.dR  # 控制增量成本
        self.n = mpcParameters.n  # 状态维度
        self.d = mpcParameters.d  # 控制维度
        self.Fx = mpcParameters.Fx  # 状态约束矩阵
        self.Fu = mpcParameters.Fu  # 控制约束矩阵
        self.bx = mpcParameters.bx  # 状态约束上界
        self.bu = mpcParameters.bu  # 控制约束上界
        self.xRef = mpcParameters.xRef  # 期望状态(轨迹跟踪目标)
        self.m = predictiveModel.m  # 每个节点的子分支数量

        self.slacks = mpcParameters.slacks  # 是否添加松弛变量
        self.slackweight = None  # 每条状态约束的概率权重
        self.timeVarying = mpcParameters.timeVarying  # 线性化模型是否随时间更新
        self.predictiveModel = predictiveModel  # 外部提供的预测模型
        self.osqp = None  # OSQP 求解器句柄
        self.BT = None  # 分支树根节点
        self.totalx = 0  # 所有节点的状态变量数量
        self.totalu = 0  # 所有节点的控制变量数量
        self.ndx = {}  # 每个节点在大状态向量中的起始索引
        self.ndu = {}  # 每个节点在大控制向量中的起始索引

        self.xPred = None  # 求解后得到的状态预测
        self.uPred = None  # 求解后得到的控制预测
        self.xLin = None  # 线性化基准状态
        self.uLin = None  # 线性化基准控制
        self.OldInput = np.zeros(self.d)  # 上一次应用的控制输入,用于平滑项

        # 初始化计时器,便于后续性能评估
        startTimer = datetime.datetime.now()
        endTimer = datetime.datetime.now()
        deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        self.linearizationTime = deltaTimer
        self.timeStep = 0  # 控制器调用计数器

    def inittree(self, x, z):
        """
        初始化分支树:从当前状态 x 与环境状态 z 出发生成根节点。

        - 每个节点保存上一轮求解得到的 x、u 轨迹,作为本轮线性化的基准。
        - 调用预测模型给出线性化矩阵 (A,B,C) 与子节点的参考轨迹。
        """
        u = np.zeros(self.d)
        # 根节点的状态/控制轨迹只有一步(当前时刻),概率为 1
        self.BT = BranchTree(np.reshape(x, [1, self.n]), np.reshape(z, [1, self.n]), np.reshape(u, [1, self.d]), 1, 0)
        q = [self.BT]  # BFS 队列
        countx = 0  # 累加全局状态向量的长度
        countu = 0  # 累加全局控制向量的长度
        self.uLin = np.reshape(u, [1, self.d])  # 初始化线性化基准
        self.xLin = np.reshape(x, [1, self.n])

        self.ndx[self.BT] = countx  # 根节点在大状态向量中的起点
        self.ndu[self.BT] = countu
        A, B, C, xp = self.predictiveModel.dyn_linearization(x, u)
        self.BT.dynmatr[0] = (A, B, C)
        countx += self.BT.xtraj.shape[0]
        countu += self.BT.xtraj.shape[0]

        while len(q) > 0:
            currentbranch = q.pop(0)

            if currentbranch.depth < self.NB:
                # 预测障碍物轨迹与分支概率
                zPred = self.predictiveModel.zpred_eval(currentbranch.ztraj[-1])
                p, dp = self.predictiveModel.branch_eval(currentbranch.xtraj[-1], currentbranch.ztraj[-1])
                currentbranch.p = p
                currentbranch.dp = dp
                for i in range(0, self.m):
                    # 为每个分支创建新的轨迹容器
                    xtraj = np.zeros((self.N, self.n))
                    utraj = np.zeros((self.N, self.d))
                    newbranch = BranchTree(
                        xtraj,
                        zPred[:, self.n * i:self.n * (i + 1)],
                        utraj,
                        p[i] * currentbranch.w,
                        currentbranch.depth + 1
                    )
                    # 使用父节点末端状态做一次前向仿真,得到该子节点的第一步预测
                    A, B, C, xp = self.predictiveModel.dyn_linearization(
                        currentbranch.xtraj[-1],
                        currentbranch.utraj[-1]
                    )
                    newbranch.xtraj[0] = xp
                    for t in range(0, self.N):
                        # 对每一步 (x,u) 做线性化,并保存 (A,B,C)
                        A, B, C, xp = self.predictiveModel.dyn_linearization(newbranch.xtraj[t], newbranch.utraj[t])
                        newbranch.dynmatr[t] = (A, B, C)
                        if t < self.N - 1:
                            newbranch.xtraj[t + 1] = xp

                    # 为新节点分配在全局向量中的偏移
                    self.ndx[newbranch] = countx
                    self.ndu[newbranch] = countu

                    # 保存线性化基准轨迹
                    self.xLin = np.vstack((self.xLin, newbranch.xtraj))
                    self.uLin = np.vstack((self.uLin, newbranch.utraj))
                    if newbranch.depth == self.NB:
                        # 叶子节点需要额外多一个状态用于终端约束
                        countx += (newbranch.xtraj.shape[0] + 1)
                    else:
                        countx += newbranch.xtraj.shape[0]
                    countu += newbranch.xtraj.shape[0]
                    currentbranch.addchild(newbranch)
                    q.append(newbranch)
        self.totalx = countx
        self.totalu = countu
        # slackweight 的长度 = (每个状态约束个数)*(所有状态变量个数)
        self.slackweight = np.zeros(self.totalx * (self.Fx.shape[0] + 1))

    def buildEqConstr(self):
        """
        构建线性等式约束 G * z = E * x(t) + L。

        - G = [Gx  Gu] 负责串联各个节点的动力学约束。
        - E 把当前实测状态注入到 QP 中,只在根节点出现一次。
        - L 存放线性化模型的常数项 C。
        """
        Gx = np.eye(self.totalx * self.n)  # 主对角为 1,用于复制 x(k+1)
        Gu = np.zeros((self.totalx * self.n, self.totalu * self.d))

        E = np.zeros((self.totalx * self.n, self.n))
        E[0:self.n] = np.eye(self.n)  # 只有根节点受真实状态影响

        L = np.zeros(self.totalx * self.n)
        self.E = E

        for branch in self.ndx:
            l = branch.xtraj.shape[0]
            ndx = self.ndx[branch]
            ndu = self.ndu[branch]
            for t in range(1, l):
                # 对应 x_{t+1} = A x_t + B u_t + C
                A, B, C = branch.dynmatr[t - 1]
                Gx[(ndx + t) * self.n:(ndx + t + 1) * self.n,
                   (ndx + t - 1) * self.n:(ndx + t) * self.n] = -A
                Gu[(ndx + t) * self.n:(ndx + t + 1) * self.n,
                   (ndu + t - 1) * self.d:(ndu + t) * self.d] = -B
                L[(ndx + t) * self.n:(ndx + t + 1) * self.n] = C
            A, B, C = branch.dynmatr[-1]
            if branch.depth < self.NB:
                # 子节点的初始状态由父节点最后一个时刻推动
                for child in branch.children:
                    ndxc = self.ndx[child]
                    Gx[ndxc * self.n:(ndxc + 1) * self.n,
                       (ndx + l - 1) * self.n:(ndx + l) * self.n] = -A
                    Gu[ndxc * self.n:(ndxc + 1) * self.n,
                       (ndu + l - 1) * self.d:(ndu + l) * self.d] = -B
                    L[ndxc * self.n:(ndxc + 1) * self.n] = C
            else:
                # 叶子节点额外增加终端状态 x_{l}
                Gx[(ndx + l) * self.n:(ndx + l + 1) * self.n,
                   (ndx + l - 1) * self.n:(ndx + l) * self.n] = -A
                Gu[(ndx + l) * self.n:(ndx + l + 1) * self.n,
                   (ndu + l - 1) * self.d:(ndu + l) * self.d] = -B
                L[(ndx + l) * self.n:(ndx + l + 1) * self.n] = C
        self.L = L

        if self.slacks:
            # 如果有松弛变量,需要在等式约束中补零列
            self.G = np.hstack((Gx, Gu, np.zeros((Gx.shape[0], self.slackweight.shape[0]))))
        else:
            self.G = np.hstack((Gx, Gu))

    def updatetree(self, x, z):
        """
        在下一次求解前刷新已有分支树:
        - 把上一轮求解得到的控制序列右移,实现 warm start。
        - 重新线性化每个节点的动力学。
        - 更新分支概率与障碍物预测轨迹。
        """
        for branch in self.ndx:
            l = branch.utraj.shape[0]
            # 把上一轮求出的控制序列向前平移一格以 warm start
            branch.utraj[0:l - 1] = self.uLin[self.ndu[branch] + 1:self.ndu[branch] + l]
            if branch.depth < self.NB:
                idx = np.argmax(branch.p)
                ndu = self.ndu[branch.children[idx]]
                branch.utraj[-1] = self.uLin[ndu]  # 末端控制继承概率最大的子分支
            else:
                branch.utraj[-1] = branch.utraj[-2]
        self.BT.ztraj = np.reshape(z, [1, self.n])
        self.BT.xtraj = np.reshape(x, [1, self.n])
        for i in range(0, self.BT.xtraj.shape[0]):
            A, B, C, xp = self.predictiveModel.dyn_linearization(self.BT.xtraj[i], self.BT.utraj[i])
            self.BT.dynmatr[i] = (A, B, C)
        q = [self.BT]

        while len(q) > 0:
            currentbranch = q.pop(0)
            if currentbranch.depth < self.NB:
                # 更新每个节点的概率与障碍物轨迹
                zPred = self.predictiveModel.zpred_eval(currentbranch.ztraj[-1])
                p, dp = self.predictiveModel.branch_eval(currentbranch.xtraj[-1], currentbranch.ztraj[-1])
                currentbranch.p = p
                currentbranch.dp = dp
                for i in range(0, self.m):
                    child = currentbranch.children[i]
                    child.w = currentbranch.w * p[i]
                    child.ztraj = zPred[:, i * self.n:(i + 1) * self.n]
                    A, B, C, xp = self.predictiveModel.dyn_linearization(currentbranch.xtraj[-1],
                                                                         currentbranch.utraj[-1])
                    child.xtraj[0] = xp
                    for t in range(0, self.N):
                        A, B, C, xp = self.predictiveModel.dyn_linearization(child.xtraj[t], child.utraj[t])
                        child.dynmatr[t] = (A, B, C)
                        if t < self.N - 1:
                            child.xtraj[t + 1] = xp

                    q.append(child)

    def buildCost(self):
        """
        构建二次型成本 H、线性项 q:
        - 对每个场景节点累加状态成本(含概率权重)。
        - 对输入引入 R 和 dR,并加入上一时刻控制作为平滑偏差。
        - 若启用松弛变量,则在 H 中追加对应的对角块。
        """
        listQ = [None] * (self.totalx)
        Hu = np.zeros([self.totalu * self.d, self.totalu * self.d])
        qx = np.zeros(self.totalx * self.n)
        dQ = self.Q * 0.5  # 用于近似积分中点项
        for branch in self.ndx:
            ndx = self.ndx[branch]
            ndu = self.ndu[branch]
            l = branch.utraj.shape[0]
            for i in range(0, l - 1):
                listQ[ndx + i] = (dQ + self.Q) * branch.w
                qx[(ndx + i) * self.n:(ndx + i + 1) * self.n] = -2 * branch.w * (
                        np.dot(self.xRef, self.Q) + np.dot(branch.xtraj[i], dQ))
                Hu[(ndu + i) * self.d:(ndu + i + 1) * self.d,
                (ndu + i) * self.d:(ndu + i + 1) * self.d] = branch.w * self.R

            if branch.depth < self.NB:
                # 非叶子节点的最后一步还需连接到子节点,以概率加权的方式考虑未来代价
                Hu[(ndu + l - 1) * self.d:(ndu + l) * self.d,
                (ndu + l - 1) * self.d:(ndu + l) * self.d] = branch.w * self.R

                listQ[ndx + l - 1] = (dQ + self.Q) * branch.w
                childJ = np.zeros(self.m)  # 子分支的累积代价,用于近似期望
                for j in range(0, self.m):
                    childJ[j] = branch.children[j].J

                qx[(ndx + l - 1) * self.n:(ndx + l) * self.n] = branch.w * (
                        -2 * np.dot(self.xRef, self.Q) - 2 * np.dot(branch.xtraj[-1], dQ) + np.dot(childJ,
                                                                                                   branch.dp))

            else:
                # 叶子节点:加入终端成本 Qf
                Hu[(ndu + l - 1) * self.d:(ndu + l) * self.d,
                (ndu + l - 1) * self.d:(ndu + l) * self.d] = branch.w * self.R
                listQ[ndx + l - 1] = (dQ + self.Q) * branch.w
                listQ[ndx + l] = self.Qf * branch.w
                qx[(ndx + l - 1) * self.n:(ndx + l) * self.n] = -2 * branch.w * (
                        np.dot(self.xRef, self.Qf) + np.dot(branch.xtraj[-1], dQ))

        Hx = linalg.block_diag(*listQ)
        qu = np.zeros(self.totalu * self.d)
        qu[0:self.d] = -2 * self.OldInput @ self.dR  # 控制平滑项:惩罚与上一控制的偏差

        q = np.append(qx, qu)  # 合并状态与控制的线性项

        if self.slacks:
            quadSlack = self.Qslack[0] * np.eye(self.slackweight.shape[0])
            linSlack = self.Qslack[1] * self.slackweight
            self.H = linalg.block_diag(Hx, Hu, quadSlack)
            self.q = np.append(q, linSlack)
        else:
            self.H = linalg.block_diag(Hx, Hu)
            self.q = q
        self.H = 2 * self.H  # CVX/QP 约定成本为 (1/2)x^T H x,因此需乘 2

    def buildIneqConstr(self):
        """
        构建不等式约束 F z <= b:
        - 状态部分包括碰撞约束与物理限制；
        - 控制部分为输入幅值约束；
        - 可选的松弛变量仅作用于状态约束。
        """
        Nc = self.Fx.shape[0] + 1  # 每个节点的状态约束 = 1 条碰撞约束 + Fx 行数
        slackweight_x = np.zeros(self.totalx * Nc)

        Fxtot = np.zeros([Nc * self.totalx, self.totalx * self.n])
        bxtot = np.zeros(Nc * self.totalx)
        for branch in self.ndx:
            l = branch.utraj.shape[0]
            for i in range(0, l):
                # 碰撞约束线性化:h(x,z) + dh * (x - x0) >= 0
                h, dh = self.predictiveModel.col_eval(branch.xtraj[i], branch.ztraj[i])
                idx = self.ndx[branch] + i
                Fxtot[idx * Nc:(idx + 1) * Nc, idx * self.n:(idx + 1) * self.n] = np.vstack((-dh, self.Fx))
                bxtot[idx * Nc:(idx + 1) * Nc] = np.append(h, self.bx)
                slackweight_x[idx * Nc:(idx + 1) * Nc] = branch.w  # 松弛成本按概率加权

        self.slackweight = slackweight_x
        rep_b = [self.Fu] * (self.totalu)
        Futot = linalg.block_diag(*rep_b)
        butot = np.tile(np.squeeze(self.bu), self.totalu)

        F_hard = linalg.block_diag(Fxtot, Futot)

        if self.slacks:
            nc_x = Fxtot.shape[0]  # 仅状态约束可松弛
            # 在原有约束矩阵右侧附加 -I,实现 (Fx)x - s <= b
            addSlack = np.zeros((F_hard.shape[0], nc_x))
            addSlack[0:nc_x, 0:nc_x] = -np.eye(nc_x)
            # 再添加 s >= 0
            I = - np.eye(nc_x)
            Zeros = np.zeros((nc_x, F_hard.shape[1]))
            Positivity = np.hstack((Zeros, I))

            self.F = np.vstack((np.hstack((F_hard, addSlack)), Positivity))
            self.b = np.hstack((bxtot, butot, np.zeros(nc_x)))
        else:
            self.F = F_hard
            self.b = np.hstack((bxtot, butot))

    def updateIneqConstr(self):
        """在树结构不变的情况下,仅更新线性化后的碰撞约束,降低计算负担。"""
        Nc = self.Fx.shape[0] + 1
        for branch in self.ndx:
            l = branch.utraj.shape[0]
            for i in range(0, l):
                h, dh = self.predictiveModel.col_eval(branch.xtraj[i], branch.ztraj[i])
                idx = self.ndx[branch] + i
                self.F[idx * Nc, idx * self.n:(idx + 1) * self.n] = -dh
                self.b[idx * Nc] = h
                self.slackweight[idx * Nc:(idx + 1) * Nc] = branch.w

    def solve(self, x, z, xRef=None):
        """
        主入口:根据当前状态 x 与环境 z 重新构建/更新 QP 并求解控制量。

        参数:
            x: 自车(ego)当前状态
            z: 其他交通参与者/障碍物状态
            xRef: 可选的参考状态,若提供则覆盖默认值
        """

        if not xRef is None:
            self.xRef = xRef
        if self.BT is None:
            self.inittree(x, z)
            # self.buildIneqConstr()
        else:
            self.updatetree(x, z)
            # self.updateIneqConstr()
        '''
        以下代码演示如何搭建并求解完整的 QP。
        若要实际运行,将此段注释去掉即可。
        self.buildCost()
        self.buildEqConstr()

        self.H_FTOCP = sparse.csc_matrix(self.H)
        self.q_FTOCP = self.q
        self.F_FTOCP = sparse.csc_matrix(self.F)
        self.b_FTOCP = self.b
        self.G_FTOCP = sparse.csc_matrix(self.G)
        self.E_FTOCP = self.E
        self.L_FTOCP = self.L
        # 调用求解器,完成 QP 求解
        startTimer = datetime.datetime.now()
        self.osqp_solve_qp(self.H_FTOCP, self.q_FTOCP, self.F_FTOCP, self.b_FTOCP, self.G_FTOCP,
                           np.add(np.dot(self.E_FTOCP, x), self.L_FTOCP))
        self.unpackSolution()
        endTimer = datetime.datetime.now();
        deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        # 可在此打印 self.solverTime.total_seconds() 查看求解耗时

        # 将解决方案的第一步控制作为“上一控制”,以便加入 dR 惩罚
        self.OldInput = self.uPred[0, :]
        '''
        self.timeStep += 1

    def addTerminalComponents(self):
        """在需要额外终端约束时,可调用该函数向 QP 中追加对应块。"""
        self.H_FTOCP = sparse.csc_matrix(self.H)
        self.q_FTOCP = self.q
        self.F_FTOCP = sparse.csc_matrix(self.F)
        self.b_FTOCP = self.b
        self.G_FTOCP = sparse.csc_matrix(self.G)
        self.E_FTOCP = self.E
        self.L_FTOCP = self.L

    def unpackSolution(self):
        """把 QP 求解结果展开成多分支的 x、u 轨迹,供下次线性化使用。"""
        if self.feasible:
            self.xPred = np.squeeze(
                np.transpose(np.reshape((self.Solution[np.arange(self.totalx * self.n)]), (-1, self.n)))).T
            # 先 reshape 成 [totalx, n],再恢复成 (时间 × 状态) 的矩阵
            self.uPred = np.squeeze(np.transpose(
                np.reshape((self.Solution[self.totalx * self.n + np.arange(self.totalu * self.d)]), (-1, self.d)))).T
            # 同理处理控制变量
            self.xLin = self.xPred
            self.uLin = self.uPred
            self.uLin = np.vstack((self.uLin, self.uLin[-1]))  # 末尾重复一行,便于 warm start

    def BT2array(self):
        """将分支树打平成数组,方便画图或调试。"""
        ztraj = []
        xtraj = []
        utraj = []
        branch_w = []
        q = [self.BT]
        while (len(q) > 0):
            curr = q.pop(0)  # BFS 逐层展开
            for child in curr.children:
                branch_w.append(child.w)  # 保存分支概率
                ztraj.append(np.vstack((curr.ztraj[-1], child.ztraj)))  # 拼接父子轨迹,便于可视化
                xtraj.append(np.vstack((curr.xtraj[-1], child.xtraj)))
                utraj.append(np.vstack((curr.utraj[-1], child.utraj)))
                q.append(child)
        return xtraj, ztraj, utraj, branch_w

    def osqp_solve_qp(self, P, q, G=None, h=None, A=None, b=None, initvals=None):
        """
        使用 OSQP 求解标准形式的二次规划问题:

        minimize  (1/2) * x^T P x + q^T x
        subject to G x <= h, A x = b

        参数已经按稀疏矩阵格式准备好,此函数只负责调用 OSQP 并缓存结果。
        """
        qp_A = vstack([G, A]).tocsc()  # 把不等式与等式堆叠成一套稀疏矩阵
        l = -inf * ones(len(h))  # 不等式的下界为 -inf(即仅有上界约束)
        qp_l = hstack([l, b])
        qp_u = hstack([h, b])

        self.osqp = OSQP()

        self.osqp.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False, polish=True)
        if initvals is not None:
            self.osqp.warm_start(x=initvals)
        res = self.osqp.solve()
        if res.info.status_val == 1:
            self.feasible = 1  # OSQP 返回 1 代表成功找到最优解
        else:
            self.feasible = 0  # 其余状态视为不可行

        self.Solution = res.x
