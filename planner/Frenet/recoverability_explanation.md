# Recoverability 说明

这个文档说明当前代码中 `recoverable shared plan`、`recoverability_indicator` 等概念到底代表什么。

## 1. shared plan 是什么

在当前实现里，规划分成两层：

- **shared plan**：先规划出一段“所有场景共用”的主轨迹前半段。
- **contingent plan**：当后续真实场景逐渐明确后，再针对某个具体场景接上的后半段轨迹。

代码位置：

- `planner/Frenet/frenet_planner.py:827`
- `planner/Frenet/frenet_planner.py:853`

也就是说，一个完整的 `final_plan` 结构大致是：

- `final_plan["shared_plan"]`
- `final_plan[0]`：针对第 0 个 credible joint scenario 的 contingent plan
- `final_plan[1]`：针对第 1 个 credible joint scenario 的 contingent plan
- ...

## 2. recoverable shared plan 是什么

当前代码里的定义非常具体：

> 如果一条 `shared plan` 在当前 **credible joint scenario set** 中，对每一个可信联合场景都至少存在一条有效的 contingent plan，那么这条 shared plan 就叫 **recoverable shared plan**。

对应判断函数：

- `planner/Frenet/frenet_planner.py:740`

当前逻辑：

- 如果当前根本不需要 contingency，直接视为 `recoverable = True`
- 如果当前 credible set 为空，也直接视为 `recoverable = True`
- 否则，必须对 credible set 里的每一个 mode 都找到可行 contingent plan
- 只要缺一个 mode，对应 shared plan 就是 `recoverable = False`

## 3. 为什么要这样定义

因为 shared plan 是先执行出去的那一段。

如果这段 shared plan 一旦执行之后，面对某个仍然“可信”的后续场景，已经没有任何有效后手可以接，那这条 shared plan 就是不稳妥的。

所以 recoverability 的核心含义是：

> **现在先走这一步，后面是否还留有足够的应对空间。**

这是一种“保留后手”的约束，不是单纯看当前这一瞬间风险低不低。

## 4. 它不等于什么

`recoverable shared plan` 不等于：

- 风险一定最低
- 当前一定最安全
- 一定不会碰撞
- 代价一定最小

它只表示：

- 对当前可信场景集来说，这条 shared plan **后续可接续**
- 不会出现“shared 先走出去了，但某个可信场景下已经无路可走”的情况

## 5. 在代码中怎么使用

当前实现中，只有 `recoverable == True` 的 `final_plan` 才会进入最终候选集合：

- `planner/Frenet/frenet_planner.py:923`

也就是说，最终排序选优之前，先做了一层 recoverability 过滤。

所以主车最终不会从“不可恢复”的 shared plan 里选解。

## 6. recoverability_history 里的几个量分别是什么意思

### `shared_plan_count`

当前时刻通过 shared 阶段有效性筛选后的 shared plan 数量。

### `recoverable_shared_plan_count`

这些 shared plan 里，有多少条对当前 credible set 是“全覆盖可恢复”的。

### `recoverability_ratio`

定义为：

`recoverable_shared_plan_count / shared_plan_count`

表示当前时刻有效 shared plan 中，有多大比例是可恢复的。

### `recoverability_indicator`

当前新增的二值指示量，定义为：

- `1`：当前时刻至少存在一条 recoverable shared plan
- `0`：当前时刻一条 recoverable shared plan 都没有

它回答的是一个更直接的问题：

> **此刻系统还有没有“至少一个带后手的选择”？**

## 7. 一个直观例子

假设当前 credible joint scenario set 里有 3 个可信场景：

- 场景 A
- 场景 B
- 场景 C

对某条 shared plan：

- A 下能找到 contingent plan
- B 下能找到 contingent plan
- C 下找不到 contingent plan

那么这条 shared plan 就是：

- `recoverable = False`

因为它不能覆盖全部 credible 场景。

只有当：

- A / B / C 三种场景下都能接出有效 contingent plan

时，这条 shared plan 才是：

- `recoverable = True`

## 8. 当前实现的一个重要特点

现在 recoverability 是基于 **credible joint scenario set** 检查的，而不是基于“所有联合场景”检查的。

这意味着：

- 我们不要求对极小概率场景全部保守覆盖
- 只要求对“累计后验概率至少覆盖 `1 - a`”的那组最小可信场景集做可恢复性保证

这样做会比“对所有联合场景全覆盖”更实用，也更不容易过度保守。

## 9. 一句话总结

当前代码里：

> **recoverable shared plan = 一条已经通过 shared 阶段筛选的主轨迹，并且对于当前 credible joint scenario set 中的每一个可信场景，都还存在至少一条有效 contingent 轨迹可以接上。**

