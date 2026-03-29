按**当前代码真实流程**讲，**每个规划周期的分支时刻 `branch time` 是先确定、再用于生成 shared 段轨迹的**，而且它是一个**该周期全局统一的分支时刻**，不是对每一条 shared plan 单独选的。

**核心结论**
- 本周期先做多模态预测，得到各障碍车的 mode 概率与未来分布。见 `planner/Frenet/frenet_planner.py:546`
- 再构造**联合场景集**，并截取**可信联合场景集**。见 `planner/Frenet/frenet_planner.py:630`、`planner/Frenet/frenet_planner.py:651`
- 然后仅基于这个可信联合场景集，扫描一组候选分支时刻，计算每个候选时刻的**场景可分性**。
- 选取**最早满足阈值**的时刻作为本周期分支时刻；若都不满足，则退回**最晚候选时刻**。见 `planner/Frenet/frenet_planner.py:1461`
- 选完后立刻令 `shared_t_list = [selected_branch_time]`，这意味着：**本周期 shared 段长度就等于这个分支时刻**。见 `planner/Frenet/frenet_planner.py:677`

**1. 每个周期一开始，默认分支时刻是什么**
- 代码一开始先把 shared horizon 设成 `self.frenet_parameters["t_list"]` 的最大值。见 `planner/Frenet/frenet_planner.py:529`
- 当前配置里 `planner/Frenet/configs/contingency.json:4` 给的是 `t_list = [2.6]`
- 所以如果后面自适应逻辑失效，默认就会退回到 **2.6 s**

**2. 候选分支时刻是怎么来的**
候选时刻由 `_get_adaptive_branching_config()` 生成。见 `planner/Frenet/frenet_planner.py:1297`

当前你的配置是：
- `enabled = true`
- `min_branch_time = 0.4`
- `candidate_dt = 0.1`
- `separability_threshold = 1.0`
见 `planner/Frenet/configs/contingency.json:14`

因为没有显式给 `max_branch_time`，代码默认取：
- `max(self.frenet_parameters["t_list"]) = 2.6`
见 `planner/Frenet/frenet_planner.py:1306`

所以当前这一版里，候选分支时刻实际就是：
- `0.4, 0.5, 0.6, ..., 2.6`

对应代码：
- 生成候选时间网格：`planner/Frenet/frenet_planner.py:1346`
- 阈值读取：`planner/Frenet/frenet_planner.py:1361`

**3. 为什么不是对所有联合场景都算，而是只看可信联合场景集**
先把所有多模态障碍物做笛卡尔积，得到联合场景 `joint_mode_selections`。见 `planner/Frenet/frenet_planner.py:631`

然后用联合后验概率选一个**最小覆盖 95% 概率质量**的可信子集：
- `alpha = 0.05`
- 即累计概率至少覆盖 `1 - alpha = 0.95`
见 `planner/Frenet/frenet_planner.py:651`、`planner/Frenet/frenet_planner.py:1242`

所以后续分支时刻选择，不是看全部联合场景，而是看：
- `credible_joint_mode_selections`

这也是为什么它叫“可信场景集驱动的自适应分支”。

**4. 某个候选时刻的“可分性”到底怎么算**
核心函数是 `_compute_joint_separability_at_step()`。见 `planner/Frenet/frenet_planner.py:1394`

对某个候选时刻 `t_k`：
- 先转成离散预测步 `future_step_idx = round(t_k / dt)`。见 `planner/Frenet/frenet_planner.py:1491`
- 然后遍历可信联合场景集中的**所有两两场景对**
- 对每一对场景：
  - 遍历其中涉及的每个 obstacle
  - 取该 obstacle 在这两个场景下对应 mode 的**均值轨迹**和**协方差轨迹**
  - 在 `future_step_idx` 那一帧上，计算这两个高斯分布的 **Bhattacharyya distance**
  - 把各 obstacle 的距离加起来，得到这对联合场景的距离
见 `planner/Frenet/frenet_planner.py:1413`、`planner/Frenet/frenet_planner.py:1447`

最后：
- 该候选时刻的整体可分性 `Sep_k`
- 取的是**可信联合场景集里最难区分那一对场景的距离**，即 pairwise distance 的最小值
见 `planner/Frenet/frenet_planner.py:1403`、`planner/Frenet/frenet_planner.py:1457`

所以本质上是：
- **最坏场景对可分性准则**

**5. 最终分支时刻怎么选**
核心函数 `_select_adaptive_branch_time()`。见 `planner/Frenet/frenet_planner.py:1461`

流程是：
- 先对所有 `candidate_times` 逐个算 `separability_series`
- 然后按时间从早到晚扫描
- 找到**第一个满足**
  - `Sep_k >= separability_threshold`
  的候选时刻
- 当前阈值就是 `1.0`
见 `planner/Frenet/frenet_planner.py:1499`

如果找到了：
- `selection_reason = "first_separable_candidate"`
见 `planner/Frenet/frenet_planner.py:1502`

如果一个都没找到：
- 就退回最后一个候选时刻，即当前配置下通常是 `2.6 s`
- `selection_reason = "fallback_latest_branch_time"`
见 `planner/Frenet/frenet_planner.py:1500`

**6. 哪些情况下根本不会进入“真正的自适应选择”**
代码里有三种直接回默认值的情况：

- 自适应分支没开：`adaptive_branching_disabled`
- 没有预测：`no_predictions`
- 可信联合场景数少于 2：`insufficient_credible_joint_modes`

见 `planner/Frenet/frenet_planner.py:1478`、`planner/Frenet/frenet_planner.py:1481`、`planner/Frenet/frenet_planner.py:1485`

这三种情况下，返回的都是默认分支时刻：
- `default_branch_time = max(candidate_times)`
见 `planner/Frenet/frenet_planner.py:1468`

**7. 这个分支时刻在本周期里怎么被用掉**
一旦选出来，立刻执行：

- `shared_branch_time = selected_branch_time`
- `shared_t_list = [shared_branch_time]`
- `shared_start_idx = round(shared_branch_time / dt)`

见 `planner/Frenet/frenet_planner.py:677`

这意味着：
- 本周期 shared trajectory 的时长就只有这一种
- 后面生成 shared trajectories 时，统一按这个时长生成
见 `planner/Frenet/frenet_planner.py:688`

然后在 contingent planning 阶段：
- 所有 contingent branch 都从 `shared_start_idx` 开始接上
见 `planner/Frenet/frenet_planner.py:921`

所以你可以把它理解成：
- **先决定“在哪个时刻切开 shared / branch”**
- 再按这个切点去生成 shared 段和各场景对应的 branch 段

**8. 可恢复性是在什么时候起作用的**
这个很关键：
- **可恢复性不参与分支时刻选择**
- 它是在分支时刻已经确定之后，shared 轨迹已经生成之后，才开始检查的

具体顺序是：
1. 先选 `branch time`
2. 再生成 shared trajectories
3. 再对每条 shared trajectory，在每个 credible joint mode 下生成 contingent branch
4. 若某条 shared trajectory 不能为所有可信场景都提供后续可行 branch，则判为 `recoverable=False`
见 `planner/Frenet/frenet_planner.py:759`、`planner/Frenet/frenet_planner.py:906`、`planner/Frenet/frenet_planner.py:932`

所以当前代码逻辑是：
- **分支时刻由“场景可分性”选**
- **shared 轨迹保留与否由“可恢复性”筛**

**9. 一句最凝练的数学化理解**
本周期分支时刻 `t_b` 的选择规则，当前代码等价于：

- 在候选集合 `T = {0.4, 0.5, ..., 2.6}` 中
- 对每个 `t in T` 计算可信联合场景集的最坏对间可分性 `Sep(t)`
- 取最早满足 `Sep(t) >= 1.0` 的 `t`
- 若不存在，则取 `max(T)`

**10. 为什么这套机制合理**
它背后的直觉是：
- 太早分支：场景还没拉开，分了也没信息价值
- 太晚分支：虽然更容易区分，但 shared 段过长，会压缩后续 contingent 的反应空间
- 所以当前代码选的是：
  - **最早达到“足够可分”标准的时刻**

这就是“自适应分支时刻”相对固定 1.4 s 的核心差别。

