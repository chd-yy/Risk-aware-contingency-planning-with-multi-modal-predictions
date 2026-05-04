# Adaptive Branching 说明

这个文档说明当前代码里“自适应分支时刻（adaptive branching time）”是如何选出来的。

当前实现采用的是**方案 A（全局近似版）**：

- 不是对每一条 `shared plan` 单独选分支时刻
- 而是先基于当前可信联合场景集，计算一个**全局的分支时刻**
- 然后所有本时刻生成的 shared trajectories 都使用这个统一的 branch time

---

## 1. 当前实现放在什么位置

主调用链在：

- `planner/Frenet/frenet_planner.py:539`
- `planner/Frenet/frenet_planner.py:666`
- `planner/Frenet/frenet_planner.py:670`

核心函数在：

- 配置读取：`planner/Frenet/frenet_planner.py:1274`
- 场景可分性计算：`planner/Frenet/frenet_planner.py:1371`
- 分支时刻选择：`planner/Frenet/frenet_planner.py:1438`
- 历史记录：`planner/Frenet/frenet_planner.py:1497`

---

## 2. 整体流程

每个 planning timestep，当前代码按下面流程选分支时刻：

1. 先做多模态预测，得到每个 obstacle 的：
   - mode 均值轨迹 `pos_list`
   - mode 协方差序列 `cov_list`
   - mode 概率 `mode_prob`
2. 构造联合场景集合 `joint_mode_selections`
3. 从中截取可信联合场景集 `credible_joint_mode_selections`
4. 在一组候选未来时刻上计算 `Sep_k`
5. 选取**最早满足** `Sep_k >= eps_sep` 的候选时刻
6. 如果没有任何候选时刻满足阈值，就退回到**最晚候选时刻**
7. 用这个选中的 `branch time` 作为本 timestep 的 shared horizon

---

## 3. 输入数据从哪里来

### 3.1 单障碍物多模态预测

在预测阶段，代码会构造每个障碍物的双模态预测（目前是 `yield/challenge`）：

- `planner/Frenet/frenet_planner.py:597`
- `planner/Frenet/utils/prediction_helpers.py:887`

每个障碍物最终会有：

- `pos_list[mode][future_step] = [x, y]`
- `cov_list[mode][future_step] = 2x2 covariance`
- `mode_prob[mode]`

---

### 3.2 联合场景集合

如果有多个障碍物，每个障碍物各有多个 mode，那么代码会做笛卡尔积，构造联合场景：

- `planner/Frenet/frenet_planner.py:623`

例如：

- obstacle 20044: `yield/challenge`
- obstacle 20087: `yield/challenge`

那么联合场景就是：

- `{20044: yield, 20087: yield}`
- `{20044: yield, 20087: challenge}`
- `{20044: challenge, 20087: yield}`
- `{20044: challenge, 20087: challenge}`

---

### 3.3 可信联合场景集

代码不会直接对所有联合场景做分支时刻选择，而是先取**可信联合场景集**：

- `planner/Frenet/frenet_planner.py:644`
- `planner/Frenet/frenet_planner.py:1231`

当前可信集定义为：

- 先按联合后验概率从大到小排序
- 取元素最少的一组场景
- 使其累计概率至少覆盖 `1 - alpha`

其中：

- `alpha = 0.05`

所以当前默认是覆盖至少 `95%` 的后验概率质量。

---

## 4. 候选分支时刻是怎么来的

候选时刻配置在：

- `planner/Frenet/configs/contingency.json:14`

当前默认配置：

- `enabled = true`
- `min_branch_time = 0.4`
- `candidate_dt = 0.1`
- `separability_threshold = 1.0`

候选时刻的构造逻辑在：

- `planner/Frenet/frenet_planner.py:1274`

规则如下：

1. 如果显式提供了 `candidate_times`，就直接用它
2. 否则，如果 `self.frenet_parameters["t_list"]` 有多个值，就从里面筛
3. 否则，就按
   - 从 `min_branch_time`
   - 到 `max_branch_time`
   - 步长 `candidate_dt`
   均匀生成

当前代码里如果没有显式配置 `max_branch_time`，会默认取：

- `max(self.frenet_parameters["t_list"])`

也就是说，**分支时刻的搜索上限默认不会超过 shared horizon 上限**。

---

## 5. `Sep_k` 是怎么计算的

### 5.1 单对联合场景之间的距离

代码使用的是 **Bhattacharyya distance**：

- `planner/Frenet/frenet_planner.py:1343`

对两个联合场景 `theta_a` 和 `theta_b`，代码会：

1. 遍历这两个联合场景涉及的所有 obstacle
2. 对每个 obstacle，在同一个未来步 `k`：
   - 取 `theta_a` 下对应 mode 的均值/协方差
   - 取 `theta_b` 下对应 mode 的均值/协方差
   - 计算这两个高斯分布的 Bhattacharyya distance
3. 把所有 obstacle 的距离加起来，得到这两个联合场景在步 `k` 的总距离

对应实现：

- `planner/Frenet/frenet_planner.py:1381`
- `planner/Frenet/frenet_planner.py:1424`

---

### 5.2 整个可信场景集的 `Sep_k`

对于一个给定未来步 `k`，代码会：

1. 枚举可信场景集内所有两两组合
2. 分别计算它们的 pairwise distance
3. 取其中的**最小值**

对应实现：

- `planner/Frenet/frenet_planner.py:1371`
- `planner/Frenet/frenet_planner.py:1434`

所以当前定义是：

> `Sep_k = 可信联合场景集内最难区分那一对场景，在未来步 k 的距离`

这和论文里“用最难区分场景对作为整体可分性指标”的思想是一致的。

---

## 6. 分支时刻怎么选

真正做选择的是：

- `planner/Frenet/frenet_planner.py:1438`

逻辑非常直接：

1. 先拿到候选时刻列表 `candidate_times`
2. 对每个候选时刻 `t_k`：
   - 转成未来离散步 `future_step_idx`
   - 计算该时刻的 `Sep_k`
3. 得到一条 `separability_series`
4. 从前往后扫描
5. 选取第一个满足

`Sep_k >= separability_threshold`

的候选时刻

如果一个都没有满足，则：

- 选择最后一个候选时刻
- `selection_reason = "fallback_latest_branch_time"`

如果找到了最早满足阈值的候选，则：

- `selection_reason = "first_separable_candidate"`

---

## 7. 什么情况下不会真的做自适应选择

函数 `_select_adaptive_branch_time(...)` 里有几个直接返回的情况：

- `planner/Frenet/frenet_planner.py:1455`

### 7.1 adaptive branching 没开启

返回：

- `selection_reason = "adaptive_branching_disabled"`

### 7.2 当前没有预测

返回：

- `selection_reason = "no_predictions"`

### 7.3 可信联合场景数少于 2

返回：

- `selection_reason = "insufficient_credible_joint_modes"`

因为如果可信场景只有 0 个或 1 个，就不存在“场景之间能否区分”的问题，也就没法定义真正的 pairwise separability。

在这些情况下，代码都会退回到默认分支时刻：

- `default_branch_time = max(candidate_times)`

---

## 8. 选出来之后怎么用

选出的 `branch time` 会立刻替换 shared 段时长：

- `planner/Frenet/frenet_planner.py:670`
- `planner/Frenet/frenet_planner.py:671`

也就是：

- `shared_t_list = [selected_branch_time]`

后面生成 shared trajectories 时，使用的就是这个新的时长：

- `planner/Frenet/frenet_planner.py:697`
- `planner/Frenet/frenet_planner.py:707`

并且 contingent 评估的起始索引也会跟着改：

- `planner/Frenet/frenet_planner.py:672`

这意味着：

> 当前代码中，自适应分支时刻不仅是“记录一个指标”，而是真的改变了 shared horizon 的长度。

---

## 9. 当前实现和论文完整版的差别

当前实现是**近似版方案 A**，不是严格版。

### 当前方案 A

- 每个 planning timestep 只选**一个全局 branch time**
- 这个时刻只和：
  - 当前可信联合场景集
  - 当前多模态预测
  - `Sep_k` 阈值
  有关

然后所有 shared trajectories 都使用同一个 branch time。

### 论文严格版

更严格的做法应该是：

- 对每一条 `shared plan`
- 对每一个候选分支时刻 `k`
- 同时检查：
  - `Sep_k >= eps_sep`
  - 该 shared 前缀在该 `k` 是否仍然对所有可信场景可恢复

然后得到：

- `t_b^*(tau^sh)`

也就是**每条 shared 轨迹自己的最优分支时刻**。

当前代码还没做到这一层。

---

## 10. 当前实现的优点和局限

### 优点

- 改动小，容易接入现有代码
- 直接复用已有：
  - 多模态预测
  - 联合场景
  - 可信场景集
  - recoverability 框架
- 计算量明显低于“每条 shared 轨迹逐一选分支时刻”

### 局限

- 还没有把 **recoverability** 真正耦合进分支时刻选择本身
- 当前选时刻只看 `Sep_k`
- recoverability 仍然是在后面筛选 final plan 时单独检查

所以严格来说，当前实现更接近：

> “基于可分性的全局 branch time 选择 + 后续 recoverability 过滤”

而不是：

> “可分性—可恢复性联合决定分支时刻”

---

## 11. 当前导出的数据是什么意思

自适应分支的离线输出包括：

- `planner/Frenet/results/eval/belief_plots/<scenario>_adaptive_branching.json`
- `planner/Frenet/results/eval/belief_plots/<scenario>_adaptive_branching_summary.png`
- `planner/Frenet/results/eval/belief_plots/<scenario>_adaptive_branching_separability_heatmap.png`

其中 JSON 里每个 timestep 记录：

- `selected_branch_time`
- `selected_branch_step`
- `selected_separability`
- `separability_threshold`
- `candidate_times`
- `separability_series`
- `selection_reason`

热力图含义：

- 横轴：planning timestep
- 纵轴：候选 branch time
- 颜色：对应 `Sep_k`
- 红线：最终选中的 branch time

---

## 12. 一句话总结

当前代码里的自适应分支时刻选择可以概括为：

> 在当前可信联合场景集上，沿一组候选未来时刻计算场景可分性 `Sep_k`，然后选取最早达到阈值 `eps_sep` 的时刻作为本 timestep 的统一 branch time；若都未达到，则回退到最晚候选时刻。

