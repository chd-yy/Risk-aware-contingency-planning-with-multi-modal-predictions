# 当前代码中 shared / branch / final plan 的 cost 是怎么确定的

这个文档基于当前分支代码整理，回答三个问题：

1. 共享段轨迹（shared plan）的 cost 是怎么来的  
2. 分支段轨迹（contingent / branch plan）的 cost 是怎么来的  
3. 最终整条 plan 的 cost 是怎么合成并用于选最优方案的  


## 1. 总体结论

当前代码里的 cost 结构可以概括成下面这个形式：

\[
J_{\text{final}} = J_{\text{shared}} + \sum_{m \in \Omega_c} w_m \, J_{\text{branch}}^{(m)}
\]

其中：

- \(J_{\text{shared}}\)：共享段轨迹的 cost
- \(J_{\text{branch}}^{(m)}\)：在联合场景/联合模式 \(m\) 下，对应分支段最优轨迹的 cost
- \(w_m\)：该联合模式的 posterior weight
- \(\Omega_c\)：当前时刻的可信联合场景集（credible joint set）

也就是说：

- **共享段 cost**：先单独算
- **每个模式下的分支段 cost**：也单独算
- **最终 plan cost**：用“共享段 cost + 各模式分支段 cost 的加权和”


## 2. 共享段轨迹 cost 的来源

共享段轨迹先在 `planner/Frenet/frenet_planner.py` 里生成，然后通过：

- `planner/Frenet/frenet_planner.py:794`
- `planner/Frenet/utils/frenet_functions.py:993`

调用 `sort_frenet_trajectories(...)` 来完成：

1. 基础 validity 检查  
2. 风险计算  
3. 风险相关 validity 检查  
4. 对有效轨迹计算 cost  


### 2.1 shared plan 的 cost 只对 valid_level = 10 的轨迹计算

在 `sort_frenet_trajectories(...)` 中：

- 先用 `check_validity_basic(...)` 做基础可行性检查
- 再用 `calc_risk(...)` 算风险
- 再用 `check_risk_validity(...)` 过滤风险超限轨迹

只有 `valid_level == 10` 的轨迹，才会进入真正的 cost 计算：

- `planner/Frenet/utils/frenet_functions.py:1161`


### 2.2 shared plan 的 cost 核心由 `calc_trajectory_costs(...)` 决定

真正的 cost 计算在：

- `planner/Frenet/utils/calc_trajectory_cost.py:36`

该函数会生成一个 `cost_dict`，然后按权重加权求和，得到：

\[
J = \sum_k \lambda_k \, c_k
\]

这里：

- \(c_k\)：每个 cost 分项
- \(\lambda_k\)：对应权重


## 3. shared / branch 共用同一套单条轨迹 cost 函数

这一点很关键：

- **共享段轨迹 cost**
- **分支段轨迹 cost**

本质上都是调用同一个函数：

- `calc_trajectory_costs(...)`

区别不在 cost 公式本身，而在于：

- 轨迹时域不同
- 起点不同
- 条件预测场景不同


## 4. 单条轨迹的 cost 分项有哪些

`calc_trajectory_costs(...)` 当前可能包含这些 cost 分项：

- `risk_cost`
- `visible_area`
- `lon_jerk`
- `lat_jerk`
- `velocity`
- `dist_to_global_path`
- `travelled_dist`
- `dist_to_goal_pos`
- `dist_to_lane_center`

对应代码见：

- `planner/Frenet/utils/calc_trajectory_cost.py:80`
- `planner/Frenet/utils/calc_trajectory_cost.py:197`
- `planner/Frenet/utils/calc_trajectory_cost.py:331`
- `planner/Frenet/utils/calc_trajectory_cost.py:360`


### 4.1 风险 cost `risk_cost`

风险项内部又分成：

- `bayes`
- `equality`
- `maximin`
- `ego`

内部先合成为：

\[
J_{\text{risk}} =
\lambda_{\text{bayes}} J_{\text{bayes}}
 + \lambda_{\text{equality}} J_{\text{equality}}
 + \lambda_{\text{maximin}} J_{\text{maximin}}
 + \lambda_{\text{ego}} J_{\text{ego}}
\]

然后把它作为 `cost_dict["risk_cost"]` 写入总代价项。

对应代码：

- `planner/Frenet/utils/calc_trajectory_cost.py:102`


### 4.2 其它常规 cost

除了风险项以外，还可能包含：

- `velocity`：当前实现里近似是 `10 - mean(traj.v)`
- `dist_to_global_path`：平均横向偏离全局路径
- `lon_jerk` / `lat_jerk`：纵向/横向 jerk 平滑性
- `travelled_dist`：轨迹长度
- `dist_to_goal_pos`：终点到目标区域/目标位置距离
- `dist_to_lane_center`：偏离车道中心线程度
- `visible_area`：遮挡/可见区域相关代价


## 5. 当前默认权重下，真正起作用的 cost 项

如果没有从外部额外传入 `weights`，planner 默认加载：

- `planner/Frenet/configs/weights.json:1`

当前默认权重是：

- `bayes = 33.3`
- `equality = 33.3`
- `maximin = 33.3`
- `ego = 10.0`
- `risk_cost = 1.0`
- `dist_to_global_path = 1.0`

而这些项当前为 0：

- `velocity = 0.0`
- `visible_area = 0.0`
- `lon_jerk = 0.0`
- `lat_jerk = 0.0`
- `travelled_dist = 0.0`
- `dist_to_goal_pos = 0.0`
- `dist_to_lane_center = 0.0`
- `responsibility = 0.0`

所以在**当前默认配置**下，单条轨迹真正参与排序的主要就是：

\[
J \approx J_{\text{risk}} + J_{\text{dist-to-global-path}}
\]

其中：

- `J_risk` 本身又是 `bayes/equality/maximin/ego` 的加权和
- 再加上 `dist_to_global_path`


## 6. shared plan 的 horizon 也会影响它的 cost

shared plan 不是固定长度。

它的时域由当前选中的 branch time 决定：

- `planner/Frenet/frenet_planner.py:670`
- `planner/Frenet/frenet_planner.py:672`

所以共享段 cost 实际上是：

- 在“当前 branch time 截断出来的共享时域”上计算出来的

这意味着：

- branch time 越长，shared plan 的 cost 评估区间越长
- branch time 越短，shared plan 的 cost 只覆盖更前面一段


## 7. 分支段轨迹（branch / contingent plan）cost 是怎么来的

对于每一条 valid shared plan，代码都会：

1. 从 shared plan 的终点状态出发  
2. 再生成一批 contingent trajectories  
3. 对每个 credible joint mode 单独筛选最优分支轨迹  

对应代码：

- `planner/Frenet/frenet_planner.py:844`
- `planner/Frenet/frenet_planner.py:871`
- `planner/Frenet/frenet_planner.py:903`


### 7.1 每个 branch mode 下的 cost 也是独立排序出来的

对某个 `mode_selection`，会调用：

- `planner/Frenet/frenet_planner.py:904`

即再次使用：

- `sort_frenet_trajectories(...)`

但这次传入了两个关键条件：

- `start_idx = shared_start_idx`
- `mode_num = mode_selection`

这意味着分支段 cost 是：

- **从 shared 段结束以后开始算**
- **在指定联合模式条件下算**

所以某个 mode 下最终保留下来的 branch cost 是：

\[
J_{\text{branch}}^{(m)} = \min_{\tau \in \mathcal{T}^{(m)}} J(\tau)
\]

其中：

- \(\mathcal{T}^{(m)}\)：该联合模式 \(m\) 下所有 valid contingent trajectories


## 8. 最终 final plan 的 cost 是怎么合成的

所有 recoverable 的 shared plan 会进入 `ft_final_list`。

然后在这里合成最终总代价：

- `planner/Frenet/frenet_planner.py:964`

当前代码是：

\[
J_{\text{final}} = J_{\text{shared}} + \sum_{m \in \Omega_c} w_m J_{\text{branch}}^{(m)}
\]

代码对应逻辑：

1. 先设：
   - `plan['cost'] = plan['shared_plan'].cost`
2. 再遍历 credible joint modes：
   - `plan['cost'] += mode_weight * plan[mode_num].cost`


### 8.1 只有 recoverable 的 shared plan 才能进入最终排序

这一步很重要：

- 如果某条 shared plan 不能为所有 credible joint modes 找到可行分支
- 那么它会被标记为 `recoverable = False`
- 并且**不会进入** `ft_final_list`

对应代码：

- `planner/Frenet/frenet_planner.py:926`
- `planner/Frenet/frenet_planner.py:936`

因此，最终参与比较的不是“所有 shared plan”，而是：

- **所有 recoverable 的 shared plan 及其对应完整 contingency plan**


## 9. `mode_weight` 是怎么来的

单障碍物 belief 先来自每个 obstacle 的 mode posterior。  
然后多个 obstacle 组合成联合模式后，联合模式权重按乘积计算：

- `planner/Frenet/frenet_planner.py:1161`

即：

\[
w_m \propto \prod_i p_i(m_i)
\]

然后对所有 joint modes 归一化。

接着再从中取可信场景集（credible joint set）：

- `planner/Frenet/frenet_planner.py:1249`

规则是：

- 按 joint weight 从大到小排序
- 取最少个模式
- 直到累计概率质量至少达到 \(1-\alpha\)

当前：

- `alpha = 0.05`

即可信场景集至少覆盖 95% 概率质量。


## 10. 一个很关键的实现细节：credible set 里的权重没有再次归一化

这里要特别注意：

- `_compute_joint_mode_weights(...)` 会先对**所有 joint modes**归一化
- `_compute_credible_joint_set(...)` 只是截取前若干个高概率模式
- 但**不会**把 credible set 内部的权重重新归一化到和为 1

所以最终用来合成 `final plan cost` 的：

- `credible_joint_mode_weights`

其实是“原始 joint posterior 的截断子集”，不是“在 credible set 内重新归一化后的条件概率”。

因此当前最终 cost 更准确地说是：

- **credible set 上的截断期望 cost**

而不是：

- **credible set 条件下的归一化期望 cost**


## 11. 当前代码中的一个重要实现现象：`factor` 没有真正参与排序

在 `calc_trajectory_costs(...)` 里会先算一个：

- `factor = get_cost_factor(...)`

并且 `output_dict` 里也保存了：

- `total_cost = factor * cost`

但是函数最终返回的是：

- `return cost, output_dict`

而不是：

- `return factor * cost, output_dict`

对应代码：

- `planner/Frenet/utils/calc_trajectory_cost.py:450`

这意味着当前实际用于排序的：

\[
J_{\text{rank}} = \sum_k \lambda_k c_k
\]

而不是：

\[
J_{\text{rank}} = \text{factor} \cdot \sum_k \lambda_k c_k
\]

所以：

- `get_cost_factor()` 目前只体现在 `cost_dict/output_dict` 里
- **没有真正影响 shared / branch / final plan 的排序结果**


## 12. 风险 fallback 也会影响 shared / branch 的 cost

如果没有 `valid_level == 10` 的轨迹，但存在 `valid_level == 3` 的轨迹，代码会走一个 fallback：

- `planner/Frenet/utils/frenet_functions.py:1142`

此时 cost 不再走 `calc_trajectory_costs(...)`，而是用：

\[
J_{\text{fallback}} =
4 \cdot \text{peak\_risk}
 + 2 \cdot \text{total\_risk}
 + 0.5 \cdot \text{terminal\_speed}
 + 0.1 \cdot \text{final\_offset}
\]

这类轨迹会打上：

- `used_risk_fallback = True`

因此当前 shared / branch cost 其实有两套来源：

1. 正常 cost：`calc_trajectory_costs(...)`
2. fallback cost：`_calc_risk_fallback_cost(...)`


## 13. 最终最优 plan 是怎么选出来的

对 `ft_final_list` 计算完 `plan['cost']` 以后：

- `planner/Frenet/frenet_planner.py:978`

按总代价升序排序：

\[
\text{best\_plan} = \arg\min J_{\text{final}}
\]

然后：

- `best_plan = ft_final_list[0]`

最后执行的也是这个 `best_plan` 的共享段：

- `planner/Frenet/frenet_planner.py:1131`


## 14. 一句话总结

当前分支里三层 cost 的关系是：

- **共享段 cost**：对 shared trajectory 本身做 validity + 单轨迹 cost 排序得到
- **分支段 cost**：对每个 credible joint mode 下的 contingent trajectory 单独排序得到
- **最终 plan cost**：`共享段 cost + credible joint modes 下 branch cost 的加权和`

如果用一句更数学化的话概括，就是：

\[
\boxed{
J_{\text{final}} = J_{\text{shared}} + \sum_{m \in \Omega_c} w_m J_{\text{branch}}^{(m)}
}
\]

其中当前代码还有两个很重要的实现特点：

1. `credible_joint_mode_weights` 是**截断 posterior**，不是 credible set 内再归一化后的概率  
2. `get_cost_factor()` **当前没有真正参与轨迹排序**

