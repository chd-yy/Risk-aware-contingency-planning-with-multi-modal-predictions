# 实验参数、模型参数与规划参数总表

## 1. 文档范围与参数来源

本文档整理当前仓库中与交叉口意图规划实验直接相关的三类参数：

1. **实验参数**：场景集、评测脚本、输出与日志控制参数；
2. **模型参数**：意图模式数、belief 更新参数、预测模型入口参数、伤害模型参数；
3. **规划参数**：共享段/分支段采样、风险约束、代价权重、自适应分支、recoverability 等。

当前代码中参数的真实来源包括：

- 配置文件：`planner/Frenet/configs/planning_fast.json`
- 配置文件：`planner/Frenet/configs/contingency.json`
- 配置文件：`planner/Frenet/configs/risk.json`
- 配置文件：`planner/Frenet/configs/prediction.json`
- 配置文件：`planner/Frenet/configs/weights.json`
- 配置文件：`planner/Frenet/configs/weights_standard.json`
- 配置文件：`planner/Frenet/configs/weights_ethical.json`
- 配置文件：`planner/Frenet/configs/weights_ego.json`
- 配置文件：`planner/Frenet/configs/harm_parameters.json`
- 参数加载器：`planner/Frenet/configs/load_json.py`
- 规划器主逻辑：`planner/Frenet/frenet_planner.py`
- 实验脚本：`test_fot_junction/evaluate_vv_batch.py`
- 模式粒度实验脚本：`test_fot_junction/evaluate_intent_mode_granularity.py`
- recoverability 对比实验脚本：`test_fot_junction/evaluate_recoverability_stress.py`

需要特别说明：

- `load_json.py` 会把 `planning_fast.json` 和 `contingency.json` 中的 `d_list` 从 `{d_max_abs, n}` 展开成 `np.linspace(...)`；
- 但 `frenet_planner.py` 在实际运行时又对共享段和分支段横向采样进行了代码级覆盖，因此**最终生效值以代码为准**；
- 当前仓库中只保留了预测模型入口配置 `prediction.json`，但 `prediction/configs/best_config.json` 与 `prediction/trained_models/best_model.tar` 不在当前工作区内，因此无法进一步追溯该网络的完整结构超参数。

---

## 2. 实验层参数

### 2.1 场景与数据集参数

批量实验脚本默认使用以下数据：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `base_scenario` | `recorded/hand-crafted/BRA_VilaVelha-92_1_T-10.xml` | 基准场景，始终排在 100 场景中的第 1 个 |
| `sample_dir` | `recorded/hand-crafted/vv_samples` | 其余 99 个扰动样本场景目录 |
| `expected_scenarios` | `100` | 预期总场景数 = 1 个 base + 99 个 sample |
| `fps` | `10` | 回放/导出 GIF 的帧率 |
| `limit` | `None` | 若设置，则只评测排序后的前 `N` 个场景 |

### 2.2 `evaluate_vv_batch.py` 支持的 CLI 参数

| 参数 | 类型 | 默认值 | 作用 |
|---|---|---|---|
| `--base-scenario` | `str` | 上述 base 场景 | 指定基准场景 |
| `--sample-dir` | `str` | 上述 sample 目录 | 指定样本目录 |
| `--output-dir` | `str` | `planner/Frenet/results/vv_batch_eval` | 指定实验输出目录 |
| `--planning-config` | `str` | `planning_fast.json` | 共享段配置 |
| `--contingency-config` | `str` | `contingency.json` | 分支段配置 |
| `--risk-config` | `str` | `risk.json` | 风险配置 |
| `--fps` | `int` | `10` | GIF 输出帧率 |
| `--experiment-tag` | `str` | `""` | 结果/日志标记 |
| `--minimal-output` | `flag` | `False` | 极简输出，只保留必要指标 |
| `--c-omega-trace-dir` | `str` | `None` | 保存逐场景 `C_Omega` 详细计算过程 |
| `--intent-mode-count` | `int` | `2` | 每个障碍车意图模式数，仅允许 `2/3/4` |
| `--recoverability-enabled` | `true/false` | `None` | 强制打开/关闭 recoverability |
| `--longitudinal-a-max-scale` | `float` | `None` | 缩放车辆纵向最大加速度 |
| `--lateral-a-max-scale` | `float` | `None` | 缩放车辆横向最大加速度 |
| `--longitudinal-v-max-scale` | `float` | `None` | 缩放车辆纵向最大速度 |
| `--limit` | `int` | `None` | 只跑前 `N` 个场景 |

### 2.3 实验脚本内的输出控制

当 `--minimal-output` 打开时，脚本会自动切换到极简模式：

| 选项 | 极简模式行为 |
|---|---|
| `show_visualization` | 强制关闭 |
| `risk_dict.figures.create_figures` | 强制关闭 |
| `risk_dashboard` | 强制关闭 |
| `collision_report` | 强制关闭 |
| `record_intent_history` | 关闭 |
| `record_clearance_history` | 关闭 |
| 导出 GIF | 不导出 |
| `metrics_per_scenario.csv` 字段 | 仅保留 `scenario, scenario_name, success, t_c_s` |

### 2.4 意图模式粒度实验参数

`test_fot_junction/evaluate_intent_mode_granularity.py` 中固定的三组设置为：

| 设置名 | `intent_mode_count` |
|---|---:|
| `mode2` | `2` |
| `mode3` | `3` |
| `mode4` | `4` |

### 2.5 Recoverability 对比实验参数

`test_fot_junction/evaluate_recoverability_stress.py` 中当前只定义了一个工况：

| 参数 | 值 |
|---|---|
| `regime` | `nominal` |
| `vehicle_overrides` | `{}` |

对比方法为：

| 方法 | `recoverability_enabled` |
|---|---:|
| `ours` | `True` |
| `ours_wo_Rec` | `False` |

### 2.6 当前代码/实验上下文中涉及的主要模式组合

结合前述实验流程，当前最核心的三组模式实验可整理为：

| 实验 | 内核/实现 | `intent_mode_count` | recoverability |
|---|---|---:|---:|
| `mode2` | `oldlike` | `2` | 开启 |
| `mode3` | 当前 `ours` | `3` | 开启 |
| `mode4` | 当前 `ours` | `4` | 开启 |

---

## 3. 规划器运行参数

### 3.1 基础运行参数

来自 `planning_fast.json` 与 `frenet_planner.py` 的基础运行设定如下：

| 参数 | 值 | 说明 |
|---|---:|---|
| `mode` | `risk` | 规划器运行在风险感知模式下 |
| `dt` | `0.1 s` | 时间离散步长 |
| `vehicle_type` | `bmw_320i` | 自车车辆模型 |
| `timing_enabled` | `true` | 开启计时 |
| `show_visualization` | `false`（批量评测下通常为假） | 是否可视化 |
| `sensor_radius` | `11155.0 m` | 感知半径，近似等价于全局感知 |

### 3.2 车辆动态缩放参数

这些参数不在 JSON 中默认启用，但可由实验脚本注入到 `evaluation_settings.vehicle_param_overrides`：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `longitudinal_a_max_scale` | `None` | 纵向最大加速度缩放 |
| `lateral_a_max_scale` | `None` | 横向最大加速度缩放 |
| `longitudinal_v_max_scale` | `None` | 最大纵向速度缩放 |

---

## 4. 共享段（shared segment）参数

### 4.1 名义配置值

来自 `planner/Frenet/configs/planning_fast.json`：

| 参数 | 值 |
|---|---:|
| `t_list` | `[1.4]` |
| `v_list_generation_mode` | `linspace` |
| `n_v_samples` | `7` |
| `d_list.d_max_abs` | `1.8` |
| `d_list.n` | `10` |
| `dt` | `0.1` |
| `v_thr` | `3.0 m/s` |

### 4.2 由 `load_planning_json()` 展开的值

加载后，配置中的横向采样会先变为：

| 参数 | 展开结果 |
|---|---|
| `d_list` | `np.linspace(-1.8, 1.8, 10)` |

### 4.3 运行时实际生效值

在 `frenet_planner.py` 中，shared trajectory 生成时实际使用：

| 参数 | 实际值 | 说明 |
|---|---|---|
| `shared_t_list` | 自适应分支结果，初始上界来自 `[1.4]` | 分支时刻可被动态缩短 |
| `d_list` | `np.linspace(-1.75, 1.75, 7)` | 代码覆盖 JSON 横向采样 |
| `n_v_samples` | `7` | 与 JSON 一致 |
| `dt` | `0.1` | 与 JSON 一致 |
| `v_thr` | `3.0` | 与 JSON 一致 |

### 4.4 共享段速度采样边界

shared segment 每一规划周期内的速度范围按当前自车状态动态计算：

| 量 | 计算方式 |
|---|---|
| `current_v` | 当前自车速度 |
| `max_acceleration` | 车辆纵向最大加速度 |
| `shared_t_min` | `min(shared_t_list)` |
| `shared_t_max` | `max(shared_t_list)` |
| `max_v` | `min(current_v + (max_acceleration / 2) * shared_t_max, v_max)` |
| `min_v` | `max(0.01, current_v - max_acceleration * shared_t_min)` |

之后调用 `get_v_list(...)`，在上述区间内按 `linspace` 方式取 `7` 个末速度样本。

---

## 5. 分支段（contingent segment）参数

### 5.1 名义配置值

来自 `planner/Frenet/configs/contingency.json`：

| 参数 | 值 |
|---|---:|
| `mode` | `risk` |
| `t_list` | `[2.6]` |
| `v_list_generation_mode` | `linspace` |
| `n_v_samples` | `5` |
| `d_list.d_max_abs` | `2.0` |
| `d_list.n` | `3` |
| `dt` | `0.1` |
| `v_thr` | `3.0 m/s` |

### 5.2 由 `load_contingency_json()` 展开的值

| 参数 | 展开结果 |
|---|---|
| `d_list` | `np.linspace(-2.0, 2.0, 3)` |

### 5.3 运行时实际生效值

根据现有代码说明与现有参数文档，contingent planning 实际使用的横向采样为：

| 参数 | 实际值 |
|---|---|
| `t_list` | `[2.6]` |
| `d_list` | `np.linspace(-1.75, 1.75, 6)` |
| `n_v_samples` | `5` |
| `dt` | `0.1` |
| `v_thr` | `3.0` |

### 5.4 自适应分支参数

来自 `contingency.json` 与 `frenet_planner.py`：

| 参数 | 值 | 说明 |
|---|---:|---|
| `enabled` | `true` | 是否启用自适应分支 |
| `min_branch_time` | `0.4 s` | 最早允许分支时刻 |
| `candidate_dt` | `0.1 s` | 分支候选时间间隔 |
| `separability_threshold` | `1.0` | 触发足够模式分离的阈值 |
| `max_branch_time` | 默认取 `max(shared_t_list)` | 若未显式配置，则退化到共享段上界 |
| `candidate_times` | 若未显式给出，则为 `np.arange(min_branch_time, max_branch_time + 0.5*candidate_dt, candidate_dt)` | 候选分支时刻集合 |

### 5.5 Recoverability 开关

来自 `contingency.json`：

| 参数 | 值 |
|---|---:|
| `recoverability.enabled` | `true` |

若批量脚本传入 `--recoverability-enabled false`，则会在运行前覆盖该项。

---

## 6. 多模态预测与意图模型参数

### 6.1 预测模型入口参数

来自 `planner/Frenet/configs/prediction.json`：

| 参数 | 值 |
|---|---|
| `pred_config_path` | `prediction/configs/best_config.json` |
| `pred_model_path` | `prediction/trained_models/best_model.tar` |
| `gpu` | `"0"` |
| `min_obs_length` | `0` |
| `on_pred_learn_method` | `null` |
| `on_pred_horizon` | `null` |
| `on_lr` | `null` |
| `on_pred_learn_density` | `null` |
| `online_layer` | `null` |
| `on_loss` | `null` |
| `on_optimizer` | `null` |
| `on_train_loss_th` | `null` |

说明：

- 当前工作区内缺少 `prediction/` 目录，因此只能确认**运行入口配置**；
- 预测网络内部结构、训练超参数、损失函数细节目前无法从本仓库还原。

### 6.2 模式数参数

在 `frenet_planner.py` 中：

| 参数 | 值 |
|---|---:|
| `intent_mode_count` 默认值 | `2` |
| 合法模式数 | `{2, 3, 4}` |
| 非法值处理 | 强制回退为 `2` |

### 6.3 各模式定义

当前代码中的模式语义如下：

| 模式数 | 模式标签 |
|---|---|
| `2` | `yield`, `challenge` |
| `3` | `yield_strong`, `yield_soft`, `challenge` |
| `4` | `yield_strong`, `yield_soft`, `challenge_soft`, `challenge_hard` |

### 6.4 模式轨迹生成参数

在 `planner/Frenet/utils/prediction_helpers.py` 中，障碍车模式轨迹由基础速度推导：

| 参数 | 计算方式 |
|---|---|
| `challenge_ds` | `base_speed * dt` |
| `yield_ds` | `min(challenge_ds * 0.35, 1.0)` |
| `challenge_hard_ds` | `max(challenge_ds * 1.15, challenge_ds + 0.15)` |
| `challenge_soft_ds` | `max(challenge_ds * 0.90, yield_ds + 0.10)` |
| `yield_soft_ds` | `min(max(yield_ds * 1.45, 0.55 * challenge_ds), challenge_soft_ds * 0.92)` |
| `yield_strong_ds` | `min(yield_ds, max(0.20, 0.30 * challenge_ds))` |

对应输出：

| 模式数 | 输出轨迹集合 |
|---|---|
| `2` | `[yield_traj, challenge_traj]` |
| `3` | `[yield_strong, yield_soft, challenge]` |
| `4` | `[yield_strong, yield_soft, challenge_soft, challenge_hard]` |

### 6.5 belief 更新参数

`update_interaction_mode_belief(...)` 中当前固定参数如下：

| 参数 | 值 | 作用 |
|---|---:|---|
| `forgetting_factor` | `0.72` | 历史 belief 保留比例 |
| `neutral_prior` | `1 / mode_count` | 均匀先验 |
| 后验回拉系数 | `0.92` | 保留当前后验 |
| 中性先验回拉系数 | `0.08` | 防止 belief 过早塌缩 |

belief 更新的核心计算流程可概括为：

1. 用 `0.72 * prior + 0.28 * neutral_prior` 对历史先验做遗忘更新；
2. 先做 coarse 的 `yield/challenge` 二类似然估计；
3. 若为 `3-mode` 或 `4-mode`，再在 coarse 组内继续细分；
4. 最终再做一次 `0.92 * posterior + 0.08 * neutral_prior` 的平滑回拉。

### 6.6 预测时域参数

`pred_horizon` 在 `frenet_planner.py` 中按以下方式计算：

| 参数 | 值 |
|---|---|
| shared 时域上界 | `max(self.frenet_parameters["t_list"]) = 1.4` |
| contingent 时域上界 | `max(self.contingency_parameters["t_list"]) = 2.6` |
| `dt` | `0.1` |
| `pred_horizon` | `max(int(1.4 / 0.1), int(2.6 / 0.1), 1) + 1 = 27` |

因此当前实验统一采用 **27 个离散点** 的预测时域。

---

## 7. 联合模式、可信场景集与分支选择参数

### 7.1 联合模式集合

若当前可见动态障碍物中有多辆车具备多模态轨迹，则联合场景由各障碍物模式做笛卡尔积得到：

| 参数 | 说明 |
|---|---|
| `multimodal_obstacle_ids` | 当前具有多模态预测的障碍车 ID 列表 |
| `multimodal_mode_ranges` | 各车模式索引范围 |
| `joint_mode_selections` | `product(*multimodal_mode_ranges)` 生成的联合模式集合 |

### 7.2 联合模式权重

联合模式权重通过各障碍车 mode belief 相乘得到：

| 参数 | 说明 |
|---|---|
| `joint_weight` | 对每个障碍物取对应模式概率并连乘 |

### 7.3 可信联合场景集参数

规划器中固定：

| 参数 | 值 |
|---|---:|
| `CREDIBLE_SET_ALPHA` | `0.05` |
| 目标覆盖概率 | `1 - alpha = 0.95` |

可信集构造方法为：

1. 按联合模式权重从大到小排序；
2. 依次累加概率；
3. 直到累计概率达到 `0.95` 为止；
4. 该最小前缀集合即 `credible joint scenario set`。

### 7.4 自适应分支记录量

代码中会维护如下历史量：

| 字段 | 含义 |
|---|---|
| `selected_branch_time` | 本周期选定的分支时刻 |
| `selected_branch_step` | 对应离散步 |
| `selected_separability` | 该时刻模式可分离度 |
| `separability_threshold` | 分支阈值 |
| `candidate_times` | 所有候选分支时刻 |
| `separability_series` | 每个候选时刻的分离度曲线 |
| `selection_reason` | 选择原因 |

---

## 8. Recoverability 相关参数

### 8.1 Recoverability 配置

| 参数 | 值 |
|---|---:|
| `recoverability.enabled` | `true`（默认） |

### 8.2 Recoverability 判定逻辑

当前实现中，一条 shared plan 被判定为 recoverable，当且仅当：

- 对当前 `credible joint scenario set` 中的**每一个**可信联合场景；
- 都至少存在一条可行的 contingent plan 能够接续。

### 8.3 Recoverability 过程记录字段

代码中保存的字段包括：

| 字段 | 含义 |
|---|---|
| `shared_plan_count` | 当前有效 shared plan 数量 |
| `recoverable_shared_plan_count` | 其中可恢复的 shared plan 数量 |
| `credible_set_size` | 当前可信集大小 |
| `recoverability_indicator` | 当前是否存在至少一条可恢复 shared plan |
| `recoverability_activation_indicator` | 当前是否触发 recoverability 压力 |
| `selected_plan_recoverable_indicator` | 最终选中方案是否 recoverable |
| `recoverability_enforced` | 当前周期是否显式施加 recoverability 过滤 |

### 8.4 与实验指标对应的量

实验输出中常见的两个 recoverability 指标可对应为：

| 指标 | 含义 |
|---|---|
| `recoverability_activation_ratio` | 全任务过程中 recoverability 被激活的时间占比 |
| `selected_plan_unrecoverable_ratio` | 最终被选中的方案中，不可恢复方案所占比例 |

---

## 9. 风险评估参数

来自 `planner/Frenet/configs/risk.json`：

| 参数 | 值 | 说明 |
|---|---:|---|
| `harm_mode` | `log_reg` | 伤害模型类型 |
| `trajectory_risk` | `max` | 轨迹风险聚合方式 |
| `max_acceptable_risk` | `0.05` | 最大可接受风险阈值 |
| `max_acceptable_trajectory_collision_prob_upper_bound` | `0.20` | 碰撞概率上界 |
| `multiple_cost_functions` | `false` | 不启用多 cost 函数 |
| `scale_factor_time` | `0.9` | 时间尺度因子 |
| `crash_angle_accuracy` | `10` | 碰撞角度离散精度 |
| `crash_angle_simplified` | `true` | 使用简化角度模型 |
| `fast_prob_mahalanobis` | `false` | 不启用快速 Mahalanobis 近似 |
| `sensor_occlusion_model` | `false` | 不建模传感器遮挡 |
| `occlusion_mode` | `false` | 不启用遮挡模式 |
| `ignore_angle` | `false` | 不忽略碰撞角度 |
| `sym_angle` | `true` | 采用对称角区域 |
| `reduced_angle_areas` | `true` | 采用简化角区域划分 |
| `figures.create_figures` | `false` | 默认不画风险图 |
| `figures.number_plotted_trajectories` | `3` | 图中显示轨迹数量 |
| `risk_dashboard` | `false` | 默认关闭风险仪表盘 |
| `collision_report` | `true` | 默认记录碰撞报告 |

---

## 10. 代价权重参数

### 10.1 默认权重 `weights.json`

| 代价项 | 权重 |
|---|---:|
| `bayes` | `33.3` |
| `equality` | `33.3` |
| `maximin` | `33.3` |
| `responsibility` | `0.0` |
| `ego` | `10.0` |
| `risk_cost` | `1.0` |
| `visible_area` | `0` |
| `lon_jerk` | `0.0` |
| `lat_jerk` | `0.0` |
| `velocity` | `0.0` |
| `dist_to_global_path` | `1.0` |
| `travelled_dist` | `0.0` |
| `dist_to_goal_pos` | `0.0` |
| `dist_to_lane_center` | `0.0` |

### 10.2 `weights_standard.json`

| 代价项 | 权重 |
|---|---:|
| `bayes` | `0.0` |
| `equality` | `0.0` |
| `maximin` | `0.0` |
| `responsibility` | `0.0` |
| `ego` | `0.0` |
| `risk_cost` | `1.0` |
| `visible_area` | `0` |
| `lon_jerk` | `0.0` |
| `lat_jerk` | `0.0` |
| `velocity` | `1.0` |
| `dist_to_global_path` | `10.0` |
| `travelled_dist` | `0.0` |
| `dist_to_goal_pos` | `0.0` |
| `dist_to_lane_center` | `0.0` |

### 10.3 `weights_ethical.json`

| 代价项 | 权重 |
|---|---:|
| `bayes` | `33.3` |
| `equality` | `33.3` |
| `maximin` | `33.3` |
| `responsibility` | `0.0` |
| `ego` | `0.0` |
| `risk_cost` | `250` |
| `visible_area` | `0` |
| `lon_jerk` | `0.0` |
| `lat_jerk` | `0.0` |
| `velocity` | `1.0` |
| `dist_to_global_path` | `10.0` |
| `travelled_dist` | `0.0` |
| `dist_to_goal_pos` | `0.0` |
| `dist_to_lane_center` | `0.0` |

### 10.4 `weights_ego.json`

| 代价项 | 权重 |
|---|---:|
| `bayes` | `0.0` |
| `equality` | `0.0` |
| `maximin` | `0.0` |
| `responsibility` | `0.0` |
| `ego` | `100.0` |
| `risk_cost` | `250` |
| `visible_area` | `0` |
| `lon_jerk` | `0.0` |
| `lat_jerk` | `0.0` |
| `velocity` | `1.0` |
| `dist_to_global_path` | `10.0` |
| `travelled_dist` | `0.0` |
| `dist_to_goal_pos` | `0.0` |
| `dist_to_lane_center` | `0.0` |

### 10.5 责任项激活条件

规划器中只有当：

| 条件 | 结果 |
|---|---|
| `weights["responsibility"] > 0` | 启用 reachable set，并计算 responsibility 相关项 |
| 否则 | `responsibility = False`，不启用该模块 |

---

## 11. 伤害模型参数（`harm_parameters.json`）

### 11.1 `log_reg` 系数

#### 11.1.1 `complete_angle_areas`

| 参数 | 值 |
|---|---:|
| `const` | `-4.626` |
| `speed` | `0.189` |
| `Imp_1` | `-0.039` |
| `Imp_2` | `0.018` |
| `Imp_3` | `0.459` |
| `Imp_4` | `-0.125` |
| `Imp_5` | `-1.413` |
| `Imp_6` | `-0.116` |
| `Imp_7` | `-1.782` |
| `Imp_8` | `-0.434` |
| `Imp_9` | `0.482` |
| `Imp_10` | `0.142` |
| `Imp_11` | `0.400` |

#### 11.1.2 `reduced_angle_areas`

| 参数 | 值 |
|---|---:|
| `const` | `-4.476` |
| `speed` | `0.179` |
| `driver_side` | `0.250` |
| `right_side` | `0.259` |
| `rear` | `-0.445` |

#### 11.1.3 `ignore_angle`

| 参数 | 值 |
|---|---:|
| `const` | `-4.591` |
| `speed` | `0.185` |

#### 11.1.4 `complete_sym_angle_areas`

| 参数 | 值 |
|---|---:|
| `const` | `-4.620` |
| `speed` | `0.189` |
| `Imp_1_11` | `0.209` |
| `Imp_2_10` | `0.086` |
| `Imp_3_9` | `0.470` |
| `Imp_4_8` | `-0.259` |
| `Imp_5_7` | `-1.590` |
| `Imp_6` | `-0.118` |

#### 11.1.5 `reduced_sym_angle_areas`

| 参数 | 值 |
|---|---:|
| `const` | `-4.457` |
| `speed` | `0.177` |
| `side` | `0.244` |
| `rear` | `-0.431` |

### 11.2 `ref_speed` 系数

#### 11.2.1 `complete_angle_areas`

| 参数 | 值 |
|---|---:|
| `ref_speed_1` | `52.886` |
| `ref_speed_2` | `51.995` |
| `ref_speed_3` | `39.992` |
| `ref_speed_4` | `56.450` |
| `ref_speed_5` | `107.092` |
| `ref_speed_6` | `52.623` |
| `ref_speed_7` | `123.535` |
| `ref_speed_8` | `68.055` |
| `ref_speed_9` | `40.475` |
| `ref_speed_10` | `47.301` |
| `ref_speed_11` | `42.249` |
| `ref_speed_12` | `48.666` |
| `exp` | `1.592` |

#### 11.2.2 `reduced_angle_areas`

| 参数 | 值 |
|---|---:|
| `ref_speed_front` | `51.285` |
| `ref_speed_right_side` | `46.452` |
| `ref_speed_rear` | `66.953` |
| `ref_speed_driver_side` | `47.115` |
| `exp` | `1.531` |

#### 11.2.3 `ignore_angle`

| 参数 | 值 |
|---|---:|
| `ref_speed` | `51.144` |
| `exp` | `1.570` |

#### 11.2.4 `complete_sym_angle_areas`

| 参数 | 值 |
|---|---:|
| `ref_speed_1_11` | `46.717` |
| `ref_speed_2_10` | `49.427` |
| `ref_speed_3_9` | `40.298` |
| `ref_speed_4_8` | `61.349` |
| `ref_speed_5_7` | `115.139` |
| `ref_speed_6` | `52.787` |
| `ref_speed_12` | `48.783` |
| `exp` | `1.589` |

#### 11.2.5 `reduced_sym_angle_areas`

| 参数 | 值 |
|---|---:|
| `ref_speed_front` | `51.287` |
| `ref_speed_side` | `46.774` |
| `ref_speed_rear` | `66.956` |
| `exp` | `1.531` |

### 11.3 其他伤害模型参数

#### 11.3.1 `gidas`

| 参数 | 值 |
|---|---:|
| `const` | `-5.820` |
| `speed` | `0.292` |

#### 11.3.2 `pedestrian`

| 参数 | 值 |
|---|---:|
| `const` | `3.164` |
| `speed` | `0.288` |

#### 11.3.3 `pedestrian_MAIS2+`

| 参数 | 值 |
|---|---:|
| `const` | `1.786` |
| `speed` | `0.259` |

---

## 12. 整个规划流程中涉及的关键中间量

为了让实验参数与算法流程对应起来，当前规划周期内的关键变量可概括如下：

### 12.1 当前状态量

| 变量 | 含义 |
|---|---|
| `c_s` | Frenet 纵向位置 |
| `c_s_d` | Frenet 纵向速度 |
| `c_s_dd` | Frenet 纵向加速度 |
| `c_d` | Frenet 横向偏移 |
| `c_d_d` | Frenet 横向速度 |
| `c_d_dd` | Frenet 横向加速度 |
| `current_v` | 当前自车速度 |

### 12.2 预测与 belief 中间量

| 变量 | 含义 |
|---|---|
| `visible_obstacle_ids` | 传感器半径内的障碍物 |
| `visible_dynamic_obstacle_ids` | 其中的动态障碍物 |
| `base_predictions` | 基础单模态预测 |
| `predictions` | 多模态 GMM 预测 |
| `prediction_belief` | 每个障碍物的 mode probability |
| `joint_mode_selections` | 全部联合模式 |
| `credible_joint_mode_selections` | 可信联合模式集合 |
| `credible_joint_mode_weights` | 可信联合模式权重 |

### 12.3 轨迹生成与选择中间量

| 变量 | 含义 |
|---|---|
| `shared_branch_time` | 共享段长度/分支时刻 |
| `shared_start_idx` | 分支开始离散步 |
| `v_list` | 当前周期末速度采样列表 |
| `ft_list` | shared segment 候选轨迹集合 |
| `final_plan` | 一条 shared plan 及其全部 contingent 接续结果 |
| `recoverable` | 该 shared plan 是否对可信集全覆盖可恢复 |

### 12.4 记录与日志中间量

| 变量 | 含义 |
|---|---|
| `obstacle_belief_history` | 每辆障碍车 belief 演化历史 |
| `joint_belief_history` | 联合模式权重演化历史 |
| `credible_joint_history` | 可信集大小、权重、标签、累计概率 |
| `recoverability_history` | recoverability 激活与选解历史 |
| `adaptive_branching_history` | 分支时刻与分离度历史 |
| `execution_dynamics_history` | ego 速度与风险指示量 |

---

## 13. 实验输出指标字段

当前批量实验脚本中，完整 `metrics_per_scenario.csv` 支持以下字段：

| 字段 | 含义 |
|---|---|
| `scenario` | 场景序号 |
| `scenario_name` | 场景名称 |
| `success` | 是否成功完成任务 |
| `collision` | 是否发生碰撞 |
| `reason_for_failure` | 失败原因 |
| `gif_path` | 对应 GIF 路径 |
| `task_time_s` | 任务总完成时间 |
| `avg_speed_mps` | 平均速度 |
| `min_clearance_m` | 最小净距 |
| `t_c_s` | 平均单周期计算时间 |
| `t95_s` | 单周期计算时间 95 分位 |
| `Omega_bar` | 平均可信集大小 |
| `C_Omega` | 可信集切换/复杂度相关指标 |
| `C_Omega_coarse` | 将细粒度模式合并后的 coarse 版本 |
| `URR` | 不可恢复风险相关统计量 |
| `recoverability_activation_ratio` | recoverability 激活占比 |
| `selected_plan_unrecoverable_ratio` | 选中方案不可恢复占比 |

在极简模式下，仅保留：

| 字段 | 含义 |
|---|---|
| `scenario` | 场景序号 |
| `scenario_name` | 场景名称 |
| `success` | 是否成功 |
| `t_c_s` | 平均单周期计算时间 |

---

## 14. 使用建议

若后续需要在论文中引用，建议把本文档拆成三层表达：

1. **正文主表**：只保留最核心的实验设置、共享段/分支段参数、风险阈值、模式数设置；
2. **附录表**：保留所有权重与伤害模型系数；
3. **代码附录说明**：单独注明“JSON 名义值”与“代码实际生效值”的差异，尤其是 shared/contingent 的横向采样覆盖问题。

这样最符合论文写作习惯，也最容易让审稿人快速定位“实验到底怎么配的”。
