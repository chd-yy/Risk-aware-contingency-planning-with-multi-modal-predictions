# Recoverability 100 场景对比结果（2026-04-29）

- 场景数：每组 `100` 个场景
- 方法：`ours` 与 `ours_wo_Rec_strict_clean`
- 动力学受限设置：
  - `limited_mild`
  - `limited_medium`
  - `limited_hard`

## 指标汇总表

| 动力学受限强度 | 方法 | 成功率 SR | 碰撞率 CR | 平均速度 v_bar (m/s) | 最小间距 d_min (m) | t_c (s) | t95 (s) | URR | recoverability_activation_ratio | selected_plan_unrecoverable_ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| limited_mild | `ours` | 0.9200 | 0.0800 | 11.2272 | 3.1780 | 1.1891 | 2.5651 | 0.0517 | 0.2086 | 0.0003 |
| limited_mild | `ours_wo_Rec_strict_clean` | 0.9000 | 0.1000 | 11.2786 | 3.2041 | 0.7836 | 1.4109 | 0.0000 | - | - |
| limited_medium | `ours` | 0.9100 | 0.0900 | 11.1494 | 3.1829 | 1.0098 | 2.4598 | 0.1125 | 0.3235 | 0.0034 |
| limited_medium | `ours_wo_Rec_strict_clean` | 0.8400 | 0.1600 | 11.1925 | 3.1617 | 0.6721 | 1.2739 | 0.0000 | - | - |
| limited_hard | `ours` | 0.8200 | 0.1800 | 10.8127 | 3.0942 | 0.9010 | 2.3969 | 0.2397 | 0.4432 | 0.0766 |
| limited_hard | `ours_wo_Rec_strict_clean` | 0.7700 | 0.2300 | 10.9157 | 3.0887 | 0.5922 | 1.2191 | 0.0000 | - | - |

## 按受限强度分组展示

### limited_mild

| 方法 | SR | CR | v_bar | d_min | t_c | t95 | URR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ours` | 0.9200 | 0.0800 | 11.2272 | 3.1780 | 1.1891 | 2.5651 | 0.0517 |
| `ours_wo_Rec_strict_clean` | 0.9000 | 0.1000 | 11.2786 | 3.2041 | 0.7836 | 1.4109 | 0.0000 |

### limited_medium

| 方法 | SR | CR | v_bar | d_min | t_c | t95 | URR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ours` | 0.9100 | 0.0900 | 11.1494 | 3.1829 | 1.0098 | 2.4598 | 0.1125 |
| `ours_wo_Rec_strict_clean` | 0.8400 | 0.1600 | 11.1925 | 3.1617 | 0.6721 | 1.2739 | 0.0000 |

### limited_hard

| 方法 | SR | CR | v_bar | d_min | t_c | t95 | URR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ours` | 0.8200 | 0.1800 | 10.8127 | 3.0942 | 0.9010 | 2.3969 | 0.2397 |
| `ours_wo_Rec_strict_clean` | 0.7700 | 0.2300 | 10.9157 | 3.0887 | 0.5922 | 1.2191 | 0.0000 |

## 简要观察

- 三种受限强度下，`ours` 的 `SR` 都高于 `ours_wo_Rec_strict_clean`
- 随着动力学约束增强，`ours` 的 `recoverability_activation_ratio` 逐步升高
- 严格消融版的 `URR` 始终为 `0`，说明其不包含 recoverability 约束所对应的恢复性行为
