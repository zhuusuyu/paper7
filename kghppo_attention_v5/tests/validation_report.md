# V5 基本验证记录

执行日期：2026-11-10。Python：`/home/hnu/anaconda3/envs/turtlebot3_ws/bin/python`。

## 实际完成

- `test_frontier_ddqn.py`：**22 项通过**，完整运行约 7.3 秒（包含五场景离屏绘图）。补充的真实 Teacher 标签安全 Mask 断言也单独通过。
- 原始七份源码/README备份 SHA256 均与备份清单一致；launch 完整复制，runs 为指向原记录的目录链接。
- AST 对照：原低层 Actor、Critic、FuzzyKnowledge、HyperFusion、PPOAgent、compute_gae、goal_observation，以及原 SubgoalSpace/SubgoalProgress 不变。低层四个模块在高层更新后权重逐项相同且 requires_grad=False。
- `eval.py`、`turtlebot3_env.py` 与备份逐字节一致。旧基线入口 `legacy_high/main.py --help` 可运行；没有运行旧模型长评估。
- 新高层 225218 个参数；B=1/64 前后向形状正确、梯度有限。Q 优势均值覆盖全部225个输出。
- 地图坐标、半开边界、seen/零证据、同帧唯一更新、命中优先、最大量程不命中、越界明确报错、精确终点元数据和归一化通过。
- Dijkstra 米制距离、对角禁止穿墙、四邻接 Frontier、地图边缘信息收益分母、占用格心距离膨胀、点到路径线段距离通过。
- Teacher 的精确终点、已知目标路线、超过3.5 m的可达Frontier、死胡同向背离最终目标的出口回退、弃权和确定性并列选择通过。五场景交互中实际动作及有效教师标签都在同一次候选 Mask 内。
- Masked CE 中无效动作/无标签样本不产生监督梯度，全部无标签返回有限零；Masked Double DQN 在线选/目标估值、`gamma^(k/25)`、终止无bootstrap、超时bootstrap及空下一Mask通过。
- Replay 独立快照、uint8编解码、环形恢复、优化器/目标网络/更新计数恢复通过。每条记录7745字节；容量50000总数组387250000字节（约387.25 MB/369.3 MiB），其中两端地图360 MB。不含网络/采样临时张量。
- 正常 Student 评估用“Teacher构造/搜索调用即报错”桩验证：不创建/调用Teacher，不调用其Dijkstra。仍正常构建在线地图；Sim2D真值距离场只用于奖励/统计。
- 受阻无动作单独终止、−100附加罚、不bootstrap且不冒充超时；初始化空Mask明确报错。Teacher诊断弃权不回退Student。
- 五场景 Matplotlib Agg 离屏渲染、坐标轴/图元、关窗后继续step通过，未出现原 PatchCollection 图例警告。未验证真实Tk窗口交互；离屏截图仅为图元检查。

## 小预算实际流程

所有产物放 `/tmp/frontier_v5_*`，不混入用户原有 runs。

1. 16个训练交互步，保存间隔8，单次验证1回合、超时4步：保存 step_000008、step_000016、last，CSV两行，更新3次，同步1次。
2. 24个训练步，保存间隔12，单次验证1回合、超时2步，warmup2、每条更新1次、同步间隔2、衰减2决策：12个决策、11次梯度更新、5次同步。采集λ为 `[0.5,0.5,0.5,0.25,0,…]`；仅前4个决策查询Teacher，之后完全停止生成标签。保存/评估2次，无最终重复行。
3. 从第2项 last 恢复到28步：累计14决策、13更新，未重新衰减或重新生成标签，最终CSV仅一行本次模型。
4. 加载第2项模型做手动Student评估：trace中Teacher查询全False、标签−1、ε=0。
5. `--guidance off` 8步流程：Teacher查询0，DDQN更新3次，正常保存/评估。
6. Teacher独立诊断1回合、4步：单独 `teacher_diagnostic_*` CSV，不改Student排名。
7. 检查TensorBoard event：训练回报、结果率、Student评估、TD/CE/λ及教师统计存在；CSV行数等于成功保存的step模型数。

这些极短任务刻意限制为2或4个底层step，评估全部超时，**不构成学习能力或导航成功率证据**。

## 0.2 m 建图量化实测

执行 `check_mapping.py`，原始24条结果保存在 `mapping_results.json`。

使用理想无噪声LiDAR，在测试用通道中心线采样多帧，检测公开三值图经过指定格心膨胀后的连通性。测试夹具读取几何仅用于生成雷达和采样位姿，Teacher仍只读三值图。此测试不是低层真实行走验收，也不是单次初始扫描的可达性结论。

| 场景 | 宽度(m) | 对齐偏移(m) | 传感器建图连通 |
|---|---:|---|---|
| 直通道 | 0.3 | 0 / 0.1 | **不连通 / 连通** |
| 直通道 | 0.4、0.6、0.8、1.2 | 0 / 0.1 | 全连通 |
| 门洞 | 0.3、0.4、0.6、0.8、1.2 | 0 / 0.1 | 全连通 |
| 原 narrow 验证任务 | seeds=5000004/5000009/5000014/5000019 | 原布局 | 四个均连通 |

0.3 m直通道已经显示对齐敏感性；门洞图上连通也不等于机器人具有足够连续净空。没有放宽原连续扫掠Mask或修改场景难度。量化后不可规划时Teacher弃权。

## 未执行与限制

- 当前 PyTorch 环境 `torch.cuda.is_available()` 为False，本次只验证CPU，没有CUDA运行结论。
- 无可用Xvfb，窗口检查用Agg离屏；真实桌面Tk窗口、Gazebo在线通信尚未在本次重构后联调。
- 未执行20万步完整训练、长验证/独立test或多种子实验，没有宣称成功率改善。
- 地图固定 `[-6,6)²`；例如House目标x=−6.5明确不兼容，观测越界不会裁剪。
- Student没有访问记忆或时序输入，Teacher可能弃权或次优；这些是本次简化结构的实际限制。
