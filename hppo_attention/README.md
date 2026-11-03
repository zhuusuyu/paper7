# HPPO-Attention：二维训练高层，冻结原低层策略

高层学习选择临时子目标，原 PPO-Attention 负责输出线速度、角速度。所有实现位于本目录，
不导入其他算法目录的模块。原低层 PPO 类和网络实现保留，已有模型、日志和 `eval.py` 保留。

## 训练与可视化

二维训练只需要 Python 3.8、NumPy、PyTorch、TensorBoard，不需要启动 ROS/Gazebo。
默认 CUDA，不可用时明确报错；CPU 使用 `--device cpu`。

```bash
cd /home/hnu/turtlebot3_ws/src/paper_7/hppo_attention
/home/hnu/anaconda3/envs/turtlebot3_ws/bin/python main.py
# 开启窗口，需安装 Matplotlib、Tkinter 并有可用的图形桌面
/home/hnu/anaconda3/envs/turtlebot3_ws/bin/python main.py --render on
```

默认训练 **200000 个底层交互 step**，每步 0.2 秒仿真时间。低层模型固定为：

```text
runs/attention_paper/step_180000.pt
```

`--low-checkpoint` 可在新训练时指定其他完整 PPO_ATTENTION v1 模型。
冻结原神经 Actor、空间注意力、模糊知识、超网络及 Critic；低层始终使用确定性融合策略。
引导下限按低层 checkpoint 的固定训练步数计算，180000 步模型的下限为零，学习到的知识融合仍保留。
只更新高层 Dueling Double DQN，不微调低层，不使用教师控制器。

窗口显示墙体、机器人与朝向、完整雷达、最终目标、当前子目标、轨迹，以及步数、距离、执行段时间、
换点原因和执行段奖励。奖励在整段结束后才有数值，执行中显示 `pending`。
坐标保持 **向上为 +x，向左为 +y**。窗口没有人为等待，关闭后训练继续，不自动重开；关闭显示时不导入绘图库。

短流程检查和单类场景训练：

```bash
python main.py --max-train-steps 320 --save-interval 160 --timeout-steps 40 \
  --eval-episodes 5 --batch-size 4 --warmup 4 --replay-capacity 128
python main.py --scenario u_wall --render on
```

## 二维场景与观测

`sim2d_env.py` 实现差速运动学、矩形墙体射线雷达、连续子步碰撞检测、场景生成及可选窗口。
每个动作内部以不超过 0.01 秒的子步推进，不瞬移至子目标。
机器人足迹用半径 0.105 m 的圆近似，雷达位于车体中心后方 0.032 m；量程为 3.5 m，噪声标准差 0.01 m。
这是运动学近似，不模拟 Gazebo 的轮胎接触、惯性和全部驱动延迟。

默认五类场景轮流采样，边界为 12×12 m；门宽、通道宽度、墙体尺寸和旋转随任务种子变化。

| `--scenario` | 内容 |
|---|---|
| `all` | 五类依次采样，默认值 |
| `u_wall` | 起点位于 U 型墙内，必须退出后绕到墙背后的目标 |
| `dead_end` | 起点位于封闭支路，必须退回入口再绕行 |
| `rooms` | 四个房间，通过多个门口连接 |
| `door` | 一道分隔墙，不同位置与宽度的门洞 |
| `narrow` | 0.6～1.2 m 净宽的 S 型连续转弯通道 |

生成任务时检查起终点余量和膨胀后连通性；不循环重试、不自动修复布局。
距离奖励使用 0.2 m 栅格、0.15 m 欧氏膨胀、八邻接且禁止对角穿墙。
每个任务预计算一次目标距离场，查询映射后最近空闲格心的距离；不增加真实位置到格心的连接距离。
地图仅用于任务验证、奖励和评估指标，不提供给高层选点或低层控制。

高低层每帧均是 `float32[364]`：360 束雷达截到 `[0,3.5]` 后除以 3.5，
实际线速度/上限、实际角速度/上限、目标距离/`5√2`、目标方向误差/π。
保持低层 checkpoint 的真实雷达角度，不把最后一束改成另一种 1° 排列。

- 高层：最近四帧拼接为 1456 维，每帧指向**最终目标**。
- 低层：当前单帧 364 维，指向**当前子目标**，速度上限和动作时长读取低层配置。
- 每个底层 step 更新四帧；reset 用首帧填满，换子目标保留历史，评估后重新初始化历史。
- 成功：最终目标实际距离严格 `<0.1 m`，且没有碰撞。
- 碰撞：雷达最小距离严格 `<0.15 m`，或足迹碰撞，碰撞优先。
- 默认 800 步超时，超时不额外加惩罚。

## 高层选择、换点与训练

225 个候选是 15 个 `[0,3.5] m` 距离区间中心 × 15 个方向 `0°,24°,…,336°`。
索引为距离索引×15+方向索引；最大区间中心约 3.383 m，不是 225 个独立角度。

安全掩码只用当前雷达，检查候选周围 0.15 m 圆盘是否位于已确认空闲区域，计入雷达安装偏移与角度采样余量。
普通候选距离须大于 0.5 m，量程边缘、遮挡和未知区域不可选。
最终目标在量程内时，用它替换最近的一个候选；它仍需通过安全检查，仍由高层选择，允许小于 0.5 m，
以保证离散动作不限制最终 0.1 m 的到达精度。
随机探索、贪心动作和 Double DQN 的下一动作选择共用掩码。
全部候选无效时报告场景、位姿及雷达/掩码统计并报错，不注入备用动作。

子目标转换为选点时的世界坐标，执行中固定。每段至少执行一个底层 step：

1. 最终成功、碰撞或完整回合超时，结束回合。
2. 普通子目标距离 `<0.3 m` 时换点；精确最终目标用 `<0.1 m`。
3. 持续 3 秒无有效进展时换点。历史最小距离累计改善 0.05 m，或历史最小绝对方向误差累计改善 15°，才刷新进展时间。
4. 达到 `max(6, 3+2*max(d0-到点阈值,0)/v_max+2*abs(h0)/w_max)` 秒时换点。

不使用固定 30 秒切换，不因暂时远离最终目标而换点；换点不重置机器人或清空历史。

每条 Replay 经验对应一整段子目标执行，执行 k 个底层 step 的奖励为：

```text
r_arrival   = +100，成功
r_collision = -100，碰撞
r_step      = -0.1 * k
r_dist      = 5 * (段开始到最终目标的A*距离 - 段结束距离)
r_out       = 0，安全掩码已过滤不合适候选；不另扣停滞或时间惩罚
```

保留论文“不合适候选 -20”的语义，但当前被硬掩码过滤后正常经验不会触发这一项，不能把停滞伪装成该惩罚。

网络为 `1456→256→256` Tanh，共享特征接价值头和 225 维优势头。
Adam 学习率 3e-4，batch 64，Replay 50000，warmup 256 条高层经验，梯度裁剪 0.5。
每条经验更新一次，每 200 次更新同步目标网络；Huber 损失，γ=0.99，单位为一次高层决策。
ε 在前 100000 个底层训练 step 从 1.0 降至 0.05。成功、碰撞不 bootstrap，超时及保存节点人为截断保留 bootstrap。
不加 HER、GRU、低层适配或规划器接管；四帧提供短期变化信息，不等同于长期导航记忆。

## 保存、恢复与独立评估

每 10000 个训练 step 保存后验证 20 个固定任务，每类 4 个。
严格按训练步触发，执行段中途遇到节点则用真实状态收尾；未完成回合不计入训练回报及三率。
训练种子从 `seed+任务序号` 开始，验证使用 `seed+1000000+序号`，独立测试使用 `seed+2000000+序号`。
评估不更新参数、Replay、训练计数或最近100回合窗口。

`runs/hppo_attention_<UTC时间>/` 包含 `step_*.pt`、最终 `last.pt`、`checkpoint_metrics.csv` 和 TensorBoard。
Checkpoint 使用 `HPPO_ATTENTION` v1，包含在线/目标网络、优化器、有效 Replay、完整冻结低层、配置、计数、窗口和随机数状态。
可以独立评估；恢复从新回合开始，不恢复仿真中的中间位姿。
恢复读取已保存的学习超参数；训练任务 `--seed`、`--scenario`、`--timeout-steps` 须与原运行一致。

```bash
python main.py --checkpoint runs/<运行目录>/last.pt --max-train-steps 300000
python main.py --mode eval --checkpoint runs/<运行目录>/last.pt --render on
# 独立100个未见任务：每类20个
python main.py --mode eval --eval-suite test --checkpoint runs/<运行目录>/last.pt
# 同一测试集上的“低层直接追最终目标”对照
python main.py --mode eval --eval-suite test --low-baseline --checkpoint runs/<运行目录>/last.pt
# 只测冻结低层的可通视局部任务，默认20个；不训练高层
python main.py --mode eval --eval-suite local --low-baseline
```

手动评估生成一份 `evaluation_<UTC时间>.csv`，包含场景、种子、结果、步数、最终距离、路径长度及 SPL。
`checkpoint_metrics.csv` 按验证成功率降序、训练 step 降序排序，包含训练窗口三率及各场景验证成功率。
终端只打印完整评估的三率汇总；TensorBoard 只记录三张图：

- `episode/reward`，横轴完整训练 episode。
- `episode/outcome_rate_100`，横轴完整训练 episode。
- `eval/outcome_rate`，横轴底层训练 step。

```bash
tensorboard --logdir runs
```

## Gazebo 同几何迁移检查

训练使用二维环境，Gazebo 入口仅用于评估，不创建地图或计算 A*。
先导出单个二维布局（打印对应起点、目标和朝向），再启动同几何场景：

```bash
python main.py --scenario u_wall --scenario-seed 3 --export-world /tmp/hppo_u.world
source /opt/ros/noetic/setup.bash
source /home/hnu/turtlebot3_ws/devel/setup.bash
roslaunch ./launch/sim2d_gazebo.launch world:=/tmp/hppo_u.world
```

启动资源使用原 burger 模型和 3.5 m 雷达，模型名 `turtlebot3`。其他场景可根据导出结果给 launch 传入
`x:=... y:=... yaw:=...`，其中 yaw 为弧度，确保初次生成机器人时不在墙内。
另一个终端加载高层模型，自动设置导出布局的起终点：

```bash
python main.py --backend gazebo --mode eval --scenario u_wall --scenario-seed 3 \
  --checkpoint runs/<运行目录>/last.pt --eval-episodes 1
```

已有 House 场景可指定实际世界坐标，`--yaw` 单位为度：

```bash
python main.py --backend gazebo --mode eval --checkpoint runs/<运行目录>/last.pt \
  --start -3 1 --yaw 180 --goal -6.5 -2.5 --eval-episodes 1
```

保持暂停状态发布动作→继续→等待0.2秒仿真时间→暂停→读取新消息。
动作实际时间可有服务调度误差，事件计时累计实际动作窗口，不包含暂停期间的推理时间。
普通动作间不发零速度；结束、重置、退出时停车，幂等清理最终目标 `ppo_goal` 和子目标 `hppo_subgoal`。
最终目标为绿色，子目标为紫色。Gazebo 中不提供全局路径，CSV 的 SPL 记为0（不计算参考最短路）。

原 `eval.py` 保留为**仅低层**的 House 固定路线评估；评估完整分层策略请使用上述 `main.py --backend gazebo`。

## 实现检查

已检查：几何射线、随机布局连通性、距离场、掩码、四帧副本、运动学、换点、带掩码 Double DQN、
低层冻结、截断 bootstrap、checkpoint 恢复、按步保存和训练/评估隔离。
已在独立嵌套窗口验证 CUDA 可视化训练与关窗继续运行，并在隔离端口对五种导出场景完成 Gazebo 短评估及退出清理。
短训练只验收实现；完整20万步后的复杂场景成功率和Gazebo迁移效果需独立测试，不预先保证。
