# KGHPPO-Attention：二维训练高层，冻结原低层策略

高层通过可训练模糊知识、超网络融合和 Double DQN 选择临时子目标，原 PPO-Attention 负责输出线速度、角速度。
参考 NeurIPS 2024《Integrating Suboptimal Human Knowledge with Hierarchical Reinforcement Learning》
第3.1–3.2节的知识控制器与超网络融合思想，改为单机器人225候选，不加入多智能体分组控制器。所有实现位于本目录，
不导入其他算法目录的模块。原低层 PPO 类和网络实现保留，已有模型、日志和 `eval.py` 保留。

## 训练与可视化

二维训练只需要 Python 3.8、NumPy、PyTorch、TensorBoard，不需要启动 ROS/Gazebo。
默认 CUDA，不可用时明确报错；CPU 使用 `--device cpu`。

```bash
cd /home/hnu/turtlebot3_ws/src/paper_7/kghppo_attention
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
共同更新高层神经Q、模糊知识和超网络；低层完全不变。知识只给候选偏好，不直接控制速度。
本算法高层从头训练，`hppo_attention/` 保留作对照，不读取其高层模型或Replay。

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
三个高层模块共同使用Adam，学习率 3e-4，batch 64，Replay 50000，warmup 256 条高层经验，梯度裁剪 0.5。
每条经验更新一次，每 200 次完整同步三个目标模块；TD使用Huber损失，γ=0.99，单位为一次高层决策。
ε 在前 100000 个底层训练 step 从 1.0 降至 0.05；随机分支使用下述知识/均匀混合分布。成功、碰撞不 bootstrap，超时及保存节点人为截断保留 bootstrap。
不加 HER、GRU、低层适配或规划器接管；四帧提供短期变化信息，不等同于长期导航记忆。

## 高层知识、融合和模仿

知识只读取现有四帧观测和候选点，不读取A*、地图、场景名称、真实碰撞几何或访问记录。
相同知识同时用于二维训练和Gazebo评估。原空间注意力属于冻结低层，高层保留原MLP，不新增时序网络。

候选特征为：精确最终目标标记E、方向一致性A、距离/3.5的L、保守净空C、方向宽敞度O、目标受阻B、低运动程度M。
A=(1+cos(候选方向−目标方向))/2；C=clip((到雷达端点的最近距离−0.15)/0.55,0,1)，
包括量程边界，属于保守观测指标而非已知全局障碍距离；O为候选方向±15°雷达均值/3.5。
取目标方向最近两束雷达的最小值d_h，B=clip((min(目标距离,3)+0.15−d_h)/0.70,0,1)。
M=clip(1−max(四帧平均|v|/0.03,四帧平均|w|/0.15),0,1)，只表示短时低运动，不是长期停滞记忆。

| 规则 | 条件 | 偏好后件 |
|---|---|---:|
| 1 | 是精确最终目标 | +3 |
| 2 | 目标方向畅通、方向一致、净空高 | +1 |
| 3 | 目标方向畅通、方向一致、距离长 | +1 |
| 4 | 目标方向受阻、方向宽敞、净空高 | +2 |
| 5 | 目标方向受阻、运动程度低、方向宽敞 | +2 |
| 6 | 净空低、不是精确最终目标 | −2 |
| 7 | 距离短、不是精确最终目标 | −1 |
| 8 | 目标方向受阻、方向不宽敞 | −1 |

高/低隶属度是中心1/0、宽度0.35的高斯函数。19个实际前件、8条规则各有一个独立的
`2*sigmoid(logit)`乘子，初始化为1，共27个可训练标量。前件、中心、宽度、后件固定。
min合取和归一化加权在log空间计算；知识分数K为后件的激活加权平均。

置信度使用覆盖度、一致性和候选区分度：

```text
V_j = sum_r w_rj * ((后件_r-K_j)/3)^2
C_K = (1-exp(-mean_valid(sum_r e_rj))) * exp(-4*mean_valid(V_j)) * tanh(std_valid(K))
```

不确定或无法区分有效候选时置信度降低；只有一个有效动作时无需知识介入。
所有动作沿用原雷达安全掩码，知识不生成新的坐标，也不覆盖安全检查。

```text
Omega(t) = high_guidance_floor_initial * max(1-t/high_guidance_floor_steps,0)
g = softmax(HyperNet(stopgrad(context)))[knowledge]
alpha = stopgrad(C_K) * (Omega + (1-Omega)*g)
```

超网络为13→64→64→2 Tanh，末层初始化为零，初始g=0.5。
上下文由最新帧速度/目标四维、四方向风险、B、M、置信度及有界化的Q/知识标准差组成。
风险阈值与低层相同；Q标准差使用tanh(log1p(sigma_N))，知识标准差使用tanh(std(K))。
所有上下文停止梯度，超网络生成两个专家融合权重。

为对齐知识与Q值量级，均值/方差只统计有效候选：

```text
Z_K = (K-mean(K)) / sqrt(var(K)+1e-6)
sigma_N = sqrt(var(Q_N)+1e-6)
Q_H = mean(Q_N) + (1-alpha)*(Q_N-mean(Q_N)) + alpha*stopgrad(sigma_N)*Z_K
```

知识改变相对偏好，保留神经Q的平均价值；alpha为零时直接返回原Q值。
贪心选择有效候选中最大的Q_H。随机探索采用
`(1-alpha)*Uniform_valid + alpha*softmax_valid(Z_K)`，温度固定1，仍保留所有安全候选的探索机会。
Double DQN由在线完整融合网络选下一动作，由目标完整融合网络估值。

```text
P_K = softmax_valid(Z_K)
P_N = softmax_valid((Q_N-mean(Q_N))/stopgrad(sigma_N))
L = L_TD + high_imitation_coef * mean(stopgrad(alpha)*KL(stopgrad(P_K) || P_N))
```

TD更新全部高层模块；模仿项只更新神经Q，不更新模糊知识或超网络。
模仿系数默认1，不另加独立时间衰减；实际权重跟随alpha。
**10000步后只取消初始下限，不关闭知识分支，也不强制模仿项归零。**
`--high-guidance-floor-initial 0`同样不代表关闭知识，`--high-imitation-coef 0`可关闭模仿项。

| 参数 | 默认值 |
|---|---:|
| `--high-guidance-floor-initial` | 0.8 |
| `--high-guidance-floor-steps` | 10000 |
| `--high-imitation-coef` | 1.0 |

高层t从0计数；低层一直按其原180000步配置推理。评估固定使用高层checkpoint累计步数，回合内不推进下限。
Replay保存采集时Omega(t)和真实段结束时Omega(t+k)，更新使用原下限重算当前模型的知识和融合，
不保存冻结的旧融合权重，不使用更新时的新下限替代历史上下文。终止状态不计算下一动作；超时和保存截断保留bootstrap。
恢复/评估自动读取checkpoint的三个知识超参数，不修改已有运行记录。

## 保存、恢复与独立评估

每 10000 个训练 step 保存后验证 20 个固定任务，每类 4 个。
严格按训练步触发，执行段中途遇到节点则用真实状态收尾；未完成回合不计入训练回报及三率。
训练种子从 `seed+任务序号` 开始，验证使用 `seed+1000000+序号`，独立测试使用 `seed+2000000+序号`。
评估不更新参数、Replay、训练计数或最近100回合窗口。

`runs/kghppo_attention_<UTC时间>/` 包含 `step_*.pt`、最终 `last.pt`、`checkpoint_metrics.csv` 和 TensorBoard。
Checkpoint 使用 `KGHPPO_ATTENTION` v1，包含在线/目标网络、优化器、有效 Replay、完整冻结低层、配置、计数、窗口和随机数状态。在线与目标网络都包含神经Q、模糊规则和超网络；
规则buffer保存前件掩码、中心、宽度、后件、候选及雷达角度；
Replay包含当前/下一掩码、采集时及真实段结束时的两个引导下限。旧HPPO模型与Replay不隐式迁移。
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
`checkpoint_metrics.csv` 按验证成功率降序、训练 step 降序排序，包含训练窗口三率、各场景验证成功率及checkpoint当前的 `high_guidance_floor`。
终端只打印完整评估的三率汇总；TensorBoard 记录五张图：

- `episode/reward`，横轴完整训练 episode。
- `episode/outcome_rate_100`，横轴完整训练 episode。
- `eval/outcome_rate`，横轴底层训练 step。
- `high_guidance/weights`：floor、alpha、preference、confidence，横轴底层训练step，每次高层更新记录。
- `high_optimization/loss`：TD、加权模仿项、总损失，横轴底层训练step，每次高层更新记录。

权重图的 `floor` 是本次Replay minibatch里历史采集下限的均值；CSV记录当前训练时刻下限。
即使当前已经超过10000步，旧经验仍可能使图里的floor大于零，不代表采样时的衰减停止。

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

已检查：27个知识参数、规则偏好、Q量级、置信度、衰减端点、探索掩码、TD/模仿梯度隔离，
以及几何射线、随机布局连通性、距离场、掩码、四帧副本、运动学、换点、带掩码 Double DQN、
低层冻结、截断 bootstrap、checkpoint 恢复、按步保存和训练/评估隔离。
已在独立嵌套窗口验证 CUDA 可视化训练与关窗继续运行，并在隔离端口对五种导出场景完成 Gazebo 短评估及退出清理。
短训练只验收实现；完整20万步后以与HPPO相同的20个任务选模型，再测独立100个任务。
分别报告五类成功、碰撞和超时率，不预先保证超过HPPO当前45%的最佳验证成功率。

本次实现验收使用320步CUDA短训练（测试下限周期缩为200步），验证后恢复到340步；默认参数仍为20万步和10000步下限周期。
五类Gazebo接口测试每类最多20步，实际为4次超时、1次碰撞；未完成完整20万步训练，不将短联调当作性能提升。
