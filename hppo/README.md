# TurtleBot3 HPPO：四帧高层选点与单帧底层 PPO

仅在 `hppo/` 实现，两阶段均运行在 ROS1 Noetic / Gazebo 中。保留四个实现文件：
`main.py` 管训练、评估与高层执行段；`ppo_agent.py` 包含底层 PPO 和高层 Dueling Double DQN；
`util.py` 包含网络、两种 buffer、四帧窗口、候选子目标、地图及奖励；`turtlebot3_env.py` 管 ROS。
不导入 `ppo/`、`kgppo/` 或论文项目的 Python 模块。设计参考仓库的 `分层.pdf`，
首版不添加 HER、历史地图、LSTM/GRU 或论文的底层障碍编码。

## 启动与自动训练

依赖：ROS1 Noetic、Gazebo、Python 3.8、NumPy、PyTorch、TensorBoard、rospkg。
可选窗口使用 Matplotlib TkAgg，需要 Tkinter 和图形桌面；关闭窗口功能时不导入 Matplotlib。
本机解释器：`/home/hnu/anaconda3/envs/turtlebot3_ws/bin/python`。

先启动 Gazebo：

```bash
source /opt/ros/noetic/setup.bash
source /home/hnu/turtlebot3_ws/devel/setup.bash
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_gazebo turtlebot3_stage_4.launch
```

另一个终端进入项目，使用已安装上述依赖的 Python 环境：

```bash
source /opt/ros/noetic/setup.bash
source /home/hnu/turtlebot3_ws/devel/setup.bash
cd /home/hnu/turtlebot3_ws/src/paper_7/hppo
python main.py
```

默认 `--stage auto --mode train --device cuda`，总预算为 **200000 个底层训练交互 step**。
一个进程先训练底层，能力达标后自动冻结并训练高层；评估交互另行计数，不进入训练预算。
不需要手动选择阶段或中间模型。默认最早第 30000 步检查底层，预训练上限为首次运行总预算的一半
（默认 100000 步），剩余预算用于高层学习；不是固定 30000 步强制交接。

```bash
python main.py --pretrain-steps 30000 --max-train-steps 200000
python main.py --checkpoint runs/auto_<时间>/last.pt
python main.py --mode eval --checkpoint runs/auto_<时间>/last.pt
```

`--pretrain-steps` 为最早允许交接步数，首次训练须满足
`0 < pretrain_steps <= max_train_steps // 2 < max_train_steps`。
到最早交接步数后，每个保存节点先更新剩余 PPO rollout，再用同一套底层参数检查两套固定局部任务，
每套 20 回合，各自至少成功 19/20 才交接。任务由种子固定，覆盖四个距离段
`[0.75,1.25]、(1.25,2]、(2,2.75]、(2.75,3.5] m`，以及五种初始相对方向
`−160°、−90°、0°、90°、160°`；两套起终点对不重复。评估使用确定性底层动作。
这些局部评估不写训练曲线，不改变参数、训练窗口或 Replay，不计训练步数。

通过后同步底层 Actor 到高层的独立副本，冻结原 Actor、Critic，记录 `switch_step`。
高层 Replay 从空开始，ε 从高层第 0 个交互步计时，reset 完整四角任务并重新填充四帧历史。
未结束的局部回合被截断：保留真实下一观测 bootstrap，不记录回报、超时或三率。
自然结束的局部回合正常记录；交接时清空最近 100 回合窗口，累计 episode 编号继续。

到预训练上限仍未通过，先更新并完成最后一次局部检查，保存模型、局部结果和 `last.pt`，
以“底层能力未达标”报错停止，**不强行冻结底层训练高层**。自定义上限若不落在保存节点，
通过时只交接，不额外保存/全局评估；未通过时仍保存失败状态及局部结果。

每 10000 个累计训练 step 保存后评估 20 回合完整四角导航。预训练期间使用当前底层副本和尚未训练的
高层；高层阶段使用当前分层策略。正常完成默认预算有 20 次保存评估；提前未达标停止会减少次数。
保存节点和交接重合时，先完成底层更新、检查和交接，再保存、评估一次。
周期评估样本不进入训练 buffer 或 Replay，也不改变训练计数和窗口。没有逐回合终端日志。

恢复会读取实际阶段、上限和交接步数，从新回合继续，不恢复 Gazebo 中间状态或未更新的 PPO rollout。
**预训练上限沿用首次运行的值**，延长总预算不会重新计算上限。训练恢复的 `--pretrain-steps` 须与模型一致；
网络宽度、速度上限、雷达规格也须匹配，Replay 容量须匹配。学习率使用本次命令的设置。
未达标而停止的运行不能直接续成高层，需重新设计训练预算后从头开始。
CUDA 不可用时明确报错；使用 CPU 需显式传 `--device cpu`。一个进程只控制一个环境。

极短预算只适合检查采样、更新、保存及“不达标停止”，不能在几个 step 内训练出合格底层。例如：

```bash
python main.py --max-train-steps 4 --pretrain-steps 1 --save-interval 2 --eval-episodes 1 --timeout-steps 1 --horizon 2 --epochs 1 --batch-size 2 --high-warmup 2 --replay-capacity 32
```

### 保留独立阶段模式

原有 `low/high` 模式仍可用于独立实验，它们继续使用独立计数。旧时序 checkpoint 仅允许评估，不允许继续训练。
底层模式评估局部任务，高层模式评估完整导航。自动模式使用新的 `HPPO_AUTO` checkpoint，
不隐式迁移独立阶段模型。

```bash
python main.py --stage low
python main.py --stage high --low-checkpoint runs/low_<时间>/last.pt
python main.py --stage low --checkpoint runs/low_<时间>/last.pt --max-train-steps 200000
python main.py --stage high --checkpoint runs/high_<时间>/last.pt --max-train-steps 200000
python main.py --stage high --mode eval --checkpoint runs/high_<时间>/last.pt
```

`--low-checkpoint` 仅用于启动独立高层训练，不能与 `--checkpoint` 同时使用，也不用于自动模式。

## 观测与四帧历史

每帧均为 `float32[364]`：

| 索引 | 内容与归一化 |
|---|---|
| `0:360` | 完整 360 束雷达，原消息顺序，截到 `[0,3.5]` m 后除以 3.5 |
| `360` | 实际线速度 / `v_max` |
| `361` | 实际角速度 / `w_max` |
| `362` | 目标欧氏距离 / 地图对角线 `5√2` m |
| `363` | 目标相对方向规整到 `[-π,π)` 后除以 π |

不使用 IMU。雷达正无穷按最大量程处理，NaN、负无穷和负值按 0 处理；零值参与碰撞判定。
真实雷达的 360 束整圈规格在环境初始化时检查一次，并读取实际角度。

**底层始终使用当前单帧 364 维**，最后两维参考当前局部目标或高层子目标。
**高层使用最终目标观测的连续四帧** `[s(t-3),s(t-2),s(t-1),s(t)]`，展平为 1456 维。
窗口每个底层环境 step 更新，不按高层决策次数采样。帧间仿真间隔会包含动作超时和零速度计算等待，不承诺固定 0.6 秒。
reset 用新首帧填满四个位置；换子目标不清空历史。训练、评估各回合重置历史，互不混入。
Replay 保存四帧独立快照，不引用可变窗口。`TurtleBot3Env.state_dim` 仍为 364。

## 底层预训练

从地图空闲格心选择直线可通行的起终点对；启动时预计算，线段与占用格接触即不可通行。
运行时直接采样，不循环重试。按底层累计训练步数执行固定课程：

| 底层交互步数 | 目标距离 | 初始相对方向 |
|---|---|---|
| `<10000` | 0.75～1.25 m | ±45° 均匀采样 |
| `10000～19999` | 0.75～2 m | ±90° 均匀采样 |
| `>=20000` | 0.75～3.5 m | 完整 ±180° |

课程只影响新回合的采样，不改动作或 PPO 更新。底层能力检查使用固定的完整距离/方向任务，
不受当前课程影响；独立 `--stage low --mode eval` 也使用完整课程范围。

底层 Actor/Critic、tanh 高斯策略和 PPO 更新与原 PPO 相同：两层 256 单元 Tanh MLP，
两维全局可学习 `log_std` 初始为 -0.5。训练采样，评估使用均值，保存 tanh 前样本并修正 Jacobian。

```text
v = (tanh(u_v) + 1) * v_max / 2    默认 [0, 0.2] m/s
w = tanh(u_w) * w_max              默认 [-1.5, 1.5] rad/s
r_L = 100*到达子目标 - 0.1 + 5*(前一步子目标A*距离 - 当前子目标A*距离) - 100*碰撞
```

到达条件为目标欧氏距离 `<0.5 m` 且未碰撞；碰撞为完整雷达最小距离 `<0.15 m`，同时发生时碰撞优先。
默认 800 step 超时，不另加超时惩罚。到达/碰撞不 bootstrap，超时 bootstrap 并切断跨回合 GAE。
优势标准化、价值 MSE、梯度裁剪、rollout 截断及末尾剩余样本更新沿用原实现。

| 底层参数 | 默认值 |
|---|---|
| `--gamma` / `--gae-lambda` / `--clip-ratio` | 0.99 / 0.95 / 0.2 |
| `--horizon` / `--batch-size` / `--epochs` | 2048 / 64 / 10 |
| `--actor-lr` / `--critic-lr` | 3e-4 / 3e-4 |
| `--net-width` / `--max-grad-norm` | 256 / 0.5 |

## 高层子目标与执行段

15 个等距区间覆盖 `[0,3.5] m`，候选半径为区间中心；15 个角度中心为 `0°,24°,…,336°`。
索引为 `距离索引*15+角度索引`，正角度向机器人左侧，网络固定输出 225 个 Q 值。
选点时将相对坐标转换为世界坐标，执行期间不随机器人运动改变。

安全掩码只读取最新雷达：排除距离 `<=0.5 m` 的候选，要求候选周围 0.15 m 圆盘位于已观测空闲区域。
实现保守地要求覆盖该圆盘的雷达角度（含边界相邻光束）量程均超过 `候选半径+0.15 m`。
遮挡、未知和安全间距不足的候选被屏蔽；部分近距及最外圈候选因此不可选，输出维度仍为 225。
全局地图不参与掩码。全部候选无效时报告位姿、雷达和掩码并报错清理，不选择备用动作。

高层训练与评估使用独立冻结的底层 Actor 副本，按高斯均值映射为实际动作。
自动预训练期间，每次保存前同步当前底层参数，评估不会冻结正在训练的原始 PPO Actor。
每段至少执行一个底层 step，再按以下优先级检查：

1. 最终目标成功、碰撞或完整回合超时，结束导航回合。
2. 当前子目标距离 `<0.5 m`，换子目标。
3. 累计动作执行时间达到 3 秒且没有有效进展，换子目标。
4. 达到本段自适应执行上限，换子目标。

有效进展参考**当前子目标**：历程最小距离相对进展锚点累计改善至少 0.05 m，或历程最小绝对方向误差
累计改善至少 15°，任一满足便更新锚点及时间。正常原地转向可刷新进展；重复摆动不会反复刷新计时。
机器人绕行时可以暂时远离最终目标，不因此被判停滞。移除拥挤度阈值及固定 30 秒换点。
上述计时和自适应上限只累计每步实际 `sim_duration`，不计零速度下的推理、绘图及模型更新时间。
每段从零开始计时；选点时按初始距离 d₀ 和弧度方向误差 h₀ 计算固定上限：

```text
T_max = max(6, 3 + 2*max(d₀-0.5, 0)/v_max + 2*abs(h₀)/w_max) 秒
```

换子目标不重置机器人、不清空四帧历史、不发送额外零速度。
完整导航起终点从四个 `(±2,±2)` 角点随机选两个不同点，跨回合可重复；
只有最终到达、碰撞或默认累计 800 个底层 step 超时才结束导航回合。

高层一条经验覆盖一次子目标选择及其实际执行段，奖励为：

```text
r_arrival   = 100，实际到达最终目标且未碰撞，否则 0
r_step      = -0.1*k，k 是本段实际执行的底层 step 数
r_dist      = 5*(执行段开始的最终目标A*距离 - 实际执行段结束的最终目标A*距离)
r_collision = -100，碰撞时，否则 0
r_out       = -20，选择不合适子目标时，否则 0
r_H         = 上述五项之和
```

硬掩码阻止选择不合适候选，因此首版实际有效动作的 `r_out=0`，不额外添加近障惩罚带。
不累计本段逐步 PPO 奖励，不用“假定已到达所选子目标”的距离。
停滞、自适应时限换点和完整回合超时均不新增额外惩罚，也不触发 -20。

高层使用两层 Tanh MLP，输入 **1456**，输出 `Q=V+A-mean(A)`，共 225 个值。
在线网络带掩码选择下一动作，目标网络估值，采用 Huber 损失。

| 高层参数 | 默认值 |
|---|---|
| `--net-width` / `--high-lr` | 256 / 3e-4 |
| `--batch-size` / `--max-grad-norm` | 64 / 0.5 |
| `--gamma` | 0.99，单位为一次高层决策 |
| `--replay-capacity` / `--high-warmup` | 50000 / 256 条高层经验 |
| `--target-update` | 每 200 次梯度更新硬同步目标网络 |
| `--epsilon-decay-steps` / `--epsilon-final` | 50000 底层训练 step / 0.05 |

ε 从 1.0 线性降至最终值；随机探索只采样有效候选。warmup 后每完成一条高层经验更新一次。
最终到达/碰撞不 bootstrap；普通子目标切换、800 步时间截断和保存节点截断均 bootstrap。
`--horizon`、`--epochs`、`--actor-lr`、`--critic-lr`、`--gae-lambda`、`--clip-ratio` 仅用于底层 PPO。

## Gazebo、地图与可视化

订阅 `/scan`、`/odom`，发布 `/cmd_vel`，参考本地 `jx/ros1_tb3_static_env.py` 的 APSR 交互。
初始化时继续仿真，之后 Gazebo 持续运行；推理、绘图、参数更新、保存和退出时均不暂停世界。
每步按以下顺序执行：

1. 发布当前 `[v,w]`，记录 ROS 仿真时钟。
2. 每隔 **0.05 秒仿真时间**检查动作时长，达到 `--action-duration` 后结束等待。
3. 通过 `finally` 发布零速度，再从回调缓存读取最新雷达、里程计并计算观测与奖励。

默认动作时长 **0.2 秒是等待下限**；受轮询和调度影响可能超出，不再保证精确推进 200 个物理步。
不调用 `gz` 子进程，不要求动作时长为整数物理步，不在普通 step 后额外等待传感器。
采用 `/use_sim_time=true` 和 `real_time_update_rate=0`；PPO 的动作、概率及更新算法不变，
不移植 APSR 的模糊引导、安全滤波或每回合强制首步零动作。

reset 停车并等待 0.1 秒仿真时间，调用 `/gazebo/reset_simulation`，设置机器人位姿后等待 0.15 秒，
删除并重建目标，再等待新的雷达和里程计。初始化、reset 有消息等待超时；服务失败直接上抛，不吞异常、不重试。
仿真时钟允许在 reset 后归零，高层历史和段计时也从新回合重新开始。
`info` 的 `sim_time` 为实际 ROS 仿真时钟，`sim_duration` 为本步动作窗口实测时长（reset 为 0），
`scan_age`、`odom_age` 为观测快照相对 ROS 时钟的消息年龄，单位均为秒。

目标是否需要删除以 reset 边界处 Gazebo 的实际模型列表为准，不缓存“曾生成”的布尔标记。
目标已不存在时直接生成新目标；仍存在时先删除再生成。删除服务的真实失败仍直接报错。
启动时清理历史目标，退出时停车、删除目标并释放订阅；重复清理仅执行一次，Gazebo 保持运行。

默认 world 通过 rospkg 定位 `turtlebot3_gazebo/worlds/turtlebot3_stage_4.world`。
`--world-file` 支持相对或绝对路径；Gazebo 需自行加载相同场景，该参数不切换 Gazebo。
静态地图从 SDF collision 构建，支持本地 include、组合位姿、矩形及圆柱，
沿用 0.15 m 欧氏膨胀和 0.2 m 栅格、25×25 格、八邻接及禁止对角穿墙。
默认 stage 4 占用 339 格、空闲 286 格。A* 只计映射后的空闲格心间距离，同格距离奖励为 0。

```bash
python main.py --a_star on
python main.py --mode eval --checkpoint runs/auto_<时间>/last.pt --a_star on
```

窗口向上为世界 +x，向左为 +y。底层显示局部目标 A* 与逐步 PPO 奖励；高层显示最终目标 A*、
紫色三角子目标、当前执行段的换点原因、步数、实际时长、自适应上限、无进展时间、
历程最小距离与方向误差，以及最近一段高层奖励的实际开始/结束距离和五个奖励分项。
窗口的 `active elapsed` 为累计动作执行时间，不包含零速度计算等待。
两阶段均显示实际仿真推进时长及雷达/里程计消息年龄。
高层奖励仅在执行段结束时刷新，执行途中保留上一段值。手动关窗只停止显示，训练继续。
目标 Gazebo 模型继续使用 `ppo_goal`，reset 删除重建；启动清理历史标记，退出清理当前目标。

## 保存、恢复与统计

自动模式创建 `runs/auto_<UTC时间>/`；独立模式创建 `runs/low_<UTC时间>/` 或 `runs/high_<UTC时间>/`，包含 `step_*.pt`、最终 `last.pt`、
`checkpoint_metrics.csv` 和 TensorBoard event。最终 `last.pt` 与最终 step 模型同内容，不重复记 CSV。

自动 checkpoint 标识为 `HPPO_AUTO`，包含 `phase`、`low_steps`、`high_steps`、`low_updates`、
累计 episode/训练 step/更新次数、当前阶段窗口历史，以及嵌套的完整底层 PPO 状态和高层状态。
底层状态含 Actor、Critic 和两个优化器；高层状态含在线/目标 Q 网络、优化器、有效 replay、
高层决策及更新次数、独立冻结 Actor 和配置。`phase` 为 `low/high`，不记录评估临时阶段。
新训练标注 `training_version=3`、`step_backend=gazebo_ros_continuous_apsr`，保存课程阶段、
局部门槛结果、`pretrain_limit`、实际 `switch_step` 和运行 `status`。3 万步只代表最早检查点，
`step_030000.pt` 的实际阶段取决于当次局部评估结果。旧版本允许显式评估对照，
训练恢复会拒绝其他交互版本（包括原生精确步进版本 2），避免混入旧 Replay；旧模型文件不改写。

独立底层 checkpoint 标识为 `HPPO_LOW`，包含网络、优化器、配置、计数和最近 100 回合历史，
不保存未满 rollout。高层标识为 `HPPO_HIGH`，声明观测形状 `(4,364)`，包含在线/目标 Q 网络、
优化器、replay 有效数据及环形位置、累计高层决策数、配置、计数、窗口历史，以及冻结底层
Actor 和其配置。高层模型评估/恢复不依赖原始底层 checkpoint 文件。

保存节点中途截断当前执行段，使用真实结束观测完成经验与正常到期更新，之后保存再评估。
未完成完整导航回合不记录 episode reward、不进入三率窗口；评估后重置训练回合及四帧窗口。
高层 replay 保留并恢复；低层继续沿用正常结束时更新剩余 buffer 的规则。

CSV 保留 `checkpoint,total_steps,episode,window_episodes,phase,low_steps,high_steps`、三种训练窗口率、
评估次数和三种全局评估率；自动模式增加 `pretrain_limit,switch_step,status,low_eval_episodes,`
`low_eval_bank_1_success_rate,low_eval_bank_2_success_rate,low_eval_success_rate,low_eval_collision_rate,`
`low_eval_timeout_rate,low_qualified`。底层局部结果与全局成功率分列，不混入 CSV 排名。
保存成功先写待评估行，完成评估再补填；按评估成功率降序、累计 step 降序排列，未完成评估排末尾。
窗口为当前训练阶段最近 100 个完整回合，不足时用实际数量。自动切换时清空窗口，
累计 episode 编号继续；`phase` 明确训练统计所属任务。没有完整回合时训练三率留空。
自动模式所有评估指标对应同一种完整导航任务，可以统一排名。
中断评估保留模型及待评估 CSV 行，不记录部分评估成绩。

```bash
tensorboard --logdir runs
```

仅记录三张图：`episode/reward`、`episode/outcome_rate_100`（横轴为完整训练回合编号），
以及 `eval/outcome_rate`。自动模式评估横轴是两阶段累计训练 step，独立模式按本阶段计数。
训练 reward 和窗口按当时的局部/完整导航任务记录，跨阶段不直接比较训练回报；高层回报为各段奖励之和。
评估固定使用 `seed+1000000+i`，不进入训练 buffer、replay、计数或窗口。
终端只输出完整评估汇总，例如 `eval episodes=20 success=75.0% collision=15.0% timeout=10.0%`。
标准手动评估不生成训练日志、CSV 或模型。持续运行和回调采样下不保证相同种子的物理轨迹完全一致。

### 120 回合覆盖评估

```bash
python main.py --mode eval --checkpoint runs/auto_<时间>/last.pt --eval-suite coverage
# 独立高层模型：再加 --stage high
```

覆盖所有 12 组有向角点任务，每组 10 次，共 120 回合。初始朝向采用原朝向加固定扰动
`[-15,-10,-5,0,5,10,15,-7.5,7.5,0]°`。使用确定性策略，不训练、不消耗训练预算；
完整运行后终端只显示一行三率汇总。结果单独保存为
`runs/coverage_<UTC时间>/coverage_results.csv`，含每回合起终点、初始朝向、结果、步数、仿真时长、回报及决策数。
未完成全部回合时不输出完整汇总。120/120 成功是实验目标，须用完成训练的固定最终模型实测，
不根据短流程检查宣称达成。

配置在入口集中校验，传感器、地图、服务及 checkpoint 在边界检查；不吞异常、不自动重试。
完整参数通过 `python main.py --help` 查看。短联调验证实现与运行流程，导航效果需完整训练衡量。

## 实现验证

已通过 11 项回归检查，包括目标存在时先删后建、目标已被删除时直接重建、删除真实失败上抛、
幂等清理、每步仅停车一次、异常停车、累计动作计时隔离、PPO 与基线采样/更新逻辑一致、
版本 3 恢复及版本 2 仅允许评估。

隔离 Gazebo 连续 100 次动作窗口为 0.20～0.25 秒，平均约 0.235 秒；
连续 20 次 reset 中交替主动删除旧目标，均只保留一个目标，起点及目标位置核对通过。
CUDA 短训练、两阶段保存恢复、确定性评估、窗口刷新/关闭及 Ctrl+C 清理已通过；
训练联调中也主动移除了目标，验证后续 reset 不会因本地缓存失真而重复删除。
交接成功分支使用测试评分验证流程，不代表短训练底层已真实达到两套 95% 门槛。

短联调只验证实现与运行流程；完整训练后才能判断导航成功率，不能据此宣称达到 120/120。
所有 Gazebo 联调使用隔离端口和私有图形显示，未操作现有训练进程。
