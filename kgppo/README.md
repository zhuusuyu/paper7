# TurtleBot3 可训练模糊知识引导 PPO（KGPPO）

四个实现文件：`main.py` 管训练/评估；`ppo_agent.py` 管 PPO 更新与模型；
`util.py` 放神经/模糊网络、rollout buffer、A* 地图及奖励；`turtlebot3_env.py` 管 ROS 和观测。
仅在 `kgppo/` 实现知识引导，`ppo/` 保持为基线。模糊隶属度与规则设计参考本机
`attention_final_v2`，不引入注意力、TTC、后方规则或额外安全覆盖。
参考了本机 `jx/ros1_tb3_static_env.py` 的接口与
[PPO-Continuous](https://github.com/YMUDRL/Deep-Reinforcement-Learning-Algorithms-with-Pytorch/tree/main/3.2%20PPO-Continuous)
的组织方式，运行时不导入它们的 Python 模块。

## 依赖与启动

使用 ROS1 Noetic、Gazebo、Python 3.8、NumPy、PyTorch、TensorBoard 和 rospkg。
ROS 消息/服务来自 `rospy`、`sensor_msgs`、`nav_msgs`、`geometry_msgs`、`gazebo_msgs`、`std_srvs`。
不需要 Gymnasium、SLAM、map_server 或额外的 A* 库。
可选 A* 窗口使用 Matplotlib 的 TkAgg 后端，需要 Matplotlib、Python Tkinter/Tk 和图形桌面；
本机环境已安装。默认关闭时不导入 Matplotlib，也不要求图形桌面。
本机已验证可导入这些依赖的解释器为：

```bash
/home/hnu/anaconda3/envs/turtlebot3_ws/bin/python
```

先在终端一启动默认 stage 4 场景（使用你已删除圆柱的 world 文件）：

```bash
source /opt/ros/noetic/setup.bash
source /home/hnu/turtlebot3_ws/devel/setup.bash
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_gazebo turtlebot3_stage_4.launch
```

终端二训练：

```bash
source /opt/ros/noetic/setup.bash
source /home/hnu/turtlebot3_ws/devel/setup.bash
cd /home/hnu/turtlebot3_ws/src/paper_7/kgppo
/home/hnu/anaconda3/envs/turtlebot3_ws/bin/python main.py --mode train
```

默认使用 CUDA，神经 Actor、模糊 Actor、Critic 和 PPO 更新都在 GPU 上执行。
CUDA 不可用时直接报错，不自动切回 CPU；需要使用 CPU 时显式传 `--device cpu`。
不要同时运行其他向 `/cmd_vel` 发布消息的控制程序。
修改代码后先停止旧训练进程，再启动新进程，当前进程不会自动加载修改。

需要逐步检查 A* 路径和奖励时开启窗口：

```bash
python main.py --a_star on
python main.py --mode eval --checkpoint runs/train_<时间>/last.pt --a_star on
```

`--a_star` 只接受 `on`、`off`，默认 `off`。训练、周期评估和手动评估共用一个窗口，
显示膨胀后的占用栅格、本次实际搜索扩展过的格子、最短路径、机器人实际位置和目标。
窗口方向为向上是 x 正方向、向左是 y 正方向；地图、路径和位置标记采用相同显示方向。
数值面板逐步显示回合内 step、上一步及当前 A* 距离（m）、`r_arrival`、`r_step`、`r_dist`、
`r_collision` 和总 `reward`；距离与奖励显示 6 位小数，奖励带正负号。
数值来自本步环境的实际计算，不在窗口中重算奖励。reset 时显示新距离，上一步距离和奖励显示为“—”。
窗口每个 reset/step 刷新，不逐节点播放或人为等待；PPO 更新期间保留最后一帧。
关闭窗口只停止显示，训练/评估继续且不自动重开；退出时清理窗口，图形初始化失败直接报错。
窗口里的路径是 A* 距离奖励的参考路径，机器人动作仍由 PPO 决定。
关闭功能时不记录搜索过程；逐步数值仅显示在窗口，终端仍只输出完整评估汇总。

默认总计训练 **100000 个训练交互 step**，每 **10000 step** 先保存模型，再确定性评估 **20 回合**。
评估交互不计入训练步数。`--save-interval` 同时控制保存和评估间隔，`--eval-episodes` 控制每次评估回合数。
短训练联调可以缩小步数预算、保存间隔、rollout 和更新轮数：

```bash
python main.py --mode train --max-train-steps 64 --save-interval 32 --eval-episodes 2 --timeout-steps 32 --horizon 32 --epochs 2 --batch-size 16
```

也可手动评估或继续训练，替换下面的 checkpoint 路径：

```bash
python main.py --mode eval --checkpoint runs/train_<时间>/last.pt --eval-episodes 20
python main.py --mode train --checkpoint runs/train_<时间>/last.pt --max-train-steps 200000
```

`--max-train-steps` 是包含 checkpoint 中已有步数的总训练目标，严格按环境 step 停止，
不要求最后一个 episode 完整结束。它替代原来的 `--episodes`；`--save-interval` 的单位也改为 step。
首次从头训练；`--checkpoint` 只接受完整 KGPPO checkpoint，不迁移基线 PPO 模型。
加载时网络宽度、速度上限、三个引导参数及真实雷达角度规格须与 checkpoint 相同；
如原训练改变了这些参数，加载时也传相同值。训练恢复优化器，但使用本次命令的学习率，
从新回合开始，不恢复未更新的 buffer 或 Gazebo 中间状态，同时恢复最近 100 个完整回合的结果。
评估只读取网络，不更新参数。训练重置种子按 `seed + 已完成 episode 数` 设置；
这不保证异步 Gazebo 运行完全确定。

完整 checkpoint 也可直接作为策略使用。`obs` 必须是下文定义的、已经处理过的 364 维观测：

```python
import torch
from ppo_agent import KGPPOAgent

path = "runs/train_<时间>/step_010000.pt"
saved = torch.load(path, map_location="cuda", weights_only=False)
agent = KGPPOAgent(dict(saved["config"], device="cuda"))
counters = agent.load(path)
action, _, _, alpha = agent.select_action(
    obs, counters["total_steps"], deterministic=True,
)  # [v, w]；使用该 checkpoint 对应的引导比例
```

## 观测与控制

| 状态索引 | 内容 | 处理 |
|---|---|---|
| `0:360` | 完整一圈雷达，保留消息顺序 | 距离截到 `[0,3.5]` m，再除以 3.5 |
| `360` | `/odom` 的 `twist.twist.linear.x` | 除以 `v_max` |
| `361` | `/odom` 的 `twist.twist.angular.z` | 除以 `w_max` |
| `362` | 到目标的欧氏距离 | 除以地图对角线 `5√2` m |
| `363` | 目标相对机器人朝向 | 规整到 `[-π,π)` 后除以 π |

观测是 `float32[364]`，不使用 IMU。正无穷雷达值按 3.5 m 处理，
NaN、负无穷、负距离按 0 处理。这些零值也参与碰撞判定。
环境启动时一次性验证雷达为整圈 360 束；运行中假定传感器规格不变。
当前 Gazebo 插件使用 world 来源的里程计，因此目标和 `/odom` 位姿使用相同坐标。

神经 Actor 为两层 256 单元 Tanh MLP，输出高斯均值，并学习两维全局 `log_std`（初始 -0.5）。
神经均值先与模糊动作融合，构造统一高斯策略（见下节）；从该分布采样的值经过 tanh，
再映射为 `[v,w]`：

```text
v = (tanh(u_v) + 1) * v_max / 2     默认 [0, 0.2] m/s
w = tanh(u_w) * w_max               默认 [-1.5, 1.5] rad/s
```

训练时采样，评估时对高斯均值执行相同映射。buffer 保存 tanh 前的样本，
log probability 包含 tanh 和线性映射的 Jacobian 修正。

## 模糊 actor 与引导退出

模糊输入为 `[h, rho_F, rho_L, rho_R]`：`h` 是观测中的归一化目标方向。
环境初始化时读取真实 `LaserScan.angle_min/angle_increment`，一次性构造前方
`[-45°,45°)`、左侧 `[45°,135°)`、右侧 `[-135°,-45°)` 的扇区掩码，
不假定雷达数组的第一个元素朝前。各扇区取最小距离 `d_D`，风险为
`rho_D = clip((0.70 - d_D) / (0.70 - 0.15), 0, 1)`。

高斯方向隶属度中心为 `[-0.58, 0, 0.58]`，宽度为 `[0.20, 0.16, 0.20]`；
风险低/高中心为 `[0, 1]`，宽度均为 `0.35`。中心、宽度和后件固定。
以下后件对应默认速度上限；改变上限时分别按 `v_max/0.2`、`w_max/1.5` 缩放。
未列出的条件为“不关心”。

| 条件 | v (m/s) | w (rad/s) | 固定优先级 |
|---|---:|---:|---:|
| 前低、目标居中 | 0.18 | 0 | 1 |
| 前低、目标在左 | 0.11 | +0.65 | 1 |
| 前低、目标在右 | 0.11 | −0.65 | 1 |
| 前高、左低、右高 | 0.025 | +1.00 | 2 |
| 前高、左高、右低 | 0.025 | −1.00 | 2 |
| 前高、左高、右高 | 0 | 0 | 2 |
| 前低、左高 | 0.07 | −0.75 | 2 |
| 前低、右高 | 0.07 | +0.75 | 2 |
| 前高、目标在左、左低 | 0.025 | +0.90 | 2 |
| 前高、目标在右、右低 | 0.025 | −0.90 | 2 |
| 前高、目标居中、左低 | 0.015 | +0.80 | 2 |

每个实际前件有独立乘子（28 个），每条规则有一个乘子（11 个），共 **39 个可训练标量**。
乘子均为 `sigmoid(logit)`，logit 初始为 0，即乘子 0.5；相同的初始缩放在归一化中抵消。
规则用 min 合取，再乘规则乘子和固定优先级，归一化加权后件得到模糊动作 `a_K`。
实现使用 log 隶属度、`logsigmoid` 和 softmax，避免极小覆盖度下的数值下溢。
停止规则也参加加权聚合，不硬覆盖最终动作。

令 `a_N = T(mu_N)` 为神经 Actor 确定性动作，融合物理动作均值：

```text
m = (1 - alpha) * a_N + alpha * a_K
mu_H = atanh(clip((m - action_bias) / action_scale, -1 + 1e-6, 1 - 1e-6))
policy = Normal(mu_H, exp(log_std))
```

训练从统一策略采样，评估使用该分布均值；最终速度输出端不再额外加噪声。
PPO 概率比使用实际融合策略的联合 log probability。Actor 损失为：

```text
L_actor = L_PPO + mean(alpha * mean(((a_N - stopgrad(a_K)) / [v_max, w_max]) ** 2))
```

两个 actor 均接收 PPO 主目标梯度，模仿项只更新神经 actor。
二者共用 Actor Adam 优化器和学习率，联合裁剪梯度；Critic 独立优化。

引导比例与模仿系数使用相同的 `alpha(t)`，`t` 为已完成训练 step：

```text
alpha(t) = initial * (exp(-k*t/T) - exp(-k)) / (1 - exp(-k)), t < T
alpha(t) = 0,                                            t >= T
```

| 参数 | 默认值 |
|---|---:|
| `--guidance-initial` | 0.8 |
| `--guidance-steps`（T） | 80000 |
| `--guidance-exp-rate`（k） | 3.0 |

第 0 步系数为 0.8，第 40000 步约为 0.145940，达到 80000 步后新动作完全使用神经策略。
`alpha=0` 时直接使用原神经高斯均值，不经过额外 atanh 或裁剪。
buffer 逐条保存采集时的 alpha，更新概率及模仿项都使用该值。
80000 步不额外触发更新；跨越该边界的 rollout 仍用旧样本的非零系数完成更新。
整个 minibatch 的 alpha 均为 0 后跳过模糊分支，其参数和 Adam 动量不再更新。
周期/手动评估固定使用 checkpoint 对应累计步数的系数，评估回合不推进衰减。

## 仿真步进

初始化取得传感器消息后，Gazebo 保持暂停。每步在暂停状态下发布 Agent 动作，调用
`/gazebo/unpause_physics`，通过 `rospy.sleep` 等待默认 **0.2 秒仿真时间**，再调用
`/gazebo/pause_physics`。暂停后取得本次动作开始之后的新雷达、里程计，计算观测和奖励。
不要求暂停后再产生动作结束时刻之后的消息。

正常动作之间不发送额外零速度。暂停会保留机器人的位置、速度等物理状态；
推理、PPO 更新和模型保存期间仿真不推进。仅初始化、回合结束、reset 和退出时发送停车指令。
reset 使用 `/gazebo/reset_world` 重置场景而不将仿真时钟归零，设置起终点后以零速度
短暂运行，取得新消息后重新暂停。初始化会短暂解除暂停取消息，因此也可在上次训练留下的暂停状态下启动。

默认 `/use_sim_time=true`，`real_time_update_rate=0` 可继续用于加速仿真。
0.2 秒指仿真时间，不是实际等待 0.2 秒；暂停服务和调度存在延迟，因此不是严格推进 200 个物理步。
动作期间需要产生新的雷达和里程计消息；默认雷达 5 Hz 与 0.2 秒动作相配，
若缩短动作时长，应相应提高传感器更新频率。
运行窗口通过 `try/finally` 恢复暂停，异常继续上抛，不自动重试。
环境接口：`reset(seed) -> (obs, info)`、
`step(action) -> (obs, reward, terminated, truncated, info)`、`close()`。

## 地图与奖励

默认通过 `rospkg` 查找 `turtlebot3_gazebo/worlds/turtlebot3_stage_4.world`。
可通过 `--world-file` 指定绝对或相对路径，
例如切换到 APSR：

```bash
python main.py --world-file /绝对路径/turtlebot3_jx_apsr_static.world
```

Gazebo 同时要加载对应场景，例如 APSR 使用：

```bash
roslaunch turtlebot3_jx_gazebo turtlebot3_jx_apsr_static.launch
```

地图从静态模型的 box/cylinder **collision** 几何构建，组合 include、model、link、collision
的平移与 yaw。支持展开 `model://turtlebot3_plaza` 等本地 SDF 引用，也支持 `file://` 和相对路径引用；
模型目录通过 `model.config` 定位 SDF。`model://` 依次查找 `GAZEBO_MODEL_PATH`、world 上一级的
`models/`、`~/.gazebo/models` 和 `turtlebot3_gazebo/models`。
sun/ground_plane 不作为障碍，非静态模型不写入固定 A* 底图。
当前支持 5 m × 5 m 平面场景的 box/cylinder 静态碰撞几何；不支持的几何、非平面旋转或
相对命名坐标在初始化时明确报错。
更换 world 文件时，应自行让 Gazebo 加载同一场景；该参数不会重启或改变 Gazebo。

先将原始碰撞几何按欧氏距离向外膨胀 **0.15 m**，再用 **0.2 m** 方格栅格化。
0.15 m 是总膨胀距离，不再叠加机器人半径或其他裕量。矩形膨胀包含圆角，圆柱半径增加 0.15 m；
格子与膨胀区域相交或接触即占用，不通过给占用格增加一整圈来膨胀。
格心以原点对齐，为 `-2.4,-2.2,...,2.4`，共 `25×25` 格；格子边界覆盖 `[-2.5,2.5]`。
默认 stage 4 占用 339 格、空闲 286 格；四角 `(±2,±2)` 都在格心上。
A* 用八邻接，直行 0.2 m、对角 `0.2√2` m，
禁止对角穿墙。启动时验证所有空闲格连通、四个任务角点为空闲且所有起终点组合可达。
默认 stage 4 和当前仓库的 APSR 场景均已通过连通性及全部 12 组角点任务的路径检查。
位置映射到最近格心；占用位置投影到最近空闲格，等距时按行列顺序选择。
`d_A*` 只包含格心之间的最短路，不包含位置到格心的连接距离；同格移动奖励为零。
地图仅用于奖励，不把 A* 路径作为策略观测或控制指令。
地图膨胀与雷达的 0.15 m 碰撞判定阈值独立。兼容的 KGPPO checkpoint 可直接加载；继续训练时采用当前地图的
A* 距离奖励，不改写已有模型和历史统计，因此新旧地图下的回报不宜直接比较。
起终点仍从四个 `(±2,±2)` 角点随机选择两个不同点，目标不固定为对角，允许与上一回合重复。
成功、碰撞或超时后，先保存终止步数据、更新回合计数和窗口统计。普通回合结束时先 reset
下一训练回合，再执行可能的 PPO 更新；保存节点则先执行到期更新、保存、评估，之后才 reset
下一训练回合。达到总训练步数后只完成最终评估，不再创建训练回合。
buffer 中保留的是 reset 前的真实下一观测及结束标记。

目标模型固定名为 `ppo_goal`。启动时只清理该模型及历史 `tb3_ppo_*_goal` 标记，
不清理其他场景模型。每次 reset 都先删除上一回合的目标，再直接在选定目标坐标重新生成
`ppo_goal`，不通过 `set_model_state` 移动静态目标。即使连续两轮随机到同一目标，也会重新生成，
以刷新 Gazebo 中的显示；同一时刻最多保留一个目标标记。
每次 reset 生成目标前等待 0.2 秒真实时间，让 GUI 先处理此前的删除消息（包括启动清理）；
此时仿真保持暂停，该等待不计入动作的仿真时间。
正常结束、异常和 Ctrl+C 共用同一清理流程；即使 ROS 已进入 shutdown，也会尝试删除目标。
强制终止而无法执行回调时，遗留标记在下次启动时清理。

```text
r_arrival   = +100，欧氏目标距离 < 0.5 m 且未碰撞，否则 0
r_step      = -0.1
r_dist      = 5 * (上一步 d_A* - 当前 d_A*)
r_collision = -100，全圈最小雷达距离 < 0.15 m，否则 0
reward      = r_arrival + r_step + r_dist + r_collision
```

到达、碰撞结束回合；同时发生时碰撞优先。到达步仍有 -0.1 和距离奖励，
碰撞步为 `-100.1 + r_dist`。默认 800 步超时，不额外处罚超时。
reset 初始化上一步路径距离，不为 reset 本身计算奖励。
`info` 提供位姿、目标、欧氏距离、方向、最小雷达距离、A* 距离，以及 step 的各奖励项和结束原因。
每步还提供 `step`、`previous_astar_distance` 和 `reward`，分别为回合内步数、实际用于距离奖励的
上一步 A* 距离及本步总奖励。reset 的 `step` 为 0，不生成尚未产生的奖励项。

## PPO 与输出

独立 Critic 同样使用两层 256 单元 Tanh MLP。默认参数：

| 参数 | 默认值 |
|---|---|
| 折扣 `--gamma` / GAE `--gae-lambda` | 0.99 / 0.95 |
| PPO `--clip-ratio` | 0.2 |
| `--horizon` / `--batch-size` / `--epochs` | 2048 / 64 / 10 |
| `--actor-lr` / `--critic-lr` | 0.0003 / 0.0003 |
| `--max-grad-norm` | 0.5，两个 actor 联合裁剪，Critic 独立裁剪 |
| `--max-train-steps` / `--eval-episodes` | 100000 训练 step / 每次评估 20 回合 |
| `--save-interval` | 每 10000 训练 step 保存并评估 |

使用优势标准化和价值 MSE，不添加熵奖励、学习率调度、奖励缩放。
到达/碰撞不 bootstrap；超时用 reset 前观测 bootstrap，同时切断跨回合 GAE；
rollout 截断、评估节点截断和总步数预算截断也使用下一状态价值。评估节点截断切断跨回合 GAE，
但不清空未满的 buffer；评估结束后继续收集训练样本。正常完成训练时更新剩余样本，
随后保存最终步数的 checkpoint 和 `last.pt`，并评估一次。周期节点与最终节点重合时不重复执行。

达到总步数上限时，如果最后一回合还未到达、碰撞或环境超时，则不记录这个未完成回合的
reward，也不将它当作超时放进统计窗口；它已采集的样本仍参与最终更新。
每个保存节点严格按训练 step 触发，不等待当前回合结束；被评估节点中断的未完成回合同样
不增加已完成 episode 数、不记录回合 reward，也不进入统计窗口。评估后从新训练回合开始。

每次训练创建独立的 `runs/train_<UTC时间>/`。
`runs` 相对于启动命令所在目录；按上面的命令启动时，输出位于 `kgppo/runs/`：

- `step_010000.pt`、`step_020000.pt` 等周期 checkpoint。
- 最终步数对应的 checkpoint（默认 `step_100000.pt`）和 `last.pt`，两者包含相同的最终模型。
- TensorBoard event 文件；三种比例的子目录由 TensorBoard 自动创建。
- `checkpoint_metrics.csv`：本次运行保存的各个 step 模型及对应训练窗口、独立评估统计。

checkpoint 包含 `algorithm="KGPPO"` 标识、神经 Actor、模糊 Actor 的参数及规则 buffer、Critic、
联合 Actor 优化器、Critic 优化器、配置（包括引导参数和雷达角度）、已完成 episode 数、
累计训练 step、更新次数和最近 100 个完整回合的结果。
周期保存只保存当时最新模型，不额外触发 PPO 更新；不保存未完成的 rollout。
如果保存点落在回合中间，已完成 episode 数和窗口都不包含该未完成回合；恢复后从新回合开始。

保存成功后先将最近 100 个完整训练回合的结果写入 `checkpoint_metrics.csv`，评估列留空、
`eval_episodes` 为 0；完整评估结束后补填同一行，不改写已经保存的 checkpoint。字段如下：

| 字段 | 含义 |
|---|---|
| `checkpoint` | 同目录的 `step_*.pt` 文件名 |
| `total_steps` / `episode` | 保存时累计 step / 已完成 episode 数 |
| `window_episodes` | 实际参与统计的回合数，最多 100 |
| `guidance_alpha` | 此模型在保存步数下的引导比例，也是评估固定使用的比例 |
| `success_rate_100` | 窗口成功率，范围 `0～1` |
| `collision_rate_100` / `timeout_rate_100` | 同一窗口的碰撞率 / 超时率 |
| `eval_episodes` | 完整评估的回合数，默认 20；未完成评估时为 0 |
| `eval_success_rate` | 确定性评估成功率，范围 `0～1` |
| `eval_collision_rate` / `eval_timeout_rate` | 同次评估的碰撞率 / 超时率 |

每个 step 模型一行；CSV 按评估成功率降序排列，并列时按训练 step 降序，未完成评估的行排在末尾。
尚无完整训练回合时，三个训练窗口比例留空、窗口数量写 0，仍可正常评估和排序。
`last.pt` 是最终 step 模型的同内容别名，不重复写一行。
恢复训练时使用 checkpoint 中的窗口历史，新目录的表只记录本次恢复运行新保存的模型。
不生成逐回合或逐更新的 CSV。

评估复用同一个 Gazebo 环境，使用当前系数的融合高斯均值经 tanh 和缩放生成动作，不进行随机动作采样，
不更新网络或优化器。每次使用相同的重置种子 `seed + 1000000 + i`，其中 `i=0…eval_episodes-1`，
便于比较不同 checkpoint；这不保证异步 Gazebo 仿真完全确定。评估数据不进入 buffer、训练计数或
最近 100 回合窗口。评估中断时保留已经保存的模型和未完成评估的 CSV 行，并执行环境清理。

## TensorBoard

在相同 Python 环境、项目目录中启动：

```bash
tensorboard --logdir runs
```

浏览器打开 `http://localhost:6006`。训练仅记录以下三张图：

| 标签 | 内容 | 横轴 |
|---|---|---|
| `episode/reward` | 每个完整训练 episode 的总 reward | 已完成训练 episode 编号 |
| `episode/outcome_rate_100` | 训练窗口的 `success`、`collision`、`timeout` 三条比例曲线 | 已完成训练 episode 编号 |
| `eval/outcome_rate` | 每次完整评估的 `success`、`collision`、`timeout` 三条比例曲线 | 累计训练 step |

训练窗口包含刚完成的当前回合及此前最多 99 回合；不足 100 回合时按已有数量计算。
每完成一回合就记录一次，比例范围为 `0～1`，三者相加为 1。
例如首回合碰撞，则该点的成功率、碰撞率、超时率为 `0、1、0`。
第 101 回合起移除窗口中最早的一回合。恢复训练时延续 episode 编号与 checkpoint 中的窗口历史，
并写入新的训练目录；不记录 loss、奖励分项等其他 TensorBoard 指标。

周期评估与手动 `--mode eval` 均只在完整评估结束后输出一行汇总，例如：

```text
eval episodes=20 success=75.0% collision=15.0% timeout=10.0%
```

不输出启动信息、逐回合日志、reset 坐标、保存提示或训练结束提示；未完成评估不输出汇总，
错误仍正常上抛。CSV、TensorBoard 和训练统计照常记录。
手动评估不创建输出目录、CSV、TensorBoard event 或模型文件。
完整参数通过 `python main.py --help` 查看。

配置在入口集中检查；传感器同步和 ROS 服务结果在环境边界检查。
不吞异常、不自动重试、不替换无法计算的 A* 距离。

## 实现验证

已通过 12 项数值检查和 10 项训练流程模拟检查：包括全部 39 个模糊标量的 PPO 梯度、
模仿项的梯度隔离、采样与重算概率一致、逐样本系数保存、零系数下与基线 PPO 的动作/概率/更新
等价，以及已有 Adam 动量下模糊参数保持不变。检查也覆盖完整 checkpoint 恢复、
十万步十次保存评估、CSV 系数与排序、三张 TensorBoard 图和终端汇总。

已在独立 ROS/Gazebo 端口完成 CUDA 短训练、周期评估、早期引导模型的手动评估、
退出引导后的恢复训练及评估期间 Ctrl+C 清理。联调使用 12 步训练预算、5 步保存间隔、
2 回合评估，并将衰减周期临时设为 8 步，以检查跨边界的行为；没有改变默认配置。
这些检查验证实现与运行流程，导航成功率仍需通过完整训练衡量。
