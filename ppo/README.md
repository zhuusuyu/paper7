# TurtleBot3 基础 PPO 导航

四个实现文件：`main.py` 管训练/评估；`ppo_agent.py` 管 PPO 更新与模型；
`util.py` 放网络、rollout buffer、A* 地图及奖励；`turtlebot3_env.py` 管 ROS 和观测。
参考了本机 `jx/ros1_tb3_static_env.py` 的接口与
[PPO-Continuous](https://github.com/YMUDRL/Deep-Reinforcement-Learning-Algorithms-with-Pytorch/tree/main/3.2%20PPO-Continuous)
的组织方式，运行时不导入它们的 Python 模块。

## 依赖与启动

使用 ROS1 Noetic、Gazebo、Python 3.8、NumPy、PyTorch、TensorBoard 和 rospkg。
ROS 消息/服务来自 `rospy`、`sensor_msgs`、`nav_msgs`、`geometry_msgs`、`gazebo_msgs`、`std_srvs`。
不需要 Gymnasium、SLAM、map_server 或额外的 A* 库。
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
cd /home/hnu/turtlebot3_ws/src/paper_7/ppo
/home/hnu/anaconda3/envs/turtlebot3_ws/bin/python main.py --mode train
```

默认使用 CUDA，Actor、Critic 和 PPO 更新都在 GPU 上执行。
CUDA 不可用时直接报错，不自动切回 CPU；需要使用 CPU 时显式传 `--device cpu`。
不要同时运行其他向 `/cmd_vel` 发布消息的控制程序。
修改代码后先停止旧训练进程，再启动新进程，当前进程不会自动加载修改。

默认总计训练 **100000 个环境交互 step**，每 **5000 step** 保存一次模型。
短训练联调可以缩小步数预算、保存间隔、rollout 和更新轮数：

```bash
python main.py --mode train --max-train-steps 64 --save-interval 32 --timeout-steps 32 --horizon 32 --epochs 2 --batch-size 16
```

评估和继续训练分别运行，替换下面的 checkpoint 路径：

```bash
python main.py --mode eval --checkpoint runs/train_<时间>/last.pt --eval-episodes 20
python main.py --mode train --checkpoint runs/train_<时间>/last.pt --max-train-steps 200000
```

`--max-train-steps` 是包含 checkpoint 中已有步数的总训练目标，严格按环境 step 停止，
不要求最后一个 episode 完整结束。它替代原来的 `--episodes`；`--save-interval` 的单位也改为 step。
加载时网络宽度和速度上限须与 checkpoint
相同；如原训练改变了这些参数，加载时也传相同值。训练恢复优化器，但使用本次命令的学习率，
从新回合开始，不恢复未更新的 buffer 或 Gazebo 中间状态，同时恢复最近 100 个完整回合的结果。
旧 checkpoint 没有窗口历史时，从新回合开始累计窗口。评估只读取网络，不更新参数。
随机种子按 `seed + episode - 1` 设置；这不保证异步 Gazebo 运行完全确定。

完整 checkpoint 也可直接作为策略使用。`obs` 必须是下文定义的、已经处理过的 364 维观测：

```python
import torch
from ppo_agent import PPOAgent

path = "runs/train_<时间>/step_005000.pt"
saved = torch.load(path, map_location="cuda", weights_only=False)
agent = PPOAgent(dict(saved["config"], device="cuda"))
agent.load(path)
action, _, _ = agent.select_action(obs, deterministic=True)  # [v, w]
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

Actor 为两层 256 单元 Tanh MLP，输出高斯均值，并学习两维全局 `log_std`（初始 -0.5）。
采样值经过 tanh，再映射为 `[v,w]`：

```text
v = (tanh(u_v) + 1) * v_max / 2     默认 [0, 0.2] m/s
w = tanh(u_w) * w_max               默认 [-1.5, 1.5] rad/s
```

训练时采样，评估时对高斯均值执行相同映射。buffer 保存 tanh 前的样本，
log probability 包含 tanh 和线性映射的 Jacobian 修正。

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
每次启动都会打印实际使用的 world 绝对路径。可通过 `--world-file` 指定绝对或相对路径，
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

0.5 m 方格的格心为 `-2.5,-2.0,...,2.5`，共 `11×11` 格。
与障碍相交即占用，不额外膨胀。A* 用八邻接，直行 0.5 m、对角 `0.5√2` m，
禁止对角穿墙。启动时验证所有空闲格连通、四个任务角点为空闲且所有起终点组合可达。
位置映射到最近格心；占用位置投影到最近空闲格，等距时按行列顺序选择。
`d_A*` 只包含格心之间的最短路，不包含位置到格心的连接距离；同格移动奖励为零。
地图仅用于奖励，不把 A* 路径作为策略观测或控制指令。
起终点仍从四个 `(±2,±2)` 角点随机选择两个不同点，目标不固定为对角，允许与上一回合重复。
每次 reset 后，终端打印 `reset episode=... start=(x, y) goal=(x, y)`，可直接核对本回合位置。
成功、碰撞或超时后，先保存终止步数据、更新回合计数和窗口统计，再 reset 下一回合，
之后才执行可能的 PPO 更新和模型保存。下一回合每次只 reset 一次；达到总训练步数或
评估回合上限后，不再创建下一回合。buffer 中保留的是 reset 前的终止观测及结束标记。

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
r_step      = -1
r_dist      = 上一步 d_A* - 当前 d_A*
r_collision = -100，全圈最小雷达距离 < 0.15 m，否则 0
reward      = r_arrival + r_step + r_dist + r_collision
```

到达、碰撞结束回合；同时发生时碰撞优先。到达步仍有 -1 和距离奖励，
碰撞步为 `-101 + r_dist`。默认 800 步超时，不额外处罚超时。
reset 初始化上一步路径距离，不为 reset 本身计算奖励。
`info` 提供位姿、目标、欧氏距离、方向、最小雷达距离、A* 距离，以及 step 的各奖励项和结束原因。

## PPO 与输出

独立 Critic 同样使用两层 256 单元 Tanh MLP。默认参数：

| 参数 | 默认值 |
|---|---|
| 折扣 `--gamma` / GAE `--gae-lambda` | 0.99 / 0.95 |
| PPO `--clip-ratio` | 0.2 |
| `--horizon` / `--batch-size` / `--epochs` | 2048 / 64 / 10 |
| `--actor-lr` / `--critic-lr` | 0.0003 / 0.0003 |
| `--max-grad-norm` | 0.5，Actor/Critic 分别裁剪 |
| `--max-train-steps` / `--eval-episodes` | 100000 环境 step / 20 回合 |
| `--save-interval` | 每 5000 环境 step |

使用优势标准化和价值 MSE，不添加熵奖励、学习率调度、奖励缩放。
到达/碰撞不 bootstrap；超时用 reset 前观测 bootstrap，同时切断跨回合 GAE；
rollout 截断和总步数预算截断也使用下一状态价值。正常完成训练时更新剩余样本，
随后保存最终步数的 checkpoint 和 `last.pt`。

达到总步数上限时，如果最后一回合还未到达、碰撞或环境超时，则不记录这个未完成回合的
reward，也不将它当作超时放进统计窗口；它已采集的样本仍参与最终更新。

每次训练创建独立的 `runs/train_<UTC时间>/`。
`runs` 相对于启动命令所在目录；按上面的命令启动时，输出位于 `ppo/runs/`：

- `step_005000.pt`、`step_010000.pt` 等周期 checkpoint。
- 最终步数对应的 checkpoint（默认 `step_100000.pt`）和 `last.pt`，两者包含相同的最终模型。
- TensorBoard event 文件；三种比例的子目录由 TensorBoard 自动创建。
- `checkpoint_metrics.csv`：本次运行保存的各个 step 模型及对应训练窗口统计。

checkpoint 包含 Actor、Critic、两个优化器、配置、已完成 episode 数、累计采样 step、
更新次数和最近 100 个完整回合的结果。周期保存只保存当时最新模型，不额外触发 PPO 更新。
如果保存点落在回合中间，已完成 episode 数和窗口都不包含该未完成回合；恢复后从新回合开始。

训练不自动执行周期性 eval，也不会在保存模型时插入评估。保存成功后，将最近 100 个完整
训练回合的结果写入 `checkpoint_metrics.csv`，字段如下：

| 字段 | 含义 |
|---|---|
| `checkpoint` | 同目录的 `step_*.pt` 文件名 |
| `total_steps` / `episode` | 保存时累计 step / 已完成 episode 数 |
| `window_episodes` | 实际参与统计的回合数，最多 100 |
| `success_rate_100` | 窗口成功率，范围 `0～1` |
| `collision_rate_100` / `timeout_rate_100` | 同一窗口的碰撞率 / 超时率 |

每个 step 模型一行；CSV 按成功率降序排列，并列时按 step 降序，第一行即当前最高成功率模型。
尚无完整回合的模型将三个比例留空、窗口数量写 0，排在表末尾。
`last.pt` 是最终 step 模型的同内容别名，不重复写一行。
这些比例来自训练中的滑动窗口，不是额外评估得出的模型测试成绩。
恢复训练时使用 checkpoint 中的窗口历史，新目录的表只记录本次恢复运行新保存的模型。
不生成逐回合或逐更新的 CSV。

## TensorBoard

在相同 Python 环境、项目目录中启动：

```bash
tensorboard --logdir runs
```

浏览器打开 `http://localhost:6006`。训练仅记录以下两张图，横轴都是**已完成的 episode 编号**：

| 标签 | 内容 |
|---|---|
| `episode/reward` | 每个完整 episode 的总 reward |
| `episode/outcome_rate_100` | 同图中的 `success`、`collision`、`timeout` 三条比例曲线 |

窗口包含刚完成的当前回合及此前最多 99 回合；不足 100 回合时按已有数量计算。
每完成一回合就记录一次，比例范围为 `0～1`，三者相加为 1。
例如首回合碰撞，则该点的成功率、碰撞率、超时率为 `0、1、0`。
第 101 回合起移除窗口中最早的一回合。恢复训练时延续 episode 编号与 checkpoint 中的窗口历史，
并写入新的训练目录；不记录 loss、奖励分项等其他 TensorBoard 指标。

评估只在终端输出回合 reward 和窗口比例，不创建输出目录、CSV、TensorBoard event 或模型文件。
完整参数通过 `python main.py --help` 查看。

配置在入口集中检查；传感器同步和 ROS 服务结果在环境边界检查。
不吞异常、不自动重试、不替换无法计算的 A* 距离。
