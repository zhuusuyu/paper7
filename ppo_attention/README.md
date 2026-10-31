# PPO-Attention：空间注意力与超网络知识融合

在 `ppo_attention/` 内独立实现，保持四个 Python 文件，不导入参考项目模块：
`main.py` 负责训练、评估、日志；`ppo_agent.py` 负责融合策略、PPO 更新与 checkpoint；
`util.py` 包含注意力网络、模糊知识、超网络、buffer、地图和奖励；`turtlebot3_env.py` 负责 ROS。

借鉴所附 STA-KGRL 的空间风险注意力、动作与模仿引导，以及
[NeurIPS 2024 hhk-MARL 第 3.2 节](https://papers.nips.cc/paper_files/paper/2024/file/ba29e3f830d039c3f1fa0b4dfcf19c54-Paper-Conference.pdf)
的超网络知识融合思想。这里是单机器人连续动作 PPO 的改造：只有单帧空间注意力，
没有四帧、TAGD、时序网络、多智能体控制或教师预训练。

## 启动

依赖 ROS1 Noetic、Gazebo、Python 3.8、NumPy、PyTorch、TensorBoard、rospkg。
可选 A* 窗口使用 Matplotlib TkAgg 和 Tkinter；关闭时不导入绘图库。

终端一启动与地图参数对应的场景：

```bash
source /opt/ros/noetic/setup.bash
source /home/hnu/turtlebot3_ws/devel/setup.bash
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_gazebo turtlebot3_stage_4.launch
```

终端二启动训练：

```bash
source /opt/ros/noetic/setup.bash
source /home/hnu/turtlebot3_ws/devel/setup.bash
cd /home/hnu/turtlebot3_ws/src/paper_7/ppo_attention
/home/hnu/anaconda3/envs/turtlebot3_ws/bin/python main.py
```

默认 CUDA；不可用时直接报错，CPU 需显式 `--device cpu`。一个进程控制一个环境。
默认总预算 **200000 个训练交互 step**，每 **10000 step** 保存后确定性评估 **20 回合**。
评估交互不进入训练预算。原有 `ppo/`、`kgppo/`、`hppo/` 不变。

```bash
python main.py --a_star on
python main.py --mode eval --checkpoint runs/attention_<时间>/last.pt
python main.py --checkpoint runs/attention_<时间>/last.pt --max-train-steps 300000
```

短联调示例：

```bash
python main.py --max-train-steps 13 --save-interval 5 --eval-episodes 2 \
  --timeout-steps 3 --horizon 4 --epochs 1 --batch-size 2
```

参数兼容性在加载边界检查一次。改变过网络宽度、速度上限、引导配置时，加载命令需传入相同值。
恢复优化器时采用本次命令的学习率，从新回合继续；不恢复 Gazebo 中间状态或未更新的 rollout。
修改源文件后需重启训练进程才能生效。

## 观测、动作和环境

环境接口保持 `reset(seed) -> (obs,info)`、`step(action) -> (obs,reward,terminated,truncated,info)`、`close()`。
每个观测是 `float32[364]`，动作是实际 `[v,w]`：

| 索引 | 内容 |
|---|---|
| `0:360` | 完整雷达，按原消息顺序，截到 `[0,3.5] m` 后除以 3.5 |
| `360` | 实际线速度 / `v_max` |
| `361` | 实际角速度 / `w_max` |
| `362` | 目标欧氏距离 / 地图对角线 `5√2 m` |
| `363` | 目标方向误差规整到 `[-π,π)` 后除以 π |

订阅 `/scan`、`/odom`，发布 `/cmd_vel`，不使用 IMU。
雷达正无穷视为最大量程，NaN、负无穷、负值视为零，零值参与碰撞判断。
初始化一次性验证完整 360 束规格并读取实际角度，建立神经网络几何 buffer 和模糊掩码。

沿用当前 PPO 的暂停/继续流程：在暂停状态发布动作，继续仿真并等待默认 **0.2 秒仿真时间**，
再暂停、取得本次动作开始后的新雷达和里程计。动作实际时长可能因调度超出；不保证固定物理步数。
普通动作间不额外发零速度；回合结束、reset 和退出时停车。推理、更新和保存期间保持暂停。
reset 使用 `/gazebo/reset_world` 保持时钟单调。目标统一为 `ppo_goal`，每回合删除重建，
是否存在以 reset 边界处的 Gazebo 模型列表为准，不缓存 `_goal_spawned`。
启动清理历史目标，`finally`/shutdown 幂等清理；真实服务错误直接上抛。

四角 `(±2,±2)` 随机选两个不同点作为起终点，跨回合可重复；下方起点朝上，上方朝下。
默认地图通过 rospkg 定位 `turtlebot3_gazebo/worlds/turtlebot3_stage_4.world`。
`--world-file` 支持相对或绝对路径，只指定读取的地图，不切换 Gazebo 场景。
地图继续由 collision 几何生成，0.15 m 欧氏膨胀、0.2 m 栅格、八邻接 A* 且禁止对角穿墙。
A* 仅用于奖励，不接管动作。默认地图占用 339 格，空闲 286 格。

奖励与 PPO 基线一致：

```text
r_arrival   = +100，目标实际距离 <0.5 m 且未碰撞
r_step      = -1
r_dist      = 5 * (上一步 A* 距离 - 当前 A* 距离)
r_collision = -100，完整雷达最小距离 <0.15 m
reward      = 上述四项严格相加
```

碰撞优先。默认 800 step 超时，不额外加超时惩罚。A* 距离只计映射后格心间的路径，同格移动距离奖励为零。
`--a_star on` 保留原地图、路径与实际奖励面板，向上为 +x，向左为 +y；关闭窗口后训练继续，不自动重开。

## 可训练的模糊知识

语义输入为归一化目标方向及前、左、后、右空间风险。四区域按真实角度划分：
前 `[-45°,45°)`、左 `[45°,135°)`、后 `[135°,180°)∪[-180°,-135°)`、右 `[-135°,-45°)`。

```text
rho_D = clip((0.70 - 该方向最小距离_m) / (0.70 - 0.15), 0, 1)
```

采用 STA-KGRL 表 II 的 16 条规则。下表动作列是论文原值，运行时按
`[v_max/0.22, w_max/1.5]` 缩放，未列前件表示不关心：

| 规则 | 前件 | v | w |
|---|---|---:|---:|
| 1 | 前低、目标居中 | 0.180 | 0 |
| 2 | 前低、目标左 | 0.110 | +0.650 |
| 3 | 前低、目标右 | 0.110 | -0.650 |
| 4 | 前高、左低、右高 | 0.025 | +1.000 |
| 5 | 前高、左高、右低 | 0.025 | -1.000 |
| 6 | 前高、左高、右高 | 0 | 0 |
| 7 | 前低、左高 | 0.070 | -0.750 |
| 8 | 前低、右高 | 0.070 | +0.750 |
| 9 | 前高、目标左、左低 | 0.025 | +0.900 |
| 10 | 前高、目标右、右低 | 0.025 | -0.900 |
| 11 | 前高、目标居中、左低 | 0.015 | +0.800 |
| 12 | 前低、后高 | 0.130 | 0 |
| 13 | 前低、左高、后高 | 0.090 | -0.450 |
| 14 | 前低、后高、右高 | 0.090 | +0.450 |
| 15 | 前高、左低、后高 | 0.010 | +0.900 |
| 16 | 前高、后高、右低 | 0.010 | -0.900 |

前三条优先级为 1，其余为 2。方向高斯中心 `[-0.58,0,0.58]`、宽度 `[0.20,0.16,0.20]`；
风险中心 `[0,1]`、宽度 `0.35`。中心、宽度、前件条件、优先级和后件固定。
42 个实际前件和 16 条规则各有一个 `2*sigmoid(logit)` 乘子，共 **58 个可训练标量**，初始为 1。
min 合取、优先级和归一化权重在 log 空间稳定计算，不增加零覆盖备用动作。
停止规则也参加加权，不硬覆盖最终动作。

线速度乘目标接近系数 `s_g=0.6+0.4*clip(g/0.2,0,1)`；这里 g 为观测索引 362 的**归一化目标距离**，
不是米制距离。设当前加权规则激活为 e、归一化权重为 w，缩放后的后件为 b，模糊动作 aK 为加权均值：

```text
V = sum_r w_r * mean(((b_r-aK)/[v_max,w_max])**2)
C = (1-exp(-sum_r e_r)) * exp(-4*V)
```

C 使用数值稳定的 `expm1` 计算覆盖度。模糊权重通过 PPO 主目标学习，不接受模仿项反传。

## 空间注意力 Actor

完整雷达按原顺序划为 **60 个扇区，每区 6 束**。使用真实角度，将每束距离转为除以 3.5 m 的
Cartesian 点；每区输入为 12 维点坐标、2 维扇区方向和 4 维速度/目标状态，共 18 维。

共享 `18→64→64` Tanh 编码器分别连接注意力分数头及 32 维特征头。
将规则中高风险前件涉及的方向投影到扇区；跨区域边界的扇区按圆均值方向归属，得到固定规则掩码。
当前归一化规则权重加权这些掩码形成 q，每个扇区风险 r 由六束中的最小距离计算：

```text
u_i = (1+q_i) * r_i
attention = softmax(neural_logits + attention_prior_scale * stopgrad(u))
```

注意力加权得到 32 维汇聚特征，再拼接速度/目标四维，进入 `36→net_width→net_width→2` Tanh MLP。
两维输出为神经高斯均值，另有全局可学习 `log_std`，初始 -0.5。Critic 独立保持 `364→256→256→1`。
`--net-width` 同时控制 Actor 输出 MLP 和 Critic 的宽度，默认 256；空间编码器固定为 64。
注意力偏置默认 1，不随步数机械衰减；学习的分数可以覆盖该偏置。知识注意力停止梯度，
其反传只经过神经编码器，不改变模糊规则权重。

## 超网络融合与策略优化

超网络 `13→64→64→4`、Tanh 激活，输入顺序为速度/目标四维、方向风险四维、置信度一维，
以及按动作上限归一化的神经动作和模糊动作各两维。整个输入作为停止梯度的上下文。
四个输出组成 `[动作维度, 神经/教师]` 两行权重，逐行 softmax 得到教师偏好 g。
最后一层权重与偏置初始化为零，初始 g=0.5。

```text
Omega(t) = guidance_floor_initial * max(1-t/guidance_floor_steps, 0)
alpha_j  = stopgrad(C) * [Omega + (1-Omega)*g_j]
m_j      = (1-alpha_j)*aN_j + alpha_j*aK_j
```

t 为采样动作前已完成的训练 step。默认初始下限 0.8、10000 步线性降为零。
这里是对论文下限约束的平滑改造：**下限归零后 alpha 不必为零，知识分支继续参与训练与部署**。
`--guidance-floor-initial 0` 也只是取消初始下限，不会关闭超网络和模糊知识。

用现有 tanh 速度映射 T 先计算 aN=T(muN)，再将实际速度均值 m 反映射为融合均值：

```text
muH = atanh(clip((m-action_bias)/action_scale, -1+1e-6, 1-1e-6))
```

alpha 为零的维度直接用 muN。训练从 `Normal(muH, exp(log_std))` 采样，经过 T 执行动作；
评估使用融合高斯均值。最终动作后没有额外噪声或安全过滤。
PPO 概率比使用这个真实执行分布；保存 tanh 前样本，沿用稳定的 tanh/线性映射 Jacobian 修正。

```text
L_IL    = mean[ stopgrad(alpha_j) * ((aN_j-stopgrad(aK_j))/D_j)**2 ]
L_actor = L_PPO_Clip + imitation_coef * L_IL
```

PPO 主目标更新神经 Actor、模糊权重和超网络；模仿项只更新神经 Actor（包括注意力）。
不让模仿项通过 alpha 降低自己的权重，也不让教师移动来迎合学生。
三个策略模块共享 Actor Adam 和联合梯度裁剪，Critic 用独立优化器。
保留原 PPO 的 GAE、优势标准化、价值 MSE、无熵奖励、无学习率调度等行为。

| 参数 | 默认值 |
|---|---:|
| `--guidance-floor-initial` | 0.8 |
| `--guidance-floor-steps` | 10000 |
| `--imitation-coef` | 1.0 |
| `--attention-prior-scale` | 1.0 |
| `--gamma` / `--gae-lambda` / `--clip-ratio` | 0.99 / 0.95 / 0.2 |
| `--horizon` / `--batch-size` / `--epochs` | 2048 / 64 / 10 |
| `--actor-lr` / `--critic-lr` | 3e-4 / 3e-4 |
| `--max-grad-norm` | 0.5 |

Buffer 每条样本保存原始状态、真实下一状态、tanh 前动作、旧 log probability、采样时 Omega、
实际 alpha_v/alpha_w/C 诊断、奖励及结束标记。更新重算当前超网络和知识分支，使用**该样本原来的 Omega**，
不冻结旧融合权重、不用更新时的步数替换旧 Omega。到达/碰撞不 bootstrap，超时 bootstrap 并切断跨回合 GAE。

## 保存、评估与 TensorBoard

新训练目录为 `runs/attention_<UTC时间>/`，包含每万步的 `step_*.pt`、最终 `last.pt`、
`checkpoint_metrics.csv` 和 TensorBoard event。周期节点先完成正常到期更新、保存，再评估；
不为周期评估额外更新 PPO。正常结束先更新剩余样本，最终节点不重复保存评估。
尚未结束的回合在保存节点截断，保留真实下一观测进行 bootstrap，不记录不完整回报或计为超时。
评估样本不进入 buffer、不计训练 step/episode/最近 100 回合窗口；评估后重置训练任务。
评估固定种子 `seed+1000000+i`，整个评估使用 checkpoint 步数对应的固定 Omega，不推进衰减。

Checkpoint 标识为 **PPO_ATTENTION、版本 1**，保存四个网络模块及角度/规则 buffer、两个优化器、
配置、累计步数、完整回合数、更新次数与最近 100 回合结果。完整策略必须保留模糊模块及超网络。
不隐式迁移 PPO/KGPPO 模型；本目录原来复制的 `runs/train_*` 不删除，但不能作为本算法 checkpoint 加载。

CSV 保持原字段和按评估成功率降序、累计步数降序的排序：
`checkpoint,total_steps,episode,window_episodes,success_rate_100,collision_rate_100,timeout_rate_100,`
`eval_episodes,eval_success_rate,eval_collision_rate,eval_timeout_rate`。
保存成功先写待评估行，完整评估后补填；中断评估保留该行及模型。最终 last.pt 不重复记一行。

```bash
tensorboard --logdir runs
```

共五张图，前三张沿用 PPO，后两张在每次 PPO 更新结束记录：

| 图 | 曲线 | 横轴 |
|---|---|---|
| `episode/reward` | 完整训练回合回报 | 已完成 episode |
| `episode/outcome_rate_100` | 最近最多100回合成功/碰撞/超时率 | 已完成 episode |
| `eval/outcome_rate` | 完整评估的三率 | 累计训练 step |
| `guidance/weights` | linear、angular、floor、confidence | 累计训练 step |
| `optimization/loss` | ppo、imitation、total | 累计训练 step |

引导曲线是本次 rollout **采样时实际值**的均值；损失图的 imitation 已乘 imitation_coef，total 为两项之和。
所有比例为 0～1。不足100回合以实际完整回合数为分母。终端仍只输出完整评估三率汇总，不打印逐步或逐回合日志。

直接调用已保存的完整策略：

```python
import torch
from ppo_agent import PPOAgent

path = "runs/attention_<时间>/last.pt"
saved = torch.load(path, map_location="cuda", weights_only=False)
agent = PPOAgent(dict(saved["config"], device="cuda"))
agent.load(path)
action, _, _, _, diagnostics = agent.select_action(obs, saved["total_steps"], deterministic=True)
# obs: float32[364]；action: 实际 [v,w]；diagnostics: [alpha_v,alpha_w,C]
```

## 验证口径

已通过 11 项数值回归，覆盖规则/几何掩码、58 个参数、注意力偏置及停止梯度、融合分布采样概率一致、
下限跨节点保存、三模块 PPO 梯度、仅神经 Actor 的模仿梯度、GAE 和 checkpoint 完整恢复。
地图、奖励、Critic 与 GAE 对照 PPO 基线保持不变。配置在入口集中校验，传感器和 checkpoint 在边界检查。
不重复判空、不吞异常、不自动重试，不增加备用控制路径或新的显式线程。
短训练检查仅验收实现与运行流程，成功率是否提升需完成训练后与基线比较。

已在独立 ROS 11367 / Gazebo 11387 端口及私有图形显示中完成 CUDA 短训练（13 step）、
从中间 checkpoint 恢复至 17 step、手动确定性评估、A* 窗口刷新/关闭，以及评估中 Ctrl+C 清理。
采样/更新/保存返回时的暂停状态、实际训练预算、固定评估下限、缺失目标重建均已检查。
读取 event 文件核对了五张图、横轴和下限归零后仍可保留的融合权重；CSV 节点及待评估行核对通过。
这些回合上限缩短为 3 step 的测试不衡量导航性能；未运行完整 20 万步训练或宣称优于 PPO。
其他算法目录及复制过来的历史运行记录未改动。
