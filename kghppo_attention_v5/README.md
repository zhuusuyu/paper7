# Frontier/Dijkstra 监督 DDQN + 冻结 PPO-Attention

本目录的新高层只有地图 CNN、状态 MLP 和 Dueling 双头。Student 独立 ε-greedy 选子目标；训练期 Teacher 只提供动作标签，通过辅助损失帮助学习。Teacher 不接管动作、不融合 Q、不修改奖励。正常评估和 Gazebo 部署不创建 Teacher、不运行策略用 Frontier/Dijkstra。

修改前的完整六个 Python 文件、README 和 launch 在 `legacy_high/`；`legacy_high/runs -> ../runs` 复用原模型，不复制或删除原运行记录。新旧高层权重不兼容。

## 运行

使用安装了 NumPy、PyTorch、TensorBoard 的 Python 环境；窗口另需 Matplotlib/Tk。Gazebo 还需要现有 ROS1 依赖。

```bash
cd /home/hnu/turtlebot3_ws/src/paper_7/kghppo_attention_v5

# 默认：20 万训练交互步，每万步保存后评估 20 个固定验证任务
python main.py --device cuda --render on

# 相同输入、网络和更新预算，无知识监督消融
python main.py --device cuda --guidance off --render off

# 恢复新版训练；seed/scenario/timeout 需与原运行一致
python main.py --device cuda --checkpoint runs/frontier_ddqn_<时间>/last.pt

# 纯 Student 验证；trace 保存实际选点
python main.py --mode eval --device cuda --checkpoint runs/frontier_ddqn_<时间>/last.pt \
  --eval-episodes 20 --render on --trace on

# 独立 100 个固定 test 任务，五类各 20 个
python main.py --mode eval --device cuda --checkpoint runs/frontier_ddqn_<时间>/last.pt --eval-suite test

# 独立 Teacher 诊断，不用 Student 回退、不参与 Student 模型排名
python main.py --mode eval --device cuda --eval-policy teacher --eval-episodes 20 --trace on

# 原 V2 高层，原入口和旧权重通过备份独立运行
cd legacy_high
python main.py --mode eval --device cuda --checkpoint runs/<旧运行目录>/<旧模型>.pt
```

`eval.py` 仍是原低层 House 评估脚本，不是新版高层入口，内容未修改。新版高层评估使用 `main.py --mode eval`。

## 状态和动作

高层普通状态 **3610 维**，以两个数组传递；225 维安全 Mask 单独传递：

| 状态 | 形状 | 说明 |
|---|---|---|
| 在线观测图 | `int8[60,60]` | 未知 −1、自由 0、占用 1 |
| 机器人状态 | `float32[10]` | x、y、sin(yaw)、cos(yaw)、goal_x、goal_y、v、w、精确终点槽位/224、槽位有效标记 |
| 动作 Mask | `bool[225]` | 不计入普通状态 |

地图固定 0.2 m/格，覆盖 `[-6,6)×[-6,6)`；行是 y、列是 x，格心为 −5.9 到 5.9。网络前位置除以 6，速度除以冻结低层的最大速度，其余不变。无精确替换时最后两维均为零。

225 个动作维持原定义：15 个半径区间中心 `(i+0.5)×3.5/15`，15 个方向 `j×24°`，索引 `i×15+j`。普通候选要求半径严格大于 0.5 m。保持原 LiDAR **完整扫掠 0.15 m 安全圆盘**检查、传感器偏置和角度间隔余量。最终目标在原局部范围内且通过同一检查时替换最近槽位，并在状态中公开索引/有效性；Teacher、Replay 和执行都使用同一批候选。

高层删除四帧、候选特征、VisitMemory、模糊知识、注意力和超网络。低层的 364 维单帧观测、网络、模糊、超网络及确定性动作完整保留。

## 网络各层

| 分支 | 逐层形状 |
|---|---|
| 地图 | `B×1×60×60 → Conv(1,16,k3,s2,p1) → B×16×30×30 → Conv(16,32,k3,s2,p1) → B×32×15×15 → Conv(32,64,k3,s2,p1) → B×64×8×8` |
| 地图后半段 | `AdaptiveAvgPool(4×4) → B×1024 → FC128` |
| 状态 | `B×10 → FC32` |
| 拼接 | `B×160` |
| Value | `160 → 128 → 1` |
| Advantage | `160 → 128 → 225` |

各隐藏层使用 ReLU。`Q = V + A − mean(A)`，优势均值对全部 225 个动作计算；安全 Mask 用于选动作、下一动作和教师 CE。

## 在线地图和 Teacher 数据流

1. 每回合重置地图。只用 LiDAR、位姿更新；雷达位于基座后 0.032 m，自由证据 −0.4、命中 +0.85，累计限幅 ±4。同帧同格只更新一次，命中优先；距离达到 `3.5−0.05 m` 视为无命中。
2. 独立保存 seen。未 seen 才是未知；已 seen 且负证据为自由，证据为零或正值保守归占用。Teacher 只能获得公开三值快照、机器人状态、候选坐标和安全 Mask，不访问内部证据、真值几何、访问历史或隐藏参考点。
3. 精确最终目标候选有效，直接返回 `direct_goal`。
4. 否则在公开已知自由图中运行八邻接 Dijkstra。直行 0.2 m、对角 `0.2√2 m`，不走未知，不斜穿占用角。机器人所在格不可规划时弃权，不投影到其他格。
5. 最终目标格已知可达，参考该目标，来源 `known_goal_route`。否则检测所有已知可达 Frontier：已知自由格的四邻域至少一格未知；不限制其距离必须小于 3.5 m。
6. Frontier 按 `w_path*d_known + w_goal*Euclidean_to_goal − w_info*information_gain` 最小选取。信息收益为 1 m 半径内未知比例，分母只计窗口内格；默认权重 `(1,1,1)`。
7. 从参考点反向 Dijkstra，取得距离场及参考路径。候选须原 Mask 有效、已知自由、与参考点连通；要求已知路径进展大于 0.05 m，并按 `进展 − 偏离参考路径的距离` 选正分最高者，偏离用点到路径线段的真实米制距离，并列按动作索引。
8. 无可靠推荐返回标签 −1、valid=False 和原因：`robot_not_free`、`no_reachable_frontier`、`no_connected_safe_candidate` 或 `no_positive_progress`。不生成备用动作。

规划膨胀按**占用格心**间的欧氏距离 0.15 m 判断，不向上取整成额外 0.2 m 一圈。因此默认格距下不会额外封掉相邻格；这不是连续几何安全证明。Teacher 推荐最终仍受原连续扫掠 Mask 约束。Teacher 路线仅用于标签和训练显示，不输入 Student，也不分段直接交给低层。

参数集中在入口：`--frontier-weights w_path w_goal w_info`、`--teacher-deviation-weight 1`、`--teacher-min-progress 0.05`。

## 学习和执行

- Student 的实际动作始终独立 ε-greedy，随机时在有效动作中均匀采样。Replay 记录实际动作和实际执行结果；教师动作另存，不冒充真实动作。
- Huber TD 采用 Double DQN：在线网络选下一有效动作，目标网络估值；折扣 `gamma^(k/25)`，k 是本段实际底层步数。
- `L = L_TD + lambda(n)*L_K`。`L_K` 是有效动作上的温度 CE，只对有效教师标签取平均；默认对有效 Q 标准化，标准差停止梯度且最小为 1，温度为 1。无标签 batch 的知识损失为有限零。
- `lambda(n)=0.5*max(1−max(n−warmup,0)/2000,0)`，n 为已完成高层决策数。默认 warmup=256；第一次更新仍为 0.5。λ=0 后不再生成标签，旧标签也不产生监督梯度。每组四次更新共用当前 λ。
- 配置：`--lambda-initial 0.5 --lambda-decay-decisions 2000 --teacher-temperature 1`；关闭 Q 标准化用 `--teacher-q-normalization off`。
- 默认 Replay 50000、batch64、每条经验4次更新、每200组学习同步目标网络、梯度上限0.5。训练预算20万底层交互步，评估交互另计。
- 换点保留原到达、停滞、自适应时限、终点首次可见/接近事件；这些事件放在回合状态，不依赖 VisitMemory。
- 奖励保持 `100*success −100*collision −0.1*k +5*真值最短距离改善`。Sim2D 的真值距离场仅用于奖励/SPL，Teacher/Student 不访问它。Gazebo 不构建该真值距离场。
- 真正无安全动作时结束为 `no_safe_action`，额外罚 −100，不 bootstrap；成功/物理碰撞优先。此结果独立统计，不冒充超时或碰撞。reset 即无安全动作明确报初始化错误。
- 成功/碰撞/受阻不 bootstrap；超时及保存节点截断保留真实下一状态并 bootstrap。保存节点立即切段、保存后评估，再 reset 新训练回合；未完成训练回合不计入窗口。

## 模型、记录和可视化

新版模型标识 `FRONTIER_DDQN` version1，保存状态/地图规格、在线/目标网络、优化器、有效 Replay、训练计数、随机数状态、固定低层完整 checkpoint 及来源/hash。旧高层拒绝隐式迁移，低层加载接口不变。

Replay 两端地图用 uint8 存 `0/1/2`，采样时转回 `−1/0/1` 再转 float。机器人状态 float32，Mask bool，动作/标签 int16；不存路径或距离场。容量50000时两端地图约360 MB（十进制），全部 Replay 数组约388 MB；保存仅写有效样本。

训练目录 `runs/frontier_ddqn_<UTC>/`：

- 每万步 `step_*.pt`，正常结束再写 `last.pt`；周期与最终重合时不重复评估。
- `checkpoint_metrics.csv`：训练四种结果窗口、λ、更新数、Student验证成功率/SPL/路径等。按成功率、SPL、步数降序排列。
- `eval_tasks_step_*.csv`：固定验证任务明细，validation seeds从5000000开始；test从7000000开始，默认100个。Teacher诊断单独保存 `teacher_diagnostic_*`，不进入模型排名。
- TensorBoard：训练回报/四种结果率、Student评估统计、TD/CE/加权CE/λ、Teacher来源/弃权/有效率/Student一致率及更新计数。
- `--trace on` 写逐段实际动作，训练额外写 Teacher 标签、来源和弃权原因。`config.json`、`progress.json`、`timing.json` 保存配置及计数。
- `--render on`：x向上、y向左；绿星最终目标、紫三角实际子目标、橙线轨迹、灰格已知自由、黑格观测占用；训练中另显示青色 Teacher Frontier/参考路径、粉方框 Teacher推荐。普通评估这些教师图元为空。浅色墙体只是可视化真值背景，不输入策略。关窗继续训练。

## Gazebo

ROS通信和 `eval.py` 未修改。先手动启动对应场景及机器人，再运行高层：

```bash
python main.py --mode eval --backend gazebo --device cuda \
  --checkpoint runs/frontier_ddqn_<时间>/last.pt \
  --start -3 1 --yaw 180 --goal -5 -2 --eval-episodes 1
```

起终点、机器人、传感器及实际雷达观测必须在 `[-6,6)²` 内，超出立即明确报错，不静默裁剪。因此此前终点x=−6.5的House任务不兼容本固定地图。Gazebo动作时长/暂停流程和低层归一化保持原样。

## 基本验证和限制

```bash
python tests/test_frontier_ddqn.py
python tests/check_mapping.py
```

具体已运行检查、CPU smoke结果及窄通道量化测量见 `tests/validation_report.md`；原始量化结果见 `tests/mapping_results.json`。这些检查只证明实现能运行及部分数值性质，不是导航性能实验。未自动启动完整训练，成功率需你运行后评估。
