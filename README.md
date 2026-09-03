# AlphaZero 五子棋 / Pente / 连珠

一个基于 **AlphaZero** 算法的棋类强化学习项目，统一支持三种规则：**五子棋（Gomoku）**、**Pente** 和 **连珠（Renju，塔拉山口-10）**。核心是「自对弈 → 训练 → 评估 → 晋级」的闭环，网络用 PyTorch，搜索用蒙特卡洛树搜索（MCTS），并额外提供监督预训练、ONNX 导出和命令行/GUI 对弈入口。

## 功能特性

- **AlphaZero 自对弈强化学习**：纯自我对弈生成训练数据，无需人工棋谱即可从零学习（五子棋/Pente）。
- **ResNet 策略-价值网络**：共享主干 + 策略头 + 价值头，可配置残差块数与通道数。
- **批量 MCTS**：PUCT 动作选择、Dirichlet 噪声探索、批量神经网络推理、8 种对称增强。
- **三种游戏规则**：五子棋、Pente、连珠（塔拉山口-10，含禁手与开局状态机）。
- **多进程并行**：自对弈与评估均支持多进程，加速数据生成与候选模型验证。
- **经验回放持久化**：回放缓冲区与模型快照落盘，支持断点续训。
- **监督预训练**：支持用 KataGo 蒸馏数据（renju15x）做行为克隆预热。
- **ONNX 导出**：把训练好的连珠模型导出为 ONNX，供 Electron/onnxruntime 前端推理。
- **对弈入口**：终端 `play.py` / `play_loop.py`，以及 PyGame GUI（`interface.py` / `gui.py`）。
- **GPU 加速**：训练与推理均可使用 CUDA（多进程时建议 CPU，避免 CUDA 争用）。

## 游戏规则

### 五子棋 (Gomoku)

- **目标**：率先在横、竖或斜线连成 5 子。
- **棋盘**：15×15，2 名玩家，黑子先行。
- **动作空间**：225（15×15 个落点）。

### Pente

- **目标**：连成 5 子 **或** 吃掉对手 10 颗棋子（5 对）。
- **吃子**：用 `己方 - 对手 - 对手 - 己方` 的排列包围并移除中间两颗对手棋子，每对记 1 次捕获。
- **棋盘**：15×15。
- **动作空间**：225。

### 连珠 (Renju，塔拉山口-10)

- **棋盘**：15×15，黑子先行。
- **开局**：前 5 手按塔拉山口-10 规则推进（天元 S1 → 交换 → 3×3 → 交换 → 5×5 → 交换 → 7×7 → 走法选择；走法一为 9×9 内第 5 手，走法二为十打点/十选一）。交换只改变执黑方，不改变落子颜色序列。
- **禁手**：前 5 手无禁手；第 6 手起对黑方启用 **三三、四四、长连（≥6）** 禁手，白方无禁手。
- **胜负**：黑恰好五连判胜（五连优先，即使同时构成禁手）；黑长连判负；白 ≥5 连胜。
- **动作空间**：229（225 个落点 + 4 个开局决策动作：交换接受/拒绝、走法一/走法二）。

规则由 `games/__init__.py::get_game_class` 统一注册，三种游戏共享同一套 MCTS / 训练 / 对弈接口。

## 目录结构

```
AlphaZero-Gomoku/
├── games/                # 游戏规则与状态编码
│   ├── gomoku.py
│   ├── pente.py
│   └── renju/            # 连珠：game.py / rule.py / opening_greedy.py
├── mcts/
│   ├── new_mcts_alpha.py # 神经网络引导的批量 MCTS
│   ├── mcts_pure.py      # 纯 MCTS 基线
│   └── old_mcts_alpha.py # 旧实现
├── players/              # 对弈选手封装（human / mcts / alpha / alpha2）
├── network.py            # AlphaZeroNet + PyTorchModel 封装
├── train.py              # 自对弈强化学习训练入口
├── train_supervised.py   # KataGo 蒸馏数据监督预训练
├── train_renju.sh        # 连珠训练编排脚本（监督预热 + 自对弈微调 + 断点续训）
├── export_onnx.py        # 连珠模型导出 ONNX
├── data/                 # KataGo 数据加载器
├── play.py               # 终端单局对弈
├── play_loop.py          # 终端多局对战 + 指标输出
├── interface.py          # PyGame GUI（菜单式）
├── gui.py                # PyGame 简单对弈
├── tests/                # 连珠规则 / 对局状态机测试
├── models/               # 五子棋/Pente 模型与回放缓冲区
└── models_renju/         # 连珠模型与回放缓冲区
```

## 安装

```bash
pip install -r requirements.txt
```

主要依赖：`torch`、`numpy`、`pygame`。若需监督预训练，还需准备 KataGo renju15x 蒸馏数据（见下文）。

## 快速开始

### 与 AI 对战

```bash
# 五子棋 / Pente / 连珠（第 3 个参数可选，默认 gomoku）
python play.py player_human player_alpha gomoku
python play.py player_human player_alpha renju
```

AI 选手在 `players/player_alpha.py` 中定义。未显式传入 `model_path` 时，会按规则自动选择模型目录（renju → `models_renju/`，其他 → `models/`）并加载最新的兼容快照（按文件名时间戳取最新，自动跳过棋盘尺寸 / 动作空间不匹配的模型；无兼容快照时回退到训练遗留的候选 checkpoint）。

### GUI 对弈

```bash
# 菜单式 GUI（游戏/选手选择）
python interface.py

# 简单 GUI：python gui.py <player1> <player2>
python gui.py player_human player_alpha
```

### 模型评估

让两个选手连续对战并输出指标（`metrics/` 下 JSON）：

```bash
# AI 模型互相对战
python play_loop.py player_alpha player_alpha2 50

# 与纯 MCTS 基线对战（第 4 个参数可选规则，默认 gomoku）
python play_loop.py player_alpha player_mcts 50 renju
```

### 训练

```bash
# 五子棋（默认）
python train.py --game gomoku

# Pente
python train.py --game pente

# 连珠（第 6 手起启用黑方禁手）
python train.py --game renju --model-dir models_renju
```

常用参数（`python train.py --help` 查看全部）：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--game` | `gomoku` | `gomoku` / `pente` / `renju` |
| `--num-iterations` | `5` | 训练迭代轮数 |
| `--games-per-iteration` | `8` | 每轮自对弈局数 |
| `--num-simulations` | `50` | 自对弈时 MCTS 每步模拟次数 |
| `--cpuct` | `1.2` | PUCT 探索/利用平衡因子 |
| `--batch-size` | `128` | 训练批次大小 |
| `--epochs` | `2` | 每轮训练 epoch 数 |
| `--eval-games` | `12` | 评估对局数 |
| `--eval-simulations` | `200` | 评估时 MCTS 每步模拟次数 |
| `--win-rate-threshold` | `0.55` | 候选模型接受阈值 |
| `--always-accept` | 关闭 | 跳过评估、每轮直接接受候选模型 |
| `--selfplay-workers` | `0` | 自对弈进程数（0=自动） |
| `--eval-workers` | `0` | 评估进程数（0=自动） |
| `--model-dir` | `models` | 模型/回放缓冲区目录 |
| `--pretrained-model` | 无 | 预训练模型路径（续训/warm-start） |
| `--resume-iteration` | `1` | 续训起始迭代编号 |

### 连珠训练脚本（推荐）

`train_renju.sh` 封装了监督预热 + 自对弈微调 + 断点续训：

```bash
bash train_renju.sh            # 纯自对弈（自动续训）
bash train_renju.sh sl         # 监督预热 → 自对弈微调
bash train_renju.sh sl-only    # 只跑监督预热
bash train_renju.sh fresh      # 从零开始纯自对弈
```

脚本会自动识别 `models_renju/` 下最新的 `snapshot_iter*.pt` 或 `sl_pretrain_*.pt` 快照，并从正确位置继续训练。

### 监督预训练

用 KataGo renju15x 蒸馏数据做行为克隆，预热策略与价值网络：

```bash
python train_supervised.py \
  --data-root /path/to/katago-gomoku-distill-2025.5/renju15x_label28b \
  --max-samples 2000000 --epochs 1 --batch-size 256 \
  --model-out models_renju
```

产出 `sl_pretrain_*.pt` 后，可接 `python train.py --game renju --pretrained-model <快照>` 继续自对弈微调。

### ONNX 导出

```bash
python export_onnx.py \
  --checkpoint models_renju/snapshot_iter31_xxx.pt \
  --out models_renju/model.onnx
```

导出输入为 `(1, 4, 15, 15)`（我的棋子/对手棋子/禁手生效/阶段），输出为 `policy (1, 229)` 与 `value (1, 1)`。

### 运行测试

```bash
pytest tests/ -v
# 或
python -m unittest tests.test_renju_rule tests.test_renju_game -v
```

## 工作原理

### 1. 状态编码

**五子棋 / Pente（3 通道）：**
- 通道 0：当前玩家棋子位置
- 通道 1：对手棋子位置
- 通道 2：上下文/回合平面

**连珠（4 通道）：**
- 通道 0：我的棋子位置（按实际执子色编码，交换会改变执黑方）
- 通道 1：对手棋子位置
- 通道 2：禁手生效（当前轮到黑方且已进入第 6 手时为 1.0）
- 通道 3：开局阶段指示（0=落子/中盘，0.5=交换，1=走法，0.75=十打点/选点）

### 2. 神经网络

`AlphaZeroNet` 采用共享残差主干 + 双头的结构：

- 初始卷积层（3×3）→ BatchNorm → ReLU
- N 个残差块（训练脚本默认 3 个残差块、64 通道，可配置）
- **策略头**：1×1 卷积 → 展平 → 全连接，输出 `action_size` 个 logits（五子棋/Pente=225，连珠=229）
- **价值头**：1×1 卷积 → 展平 → MLP → 标量，`tanh` 压缩到 [-1, 1]

**训练细节**（`PyTorchModel` 默认）：优化器 Adam（lr=1e-3，weight_decay=1e-4）；损失 = 价值 MSE + 策略 KL 散度；梯度裁剪范数 3.0。

### 3. MCTS

- 使用 PUCT 公式进行动作选择。
- 训练时前 N 手添加 Dirichlet 噪声（默认 alpha=0.03、epsilon=0.25）增强探索。
- 批量神经网络推理叶子节点。
- 状态键控的树（字典存储 P/V/N/W/children）。
- 8 种对称变换（4 旋转 × 2 翻转）用于数据增强。
- 连珠的十打点/选点（走法二）由策略/价值贪心解析，不进 MCTS。

### 4. 训练流程

每次迭代：

1. **自对弈**：用候选模型 + MCTS 生成对局，记录（状态，策略，胜负）。
2. **存储**：样本（含对称增强）写入经验回放缓冲区，并持久化到 `replay_buffer_latest.pkl`。
3. **训练**：从缓冲区采样小批次，更新策略与价值网络。
4. **评估**：候选模型与当前最佳模型对战（轮流先手）。
5. **接受**：胜率 ≥ 阈值（默认 55%）则晋级为最佳模型，否则从最佳模型恢复。

模型快照按 `snapshot_iter{N}_{时间戳}.pt` 保存，缓冲区每次迭代覆盖保存，二者共同支持断点续训。

## 参考

- [AlphaGo Zero: Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270)
- [AlphaZero: A general reinforcement learning algorithm](https://arxiv.org/abs/1712.01815)
- [alpha-zero-general](https://github.com/suragnair/alpha-zero-general)

更详细的训练参数调优说明见 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)。
