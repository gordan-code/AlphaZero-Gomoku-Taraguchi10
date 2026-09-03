#!/usr/bin/env bash
# ============================================================================
# RenjuMaster / AlphaZero-Gomoku —— 连珠（Taraguchi-10）训练脚本
#
# 用法：
#   bash train_renju.sh               # 纯自对弈（自动续训：检测到快照就继续）
#   bash train_renju.sh sl            # 一键：监督预热 → 自动读最新快照 → 自对弈微调
#   bash train_renju.sh sl-only       # 只跑监督预热（产出 sl_pretrain_*.pt 后退出）
#   bash train_renju.sh fresh         # 纯自对弈从零（忽略历史快照）
#   bash train_renju.sh fresh sl      # 从零监督预热 + 自对弈微调
#   bash train_renju.sh sl bg         # 后台：监督预热（前台）+ 自对弈（后台写日志）
#
# 断点续训说明：
#   - 回放缓冲区（replay_buffer_latest.pkl）总是自动加载。
#   - 模型快照会自动识别两类并取最新：
#       snapshot_iter*.pt   自对弈快照   → 从 iter+1 继续
#       sl_pretrain_*.pt    监督预热快照 → 自对弈从第 1 轮 warm-start
#   - 监督预热若检测到已有快照，会 --pretrained 从它继续（warm-start），不丢历史进度。
#   - 想彻底重来就用 `fresh`。
#
# 首次在 AutoDL 上运行前，先装依赖（torch 用镜像自带的即可）：
#   pip install numpy==2.4.0 filelock fsspec Jinja2 MarkupSafe mpmath networkx sympy setuptools typing_extensions
# ============================================================================

cd "$(dirname "$0")"

# -------------------- 可调参数 --------------------
GAME=renju
ITERATIONS=68            # 训练迭代轮数（续训时表示"从当前再训这么多轮"）
GAMES_PER_ITER=20         # 每轮自对弈局数（越多越稳）
SIMULATIONS=200           # 自对弈 MCTS 每步模拟次数（越大越强、越慢）
CPUCT=1.2                 # MCTS 探索系数
BATCH_SIZE=128            # 训练批次大小
EPOCHS=2                  # 每轮训练 epoch 数
EVAL_GAMES=12             # 每轮评估对局数
EVAL_SIMS=200             # 评估时 MCTS 模拟次数
EVAL_MAX_MOVES=160        # 评估对局手数上限（超限判和，防马拉松局）
ALWAYS_ACCEPT=1           # 1=跳过评估、每轮直接接受候选（greedy 更新）；0=保留评估门槛
LOG_EVAL_GAMES=2          # always-accept 下的廉价评估局数（0=关闭，仅记日志不决策）
LOG_EVAL_SIMS=60          # 廉价评估每步模拟次数
SELPLAY_WORKERS=8         # 自对弈并行进程数（0=自动，按 CPU 核数调）
EVAL_WORKERS=8            # 评估并行进程数（0=自动）
MODEL_DIR=models_renju    # 模型 / 回放缓冲保存目录

# -------------------- 监督预热（KataGo renju15x 蒸馏数据） --------------------
# AutoDL 上请把 SL_DATA_ROOT 改成你上传后的实际路径。
SL_DATA_ROOT="/root/autodl-tmp/katago-gomoku-distill-2025.5/renju15x_label28b"
SL_MAX_SAMPLES=2000000    # 监督预热用多少局面（200 万起步；全量约 6480 万）
SL_BATCH_SIZE=512         # 监督批次大小
SL_EPOCHS=1               # 监督 epoch 数
SL_CHANNELS=64            # 与自对弈网络保持一致（train.py 默认 64/3 残差块）
SL_RES_BLOCKS=3
SL_LR=1e-3

# 取最新快照：按文件名里的时间戳（YYYYMMDD_HHMMSS）排序，不依赖文件系统 mtime。
# （文件上传/同步后 mtime 可能错乱，文件名时间戳才是可靠顺序。）
latest_snapshot() {
    ls "$1"/snapshot_iter*.pt "$1"/sl_pretrain_*.pt 2>/dev/null \
      | sed -E 's/.*_([0-9]{8}_[0-9]{6})\.pt$/\1 &/' \
      | sort -r \
      | head -1 \
      | cut -d' ' -f2-
}

# -------------------- 参数解析 --------------------
FRESH=0
MODE=""
RUN_SL=0
SL_ONLY=0
for arg in "$@"; do
    case "$arg" in
        fresh)    FRESH=1 ;;
        bg)       MODE="bg" ;;
        sl)       RUN_SL=1 ;;
        sl-only)  RUN_SL=1; SL_ONLY=1 ;;
    esac
done

# ============================================================================
# 阶段 1/2：监督预热（可选）
# ============================================================================
if [ "$RUN_SL" -eq 1 ]; then
    echo ""
    echo "=============================================================="
    echo " 阶段 1/2：KataGo renju15x 监督预热（行为克隆，学中盘禁手）"
    echo "=============================================================="

    # warm-start：若已有快照则从最新继续监督训练，不丢历史进度
    SL_PRETRAIN_FLAG=""
    if [ "$FRESH" -eq 0 ]; then
        EXISTING=$(latest_snapshot "${MODEL_DIR}")
        if [ -n "$EXISTING" ]; then
            SL_PRETRAIN_FLAG="--pretrained ${EXISTING}"
            echo "🧬 监督预热 warm-start from: $(basename "$EXISTING")（3 通道会自动迁移为 4 通道）"
        fi
    fi

    python train_supervised.py \
      --data-root "${SL_DATA_ROOT}" \
      --max-samples ${SL_MAX_SAMPLES} \
      --batch-size ${SL_BATCH_SIZE} \
      --epochs ${SL_EPOCHS} \
      --n-res-blocks ${SL_RES_BLOCKS} \
      --channels ${SL_CHANNELS} \
      --lr ${SL_LR} \
      --model-out ${MODEL_DIR} \
      ${SL_PRETRAIN_FLAG} \
      || { echo "❌ 监督预热失败，退出。"; exit 1; }

    if [ "$SL_ONLY" -eq 1 ]; then
        echo "✅ sl-only：只跑监督预热，不进入自对弈。"
        exit 0
    fi
fi

# ============================================================================
# 阶段 2/2：自对弈微调（断点续训检测）
# ============================================================================
RESUME_FLAGS=""
if [ "$FRESH" -eq 1 ] && [ "$RUN_SL" -eq 0 ]; then
    echo "🆕 已指定 fresh，纯自对弈从零开始训练。"
else
    # 取最新快照（自对弈 snapshot_iter* 或监督预热 sl_pretrain*）
    LATEST=$(latest_snapshot "${MODEL_DIR}")
    if [ -n "$LATEST" ]; then
        BN=$(basename "$LATEST")
        case "$BN" in
            snapshot_iter*.pt)
                ITER_NUM=$(echo "$BN" | sed -E 's/snapshot_iter([0-9]+)_.*/\1/')
                NEXT_ITER=$((ITER_NUM + 1))
                RESUME_FLAGS="--pretrained-model ${LATEST} --resume-iteration ${NEXT_ITER}"
                echo "🔁 检测到自对弈快照，续训：${BN}（从第 ${NEXT_ITER} 轮继续）"
                ;;
            sl_pretrain_*.pt)
                RESUME_FLAGS="--pretrained-model ${LATEST} --resume-iteration 1"
                echo "🔁 检测到监督预热快照，自对弈 warm-start：${BN}（从第 1 轮开始）"
                ;;
        esac
    else
        echo "🆕 未检测到快照，自对弈从零开始训练。"
    fi
fi

# -------------------- 组装命令 --------------------
ACCEPT_FLAGS=""
if [ "$ALWAYS_ACCEPT" = "1" ]; then
    ACCEPT_FLAGS="--always-accept"
    echo "⚡ 已启用 --always-accept：跳过正式评估，每轮直接接受候选模型。"
    if [ "$LOG_EVAL_GAMES" -gt 0 ]; then
        ACCEPT_FLAGS="$ACCEPT_FLAGS --log-eval-games ${LOG_EVAL_GAMES} --log-eval-sims ${LOG_EVAL_SIMS}"
        echo "📊 附带廉价评估（${LOG_EVAL_GAMES} 局 / ${LOG_EVAL_SIMS} 模拟），仅记日志不决策。"
    fi
fi

CMD="python train.py \
  --game ${GAME} \
  --num-iterations ${ITERATIONS} \
  --games-per-iteration ${GAMES_PER_ITER} \
  --num-simulations ${SIMULATIONS} \
  --cpuct ${CPUCT} \
  --batch-size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --eval-games ${EVAL_GAMES} \
  --eval-simulations ${EVAL_SIMS} \
  --eval-max-moves ${EVAL_MAX_MOVES} \
  ${ACCEPT_FLAGS} \
  --selfplay-workers ${SELPLAY_WORKERS} \
  --eval-workers ${EVAL_WORKERS} \
  ${RESUME_FLAGS} \
  --model-dir ${MODEL_DIR}"

# -------------------- 运行 --------------------
if [ "$MODE" = "bg" ]; then
    nohup ${CMD} > train_renju.log 2>&1 &
    echo "✅ 已后台启动自对弈，PID=$!"
    echo "   查看日志：tail -f train_renju.log"
else
    exec ${CMD}
fi
