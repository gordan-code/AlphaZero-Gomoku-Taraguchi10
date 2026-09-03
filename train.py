import argparse
import os
import time
import random
from collections import deque
from typing import List, Tuple, Optional
import numpy as np
from network import PyTorchModel
from mcts.new_mcts_alpha import MCTS
from games import get_game_class
from datetime import datetime
from copy import deepcopy
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle


# -------------------------
#  多进程自对弈 worker
# -------------------------
_SELFPLAY_MODEL = None
_SELFPLAY_MODEL_META = None  # (model_path, board_size, action_size, device)

# 评估 worker 模型缓存（每个子进程只加载一次）
_EVAL_MODEL_NEW = None
_EVAL_MODEL_BEST = None
_EVAL_MODEL_META = None  # (new_path, best_path, board_size, action_size, device)


def _resolve_greedy_opening(game, model):
    """连珠走法二的十打点/选点用策略贪心；五子棋无此阶段，直接跳过。"""
    if hasattr(game, 'phase'):
        from games.renju.opening_greedy import resolve_greedy_opening_steps
        resolve_greedy_opening_steps(game, model)


def _selfplay_worker_init(
    model_path: str,
    board_size: int,
    action_size: int,
    device: str,
    base_seed: int = 0,
    torch_num_threads: int = 1,
):
    """
    Windows 下使用 spawn：每个子进程会重新 import 本文件。
    initializer 用于设定随机种子与 PyTorch 线程数，避免 CPU 过度抢占。
    """
    try:
        import torch
        torch.set_num_threads(max(1, int(torch_num_threads)))
    except Exception:
        pass

    pid = os.getpid()
    seed = int(base_seed) + pid
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    # 每个子进程只加载一次模型（避免每个任务重复 load）
    global _SELFPLAY_MODEL, _SELFPLAY_MODEL_META
    _SELFPLAY_MODEL_META = (str(model_path), int(board_size), int(action_size), str(device))

    from network import PyTorchModel  # 延迟 import
    _SELFPLAY_MODEL = PyTorchModel(board_size=int(board_size), action_size=int(action_size), device=str(device))
    _SELFPLAY_MODEL.load(str(model_path), map_location=str(device))


def _selfplay_generate_games(
    *,
    game_name: str = "gomoku",
    board_size: int,
    n_simulations: int,
    cpuct: float,
    temp_threshold: int,
    add_dirichlet_noise: bool,
    games_to_play: int,
    use_symmetries: bool,
    max_moves: int,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    dirichlet_n_moves: int,
    device: str = "cpu",
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], dict]:
    """
    子进程入口：加载模型 -> 跑若干局自对弈 -> 返回样本与胜负统计。
    注意：为了避免 GPU 多进程争用，默认 device=cpu。
    """
    from mcts.new_mcts_alpha import MCTS
    from games import get_game_class
    GameClass = get_game_class(game_name)

    # 多进程路径：优先复用 initializer 加载的全局模型
    global _SELFPLAY_MODEL, _SELFPLAY_MODEL_META
    model = _SELFPLAY_MODEL
    if model is None:
        # 兼容：如果没有走 initializer（例如单进程/直接调用），再本地创建模型
        from network import PyTorchModel
        model = PyTorchModel(board_size=board_size, action_size=board_size * board_size, device=device)

    def temp_fn(move_number: int):
        return max(0.0, 1.0 - move_number / temp_threshold)

    winners = {0: 0, 1: 0, 2: 0}
    all_examples: List[Tuple[np.ndarray, np.ndarray, float]] = []

    for _ in range(int(games_to_play)):
        mcts_play = MCTS(
            game_class=GameClass,
            n_simulations=n_simulations,
            nn_model=model,
            cpuct=cpuct,
            dirichlet_alpha=dirichlet_alpha,
            epsilon=dirichlet_epsilon,
            apply_dirichlet_n_first_moves=dirichlet_n_moves,
            add_dirichlet_noise=add_dirichlet_noise,
        )
        game = GameClass(size=board_size)
        game.current_player = 1
        # 连珠：开局（含交换/走法/十打点）由 MCTS + 策略贪心从初始状态直接搜索

        examples, winner = play_game_and_collect(
            mcts_play,
            game,
            temp_fn,
            max_moves=max_moves,
            use_symmetries=use_symmetries,
        )
        all_examples.extend(examples)
        winners[winner] = winners.get(winner, 0) + 1

        mcts_play.clear_tree()
        del mcts_play

    # 显式释放
    # 如果是全局复用模型，不在这里释放；让子进程退出时统一回收
    gc.collect()

    return all_examples, winners

# -------------------------
#  多进程评估 worker
# -------------------------
def _eval_worker_init(
    model_new_path: str,
    model_best_path: str,
    board_size: int,
    action_size: int,
    device: str,
    base_seed: int = 0,
    torch_num_threads: int = 1,
):
    try:
        import torch
        torch.set_num_threads(max(1, int(torch_num_threads)))
    except Exception:
        pass

    pid = os.getpid()
    seed = int(base_seed) + pid
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    global _EVAL_MODEL_NEW, _EVAL_MODEL_BEST, _EVAL_MODEL_META
    _EVAL_MODEL_META = (str(model_new_path), str(model_best_path), int(board_size), int(action_size), str(device))

    from network import PyTorchModel  # 延迟 import
    _EVAL_MODEL_NEW = PyTorchModel(board_size=int(board_size), action_size=int(action_size), device=str(device))
    _EVAL_MODEL_NEW.load(str(model_new_path), map_location=str(device))

    _EVAL_MODEL_BEST = PyTorchModel(board_size=int(board_size), action_size=int(action_size), device=str(device))
    _EVAL_MODEL_BEST.load(str(model_best_path), map_location=str(device))


def _eval_play_games(
    *,
    game_name: str = "gomoku",
    board_size: int,
    n_games: int,
    start_index: int,
    n_simulations: int,
    cpuct: float,
    max_moves: int,
) -> Tuple[int, int, int]:
    """
    子进程评估：跑 n_games 局，使用 start_index 来决定交替先手。
    返回 (new_wins, draws, total_games)
    """
    from mcts.new_mcts_alpha import MCTS
    from games import get_game_class
    GameClass = get_game_class(game_name)

    global _EVAL_MODEL_NEW, _EVAL_MODEL_BEST
    model_new = _EVAL_MODEL_NEW
    model_best = _EVAL_MODEL_BEST

    new_wins = 0
    draws = 0

    for gi in range(int(n_games)):
        global_i = int(start_index) + gi

        game = GameClass(size=int(board_size))
        if not hasattr(game, "phase"):
            # 五子棋/连珠以外：随机第一手增加开局多样性
            center = int(board_size) // 2
            radius = 4  # 9×9区域 (81种可能 × 2先后手 = 162种组合)
            r1 = random.randint(center - radius, center + radius)
            c1 = random.randint(center - radius, center + radius)
            game.do_move((r1, c1))
            # 现在 current_player = 2，从第二手开始真正评估
        # 连珠（renju）：从 S1_MOVE 直接开始，开局由 MCTS 搜索

        new_starts = (global_i % 2 == 0)
        move_number = 1

        mcts_new = MCTS(
            game_class=GameClass,
            n_simulations=n_simulations,
            nn_model=model_new,
            cpuct=cpuct,
            add_dirichlet_noise=False,
        )
        mcts_best = MCTS(
            game_class=GameClass,
            n_simulations=n_simulations,
            nn_model=model_best,
            cpuct=cpuct,
            add_dirichlet_noise=False,
        )

        while not game.is_game_over():
            _resolve_greedy_opening(game, mcts_new.nn_model)
            if game.is_game_over():
                break
            if (game.current_player == 1 and new_starts) or (game.current_player == 2 and not new_starts):
                pi = mcts_new.run(game, len(game.move_history))
            else:
                pi = mcts_best.run(game, len(game.move_history))

            action = int(np.argmax(pi))
            game.do_move(action)
            move_number += 1
            if move_number > max_moves:
                break

        winner = game.get_winner()
        if winner == 0:
            draws += 1
        else:
            if (winner == 1 and new_starts) or (winner == 2 and not new_starts):
                new_wins += 1

        mcts_new.clear_tree()
        mcts_best.clear_tree()
        del mcts_new
        del mcts_best

    gc.collect()
    return int(new_wins), int(draws), int(n_games)

# 在循环内动态映射游戏名称到类

# -------------------------
#  工具函数
# -------------------------
def softmax_temperature(pi: np.ndarray, temp: float) -> np.ndarray:
    if temp <= 0:
        return pi
    logits = np.log(pi + 1e-15)
    logits = logits / temp
    exps = np.exp(logits - np.max(logits))
    p = exps / np.sum(exps)
    return p


def sample_action_from_pi(pi: np.ndarray, temp: float) -> int:
    if temp == 0:
        return int(np.argmax(pi))
    p = softmax_temperature(pi, temp)
    return int(np.random.choice(len(p), p=p))


# -------------------------
#  经验回放缓冲区
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity: int = 20000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, examples: List[Tuple[np.ndarray, np.ndarray, float]]):
        """
        添加示例列表 (state_enc, pi, z)
        state_enc: (C,H,W)
        pi: (action_size,)
        z: 标量 (-1,0,1)
        """
        for ex in examples:
            self.buffer.append(ex)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, k=batch_size)
        states, pis, zs = zip(*batch)
        states = np.stack(states, axis=0).astype(np.float32)
        pis = np.stack(pis, axis=0).astype(np.float32)
        zs = np.array(zs, dtype=np.float32).reshape(-1, 1)
        return states, pis, zs

    def __len__(self):
        return len(self.buffer)


# -------------------------
#  Buffer 持久化
# -------------------------
def save_replay_buffer(buffer: ReplayBuffer, filepath: str):
    """
    保存 ReplayBuffer 到磁盘
    只保存 buffer 内容，不保存 capacity（在加载时重新设定）
    """
    try:
        # 将 deque 转为 list 以便 pickle
        buffer_data = {
            'buffer': list(buffer.buffer),
            'capacity': buffer.capacity
        }
        with open(filepath, 'wb') as f:
            pickle.dump(buffer_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Buffer] 已保存到: {filepath} (大小: {len(buffer)} 样本)")
        return True
    except Exception as e:
        print(f"[Buffer] 保存失败: {e}")
        return False


def load_replay_buffer(filepath: str, capacity: int) -> Optional[ReplayBuffer]:
    """
    从磁盘加载 ReplayBuffer
    如果文件不存在或加载失败，返回 None
    """
    if not os.path.exists(filepath):
        print(f"[Buffer] 未找到已保存的 buffer: {filepath}")
        return None
    
    try:
        with open(filepath, 'rb') as f:
            buffer_data = pickle.load(f)
        
        # 创建新的 ReplayBuffer
        buffer = ReplayBuffer(capacity=capacity)
        
        # 恢复数据
        saved_buffer = buffer_data['buffer']
        saved_capacity = buffer_data.get('capacity', capacity)
        
        # 如果保存的容量与当前配置不同，给出警告
        if saved_capacity != capacity:
            print(f"[Buffer] 警告: 保存的容量 ({saved_capacity}) 与当前配置 ({capacity}) 不同")
        
        # 将数据添加回 buffer（deque 会自动处理 maxlen）
        for item in saved_buffer:
            buffer.buffer.append(item)
        
        print(f"[Buffer] 已加载: {filepath} (大小: {len(buffer)} 样本)")
        return buffer
    except Exception as e:
        print(f"[Buffer] 加载失败: {e}")
        return None


# -------------------------
#  自对弈单局游戏
# -------------------------
def play_game_and_collect(mcts: MCTS, game, temp_fn, max_moves=225, use_symmetries=True):
    """
    进行一局完整的游戏并返回增强后的示例：
    final_examples: (state_enc (C,H,W), pi (A,), z 标量) 的列表
    winner: 0/1/2
    """
    examples = []
    move_number = 0

    while True:
        # 走法二的十打点/选点用策略贪心（不进 MCTS，不存样本）
        _resolve_greedy_opening(game, mcts.nn_model)
        if game.is_game_over():
            break

        state_enc = game.get_encoded_state()  # 期望是视角不变的
        pi = mcts.run(game, len(game.move_history))  # 向量 (action_size,) 第二个参数是当前是第几步
        # 这个参数是让MCtS 知道当前是第几步,是不是要加入dirichlet noise，用来增强MCTS的探索能力
        # 返回一个向量 (action_size,) 每个元素是每个动作的概率
        pi_for_store = pi.copy()

        temp = temp_fn(move_number)
        action = sample_action_from_pi(pi, temp)

        # 安全回退：如果选择的动作不合法，使用 argmax
        valid_mask = game.get_valid_moves()
        if valid_mask[action] != 1.0:
            action = int(np.argmax(pi))
        # 存储 (state, pi, player)
        examples.append((state_enc, pi_for_store, int(game.current_player)))

        # 执行动作（含交换/走法决策动作）
        game.do_move(action)

        move_number += 1

        if game.is_game_over() or move_number >= max_moves:
            break

    winner = game.get_winner()  # 0/1/2

    # 将示例转换为 (state_aug, pi_aug, z)
    final_examples = []
    for state_enc, pi_vec, player in examples:
        if winner == 0:
            z = 0.0
        else:
            z = 1.0 if winner == player else -1.0

        if use_symmetries:
            syms = mcts.symmetries(state_enc, pi_vec)
            for s_aug, pi_aug in syms:
                final_examples.append((s_aug.astype(np.float32), pi_aug.astype(np.float32), z))
        else:
            final_examples.append((state_enc.astype(np.float32), pi_vec.astype(np.float32), z))

    return final_examples, winner


# -------------------------
#  模型间评估
# -------------------------
def evaluate_models(model_new: PyTorchModel,
                model_best: PyTorchModel,
                game_name: str,
                n_games: int = 20,
                n_simulations: int = 100,
                cpuct: float = 1.0,
                max_moves: int = 160) -> Tuple[int, float, int]:
    """
    在 model_new 和 model_best 之间进行 n_games 局游戏（轮流先手）。
    返回 (win_rate_of_new, draws)
    """
    # 根据 rules 选择游戏类（gomoku / renju / pente）
    GameClass = get_game_class(game_name)

    new_wins = 0
    draws = 0
    total = n_games

    for i in range(n_games):
        game = GameClass(size=model_new.board_size)

        if not hasattr(game, "phase"):
            # 五子棋/连珠以外：随机第一手增加开局多样性
            center = model_new.board_size // 2
            radius = 4  # 9×9区域 (81种可能 × 2先后手 = 162种组合)
            r1 = random.randint(center - radius, center + radius)
            c1 = random.randint(center - radius, center + radius)
            game.do_move((r1, c1))
            # 现在 current_player = 2，从第二手开始真正评估
        # 连珠（renju）：从 S1_MOVE 直接开始，开局由 MCTS 搜索

        # 确定谁先手：新模型在偶数局先手
        new_starts = (i % 2 == 0)
        move_number = 1

        # 为两个玩家创建 MCTS 实例（扩展时各自使用自己的模型）
        mcts_new = MCTS(game_class=GameClass, n_simulations=n_simulations, nn_model=model_new, cpuct=cpuct, add_dirichlet_noise=False)
        mcts_best = MCTS(game_class=GameClass, n_simulations=n_simulations, nn_model=model_best, cpuct=cpuct, add_dirichlet_noise=False)

        while not game.is_game_over():
            _resolve_greedy_opening(game, mcts_new.nn_model)
            if game.is_game_over():
                break
            # 根据当前玩家和谁先手决定谁下棋
            if (game.current_player == 1 and new_starts) or (game.current_player == 2 and not new_starts):
                pi = mcts_new.run(game, len(game.move_history))
            else:
                pi = mcts_best.run(game, len(game.move_history))

            # 确定性选择 (argmax)
            action = int(np.argmax(pi))
            game.do_move(action)
            move_number += 1
            if move_number > max_moves:
                break

        winner = game.get_winner()
        if winner == 0:
            draws += 1
        else:
            # 确定新模型是否获胜
            if (winner == 1 and new_starts) or (winner == 2 and not new_starts):
                new_wins += 1

        mcts_new.clear_tree()
        mcts_best.clear_tree()
        del mcts_new
        del mcts_best
        gc.collect()

    win_rate = new_wins / float(total)
    return new_wins, win_rate, draws


# -------------------------
#  多进程评估（外部接口）
# -------------------------
def evaluate_models_mp(
    model_new: PyTorchModel,
    model_best: PyTorchModel,
    board_size: int,
    action_size: int,
    n_games: int,
    n_simulations: int,
    cpuct: float,
    max_moves: int = 160,
    *,
    game_name: str = "gomoku",
    model_dir: str,
    num_workers: int,
    games_per_task: int = 1,
    device: str = "cpu",
    base_seed: int = 54321,
    torch_threads: int = 1,
) -> Tuple[int, float, int]:
    """
    并行评估：保存两个模型 checkpoint -> 多进程并行跑对局 -> 汇总。
    返回 (new_wins, win_rate, draws)
    """
    os.makedirs(model_dir, exist_ok=True)

    # 保存 checkpoint（避免跨进程传递模型对象）
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_new = os.path.join(model_dir, f"_eval_model_new_{ts}.pt")
    ckpt_best = os.path.join(model_dir, f"_eval_model_best_{ts}.pt")
    model_new.save(ckpt_new)
    model_best.save(ckpt_best)

    games_per_task = max(1, int(games_per_task))
    tasks = []
    remaining = int(n_games)
    start_idx = 0
    while remaining > 0:
        g = min(games_per_task, remaining)
        tasks.append((start_idx, g))
        start_idx += g
        remaining -= g

    ctx = mp.get_context("spawn")
    total_new_wins = 0
    total_draws = 0
    total_games = 0

    with ProcessPoolExecutor(
        max_workers=int(num_workers),
        mp_context=ctx,
        initializer=_eval_worker_init,
        initargs=(ckpt_new, ckpt_best, board_size, action_size, device, base_seed, torch_threads),
    ) as ex:
        futures = []
        for sidx, gcount in tasks:
            futures.append(
                ex.submit(
                    _eval_play_games,
                    game_name=game_name,
                    board_size=board_size,
                    n_games=int(gcount),
                    start_index=int(sidx),
                    n_simulations=n_simulations,
                    cpuct=cpuct,
                    max_moves=max_moves,
                )
            )

        for fut in as_completed(futures):
            nw, dr, tg = fut.result()
            total_new_wins += int(nw)
            total_draws += int(dr)
            total_games += int(tg)

    # 清理 checkpoint
    for p in (ckpt_new, ckpt_best):
        try:
            os.remove(p)
        except Exception:
            pass

    win_rate = total_new_wins / float(total_games) if total_games > 0 else 0.0
    return int(total_new_wins), float(win_rate), int(total_draws)


# -------------------------
#  主训练循环
# -------------------------
def train_alphazero(
    game_name: str = "gomoku",
    board_size: int = 15,
    num_iterations: int = 5,
    games_per_iteration: int = 8,
    n_simulations: int = 50,
    buffer_size: int = 10000,
    batch_size: int = 128,
    epochs_per_iter: int = 2,
    temp_threshold: int = 8,
    eval_games: int = 12,
    eval_mcts_simulations: int = 200,
    eval_max_moves: int = 160,
    win_rate_threshold: float = 0.55,
    always_accept: bool = False,
    log_eval_games: int = 0,
    log_eval_sims: int = 60,
    cpuct: float = 1.2,
    model_dir: str = "models",
    save_every: int = 1,
    pretrained_model_path: Optional[str] = None,  # 新参数，用于传递预训练模型
    next_iteration_continuation: int = 1,
    # --- MCTS Dirichlet噪声参数 ---
    dirichlet_alpha: float = 0.03,             # Dirichlet噪声的alpha参数
    dirichlet_epsilon: float = 0.25,           # Dirichlet噪声的混合比例
    dirichlet_n_moves: int = 30,               # 前N手添加Dirichlet噪声
    # --- 多进程自对弈参数 ---
    selfplay_num_workers: int = 0,             # 0=自动（建议 CPU 核心数-1，最多8）
    selfplay_device: str = "cpu",              # 建议 "cpu"，避免 CUDA 多进程争用
    selfplay_games_per_task: int = 1,          # 每个任务包含的自对弈局数（越大 IPC 越少）
    selfplay_base_seed: int = 12345,           # 子进程随机种子基数
    selfplay_torch_threads: int = 1,           # 每个子进程内 torch CPU 线程数
    # --- 多进程评估参数 ---
    eval_num_workers: int = 0,                 # 0=自动（建议 CPU 核心数-1，最多8）
    eval_device: str = "cpu",                  # 建议 cpu（多进程 cuda 风险高）
    eval_games_per_task: int = 1,              # 每个任务评估局数
    eval_base_seed: int = 54321,
    eval_torch_threads: int = 1,
):
    """
    核心训练流程。
    """
    os.makedirs(model_dir, exist_ok=True)

    # 根据规则计算动作空间大小（gomoku/pente=225，renju=225+4=229）
    GameClass = get_game_class(game_name)
    action_size = GameClass(size=board_size).action_size

    # 检查是否存在预训练模型
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"从以下路径加载预训练模型: {pretrained_model_path}")
        model_best = PyTorchModel(board_size=board_size, action_size=action_size)
        model_best.load(pretrained_model_path)  # 加载预训练模型
        model_candidate = PyTorchModel(board_size=board_size, action_size=action_size)
        model_candidate.net.load_state_dict(model_best.net.state_dict())
        print("预训练模型加载成功。")
    else:
        print("未找到预训练模型。初始化新模型。")
        model_best = PyTorchModel(board_size=board_size, action_size=action_size)
        model_candidate = PyTorchModel(board_size=board_size, action_size=action_size)
        # 关键：让候选模型复制最佳模型的初始权重，确保第一轮评估公平
        model_candidate.net.load_state_dict(model_best.net.state_dict())

    # 经验回放缓冲区
    buffer_filepath = os.path.join(model_dir, "replay_buffer_latest.pkl")
    
    # 尝试加载已保存的 buffer
    buffer = load_replay_buffer(buffer_filepath, capacity=buffer_size)
    
    # 如果加载失败或文件不存在，创建新的空 buffer
    if buffer is None:
        print("[Buffer] 创建新的空 buffer")
        buffer = ReplayBuffer(capacity=buffer_size)
    else:
        print(f"[Buffer] 成功加载历史 buffer，当前大小: {len(buffer)}/{buffer_size}")

    # 温度调度
    def temp_fn(move_number: int):
        return max(0.0, 1.0 - move_number / temp_threshold)

    for it in range(next_iteration_continuation, next_iteration_continuation + num_iterations):
        t0 = time.time()
        print(f"\n=== ITER {it}/{next_iteration_continuation + num_iterations - 1}: 自对弈生成 (games={games_per_iteration}, sims={n_simulations}), 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        selfplay_t0 = time.time()

        # 使用候选模型进行自对弈生成（多进程）
        winners = {0: 0, 1: 0, 2: 0}

        # 自动 worker 数：CPU 核心数-1，最多 8，最少 1
        if selfplay_num_workers and selfplay_num_workers > 0:
            num_workers = int(selfplay_num_workers)
        else:
            cpu_cnt = os.cpu_count() or 2
            num_workers = max(1, min(8, cpu_cnt - 1))

        # 1 worker 等价串行（但仍走同一套代码路径）
        max_moves = board_size * board_size
        use_symmetries = True
        add_dirichlet_noise = True

        # 并行执行
        if num_workers == 1:
            # 串行：直接使用主进程的 model_candidate（避免多余的 save/load）
            for g in range(games_per_iteration):
                mcts_play = MCTS(
                    game_class=GameClass,
                    n_simulations=n_simulations,
                    nn_model=model_candidate,
                    cpuct=cpuct,
                    dirichlet_alpha=dirichlet_alpha,
                    epsilon=dirichlet_epsilon,
                    apply_dirichlet_n_first_moves=dirichlet_n_moves,
                    add_dirichlet_noise=add_dirichlet_noise,
                )
                game = GameClass(size=board_size)
                game.current_player = 1
                # 连珠：开局由 MCTS + 策略贪心从初始状态直接搜索
                examples, winner = play_game_and_collect(
                    mcts_play, game, temp_fn, max_moves=max_moves, use_symmetries=use_symmetries
                )
                buffer.add(examples)
                winners[winner] = winners.get(winner, 0) + 1

                mcts_play.clear_tree()
                del mcts_play
                gc.collect()
        else:
            # 保存候选模型到 checkpoint，子进程从磁盘加载（避免传递不可 pickle 的模型对象）
            selfplay_ckpt_path = os.path.join(model_dir, f"_selfplay_candidate_iter{it}.pt")
            model_candidate.save(selfplay_ckpt_path)

            # 任务切分
            games_per_task = max(1, int(selfplay_games_per_task))
            tasks = []
            remaining = int(games_per_iteration)
            while remaining > 0:
                g = min(games_per_task, remaining)
                tasks.append(g)
                remaining -= g

            gen_games_done = 0
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=ctx,
                initializer=_selfplay_worker_init,
                initargs=(selfplay_ckpt_path, board_size, action_size, selfplay_device, selfplay_base_seed, selfplay_torch_threads),
            ) as ex:
                futures = []
                for gcount in tasks:
                    futures.append(
                        ex.submit(
                            _selfplay_generate_games,
                            game_name=game_name,
                            board_size=board_size,
                            n_simulations=n_simulations,
                            cpuct=cpuct,
                            temp_threshold=temp_threshold,
                            add_dirichlet_noise=add_dirichlet_noise,
                            games_to_play=int(gcount),
                            use_symmetries=use_symmetries,
                            max_moves=max_moves,
                            dirichlet_alpha=dirichlet_alpha,
                            dirichlet_epsilon=dirichlet_epsilon,
                            dirichlet_n_moves=dirichlet_n_moves,
                            device=selfplay_device,
                        )
                    )

                for fut in as_completed(futures):
                    examples, w = fut.result()
                    buffer.add(examples)
                    for k, v in w.items():
                        winners[k] = winners.get(k, 0) + int(v)
                    gen_games_done += int(sum(w.values()))

            # 可选：清理 selfplay checkpoint（保留也可以方便复现实验）
            try:
                os.remove(selfplay_ckpt_path)
            except Exception:
                pass

        selfplay_t1 = time.time()
        print(f"自对弈完成：耗时 {(selfplay_t1 - selfplay_t0)/60:.2f}分钟，胜负统计={winners}，buffer_size={len(buffer)}")

        # 如果有足够的样本，训练候选模型
        if len(buffer) >= batch_size:
            print(f"\nTraining candidate model: buffer={len(buffer)}, batch_size={batch_size}, epochs_per_iter={epochs_per_iter}")
            n_batches = max(1, len(buffer) // batch_size)
            for epoch in range(epochs_per_iter):
                epoch_t0 = time.time()
                for b in range(n_batches):
                    states_b, pis_b, zs_b = buffer.sample(batch_size)
                    loss_info = model_candidate.train_batch(states_b, pis_b, zs_b, epochs=1)
                epoch_t1 = time.time()
                print(f"  epoch {epoch+1}/{epochs_per_iter} finished in {epoch_t1 - epoch_t0:.1f}s, last_loss={loss_info}")
        else:
            print(f"训练样本不足 (buffer={len(buffer)}, 需要 {batch_size})。跳过本次迭代的训练。")

        if always_accept:
            # --always-accept：跳过正式评估，直接接受候选（greedy 更新）
            win_rate = 1.0
            new_wins = eval_games
            draws = 0
            if log_eval_games > 0:
                # 廉价评估（仅日志，不参与接受决策）
                try:
                    q_wins, q_rate, q_draws = evaluate_models(
                        model_candidate,
                        model_best,
                        game_name,
                        n_games=log_eval_games,
                        n_simulations=log_eval_sims,
                        cpuct=cpuct,
                        max_moves=eval_max_moves,
                    )
                    print(f"廉价评估（仅日志）：胜率={q_rate:.3f}（{q_wins}/{log_eval_games}），平局={q_draws}")
                except Exception as e:
                    print(f"廉价评估失败：{e}")
            else:
                print("评估已跳过（--always-accept），直接接受候选模型。")
        else:
            # 评估（精简输出：只在结束后汇总一次）
            eval_t0 = time.time()
            try:
                # 自动评估进程数（与自对弈同策略）
                if eval_num_workers and eval_num_workers > 0:
                    eval_workers = int(eval_num_workers)
                else:
                    cpu_cnt = os.cpu_count() or 2
                    eval_workers = max(1, min(8, cpu_cnt - 1))

                if eval_workers == 1:
                    new_wins, win_rate, draws = evaluate_models(
                        model_candidate,
                        model_best,
                        game_name,
                        n_games=eval_games,
                        n_simulations=eval_mcts_simulations,
                        cpuct=cpuct,
                        max_moves=eval_max_moves,
                    )
                else:
                    new_wins, win_rate, draws = evaluate_models_mp(
                        model_candidate,
                        model_best,
                        game_name=game_name,
                        board_size=board_size,
                        action_size=action_size,
                        n_games=eval_games,
                        n_simulations=eval_mcts_simulations,
                        cpuct=cpuct,
                        max_moves=eval_max_moves,
                        model_dir=model_dir,
                        num_workers=eval_workers,
                        games_per_task=eval_games_per_task,
                        device=eval_device,
                        base_seed=eval_base_seed,
                        torch_threads=eval_torch_threads,
                    )
            except Exception as e:
                # 保持可见性，但不刷屏
                print(f"评估失败：{e}")
                new_wins, win_rate, draws = 0, 0.0, 0

            eval_t1 = time.time()
            print(
                f"评估完成：耗时 {(eval_t1 - eval_t0)/60:.2f} 分钟，胜率={win_rate:.3f}（{new_wins}/{eval_games}），平局={draws}"
            )

        # 接受/拒绝
        if win_rate >= win_rate_threshold:
            print(" 候选模型被接受 -> 提升为最佳模型。")
            # 更新 model_best（深拷贝权重和优化器状态）
            model_best.net.load_state_dict(model_candidate.net.state_dict())
            model_best.optimizer.load_state_dict(model_candidate.optimizer.state_dict())
            # 从最佳模型创建新的候选模型
            model_candidate = PyTorchModel(board_size=board_size, action_size=action_size)
            model_candidate.net.load_state_dict(model_best.net.state_dict())
            model_candidate.optimizer.load_state_dict(model_best.optimizer.state_dict())
        else:
            print(" 候选模型被拒绝 -> 从最佳模型恢复候选模型。")
            # 从最佳模型恢复权重和优化器状态（保持训练连续性）
            model_candidate = PyTorchModel(board_size=board_size, action_size=action_size)
            model_candidate.net.load_state_dict(model_best.net.state_dict())
            model_candidate.optimizer.load_state_dict(model_best.optimizer.state_dict())

        # 定期保存模型快照
        if it % save_every == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = os.path.join(model_dir, f"snapshot_iter{it}_{timestamp}.pt")
            model_best.save(snapshot_path)
            print(f" 💾 Saved snapshot: {snapshot_path}")
        
        # 每轮都保存 buffer（覆盖旧的，只保留最新）
        save_replay_buffer(buffer, buffer_filepath)

        t1 = time.time()
        print(f"迭代 {it} 完成，耗时 {(t1 - t0)/60:.2f}分钟。本次迭代获胜者: {winners}")

    print("\n=== 训练完成 ===")

# -------------------------
#  命令行入口
# -------------------------
def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。所有默认值与 train_alphazero 的签名默认值保持一致。"""
    p = argparse.ArgumentParser(
        description="AlphaZero 五子棋训练入口（Gomoku/Pente/Renju）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- 游戏与规则 ---
    p.add_argument("--game", default="gomoku", choices=["gomoku", "pente", "renju"],
                   help="游戏规则。renju 第 6 手起启用黑方禁手；小测试建议用 gomoku（无禁手）")
    p.add_argument("--board-size", type=int, default=15, help="棋盘边长（N x N）")
    p.add_argument("--n-in-row", type=int, default=5,
                   help="连成几子获胜。当前 Gomoku 逻辑硬编码为 5，仅支持 5")

    # --- 训练主循环 ---
    p.add_argument("--num-iterations", type=int, default=5, help="训练迭代次数")
    p.add_argument("--games-per-iteration", type=int, default=8, help="每轮自对弈局数")
    p.add_argument("--num-simulations", type=int, default=50, help="自对弈时 MCTS 每步模拟次数")
    p.add_argument("--cpuct", type=float, default=1.2, help="MCTS 探索/利用平衡因子")
    p.add_argument("--buffer-size", type=int, default=10000, help="经验回放缓冲区容量")
    p.add_argument("--batch-size", type=int, default=128, help="训练批次大小")
    p.add_argument("--epochs", type=int, default=2, help="每次迭代的训练轮数（epochs_per_iter）")
    p.add_argument("--temp-threshold", type=int, default=8, help="温度退火阈值（前 N 手带探索温度）")

    # --- 评估 ---
    p.add_argument("--eval-games", type=int, default=12, help="每次迭代评估的对局数")
    p.add_argument("--eval-simulations", type=int, default=200, help="评估时 MCTS 每步模拟次数")
    p.add_argument("--eval-max-moves", type=int, default=160, help="评估对局手数上限（超限判和，防止马拉松局）")
    p.add_argument("--win-rate-threshold", type=float, default=0.55, help="候选模型被接受的最低胜率")
    p.add_argument("--always-accept", action="store_true", help="跳过评估，每轮直接接受候选模型（greedy 更新，省评估耗时）")
    p.add_argument("--log-eval-games", type=int, default=0, help="always-accept 下的廉价评估局数（0=关闭；仅日志不参与决策）")
    p.add_argument("--log-eval-sims", type=int, default=60, help="廉价评估每步模拟次数")

    # --- Dirichlet 噪声 ---
    p.add_argument("--dirichlet-alpha", type=float, default=0.03, help="Dirichlet 噪声 alpha")
    p.add_argument("--dirichlet-epsilon", type=float, default=0.25, help="Dirichlet 噪声混合比例")
    p.add_argument("--dirichlet-n-moves", type=int, default=30, help="前 N 手添加 Dirichlet 噪声")

    # --- 保存 / 续训 ---
    p.add_argument("--model-dir", default="models", help="模型与 buffer 保存目录")
    p.add_argument("--save-every", type=int, default=1, help="每隔多少次迭代保存一次快照")
    p.add_argument("--pretrained-model", default=None,
                   help="预训练模型路径（None/空串/none 表示从头训练）")
    p.add_argument("--resume-iteration", type=int, default=1, help="从第几次迭代继续（续训）")

    # --- 多进程自对弈 ---
    p.add_argument("--selfplay-workers", type=int, default=0,
                   help="自对弈进程数（0=自动，CPU 核数-1，最多 8）")
    p.add_argument("--selfplay-device", default="cpu", help="自对弈设备（建议 cpu，多进程 cuda 有争用风险）")
    p.add_argument("--selfplay-games-per-task", type=int, default=1, help="每个自对弈任务包含的局数")
    p.add_argument("--selfplay-threads", type=int, default=1, help="每个自对弈子进程内 torch CPU 线程数")

    # --- 多进程评估 ---
    p.add_argument("--eval-workers", type=int, default=0,
                   help="评估进程数（0=自动，CPU 核数-1，最多 8）")
    p.add_argument("--eval-device", default="cpu", help="评估设备（建议 cpu）")
    p.add_argument("--eval-games-per-task", type=int, default=1, help="每个评估任务的对局数")
    p.add_argument("--eval-threads", type=int, default=1, help="每个评估子进程内 torch CPU 线程数")

    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    # n-in-row 目前只有 5 是有效值（Gomoku.check_winner 硬编码 count >= 5）
    if args.n_in_row != 5:
        raise SystemExit(
            f"--n-in-row 目前只支持 5（游戏胜负判定硬编码为 5 连），收到 {args.n_in_row}"
        )

    # 归一化预训练模型路径：空串 / "none" 视为从头训练
    pretrained = args.pretrained_model
    if pretrained is None or str(pretrained).strip() == "" or str(pretrained).strip().lower() == "none":
        pretrained = None

    train_alphazero(
        game_name=args.game,
        board_size=args.board_size,

        num_iterations=args.num_iterations,
        games_per_iteration=args.games_per_iteration,
        n_simulations=args.num_simulations,
        cpuct=args.cpuct,

        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        epochs_per_iter=args.epochs,
        temp_threshold=args.temp_threshold,

        eval_games=args.eval_games,
        eval_mcts_simulations=args.eval_simulations,
        eval_max_moves=args.eval_max_moves,
        win_rate_threshold=args.win_rate_threshold,
        always_accept=args.always_accept,
        log_eval_games=args.log_eval_games,
        log_eval_sims=args.log_eval_sims,

        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        dirichlet_n_moves=args.dirichlet_n_moves,

        model_dir=args.model_dir,
        save_every=args.save_every,
        pretrained_model_path=pretrained,
        next_iteration_continuation=args.resume_iteration,

        selfplay_num_workers=args.selfplay_workers,
        selfplay_device=args.selfplay_device,
        selfplay_games_per_task=args.selfplay_games_per_task,
        selfplay_torch_threads=args.selfplay_threads,

        eval_num_workers=args.eval_workers,
        eval_device=args.eval_device,
        eval_games_per_task=args.eval_games_per_task,
        eval_torch_threads=args.eval_threads,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()