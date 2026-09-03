import numpy as np
import math

class MCTS:
    """
    基于AlphaZero的蒙特卡洛树搜索(MCTS)，支持批量预测。
    - 针对Gomoku（五子棋）和Pente（同类棋类游戏）进行了适配。
    - 使用神经网络的策略（policy）和价值（value）输出。
    - 支持对称变换（旋转与水平翻转）。
    """

    def __init__(self, game_class, n_simulations, nn_model, cpuct=1.0, batch_size=32, dirichlet_alpha=0.03, epsilon=0.03, apply_dirichlet_n_first_moves=10, add_dirichlet_noise=True):
        self.game_class = game_class
        self.n_simulations = n_simulations
        self.nn_model = nn_model
        self.cpuct = cpuct
        self.batch_size = batch_size
        self.dirichlet_alpha = dirichlet_alpha
        self.epsilon = epsilon
        self.apply_dirichlet_n_first_moves = apply_dirichlet_n_first_moves
        self.add_dirichlet_noise = add_dirichlet_noise

        # Tree dictionaries
        self.P = {}
        self.V = {}
        self.N = {}
        self.W = {}
        self.children = {}
        self.action_size = game_class().action_size

        # Batch prediction queue
        self.pending_states = []
        self.pending_keys = []
        self.pending_game_states = []

        # Root node key (for Dirichlet noise)
        self.root_key = None

    # -------------------------
    #  SIMETRIAS
    # -------------------------
    def symmetries(self, state, pi):
        size = state.shape[1]
        n_pos = size * size
        pos_pi = pi[:n_pos].reshape(size, size)
        dec_pi = pi[n_pos:]  # 决策动作 logits（swap/variant），与对称无关

        out = []
        for k in range(4):
            rotated_s = np.rot90(state, k, axes=(1, 2))
            rotated_pos = np.rot90(pos_pi, k)
            out.append((rotated_s, np.concatenate([rotated_pos.flatten(), dec_pi])))

            flipped_s = np.flip(rotated_s, axis=2)
            flipped_pos = np.flip(rotated_pos, axis=1)
            out.append((flipped_s, np.concatenate([flipped_pos.flatten(), dec_pi])))

        return out

    def clear_tree(self):
        # Tree dictionaries
        self.P = {}
        self.V = {}
        self.N = {}
        self.W = {}
        self.children = {}

        # Batch prediction queue
        self.pending_states = []
        self.pending_keys = []
        self.pending_game_states = []

        # Root node key (for Dirichlet noise)
        self.root_key = None

    # -------------------------
    #  MCTS RUN
    # -------------------------
    def run(self, game_state, move_number):
        self.root_key = self._state_key(game_state) # 获取当前状态的键，是bytes类型 ，是唯一标识当前状态的键，里面有棋盘数据和当前玩家
    
        # 进行N次模拟
        for _ in range(self.n_simulations):
            game_copy = game_state.clone()  # 克隆当前状态，避免修改原始状态
            self.search(game_copy, move_number)  # 搜索当前状态，返回一个值，这个值是当前状态的评估值

        # 处理所有剩余的状态
        self._predict_batch(move_number)

        s_key = self._state_key(game_state)
        counts = self.N[s_key]
        total = np.sum(counts)
        if total > 0:
            pi = counts / total
        else:
            # fallback seguro: uniformemente entre ações válidas
            valid = self.children[s_key]
            pi = valid / np.sum(valid)
        return pi

    # -------------------------
    #  SEARCH
    # -------------------------
    def search(self, game_state, move_number):
        # 终局判断
        if game_state.is_game_over():
            winner = game_state.get_winner()
            if winner == 0:
                return 0
            # 当前玩家还没有下棋，既然分出胜负，说明上一步的对手赢了。
            # 对于当前轮到的玩家来说，这是必输的局面，直接返回 -1。
            return -1

        # 走法二的十打点/选点：策略贪心（不进树搜索），原地推进到 S6_MOVE
        if hasattr(game_state, 'phase') and game_state.phase in ('V2_TEN_OFFER', 'V2_TEN_PICK'):
            from games.renju.opening_greedy import resolve_greedy_opening_steps
            resolve_greedy_opening_steps(game_state, self.nn_model)

        s_key = self._state_key(game_state)

        if s_key not in self.P: # 如果当前状态的键不在P中，表示当前状态没有被访问过
            # adiciona à fila de batch
            self.pending_states.append(game_state.get_encoded_state())
            self.pending_keys.append(s_key)
            self.pending_game_states.append(game_state.clone())  # 将当前状态克隆并添加到pending_game_states中

            # se a fila encheu, processa em batch
            if len(self.pending_states) >= self.batch_size:
                self._predict_batch(move_number) # 预测当前状态的值和策略

            #  
            if s_key not in self.P:
                valid = game_state.get_valid_moves()
                self.P[s_key] = valid / np.sum(valid) # 将当前状态的合法动作归一化，得到每个位置的概率
                self.V[s_key] = 0 # 当前状态的评估值为0
                self.N[s_key] = np.zeros_like(valid, dtype=np.float32) # 当前状态的访问次数为0
                self.W[s_key] = np.zeros_like(valid, dtype=np.float32) # 当前状态的奖励值为0
                self.children[s_key] = valid # 将当前状态的合法动作添加到当前状态的子节点中
                return self.V[s_key]

        # 已访问过的节点 → UCB选择
        valid = self.children[s_key] # 获取当前状态的合法动作 就是所有合法的位置，是一个二进制向量，长度为 action_size
        sqrt_sum = math.sqrt(np.sum(self.N[s_key])) # 计算当前状态的访问次数的平方根
        ucb = self.W[s_key] / (1 + self.N[s_key]) + self.cpuct * self.P[s_key] * sqrt_sum / (1 + self.N[s_key])
        ucb = np.where(valid == 1, ucb, -1e9) # 将当前状态的合法动作设置为UCB值，否则设置为-1e9

        action = np.argmax(ucb) # 选择UCB值最大的动作
        # 执行动作（含交换/走法决策动作）
        game_state.do_move(action)

        v = -self.search(game_state, move_number) # 递归搜索下一个状态，返回下一个状态的评估值

        self.W[s_key][action] += v # 更新当前状态的奖励值
        self.N[s_key][action] += 1 # 更新当前状态的访问次数

        return v

    # -------------------------
    #  BATCH PREDICTION
    # -------------------------
    def _predict_batch(self, move_number):
        if not self.pending_states:
            return

        X = np.stack(self.pending_states, axis=0).astype(np.float32)
        policies, values = self.nn_model.predict(X)

        for k, p, v, gs in zip(self.pending_keys, policies, values, self.pending_game_states):
            p = p.flatten()
            valid = gs.get_valid_moves()  # <-- usa o estado real armazenado na fila
            p = p * valid
            if np.sum(p) < 1e-8:
                p = valid / np.sum(valid)

            # Add Dirichlet noise ONLY at the root node
            if self.add_dirichlet_noise and k == self.root_key and move_number < self.apply_dirichlet_n_first_moves:
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(p))
                p = (1 - self.epsilon) * p + self.epsilon * noise
                p /= np.sum(p)

            self.P[k] = p
            self.V[k] = v[0] if hasattr(v, "__len__") else v
            self.N[k] = np.zeros_like(p, dtype=np.float32)
            self.W[k] = np.zeros_like(p, dtype=np.float32)
            self.children[k] = valid

        # limpar fila
        self.pending_states = []
        self.pending_keys = []
        self.pending_game_states = []

    # -------------------------
    #  STATE KEY
    # -------------------------
    def _state_key(self, game_state):
        board = game_state.board
        player = game_state.current_player
        phase = getattr(game_state, 'phase_key', lambda: '')()
        return board.tobytes() + bytes([player]) + phase.encode('utf-8')