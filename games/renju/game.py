# -*- coding: utf-8 -*-
"""连珠（Taraguchi-10）对局状态机，供 AlphaZero 训练/对弈使用。

与 ``taraguchi10-gomoku`` 的 TS FSM 语义逐条对齐：

- 棋盘 ``board[row][col]``（15×15，numpy int8），0 空 / 1 黑 / 2 白。
- 前 5 手为塔拉山口-10 开局（区域约束 + 交换 + 走法一/二），**无禁手**；
  第 5 手之后（第 6 手起）才启用黑方禁手（三三/四四/长连）。
- 交换只改变 ``black_owner``（谁执黑），不改变落子颜色序列。
- 五连优先：黑恰好五连判胜，即使同时构成禁手。
- 黑长连（>=6）判负；白 >=5 连胜。
"""

from typing import List, Optional, Tuple

import numpy as np

from games.renju.rule import BLACK, EMPTY, WHITE, Renju_Rule

SIZE = 15
CENTER = 7

# ---- 阶段（与 TS fsm.ts Phase 对齐）----
S1_MOVE = 'S1_MOVE'
S1_SWAP = 'S1_SWAP'
S2_MOVE = 'S2_MOVE'
S2_SWAP = 'S2_SWAP'
S3_MOVE = 'S3_MOVE'
S3_SWAP = 'S3_SWAP'
S4_MOVE = 'S4_MOVE'
S4_SWAP = 'S4_SWAP'          # 走法一的 E4 交换
VARIANT_CHOICE = 'VARIANT_CHOICE'
V1_S5_MOVE = 'V1_S5_MOVE'
V1_S5_SWAP = 'V1_S5_SWAP'    # 走法一的 E5 交换
V2_TEN_OFFER = 'V2_TEN_OFFER'
V2_TEN_PICK = 'V2_TEN_PICK'
S6_MOVE = 'S6_MOVE'
PLAY = 'PLAY'
OVER = 'OVER'

# 各落子阶段的中心区域半径（3×3 → r=1 … 9×9 → r=4）
_REGION = {S1_MOVE: 0, S2_MOVE: 1, S3_MOVE: 2, S4_MOVE: 3, V1_S5_MOVE: 4}

_MOVE_PHASES = {S1_MOVE, S2_MOVE, S3_MOVE, S4_MOVE, V1_S5_MOVE, S6_MOVE, PLAY}
_SWAP_PHASES = {S1_SWAP, S2_SWAP, S3_SWAP, S4_SWAP, V1_S5_SWAP}

# ---- 动作空间：225 个落子 + 4 个开局决策动作 ----
N_POS_ACTIONS = SIZE * SIZE
SWAP_ACCEPT = N_POS_ACTIONS       # 225
SWAP_REJECT = N_POS_ACTIONS + 1   # 226
VARIANT_1 = N_POS_ACTIONS + 2     # 227
VARIANT_2 = N_POS_ACTIONS + 3     # 228
N_DECISION_ACTIONS = 4


class RenjuGame:
    def __init__(self, size: int = SIZE):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        self.black_owner: int = BLACK          # 玩家 1 执黑（交换会改变）
        self.phase: str = S1_MOVE
        self.variant: Optional[int] = None
        self.offers: List[Tuple[int, int]] = []
        self.move_history: List[Tuple[int, int]] = []
        self.last_move: Optional[Tuple[int, int]] = None
        self.current_player: int = BLACK
        self.winner: Optional[int] = None
        self.result_reason: Optional[str] = None

    # ------------------------------------------------------------------
    # 派生状态（对齐 TS currentActor / regionRadius）
    # ------------------------------------------------------------------
    def _to_place_black(self) -> bool:
        return len(self.move_history) % 2 == 0

    @staticmethod
    def _other(p: int) -> int:
        return 3 - p

    def actor(self):
        """返回 ``(player, kind)``；kind ∈ move/swap/variant/offers/pick/None。"""
        if self.phase == OVER:
            return None
        to_black = self._to_place_black()
        if self.phase in _MOVE_PHASES:
            p = self.black_owner if to_black else self._other(self.black_owner)
            return (p, 'move')
        if self.phase in _SWAP_PHASES:
            p = self.black_owner if to_black else self._other(self.black_owner)
            return (p, 'swap')
        if self.phase == VARIANT_CHOICE:
            return (self.black_owner, 'variant')
        if self.phase == V2_TEN_OFFER:
            return (self.black_owner, 'offers')
        if self.phase == V2_TEN_PICK:
            return (self._other(self.black_owner), 'pick')
        return None

    def region_radius(self) -> Optional[int]:
        return _REGION.get(self.phase, None)

    def is_opening_phase(self) -> bool:
        return self.phase not in (PLAY, OVER)

    @property
    def swap_available(self) -> bool:
        """交换标志：当前是否为交换决策阶段。"""
        act = self.actor()
        return act is not None and act[1] == 'swap'

    def forbidden_enabled(self) -> bool:
        """前 5 手无禁手；第 5 手之后（第 6 手起）启用黑方禁手。"""
        return len(self.move_history) >= 5

    def phase_key(self) -> str:
        """状态键的相位部分（供 MCTS _state_key 区分交换/走法/落子阶段与执黑方）。"""
        return f"{self.phase}|{self.black_owner}"

    # ------------------------------------------------------------------
    # 禁手 / 合法着法
    # ------------------------------------------------------------------
    def _point_is_forbidden(self, r: int, c: int, rule: Renju_Rule) -> bool:
        self.board[r, c] = BLACK
        kind = rule.check_forbidden(r, c)
        self.board[r, c] = EMPTY
        return kind is not None

    def _near_stone(self, r: int, c: int, color: int, radius: int = 4) -> bool:
        """(r,c) 附近（切比雪夫距离 ≤ radius）是否存在 color 棋子。"""
        r0, r1 = max(0, r - radius), min(self.size, r + radius + 1)
        c0, c1 = max(0, c - radius), min(self.size, c + radius + 1)
        return bool(np.any(self.board[r0:r1, c0:c1] == color))

    def forbidden_moves_set(self) -> set:
        rule = Renju_Rule(self.board, self.size)
        return rule.forbidden_moves(BLACK)

    def legal_moves(self) -> List[Tuple[int, int]]:
        """当前 move 阶段的所有合法落点（区域约束 + 空位 + 黑方禁手过滤）。

        黑方禁手必然由「紧邻已有黑子」的点构成（三/四/长连都需要黑子支撑），
        远离任何黑子的空点不可能禁手，直接跳过昂贵的递归禁手判定（显著提速 MCTS）。
        """
        act = self.actor()
        if act is None or act[1] != 'move':
            return []
        radius = self.region_radius()
        black_and_forbidden = act[0] == BLACK and self.forbidden_enabled()
        rule = Renju_Rule(self.board, self.size) if black_and_forbidden else None
        moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != EMPTY:
                    continue
                if radius is not None and not self._within(r, c, radius):
                    continue
                if black_and_forbidden:
                    if self._near_stone(r, c, BLACK) and self._point_is_forbidden(r, c, rule):
                        continue
                moves.append((r, c))
        return moves

    @staticmethod
    def _within(r: int, c: int, radius: int) -> bool:
        return abs(r - CENTER) <= radius and abs(c - CENTER) <= radius

    # ------------------------------------------------------------------
    # 落子 / 交换 / 走法 / 打点 / 选点
    # ------------------------------------------------------------------
    def _place(self, r: int, c: int, color: int) -> None:
        self.board[r, c] = color
        self.move_history.append((r, c))
        self.last_move = (r, c)
        self._check_end(r, c, color)

    def apply_move(self, r: int, c: int) -> bool:
        act = self.actor()
        if act is None or act[1] != 'move':
            return False
        if (r, c) not in self.legal_moves():
            return False
        self._place(r, c, act[0])
        self._advance_move()
        self._sync_current_player()
        return True

    def apply_swap(self, accept: bool) -> bool:
        act = self.actor()
        if act is None or act[1] != 'swap':
            return False
        if accept:
            self.black_owner = self._other(self.black_owner)
        self.phase = {
            S1_SWAP: S2_MOVE,
            S2_SWAP: S3_MOVE,
            S3_SWAP: S4_MOVE,
            S4_SWAP: V1_S5_MOVE,
            V1_S5_SWAP: S6_MOVE,
        }[self.phase]
        self._sync_current_player()
        return True

    def apply_variant(self, variant: int) -> bool:
        act = self.actor()
        if act is None or act[1] != 'variant' or variant not in (1, 2):
            return False
        self.variant = variant
        self.phase = S4_SWAP if variant == 1 else V2_TEN_OFFER
        self._sync_current_player()
        return True

    def apply_offers(self, points: List[Tuple[int, int]]) -> bool:
        act = self.actor()
        if act is None or act[1] != 'offers':
            return False
        if len(points) != 10:
            return False
        for r, c in points:
            if not (0 <= r < self.size and 0 <= c < self.size):
                return False
            if self.board[r, c] != EMPTY:
                return False
        for i in range(10):
            for j in range(i + 1, 10):
                if points[i] == points[j]:
                    return False
                if points[i][0] + points[j][0] == 14 and points[i][1] + points[j][1] == 14:
                    return False  # 关于天元对称，不允许
        self.offers = list(points)
        self.phase = V2_TEN_PICK
        self._sync_current_player()
        return True

    def apply_pick(self, index: int) -> bool:
        act = self.actor()
        if act is None or act[1] != 'pick':
            return False
        if not (0 <= index < len(self.offers)):
            return False
        r, c = self.offers[index]
        self.offers = []
        self._place(r, c, BLACK)  # 第 5 手是黑子
        self.phase = S6_MOVE
        self._sync_current_player()
        return True

    def _advance_move(self) -> None:
        if self.phase == OVER:
            return
        self.phase = {
            S1_MOVE: S1_SWAP,
            S2_MOVE: S2_SWAP,
            S3_MOVE: S3_SWAP,
            S4_MOVE: VARIANT_CHOICE,
            V1_S5_MOVE: V1_S5_SWAP,
            S6_MOVE: PLAY,
            PLAY: PLAY,
        }[self.phase]

    def _sync_current_player(self) -> None:
        if self.phase == OVER:
            self.current_player = 0
            return
        act = self.actor()
        self.current_player = act[0] if act is not None else 0

    # ------------------------------------------------------------------
    # 终局判定（对齐 TS checkGameEnd，禁手只在第 6 手起生效）
    # ------------------------------------------------------------------
    def _check_end(self, r: int, c: int, color: int) -> None:
        rule = Renju_Rule(self.board, self.size)
        if color == BLACK:
            if rule.find_winning_line(r, c, BLACK, True) is not None:
                self._finish(self.black_owner, 'five')
                return
            if self.forbidden_enabled():
                kind = rule.check_forbidden(r, c)
                if kind is not None:
                    self._finish(self._other(self.black_owner), kind)
                    return
        else:
            if rule.find_winning_line(r, c, WHITE, False) is not None:
                self._finish(self._other(self.black_owner), 'five')
                return
        if len(self.move_history) >= self.size * self.size:
            self._finish(None, 'draw')

    def _finish(self, winner, reason: str) -> None:
        self.phase = OVER
        self.winner = winner
        self.result_reason = reason
        self.current_player = 0

    def is_game_over(self) -> bool:
        return self.phase == OVER

    def get_winner(self) -> int:
        return self.winner if self.winner is not None else 0

    def check_winner(self) -> int:
        """兼容纯 MCTS（mcts_pure）对终局的查询，返回 0/1/2。"""
        return self.get_winner()

    # ------------------------------------------------------------------
    # Gomoku 兼容接口（供 MCTS / players / train 复用）
    # ------------------------------------------------------------------
    @property
    def action_size(self) -> int:
        return self.size * self.size + N_DECISION_ACTIONS

    def action_to_move(self, action: int) -> Tuple[int, int]:
        return (action // self.size, action % self.size)

    def move_to_action(self, move: Tuple[int, int]) -> int:
        return int(move[0] * self.size + move[1])

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        return self.legal_moves()

    def has_legal_moves(self) -> bool:
        return len(self.legal_moves()) > 0

    def get_valid_moves(self) -> np.ndarray:
        """二进制动作向量（229 长度）：move/swap/variant 阶段分别开启对应动作。"""
        valid = np.zeros(self.action_size, dtype=np.float32)
        act = self.actor()
        if act is None:
            return valid
        if act[1] == 'move':
            for r, c in self.legal_moves():
                valid[self.move_to_action((r, c))] = 1.0
        elif act[1] == 'swap':
            valid[SWAP_ACCEPT] = 1.0
            valid[SWAP_REJECT] = 1.0
        elif act[1] == 'variant':
            valid[VARIANT_1] = 1.0
            valid[VARIANT_2] = 1.0
        # offers / pick 阶段由策略贪心 helper 处理，不进 MCTS
        return valid

    def do_move(self, move) -> bool:
        """支持 (r,c) 元组或扁平动作索引（含交换/走法决策动作）。"""
        if isinstance(move, (tuple, list)):
            return self.apply_move(int(move[0]), int(move[1]))
        action = int(move)
        if 0 <= action < N_POS_ACTIONS:
            r, c = divmod(action, self.size)
            return self.apply_move(r, c)
        if action == SWAP_ACCEPT:
            return self.apply_swap(True)
        if action == SWAP_REJECT:
            return self.apply_swap(False)
        if action == VARIANT_1:
            return self.apply_variant(1)
        if action == VARIANT_2:
            return self.apply_variant(2)
        return False

    def undo_move(self) -> None:
        # 开局含交换/走法/打点等非落子决策，无法用单一落子安全回退；
        # 训练/对弈用 clone() 展开分支，不需要 undo。此方法仅占位。
        raise NotImplementedError('RenjuGame 不支持单步 undo，请使用 clone()')

    def clone(self) -> 'RenjuGame':
        g = RenjuGame(self.size)
        g.board = self.board.copy()
        g.black_owner = int(self.black_owner)
        g.phase = self.phase
        g.variant = self.variant
        g.offers = list(self.offers)
        g.move_history = list(self.move_history)
        g.last_move = None if self.last_move is None else tuple(self.last_move)
        g.current_player = int(self.current_player)
        g.winner = self.winner
        g.result_reason = self.result_reason
        return g

    def get_state(self) -> np.ndarray:
        return self.board.copy()

    def display(self) -> None:
        """简单终端打印（兼容 play.py / play_loop.py 的展示调用）。"""
        print()
        for r in range(self.size):
            row = []
            for c in range(self.size):
                v = int(self.board[r, c])
                row.append('●' if v == 1 else ('○' if v == 2 else '·'))
            print(f"{r + 1:2} " + " ".join(row))
        print(f"当前行动方: {self.current_player} | 阶段: {self.phase} | 黑方: 玩家{self.black_owner}")

    def get_encoded_state(self) -> np.ndarray:
        """4 通道：我的棋子 / 对手棋子 / 禁手生效 / 阶段指示。

        按「当前玩家实际执子色」编码（交换会改变 black_owner，不能用玩家编号当颜色）。
        - 通道2 禁手生效：当前轮到黑且已进入第 6 手（len(move_history)>=5）时为 1.0，否则 0.0。
          对齐 KataGo 蒸馏数据 globalInputNC[:,5]（己黑=-1/己白=+1），监督预训练可复用此通道。
        - 通道3 阶段：0=落子/中盘, 0.5=交换, 1=走法, 0.75=打点/选点。
        """
        board = self.board
        my_color = BLACK if self.current_player == self.black_owner else WHITE
        opp_color = WHITE if my_color == BLACK else BLACK
        p_my = (board == my_color).astype(np.float32)
        p_opp = (board == opp_color).astype(np.float32)

        forbidden_active = 1.0 if (self.current_player == self.black_owner and self.forbidden_enabled()) else 0.0
        forbidden_plane = np.full((self.size, self.size), forbidden_active, dtype=np.float32)

        act = self.actor()
        phase_code = {'swap': 0.5, 'variant': 1.0, 'offers': 0.75, 'pick': 0.75}.get(
            act[1] if act else None, 0.0
        )
        phase_plane = np.full((self.size, self.size), phase_code, dtype=np.float32)

        return np.stack([p_my, p_opp, forbidden_plane, phase_plane], axis=0)

    # ------------------------------------------------------------------
    # 开局默认策略（训练自对弈：拒绝交换 + 走法一，把决策压缩为纯落子）
    # ------------------------------------------------------------------
    def resolve_opening_to_play(self) -> None:
        """用确定性策略把开局走完（拒绝交换、走法一），直到进入 PLAY。"""
        guard = 0
        while self.phase != PLAY and self.phase != OVER and guard < 50:
            act = self.actor()
            if act is None:
                break
            kind = act[1]
            if kind == 'swap':
                self.apply_swap(False)
            elif kind == 'variant':
                self.apply_variant(1)
            elif kind == 'offers':
                self.apply_offers(self._default_offers())
            elif kind == 'pick':
                self.apply_pick(0)
            elif kind == 'move':
                moves = self.legal_moves()
                if not moves:
                    break
                self.apply_move(*moves[0])
            guard += 1

    def _default_offers(self) -> List[Tuple[int, int]]:
        """走法二兜底：按中心距离取 10 个空点（不含中心对称对）。"""
        pts = []
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != EMPTY:
                    continue
                pts.append((r, c))
        pts.sort(key=lambda p: abs(p[0] - CENTER) + abs(p[1] - CENTER))
        chosen = []
        for p in pts:
            if len(chosen) >= 10:
                break
            if any(q[0] + p[0] == 14 and q[1] + p[1] == 14 for q in chosen):
                continue
            chosen.append(p)
        return chosen
