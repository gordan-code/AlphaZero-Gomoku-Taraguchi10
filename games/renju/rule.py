# -*- coding: utf-8 -*-
"""Renju（连珠）禁手规则判断模块（严格 RIF，与 taraguchi10-gomoku 的 TS 实现逐行对齐）。

本模块替换了原先以 ``AlphaZero_Gomoku_Renju_Rule/renju_rule.py`` 为蓝本的简化实现。
原蓝本用 ``get_stone_count`` + ``find_empty_point`` 的"连续子 + 端点"模型判定四/三，
在跳三、眠三、假三（嵌套禁手）等边角棋形上与严格 RIF 不一致。这里改为与 TS
``src/shared/forbidden.ts`` 完全相同的"整线 5/4 窗口扫描"算法，保证跨语言输出一致。

坐标约定：棋盘为 ``board[row][col]``（``list[list[int]]`` 或 ``numpy.ndarray``），
``row`` 0..size-1 从上到下，``col`` 0..size-1 从左到右。取值 ``0`` 空 / ``1`` 黑 / ``2`` 白。

核心语义（与 TS 完全一致，务必保留）：

- ``check_forbidden(row, col)`` 假定 ``(row, col)`` **已经放上黑子**，判定这一手是否禁手。
- 五连优先：恰好五连直接判胜，即使同时构成三三/四四/长连也不算禁手。
- 黑长连（>=6）为禁手（overline）。
- 双三的递归性：判断活三时，成四点本身若是禁手，则该三为"假三"不计入。
- 四/三按"棋子坐标集合"去重（同一四/三只计一次）。
"""

from typing import List, Optional, Set, Tuple

EMPTY = 0
BLACK = 1
WHITE = 2

# 兼容蓝本小写命名
empty = EMPTY
black_stone = BLACK
white_stone = WHITE

# 假三递归深度上限（与 TS MAX_DEPTH 一致）
MAX_DEPTH = 3

# 4 个方向：(dcol, drow) = 横、竖、↘、↗
DIRS: Tuple[Tuple[int, int], ...] = ((1, 0), (0, 1), (1, 1), (1, -1))


class Renju_Rule(object):
    def __init__(self, board, board_size: Optional[int] = None):
        self.board = board
        self.board_size = board_size if board_size is not None else len(board)
        self._forbidden_mask_cache = {}

    # ------------------------------------------------------------------
    # 缓存 / 批量接口
    # ------------------------------------------------------------------
    def _board_signature(self):
        b = self.board
        if hasattr(b, "tobytes"):
            return b.tobytes()
        return tuple(tuple(row) for row in b)

    def clear_cache(self) -> None:
        self._forbidden_mask_cache.clear()

    def forbidden_moves(self, stone: int = BLACK) -> Set[Tuple[int, int]]:
        """返回落子后为禁手的所有空位 ``{(row, col), ...}``（仅黑方有意义）。"""
        key = (self._board_signature(), stone)
        cached = self._forbidden_mask_cache.get(key)
        if cached is not None:
            return cached
        result: Set[Tuple[int, int]] = set()
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] == EMPTY and self.forbidden_point(row, col, stone):
                    result.add((row, col))
        self._forbidden_mask_cache[key] = result
        return result

    # ------------------------------------------------------------------
    # 基础工具
    # ------------------------------------------------------------------
    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.board_size and 0 <= col < self.board_size

    def _line_of(self, row: int, col: int, dc: int, dr: int):
        """提取过 (row,col) 的整条线（沿方向，出界即止），返回 (stones, positions)。"""
        stones = []
        positions = []
        for i in range(-7, 8):
            rr = row + dr * i
            cc = col + dc * i
            if not self._in_bounds(rr, cc):
                continue
            stones.append(self.board[rr][cc])
            positions.append((rr, cc))
        return stones, positions

    @staticmethod
    def _index_of(positions, row: int, col: int) -> int:
        for i, (rr, cc) in enumerate(positions):
            if rr == row and cc == col:
                return i
        return -1

    # ------------------------------------------------------------------
    # 连子计数与胜负
    # ------------------------------------------------------------------
    def run_length(self, row: int, col: int, color: int) -> int:
        """落子后 (row,col) 所在直线上连续同色子的最大长度（4 方向取最大，含自身）。"""
        best = 1
        for dc, dr in DIRS:
            n = 1
            i = 1
            while True:
                rr, cc = row + dr * i, col + dc * i
                if not self._in_bounds(rr, cc) or self.board[rr][cc] != color:
                    break
                n += 1
                i += 1
            i = 1
            while True:
                rr, cc = row - dr * i, col - dc * i
                if not self._in_bounds(rr, cc) or self.board[rr][cc] != color:
                    break
                n += 1
                i += 1
            if n > best:
                best = n
        return best

    def find_winning_line(self, row: int, col: int, color: int, exact: bool):
        """返回落子后某方向的连线坐标（含自身）；黑 exact=True 恰好 5，白 exact=False 则 >=5。"""
        for dc, dr in DIRS:
            cells = [(row, col)]
            i = 1
            while i < 5:
                rr, cc = row + dr * i, col + dc * i
                if not self._in_bounds(rr, cc) or self.board[rr][cc] != color:
                    break
                cells.append((rr, cc))
                i += 1
            i = 1
            while i < 5:
                rr, cc = row - dr * i, col - dc * i
                if not self._in_bounds(rr, cc) or self.board[rr][cc] != color:
                    break
                cells.insert(0, (rr, cc))
                i += 1
            length = len(cells)
            ok = (length == 5) if exact else (length >= 5)
            if ok:
                return cells
        return None

    # ------------------------------------------------------------------
    # 四的计数（严格 RIF：5 窗口扫描，黑子集合去重）
    # ------------------------------------------------------------------
    def _count_fours(self, row: int, col: int) -> int:
        total = 0
        for dc, dr in DIRS:
            stones, positions = self._line_of(row, col, dc, dr)
            pi = self._index_of(positions, row, col)
            if pi < 0:
                continue
            seen: Set[Tuple[Tuple[int, int], ...]] = set()
            n = len(stones)
            for start in range(max(0, pi - 4), min(pi, n - 5) + 1):
                black = 0
                empty_idx = -1
                ok = True
                for k in range(start, start + 5):
                    s = stones[k]
                    if s == BLACK:
                        black += 1
                    elif s == EMPTY:
                        empty_idx = k
                    else:
                        ok = False
                        break
                if not ok or black != 4 or empty_idx < 0:
                    continue
                before = stones[start - 1] if start - 1 >= 0 else WHITE
                after = stones[start + 5] if start + 5 < n else WHITE
                if before == BLACK or after == BLACK:
                    continue  # 会成长连，不构成"四"
                key = tuple(sorted(
                    positions[start + k] for k in range(5) if start + k != empty_idx
                ))
                seen.add(key)
            total += len(seen)
        return total

    # ------------------------------------------------------------------
    # 活三的计数（严格 RIF：模拟成四 + 嵌套假三递归）
    # ------------------------------------------------------------------
    def _count_open_threes(self, row: int, col: int, depth: int) -> int:
        total = 0
        for dc, dr in DIRS:
            stones, positions = self._line_of(row, col, dc, dr)
            n = len(stones)
            pi = self._index_of(positions, row, col)
            if pi < 0:
                continue
            seen: Set[Tuple[Tuple[int, int], ...]] = set()
            for ei in range(max(0, pi - 3), min(n - 1, pi + 3) + 1):
                if ei == pi or stones[ei] != EMPTY:
                    continue
                sim = list(stones)
                sim[ei] = BLACK
                found = False
                found_key = None
                for j in range(n - 3):
                    if not (sim[j] == sim[j + 1] == sim[j + 2] == sim[j + 3] == BLACK):
                        continue
                    lo = sim[j - 1] if j - 1 >= 0 else WHITE
                    hi = sim[j + 4] if j + 4 < n else WHITE
                    if lo != EMPTY or hi != EMPTY:
                        continue
                    lo2 = sim[j - 2] if j - 2 >= 0 else WHITE
                    hi2 = sim[j + 5] if j + 5 < n else WHITE
                    if lo2 == BLACK or hi2 == BLACK:
                        continue
                    if not (j <= pi <= j + 3 and j <= ei <= j + 3):
                        continue
                    if depth < MAX_DEPTH:
                        er, ec = positions[ei]
                        self.board[er][ec] = BLACK
                        is_forbidden = self.check_forbidden(er, ec, depth + 1) is not None
                        self.board[er][ec] = EMPTY
                        if is_forbidden:
                            continue  # 成四点本身是禁手 → 假三
                    found = True
                    found_key = tuple(sorted(
                        positions[k] for k in (j, j + 1, j + 2, j + 3) if k != ei
                    ))
                    break
                if found:
                    seen.add(found_key)
            total += len(seen)
        return total

    # ------------------------------------------------------------------
    # 主入口（严格对齐 TS checkForbidden）
    # ------------------------------------------------------------------
    def check_forbidden(self, row: int, col: int, depth: int = 0) -> Optional[str]:
        """判定 (row,col)（已放黑子）是否禁手。返回 'double-three'/'double-four'/'overline'/None。"""
        if self.find_winning_line(row, col, BLACK, True) is not None:
            return None  # 五连优先
        if self.run_length(row, col, BLACK) >= 6:
            return 'overline'
        if self._count_fours(row, col) >= 2:
            return 'double-four'
        if depth < MAX_DEPTH and self._count_open_threes(row, col, depth) >= 2:
            return 'double-three'
        return None

    def forbidden_point(self, row: int, col: int, stone: int = BLACK) -> bool:
        """判断黑方落子在空点 (row,col) 是否禁手（临时落子后判定并恢复）。白方恒 False。"""
        if stone != BLACK:
            return False
        if self.board[row][col] != EMPTY:
            return False
        self.board[row][col] = BLACK
        kind = self.check_forbidden(row, col)
        self.board[row][col] = EMPTY
        return kind is not None

    def is_legal_move(self, row: int, col: int, stone: int) -> bool:
        """落子是否合法：空位，且（黑方）非禁手。"""
        if self.board[row][col] != EMPTY:
            return False
        if stone == BLACK:
            return not self.forbidden_point(row, col, stone)
        return True
