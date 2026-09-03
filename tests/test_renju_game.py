# -*- coding: utf-8 -*-
"""RenjuGame 对局状态机测试：开局阶段 + 禁手时机 + 胜负判定。"""

import unittest

import numpy as np

from games.renju.game import (
    PLAY,
    S1_MOVE,
    S1_SWAP,
    S2_MOVE,
    S2_SWAP,
    S3_MOVE,
    S3_SWAP,
    S4_MOVE,
    S4_SWAP,
    S6_MOVE,
    SWAP_ACCEPT,
    SWAP_REJECT,
    VARIANT_1,
    VARIANT_2,
    VARIANT_CHOICE,
    RenjuGame,
)
from games.renju.rule import BLACK, WHITE


def _midgame_with_stones(black_pts, white_pts, move_count=6):
    """构造一个处于 PLAY 阶段（第 6 手后）的连珠对局。move_count 决定轮到谁。"""
    g = RenjuGame()
    g.phase = PLAY
    g.black_owner = BLACK
    for r, c in black_pts:
        g.board[r, c] = BLACK
    for r, c in white_pts:
        g.board[r, c] = WHITE
    g.move_history = [(0, 0)] * move_count  # 伪造手数，使禁手启用
    g.current_player = BLACK if move_count % 2 == 0 else WHITE
    return g


class RenjuGameOpeningTest(unittest.TestCase):
    def test_initial_state(self):
        g = RenjuGame()
        self.assertEqual(g.phase, S1_MOVE)
        self.assertEqual(g.actor(), (1, "move"))
        self.assertEqual(g.legal_moves(), [(7, 7)])  # 天元

    def test_variant1_phase_sequence(self):
        """走法一：S1..S4 + 交换 + 第 5 手(9×9) + 第 6 手 → PLAY。"""
        g = RenjuGame()
        seq = [
            (S1_MOVE, (1, "move"), (7, 7)),   # 第 1 手黑天元
            (S1_SWAP, (2, "swap"), None),     # 白交换
            (S2_MOVE, (2, "move"), (6, 6)),   # 第 2 手白 3×3
            (S2_SWAP, (1, "swap"), None),     # 黑交换
            (S3_MOVE, (1, "move"), (7, 6)),   # 第 3 手黑 5×5
            (S3_SWAP, (2, "swap"), None),     # 白交换
            (S4_MOVE, (2, "move"), (6, 7)),   # 第 4 手白 7×7
        ]
        for phase, actor, move in seq:
            self.assertEqual(g.phase, phase)
            self.assertEqual(g.actor(), actor)
            if move is not None:
                self.assertTrue(g.apply_move(*move))
            else:
                self.assertTrue(g.apply_swap(False))

        self.assertEqual(g.phase, VARIANT_CHOICE)
        self.assertTrue(g.apply_variant(1))
        self.assertEqual(g.phase, S4_SWAP)     # 走法一 E4 交换
        self.assertTrue(g.apply_swap(False))
        self.assertEqual(g.phase, "V1_S5_MOVE")
        self.assertTrue(g.apply_move(4, 4))    # 第 5 手 9×9
        self.assertEqual(g.phase, "V1_S5_SWAP")
        self.assertTrue(g.apply_swap(False))
        self.assertEqual(g.phase, S6_MOVE)
        self.assertTrue(g.apply_move(0, 0))    # 第 6 手
        self.assertEqual(g.phase, PLAY)
        self.assertEqual(len(g.move_history), 6)
        self.assertTrue(g.forbidden_enabled())

    def test_swap_changes_black_owner(self):
        g = RenjuGame()
        g.apply_move(7, 7)
        self.assertEqual(g.black_owner, BLACK)
        g.apply_swap(True)
        self.assertEqual(g.black_owner, WHITE)  # 白接管黑棋

    def test_variant2_offers_and_pick(self):
        g = RenjuGame()
        g.apply_move(7, 7)
        g.apply_swap(False)
        g.apply_move(6, 6)
        g.apply_swap(False)
        g.apply_move(7, 6)
        g.apply_swap(False)
        g.apply_move(6, 7)
        self.assertEqual(g.phase, VARIANT_CHOICE)
        g.apply_variant(2)
        self.assertEqual(g.phase, "V2_TEN_OFFER")

        points = [(r, c) for r in range(3, 8) for c in range(3, 5)]
        points = points[:10]
        self.assertTrue(g.apply_offers(points))
        self.assertEqual(g.phase, "V2_TEN_PICK")
        self.assertTrue(g.apply_pick(0))
        self.assertEqual(g.phase, S6_MOVE)
        self.assertEqual(len(g.move_history), 5)  # 第 5 手由选点落子

    def test_forbidden_disabled_during_opening(self):
        g = RenjuGame()
        self.assertFalse(g.forbidden_enabled())
        g.resolve_opening_to_play()
        self.assertEqual(g.phase, PLAY)
        self.assertTrue(g.forbidden_enabled())


class RenjuGameForbiddenTest(unittest.TestCase):
    def test_double_three_excluded_from_legal_moves(self):
        g = _midgame_with_stones(
            [(5, 7), (6, 7), (7, 5), (7, 6)],
            [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],
        )
        self.assertTrue(g.forbidden_enabled())
        self.assertNotIn((7, 7), g.legal_moves())
        self.assertIn((7, 7), g.forbidden_moves_set())

    def test_black_five_wins(self):
        g = _midgame_with_stones([(7, 4), (7, 5), (7, 6), (7, 8)], [(0, 0), (0, 1)])
        self.assertTrue(g.apply_move(7, 7))
        self.assertTrue(g.is_game_over())
        self.assertEqual(g.get_winner(), BLACK)

    def test_white_five_wins(self):
        g = _midgame_with_stones([(7, 4), (7, 5)], [(8, 4), (8, 5), (8, 6), (8, 8)], move_count=5)
        self.assertTrue(g.apply_move(8, 7))
        self.assertTrue(g.is_game_over())
        self.assertEqual(g.get_winner(), WHITE)

    def test_mcts_compatible_interface(self):
        g = RenjuGame()
        g.resolve_opening_to_play()
        valid = g.get_valid_moves()
        self.assertEqual(valid.shape, (229,))
        self.assertEqual(int(valid.sum()), len(g.legal_moves()))
        enc = g.get_encoded_state()
        self.assertEqual(enc.shape, (4, 15, 15))
        clone = g.clone()
        self.assertEqual(clone.phase, g.phase)
        self.assertEqual(g.action_to_move(g.move_to_action((7, 7))), (7, 7))


class RenjuGameActionSpaceTest(unittest.TestCase):
    def test_action_size_229(self):
        self.assertEqual(RenjuGame().action_size, 229)

    def test_swap_phase_mask(self):
        g = RenjuGame()
        g.apply_move(7, 7)  # S1_SWAP
        valid = set(np.nonzero(g.get_valid_moves())[0].tolist())
        self.assertEqual(valid, {SWAP_ACCEPT, SWAP_REJECT})

    def test_variant_phase_mask(self):
        g = RenjuGame()
        g.apply_move(7, 7)
        g.apply_swap(False)
        g.apply_move(6, 6)
        g.apply_swap(False)
        g.apply_move(7, 6)
        g.apply_swap(False)
        g.apply_move(6, 7)  # VARIANT_CHOICE
        valid = set(np.nonzero(g.get_valid_moves())[0].tolist())
        self.assertEqual(valid, {VARIANT_1, VARIANT_2})

    def test_do_move_decision_actions(self):
        g = RenjuGame()
        g.apply_move(7, 7)
        self.assertTrue(g.do_move(SWAP_ACCEPT))
        self.assertEqual(g.black_owner, 2)

    def test_encoding_phase_plane(self):
        g = RenjuGame()
        self.assertEqual(g.get_encoded_state().shape, (4, 15, 15))
        self.assertEqual(g.get_encoded_state()[3, 0, 0], 0.0)  # move 阶段 phase
        self.assertEqual(g.get_encoded_state()[2, 0, 0], 0.0)  # 开局无禁手
        g.apply_move(7, 7)  # swap 阶段
        self.assertEqual(g.get_encoded_state()[3, 0, 0], 0.5)

    def test_encoding_forbidden_channel(self):
        # 黑方轮到且禁手启用 → 通道2=1.0；白方轮到 → 通道2=0.0
        g = _midgame_with_stones([(7, 4), (7, 5), (7, 6)], [(0, 0), (0, 1)], move_count=6)
        self.assertEqual(g.current_player, BLACK)
        self.assertTrue(g.forbidden_enabled())
        self.assertEqual(g.get_encoded_state()[2, 0, 0], 1.0)
        g.current_player = WHITE
        self.assertEqual(g.get_encoded_state()[2, 0, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
