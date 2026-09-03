# -*- coding: utf-8 -*-
"""Renju 规则跨语言交叉验证（Python 侧）。

加载 games/renju/renju_test_cases.json，用本仓库 rule.py 跑出结果，
并与 JSON 中的 expect 比对。同一份 JSON 也会被 taraguchi10-gomoku 的 TS
测试加载，从而保证 TS 与 Python 输出完全一致。

运行方式（在 AlphaZero-Gomoku 根目录）：
    python -m unittest tests.test_renju_rule -v
    或：pytest tests/test_renju_rule.py -v
"""

import json
import sys
import unittest
from pathlib import Path

# 确保从任意 cwd 运行时都能 import `games` 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.renju.rule import BLACK, EMPTY, WHITE, Renju_Rule

CASES_PATH = Path(__file__).resolve().parent.parent / "games" / "renju" / "renju_test_cases.json"
CASES = json.loads(CASES_PATH.read_text(encoding="utf-8"))


def build_board(size, stones):
    board = [[EMPTY] * size for _ in range(size)]
    for r, c, s in stones:
        board[r][c] = s
    return board


def run_case(case):
    size = CASES["size"]
    board = build_board(size, case["stones"])
    rule = Renju_Rule(board, size)
    r, c = case["move"]
    color = case["move_color"]

    if case["type"] == "forbidden":
        if color != BLACK:
            return "legal"  # 白方无禁手
        board[r][c] = BLACK
        kind = rule.check_forbidden(r, c)
        board[r][c] = EMPTY
        return kind if kind is not None else "legal"

    # type == "win"
    board[r][c] = color
    length = rule.run_length(r, c, color)
    if color == BLACK:
        if length == 5:
            return "black-five"
        if length >= 6:
            return "black-over"
        return "none"
    if length == 5:
        return "white-five"
    if length >= 6:
        return "white-overline"
    return "none"


class RenjuRuleCrossTest(unittest.TestCase):
    def test_all_cases(self):
        self.assertTrue(len(CASES["cases"]) >= 30, "用例数量不足 30")
        for case in CASES["cases"]:
            with self.subTest(id=case["id"]):
                got = run_case(case)
                self.assertEqual(
                    got,
                    case["expect"],
                    f"{case['id']} ({case['desc']}) 期望 {case['expect']} 实得 {got}",
                )


if __name__ == "__main__":
    unittest.main()
