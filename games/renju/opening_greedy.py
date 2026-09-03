# -*- coding: utf-8 -*-
"""走法二（十打点 / 十选一）的策略贪心解析。

不进 MCTS：黑方报价 = 策略网络 top-10（去中心对称）；白方选点 = 价值网络贪心。
"""

from typing import List, Tuple

from games.renju.game import N_POS_ACTIONS, V2_TEN_OFFER, V2_TEN_PICK, RenjuGame


def top10_offers(game: RenjuGame, pos_probs) -> List[Tuple[int, int]]:
    """按策略概率取前 10 个空点（跳过与已选点关于天元对称的点）。"""
    pts = []
    for r in range(game.size):
        for c in range(game.size):
            if game.board[r, c] == 0:
                pts.append((r, c, float(pos_probs[r * game.size + c])))
    pts.sort(key=lambda x: -x[2])
    chosen: List[Tuple[int, int]] = []
    for r, c, _ in pts:
        if len(chosen) >= 10:
            break
        if any(q[0] + r == 14 and q[1] + c == 14 for q in chosen):
            continue
        chosen.append((r, c))
    return chosen


def resolve_greedy_opening_steps(game: RenjuGame, model) -> None:
    """把走法二的十打点/十选一用策略贪心走完（报价 + 选点），原地推进对局。"""
    guard = 0
    while game.phase in (V2_TEN_OFFER, V2_TEN_PICK) and guard < 20:
        act = game.actor()
        if act is None:
            break
        if act[1] == 'offers':
            probs, _ = model.predict(game.get_encoded_state()[None])
            offers = top10_offers(game, probs[0][:N_POS_ACTIONS])
            if len(offers) < 10:
                break
            game.apply_offers(offers)
        elif act[1] == 'pick':
            best_i, best_v = 0, -float('inf')
            for i in range(len(game.offers)):
                g2 = game.clone()
                g2.apply_pick(i)
                _, v = model.predict(g2.get_encoded_state()[None])
                if v[0][0] > best_v:
                    best_v = v[0][0]
                    best_i = i
            game.apply_pick(best_i)
        guard += 1
