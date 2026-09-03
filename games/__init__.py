# -*- coding: utf-8 -*-
"""游戏注册表：按规则名返回对应的游戏类，供 players / train / play 复用。"""


def get_game_class(rules: str):
    """根据规则名返回游戏类。

    支持：``gomoku`` / ``pente`` / ``renju``（塔拉山口-10，含禁手与开局状态机）。
    """
    rules = (rules or "gomoku").lower()
    if rules == "gomoku":
        from games.gomoku import Gomoku
        return Gomoku
    if rules == "pente":
        from games.pente import Pente
        return Pente
    if rules == "renju":
        from games.renju.game import RenjuGame
        return RenjuGame
    raise ValueError(f"Unsupported rules: {rules}")


__all__ = ["get_game_class"]
