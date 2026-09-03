import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from mcts.mcts_pure import MCTSGomoku
from games import get_game_class

class Player:

    def __init__(self, rules="gomoku", board_size=15, n_playout=25, c_puct=1.4):
        
        self.rules = rules.lower()
        self.board_size = board_size
        self.n_playout = n_playout
        self.game_class = get_game_class(self.rules)
        self.mcts = MCTSGomoku(n_playout=n_playout, c_puct=c_puct)

    def play(self, board, turn_number, last_opponent_move):

        game = self.game_class(size=self.board_size)

        # copia o estado atual do tabuleiro
        if isinstance(board, list):
            game.board = np.array(board, dtype=int)
        else:
            game.board = np.copy(board.board)

        # 连珠：复制完整开局状态，并在未进入中盘时先用确定性策略走完开局
        if hasattr(board, "black_owner"):
            game.black_owner = int(board.black_owner)
            game.phase = board.phase
            game.variant = board.variant
            game.offers = list(board.offers)
            game.move_history = list(board.move_history)
            game.current_player = int(board.current_player)
            game.resolve_opening_to_play()
        else:
            game.current_player = 1 if turn_number % 2 == 0 else 2
        game.last_move = last_opponent_move

        # procura a melhor jogada via MCTS
        move = self.mcts.get_move(game)
        return move