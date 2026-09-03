import numpy as np
import torch
from mcts.new_mcts_alpha import MCTS
from network import PyTorchModel
from games import get_game_class
from players import default_model_path

class Player:
    def __init__(self,
                 rules="gomoku",
                 board_size=15,
                 n_simulations=3000,
                 c_puct=1.0,
                 model_path=None,
                 nn_model=PyTorchModel):
        
        self.rules = rules.lower()
        self.board_size = board_size
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.model_path = model_path

        # 3) 先确定游戏类与动作空间（renju=229 / gomoku=225），再据此构建网络
        self.game_class = get_game_class(self.rules)
        self.action_size = self.game_class(size=self.board_size).action_size

        # 1) Criar o modelo neural (usa PyTorchModel)
        self.net = nn_model(board_size=self.board_size, action_size=self.action_size)

        # 未显式指定模型：按规则选目录（renju -> models_renju，其他 -> models）并取最新兼容快照
        if model_path is None:
            model_path = default_model_path(self.rules, board_size=self.board_size, action_size=self.action_size)
            if model_path is not None:
                print(f"[PlayerAlpha] 未指定 model_path，自动使用最新快照: {model_path}")

        if model_path is not None:
            print(f"[PlayerAlpha] Carregando o modelo: {model_path}")
            self.net.load(model_path)
        else:
            print("[PlayerAlpha] AVISO: Nenhum modelo encontrado. Usando pesos aleatórios!")

        # 2) Colocar o modelo em modo de avaliação
        self.net.net.eval()

        # 4) Criar o MCTS
        self.mcts = MCTS(
            game_class=self.game_class,
            n_simulations=self.n_simulations,
            nn_model=self.net,  # Passa o PyTorchModel para o MCTS
            cpuct=self.c_puct,
            dirichlet_alpha=0.03,
            epsilon=0.03,
            apply_dirichlet_n_first_moves=10,
            add_dirichlet_noise=False
        )

    # ------------------------------------------------------------
    #   JOGAR
    # ------------------------------------------------------------
    def resolve_opening(self, game):
        """连珠走法二的十打点/选点：用模型策略贪心原地推进。"""
        if hasattr(game, 'phase'):
            from games.renju.opening_greedy import resolve_greedy_opening_steps
            resolve_greedy_opening_steps(game, self.net)

    def play(self, board, turn_number, last_opponent_move):
        """
        Realiza uma jogada com base no estado atual do jogo, utilizando o MCTS e a rede neural para calcular a jogada.
        """
        # Cria o jogo
        game = self.game_class(size=self.board_size)

        # Copia o tabuleiro para o jogo
        if isinstance(board, list):
            game.board = np.array(board, dtype=int)
        else:
            game.board = np.copy(board.board)

        # 连珠：复制完整开局状态（black_owner / phase / variant / offers / move_history）
        if hasattr(board, "black_owner"):
            game.black_owner = int(board.black_owner)
            game.phase = board.phase
            game.variant = board.variant
            game.offers = list(board.offers)
            game.move_history = list(board.move_history)
            game.current_player = int(board.current_player)
        else:
            # 普通五子棋：根据手数推导当前玩家
            game.current_player = 1 if turn_number % 2 == 0 else 2

        # Último movimento
        game.last_move = last_opponent_move

        # 走法二的十打点/选点：策略贪心原地推进
        self.resolve_opening(game)

        # Executa o MCTS para obter a política (probabilidades das jogadas)
        pi = self.mcts.run(game,turn_number)

        # 返回扁平动作（含交换/走法决策动作 225..228）
        return int(np.argmax(pi))
