# gomoku.py
import numpy as np
from typing import List, Tuple, Optional

class Gomoku:
    """
    五子棋游戏的实现（默认15x15）。
    - 获胜条件：横、竖、斜方向连成5子
    - 内部状态说明：
        board : np.array (size x size)，0代表空，1代表玩家1，2代表玩家2
        current_player : 当前行动玩家，1或2
        move_history : 已执行落子的(r,c)坐标列表
        last_move : 上一步的坐标(r,c)，或None
    AlphaZero/MCTS相关说明:
    - action_size = size * size
    - get_valid_moves() 返回长度为 action_size 的二进制(float32)向量
    - action 索引 = r * size + c
    """

    def __init__(self, size: int = 15):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        self.current_player = 1
        self.move_history: List[Tuple[int, int]] = []
        self.last_move: Optional[Tuple[int, int]] = None

    # ----------------------
    # clonagem / cópia
    # ----------------------
    def clone(self) -> "Gomoku":
        """Retorna uma cópia profunda do jogo (útil para debugging / simulações isoladas)."""
        new_game = Gomoku(self.size)
        new_game.board = self.board.copy()
        new_game.current_player = int(self.current_player)
        new_game.move_history = list(self.move_history)
        new_game.last_move = None if self.last_move is None else tuple(self.last_move)
        return new_game

    # ----------------------
    # utilitários ação <-> movimento
    # ----------------------
    @property
    def action_size(self) -> int:
        return self.size * self.size

    def action_to_move(self, action: int) -> Tuple[int, int]:
        """Converte action index (0..size*size-1) para (r,c)."""
        r = action // self.size
        c = action % self.size
        return (r, c)

    def move_to_action(self, move: Tuple[int, int]) -> int:
        """将 (r,c) 坐标转换为 action 索引。"""
        r, c = move
        return int(r * self.size + c)

    # ----------------------
    # jogadas / undo
    # ----------------------
    def do_move(self, move) -> bool:
        """
        Executa uma jogada.
        - move: (r,c) 元组或扁平动作索引 0..size*size-1
        - retorna True se jogada for válida e aplicada, False caso contrário.
        """
        if isinstance(move, (tuple, list)):
            r, c = int(move[0]), int(move[1])
        else:
            r, c = divmod(int(move), self.size)
        if not (0 <= r < self.size and 0 <= c < self.size):
            return False
        if self.board[r, c] != 0:
            return False

        # aplica jogada
        self.board[r, c] = self.current_player
        self.move_history.append((r, c))
        self.last_move = (r, c)
        # troca jogador
        self.current_player = 3 - self.current_player
        return True

    def undo_move(self) -> None:
        """
        Desfaz a última jogada.
        - Reverte board, current_player e last_move.
        - Se não houver jogadas, não faz nada.
        """
        if not self.move_history:
            return

        last = self.move_history.pop()  # (r, c)
        r, c = last
        # limpa a casa
        self.board[r, c] = 0
        # reverte jogador (já tinha sido trocado em do_move)
        self.current_player = 3 - self.current_player
        # actualiza last_move para a nova última jogada (ou None)
        self.last_move = self.move_history[-1] if self.move_history else None

    # ----------------------
    # movimentos legais / válidos
    # ----------------------
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Lista de coordenadas vazias (útil para interface/humano)."""
        empties = np.where(self.board == 0)
        return list(zip(empties[0].tolist(), empties[1].tolist()))

    def has_legal_moves(self) -> bool:
        return np.any(self.board == 0)

    def get_valid_moves(self) -> np.ndarray:
        """
        返回一个二进制向量（float32），长度为 action_size。
        - 合法动作对应位置为 1.0，非法动作为 0.0。
        这个格式是网络和 MCTS 所期望的（策略掩码）。
        """
        valid = np.zeros(self.action_size, dtype=np.float32)
        empties = np.where(self.board == 0)
        # zip 返回 (r,c) 对
        for r, c in zip(empties[0], empties[1]):
            idx = self.move_to_action((int(r), int(c)))
            valid[idx] = 1.0
        return valid

    # ----------------------
    # estado / codificação para rede
    # ----------------------
    def get_state(self) -> np.ndarray:
        """Retorna uma cópia simples do tabuleiro (size x size)."""
        return self.board.copy()

    def get_encoded_state(self) -> np.ndarray:
        """
        Retorna o estado codificado para a rede AlphaZero: np.array [C, H, W]
        C = 3:
          - canal 0: peças do jogador 1 (1.0/0.0)
          - canal 1: peças do jogador 2 (1.0/0.0)
          - canal 2: turno (toda a matriz preenchida com 1.0 se current_player == 1, 0.0 caso contrário)
        Observação: isto corresponde à tua especificação inicial. Se quiseres
        perspectiva invariance (sempre 'eu' = canal0), avisa que eu mudo.
        """
        board = self.board
        size = self.size

        player = self.current_player
        opponent = 3 - player

        p_cur = (board == player).astype(np.float32)
        p_opp = (board == opponent).astype(np.float32)
        turn = np.full((size, size), 1.0, dtype=np.float32)  # opcionalmente 1.0 sempre

        return np.stack([p_cur, p_opp, turn], axis=0)

    # ----------------------
    # verificação de vitória (usa last_move para eficiência)
    # ----------------------
    def check_winner(self) -> int:
        """
        Verifica vencedor considerando apenas a última jogada (muito mais rápido).
        Retorna:
          0 -> sem vencedor
          1 -> jogador 1 venceu
          2 -> jogador 2 venceu
        """
        if self.last_move is None:
            return 0

        r, c = self.last_move
        player = int(self.board[r, c])
        if player == 0:
            return 0

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1

            # direção positiva
            nr, nc = r + dr, c + dc
            while 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr, nc] == player:
                count += 1
                nr += dr
                nc += dc

            # direção negativa
            nr, nc = r - dr, c - dc
            while 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr, nc] == player:
                count += 1
                nr -= dr
                nc -= dc

            if count >= 5:
                return player

        return 0

    def is_game_over(self) -> bool:
        """True se houver vencedor ou não houver jogadas legais."""
        return self.check_winner() != 0 or not self.has_legal_moves()

    def get_winner(self) -> int:
        """Retorna o vencedor (0 se não houver)."""
        return self.check_winner()

    # ----------------------
    # visualização (mantive a tua função)
    # ----------------------
    def display(self) -> None:
        """Mostra o tabuleiro no terminal, com cores e alinhamento."""
        RED = "\033[31m"   # vermelho
        BLUE = "\033[34m"  # azul
        RESET = "\033[0m"  # reset de cor

        print()

        # Cabeçalho de colunas (alinhado)
        col_numbers = "    " + " ".join(f"{i+1:2}" for i in range(self.size))
        print(col_numbers)

        # Corpo do tabuleiro
        for r in range(self.size):
            row = f"{r+1:2}  "  # número da linha + espaçamento
            for c in range(self.size):
                val = int(self.board[r, c])
                if val == 1:
                    row += f" {RED}●{RESET} "
                elif val == 2:
                    row += f" {BLUE}●{RESET} "
                else:
                    row += " - "
            print(row)

        print()
        current = "1" if self.current_player == 1 else "2"
        color_dot = f"{RED}●{RESET}" if self.current_player == 1 else f"{BLUE}●{RESET}"
        print(f"Jogador atual: {color_dot} (player {current})")
