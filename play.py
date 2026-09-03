# play.py

import sys
import importlib
import time

from games import get_game_class

# =========================================================================== #
#               Para jogar usando o play.py digite no terminal:               #
#                  python play.py player_human player_human                   #
# =========================================================================== #

RED = "\033[31m"
BLUE = "\033[34m"
RESET = "\033[0m"

# ====== DYNAMIC PLAYER LOADING ======
def load_player(module_name, rules, size):
    # aceita 'player_mcts' ou 'player_mcts.py'
    module_name = module_name.replace(".py", "").strip()
    if not module_name.startswith("players."):
        module_name = f"players.{module_name}"

    module = importlib.import_module(module_name)

    if hasattr(module, "Player"):
        return module.Player(rules, size)

    raise ValueError(f"Nenhuma classe Player encontrada em {module_name}")


# ====== GAME MENU ======
def choose_game(argv):
    # 支持命令行指定规则：python play.py p1 p2 [gomoku|renju|pente]
    if len(argv) >= 4 and argv[3].lower() in ("gomoku", "renju", "pente"):
        return argv[3].lower()
    return "gomoku"

# ====== MAIN ======
def main():
    if len(sys.argv) < 3:
        print("Uso: python play.py <player1> <player2> [gomoku|renju|pente]")
        print("Exemplo: python play.py player_human player_mcts renju")
        sys.exit(1)

    player1_name, player2_name = sys.argv[1:3]

    # Menu interativo
    game_name = choose_game(sys.argv)
    size = 15

    # Inicializa o jogo
    game = get_game_class(game_name)(size)
    if hasattr(game, "resolve_opening_to_play"):
        game.resolve_opening_to_play()

    # Carrega os jogadores
    player1 = load_player(player1_name,game_name,size)
    player2 = load_player(player2_name,game_name,size)
    players = {1: player1, 2: player2}

    print(f"\n🎮 Iniciando {game_name.capitalize()}")
    print(f"{RED}●{RESET} Jogador 1: {player1_name}")
    print(f"{BLUE}●{RESET} Jogador 2: {player2_name}\n")

    game.display()

    turn_number = 0

    # Loop principal
    while not game.is_game_over():
        turn_number += 1
        player = players[game.current_player]
        # 连珠走法二的十打点/选点：由当前玩家模型贪心解析（原地推进）
        if hasattr(player, 'resolve_opening'):
            player.resolve_opening(game)
            if game.is_game_over():
                break

        valid_move = False
        while not valid_move:
            start_time = time.time()
            try:
                move = player.play(game.clone(), turn_number, game.last_move)
            except Exception as e:
                print(f"Erro no jogador {game.current_player}: {e}")
                continue

            think_time = time.time() - start_time
            print(f"⏱️  Tempo de decisão: {think_time:.2f}s")

            if move is None:
                print("Jogador não devolveu jogada. Tente novamente.")
                continue 

            try:
                game.do_move(move)
                valid_move = True
            except ValueError as e:
                print(f"Jogada inválida: {e}")

        game.display()  # mostra o tabuleiro após cada jogada válida

    # Certifica que o último estado do tabuleiro é mostrado
    print("\nEstado final do tabuleiro:")
    game.display()

    # Mensagem de vitória

    winner = game.get_winner()
    if winner == 0:
        print("\nEmpate! Nenhum vencedor.")
    else:
        if winner == 1:
            piece = f"{RED}●{RESET}"
        else:
            piece = f"{BLUE}●{RESET}"

        print(f"\n🏆 Jogador {winner} ({piece}) venceu!")


if __name__ == "__main__":
    main()