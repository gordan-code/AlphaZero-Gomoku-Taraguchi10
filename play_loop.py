import sys
import importlib
import time
import random
import json
import numpy as np
import os
from pathlib import Path

from games import get_game_class

METRICS = Path("metrics")
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
def choose_game(argv=None):
    # 支持命令行指定规则：python play_loop.py p1 p2 n [gomoku|renju|pente]
    if argv and len(argv) >= 5 and argv[4].lower() in ("gomoku", "renju", "pente"):
        return argv[4].lower()
    return "gomoku"

def change_starting_player(player1_name, player2_name, game, game_name, size, metrics, game_iter):
    # Carrega os jogadores
    player1 = load_player(player1_name,game_name,size)
    player2 = load_player(player2_name,game_name,size)
    metrics["starting_player_per_game"][f"game_{game_iter}"] = player1_name
    players = {1: player1, 2: player2}

    print(f"\n🎮 Iniciando {game_name.capitalize()}")
    print(f"{RED}●{RESET} Jogador 1: {player1_name}")
    print(f"{BLUE}●{RESET} Jogador 2: {player2_name}\n")

    turn_number = 0

    if hasattr(game, "resolve_opening_to_play"):
        # 连珠：用确定性策略走完塔拉山口-10 开局
        game.resolve_opening_to_play()
    else:
        r = random.randint(0, 14)
        c = random.randint(0, 14)
        game.do_move((r, c))
        metrics["move_made"][player1_name][f"game_{game_iter}"].append((r,c))
        metrics["time_for_each_move"][player1_name][f"game_{game_iter}"].append(0)
    game.display()

    # Loop principal
    while not game.is_game_over():
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
                think_time = time.time() - start_time

                if game.current_player == 1:
                    metrics["move_made"][player1_name][f"game_{game_iter}"].append(move)
                    metrics["time_for_each_move"][player1_name][f"game_{game_iter}"].append(think_time)
                else:
                    metrics["move_made"][player2_name][f"game_{game_iter}"].append(move)
                    metrics["time_for_each_move"][player2_name][f"game_{game_iter}"].append(think_time)

                print(f"⏱️  Tempo de decisão: {think_time:.2f}s")
            except Exception as e:
                print(f"Erro no jogador {game.current_player}: {e}")
                continue

            if move is None:
                print("Jogador não devolveu jogada. Tente novamente.")
                continue 

            try:
                game.do_move(move)
                turn_number += 1
                valid_move = True
            except ValueError as e:
                print(f"Jogada inválida: {e}")

        game.display()  # mostra o tabuleiro após cada jogada válida

    # Certifica que o último estado do tabuleiro é mostrado
    print("\nEstado final do tabuleiro:")
    game.display()

    # Mensagem de vitória

    winner = game.get_winner()
    winner_name = None
    if winner == 0:
        print("\nEmpate! Nenhum vencedor.")
    else:
        if winner == 1:
            piece = f"{RED}●{RESET}"
            winner_name = player1_name
        else:
            piece = f"{BLUE}●{RESET}"
            winner_name = player2_name

        print(f"\n🏆 Jogador {winner} ({winner_name}) ({piece}) venceu!")

    return winner_name

def initiate_metrics(player1_name, player2_name, player1_obj, player2_obj, game_name, n_games):
    metrics = {}
    metrics["total_duration"] = 0

    player1_mcts_simulations = 0
    player2_mcts_simulations = 0
    player1_model_path = ""
    player2_model_path = ""

    if hasattr(player1_obj, "n_simulations"):
        player1_mcts_simulations = player1_obj.n_simulations
    elif hasattr(player1_obj, "n_playout"):
        player1_mcts_simulations = player1_obj.n_playout
    else:
        player1_mcts_simulations = None

    if hasattr(player2_obj, "n_simulations"):
        player2_mcts_simulations = player2_obj.n_simulations
    elif hasattr(player2_obj, "n_playout"):
        player2_mcts_simulations = player2_obj.n_playout
    else:
        player2_mcts_simulations = None

    if hasattr(player1_obj, "model_path"):
        player1_model_path = player1_obj.model_path
    else:
        player1_model_path = None

    if hasattr(player2_obj, "model_path"):
        player2_model_path = player2_obj.model_path
    else:
        player2_model_path = None

    metrics["player1"] = (player1_name, player1_mcts_simulations, player1_model_path)
    metrics["player2"] = (player2_name, player2_mcts_simulations, player2_model_path)
    metrics["game"] = game_name
    metrics["n_games"] = n_games
    metrics["total_duration_minutes"] = 0
    
    metrics["move_made"] = {player1_name: {}, player2_name: {}}
    metrics["time_for_each_move"] = {player1_name: {}, player2_name: {}}
    metrics["game_duration_seconds"] = {}
    for player in metrics["time_for_each_move"].keys():
        for i in range(1, n_games+1):
            metrics["move_made"][player][f"game_{i}"] = list()
            metrics["time_for_each_move"][player][f"game_{i}"] = list()
            metrics["game_duration_seconds"][f"game_{i}"] = 0

    metrics["wins"] = {}
    metrics["draws"] = 0
    metrics["starting_player_per_game"] = {}
    for i in range(1, n_games+1):
        metrics["starting_player_per_game"][f"game_{i}"] = None

    return metrics

def to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_safe(x) for x in obj]
    elif isinstance(obj, tuple):
        return [to_json_safe(x) for x in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def loop_for_n_games():
    # 第 4 个参数可选：规则名（gomoku/renju/pente），与 play.py 保持一致
    if len(sys.argv) < 4:
        print("Uso: python play_loop.py <player1> <player2> <n_games> [gomoku|renju|pente]")
        print("Exemplo: python play_loop.py player_human player_mcts 50 renju")
        sys.exit(1)

    player1_name, player2_name, n_games = sys.argv[1:4]
    n_games = int(n_games)

    # Menu interativo
    game_name = choose_game(sys.argv)
    size = 15

    player1 = load_player(player1_name,game_name,size)
    player2 = load_player(player2_name,game_name,size)

    wins = {player1_name: 0, player2_name: 0}
    metrics = initiate_metrics(player1_name, player2_name, player1, player2, game_name, n_games)

    start_time = time.time()

    # Inicializa o jogo
    for i in range(n_games):
        game = get_game_class(game_name)(size)
        
        if i % 2 == 0:
            game_start_time = time.time()
            winner = change_starting_player(player1_name, player2_name, game, game_name, size, metrics, i+1)
            metrics["game_duration_seconds"][f"game_{i+1}"] = time.time() - game_start_time
            if winner:
                wins[winner] += 1
            print(f"Finished game: {i+1}/{n_games}")
            time.sleep(3)
        else:
            game_start_time = time.time()
            winner = change_starting_player(player2_name, player1_name, game, game_name, size, metrics, i+1)
            metrics["game_duration_seconds"][f"game_{i+1}"] = time.time() - game_start_time
            if winner:
                wins[winner] += 1
            print(f"Finished game: {i+1}/{n_games}")
            time.sleep(3)

    metrics["total_duration_minutes"] = (time.time() - start_time)//60
    print(f"This loop took: {(time.time() - start_time)//60} minutes!")

    metrics["wins"] = wins
    metrics["draws"] = metrics["n_games"] - sum(wins.values())
    for key, value in wins.items():
        print(f"{key} won {value} times")

    player1_mcts_simulations = metrics["player1"][1]
    player2_mcts_simulations = metrics["player2"][1]

    metrics_filename = f"{player1_name}_{player1_mcts_simulations}_{player2_name}_{player2_mcts_simulations}_3.json"
    save_path = METRICS / metrics_filename
    with open(save_path, "w") as f:
        json.dump(to_json_safe(metrics), f, indent=4)

if __name__ == "__main__":
    os.makedirs("metrics", exist_ok=True)
    loop_for_n_games()
    