from game import ChessGame
from neural_network import NeuralNetwork
"""
-----------
Description
-----------
This script lets you play against an AI or have the AI play against others. After each game, the players switch colors (white and black). 
Choose from these player types: "real", "ai", "random", "minimax", "ai+minimax2", or '50_minimax_50_random'.

For "ai" or "ai+minimax2", specify the model name and decide whether to load the best checkpoint (default loads the newest). 
You can also adjust the AI's randomness (0 is deterministic, 1 is random).

For "minimax", set the search depth as an integer.

To visualize, use `use_window=True` to show the game with a time delay between moves. 
To run many games without visualization, set `use_window=False` and specify the number of games. Results will be shown at the end.

-----------
Examples
-----------
1. Play against the best AI with a depth of 2:
player1 = "ai+minimax2", player2 = "real", MODEL_NAME_1 = "pretrained", load_from_best_checkpoint_1 = True,
use_window = True, time_delay = 0.3

2. Play against the best AI with deterministic decisions:
player1 = "ai", player2 = "real", MODEL_NAME_1 = "pretrained", load_from_best_checkpoint_1 = True,
ai_decision_temperature = 0, use_window = True, time_delay = 0.3

3. AI plays 100 games against minimax with depth 2 (no visualization):
player1 = "ai", player2 = "minimax", MODEL_NAME_1 = "pretrained", load_from_best_checkpoint_1 = True,
ai_decision_temperature = 0, search_depth = 2, use_window = False, n_games = 100
-----------
"""


# choose opponents from "real", "ai", "random", "minimax", "ai+minimax2", '50_minimax_50_random'
player1 = "ai"
player2 = '50_minimax_50_random'

# if a player is an AI, specify the model name as it was saved e.g. pretrained
# and specify if the best checkpoint should be loaded
MODEL_NAME_1 = "pretrained"
MODEL_NAME_2 = "pretrained"
load_from_best_checkpoint_1 = True
load_from_best_checkpoint_2 = True

# specify ai decision temperature (0 is deterministic, 1 is more random) e.g. 0.1
ai_decision_temperature = 0.1
# if a player is minimax, then specify the search depth
search_depth = 2

# visualization
use_window = True
time_delay = 0.3
n_games = 10

print("player 1:", player1)
print("player 2:", player2)

nnet1, nnet2 = None, None
if "ai" in player1:
    nnet1 = NeuralNetwork(main_filepath="saves/" + MODEL_NAME_1 + "/ChessAi_", load_from_checkpoint=load_from_best_checkpoint_1)
if "ai" in player2:
    nnet2 = NeuralNetwork(main_filepath="saves/" + MODEL_NAME_2 + "/ChessAi_", load_from_checkpoint=load_from_best_checkpoint_2)

game = ChessGame(player_1_type=player1, player_2_type=player2, window=use_window, time_delay=time_delay,
            ai_decision_temperature=ai_decision_temperature, search_depth=search_depth, nnets=[nnet1, nnet2])
p1_wins, p2_wins, draws, _, average_game_length = game.gamePlay(n_games=n_games)

print("player 1 wins:", p1_wins, "\nplayer 2 wins:", p2_wins, "\ndraws:", draws, "\naverage game length:", average_game_length)