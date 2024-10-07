import sim.sim as sim
from tools.pick_six import generate_vgc_team, generate_team
from src.model import PokemonDQN
import numpy as np
import sim.pokemon as Pokemon
from sim.player import *
from src.train import encode_battle_state, create_decision, evaluate_battle_outcome
import argparse

from util.parser import parse_pokemon_file

# Pokemon Battle AI training with Battle class integration


def compare(episodes=100, team_1=None, team_2=None, model1=None, model2=None):

    print(team_1)
    print(team_2)
    
    agent_1 = PokemonDQN()
    agent_2 = PokemonDQN()

    agent_1.epsilon = 0
    agent_2.epsilon = 0
    

    if model1 is not None:
        agent_1.model.load_weights(model1)
    if model2 is not None:
        agent_2.model.load_weights(model2)
    win_loss = 0
    for e in range(episodes):
        # Create a battle object
        battle = sim.Battle('single', 'Red', team_1, 'Blue', team_2, debug=False)

        state_1 = encode_battle_state(battle)  # Encode battle state for side 0
        state_2 = encode_battle_state(battle)  # Encode battle state for side 1
        
        turn = 0


        
        print("Battle " + str(e) + " started")
        while not battle.ended:
            # AI 1 makes a move
            action_1 = agent_1.act(state_1)
            decision_1 = create_decision(battle.p1, action_1)  # Create move/switch decision for AI 1

            # AI 2 makes a move
            action_2 = agent_2.act(state_2)
            decision_2 = create_decision(battle.p2, action_2)  # Create move/switch decision for AI 2
            
            # Execute turn
            sim.do_turn(battle)

            turn += 1
            if turn > 100:
                print(state_1)
                sys.exit("Battle reached max number of allowed turns")

            if battle.ended:
                print(battle.winner)
                win_loss += int(battle.winner == "p1")
                print("WR:", win_loss / (e + 1))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Simulate a Pokémon battle between two teams.')

    # Add arguments for team_1_path, team_2_path, model_path, and episodes
    parser.add_argument('team_1_path', type=str, help='Path to the first team\'s Pokémon file', default='examples/henry.txt')    
    parser.add_argument('team_2_path', type=str, help='Path to the second team\'s Pokémon file', default='examples/jasper.txt')
    parser.add_argument('team_1_model', type=str, help='Path to the first team\'s model file', default='out/red_dqn.keras')    
    parser.add_argument('team_2_model', type=str, help='Path to the second team\'s model file', default='out/blue_dqn.keras')
    parser.add_argument('games', type=int, help='Number of games to play', default=10)

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    team_1_path = args.team_1_path
    team_2_path = args.team_2_path
    team_1_model = args.team_1_model
    team_2_model = args.team_2_model
    games = args.games

    team_1 = parse_pokemon_file(team_1_path)

    team_2 = parse_pokemon_file(team_2_path)

    compare(team_1=team_1, team_2=team_2, model1=team_1_model, model2=team_2_model, episodes=games)