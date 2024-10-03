from sim import sim as sim
from tools.pick_six import generate_vgc_team   
from model import PokemonDQN

# Pokemon Battle AI training with Battle class integration


def pokemon_battle_training(battle_class, episodes=1000, team_1=None, team_2=None):
    state_size = 128  # Define according to your battle state encoding
    action_size = 10  # Define according to available actions (moves, switches)
    
    agent_1 = PokemonDQN(state_size=state_size, action_size=action_size)
    agent_2 = PokemonDQN(state_size=state_size, action_size=action_size)
    
    for e in range(episodes):
        # Create a battle object
        battle = battle_class(doubles=True, debug=False)

        # Join teams to the battle
        battle.join(side_id=0, team=team_1)
        battle.join(side_id=1, team=team_2)

        state_1 = encode_battle_state(battle, 0)  # Encode battle state for side 0
        state_2 = encode_battle_state(battle, 1)  # Encode battle state for side 1
        
        done = False
        turn = 0
        
        while not done:
            # AI 1 makes a move
            action_1 = agent_1.act(state_1)
            decision_1 = create_decision(action_1)  # Create move/switch decision for AI 1
            battle.choose(0, decision_1)

            # AI 2 makes a move
            action_2 = agent_2.act(state_2)
            decision_2 = create_decision(action_2)  # Create move/switch decision for AI 2
            battle.choose(1, decision_2)
            
            # Execute turn
            battle.do_turn()
            
            # Update state for both sides
            next_state_1 = encode_battle_state(battle, 0)
            next_state_2 = encode_battle_state(battle, 1)
            
            # Check for rewards and game status
            reward_1, reward_2, done = evaluate_battle_outcome(battle)

            # Remember the experience
            agent_1.remember(state_1, action_1, reward_1, next_state_1, done)
            agent_2.remember(state_2, action_2, reward_2, next_state_2, done)
            
            # Update the states for the next turn
            state_1 = next_state_1
            state_2 = next_state_2

            turn += 1
            if done:
                print(f"Episode: {e+1}/{episodes}, Turns: {turn}, Epsilon: {agent_1.epsilon:.2f}")
                break

        # Train both agents
        agent_1.replay()
        agent_2.replay()

    # Save models after training
    agent_1.save('pokemon_dqn_1.h5')
    agent_2.save('pokemon_dqn_2.h5')

# Helper function to encode the battle state into a usable format for the model
def encode_battle_state(battle, side_id):
    # Example placeholder for state encoding logic
    # Include team info, current Pokémon, HP, moves, etc.
    state = np.random.random(128)  # Replace with actual encoding logic
    return state

# Helper function to evaluate the battle outcome and return rewards
def evaluate_battle_outcome(battle):
    # Example: Evaluate based on remaining Pokémon, damage dealt, etc.
    reward_1 = np.random.randint(-10, 10)  # Replace with actual reward logic
    reward_2 = np.random.randint(-10, 10)
    done = False  # Replace with actual game over condition
    return reward_1, reward_2, done

# Helper function to create a decision (move/switch) based on action
def create_decision(action):
    # Example placeholder: action 0 is a move, action 1 is a switch, etc.
    if action == 0:
        return data.dex.Decision('move', 0)  # Example: Move 1
    elif action == 1:
        return data.dex.Decision('switch', 0)  # Example: Switch to Pokémon 1
    # Add other actions as necessary
    return data.dex.Decision('move', 0)  # Default move action

# Example usage: simulate Pokémon battles and train DQNs
team_1 = {  # JSON representation of team 1
    "name": "Team 1",
    "pokemon": [
        {"species": "pikachu", "moves": ["thunderbolt", "quickattack", "iron_tail", "thunder"], "evs": [0, 252, 0, 0, 4, 252], "ability": "static", "item": "lightball"},
        # Add more Pokémon to the team...
    ]
}
team_2 = {  # JSON representation of team 2
    "name": "Team 2",
    "pokemon": [
        {"species": "charizard", "moves": ["flamethrower", "dragonclaw", "airslash", "roost"], "evs": [0, 252, 0, 0, 4, 252], "ability": "blaze", "item": "charizardite_x"},
        # Add more Pokémon to the team...
    ]
}

# Replace BattleClass with the actual battle class used for simulations
pokemon_battle_training(Battle, episodes=1000, team_1=team_1, team_2=team_2)