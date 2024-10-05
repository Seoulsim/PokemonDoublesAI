import sim.sim as sim
from tools.pick_six import generate_vgc_team, generate_team
from src.model import PokemonDQN
import numpy as np
import sim.pokemon as Pokemon

# Pokemon Battle AI training with Battle class integration


def pokemon_battle_training(battle_class, episodes=1000, team_1=None, team_2=None):
    state_size = 128  # Define according to your battle state encoding
    action_size = 10  # Define according to available actions (moves, switches)
    
    agent_1 = PokemonDQN(state_size=state_size, action_size=action_size)
    agent_2 = PokemonDQN(state_size=state_size, action_size=action_size)
    
    for e in range(episodes):
        # Create a battle object
        battle = battle_class(doubles=False, debug=False)

        # Join teams to the battle
        battle.join(side_id=0, team=team_1)
        battle.join(side_id=1, team=team_2)

        state_1 = encode_battle_state(battle)  # Encode battle state for side 0
        state_2 = encode_battle_state(battle)  # Encode battle state for side 1
        
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
            next_state_1 = encode_battle_state(battle)
            next_state_2 = encode_battle_state(battle)
            
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
def encode_battle_state(battle: sim.Battle):
    # Example placeholder for state encoding logic
    # Include team info, current Pokémon, HP, moves, etc.

    # Encoding by index:

    # 1: active pokemon player 1
    # 2: active pokemon player 2
    # 3 - 13: Possible field conditions (weather, terrain, spikes, trickroom, etc)
    # pokemon stats = hp, attack, defense, special attack, special defense, speed, accuracy, evasion * 12 pokemon = 
    # 14 - 110: pokemon current stats accounting for boosts in the order of maxhp, current hp, attack, defense, special attack, special defense, speed, accuracy, evasion
    # DONT ACTUALLY NEED TO ENCODE ALL OF THIS IF WE'RE TRAINING FOR ONE SPECIFIC MATCHUP
    # 
    # Can be simplified to only use the active pokemon and their stats + the hp of pokemon on bench
    # ignore PP for simplicity
    # pokemon stats = curret hp, attack, defense, special attack, special defense, speed, accuracy, evasion * 2 + current hp of bench pokemon = 10 + 16 = 26
    # 14 - 40: pokemon stats + bench stats

    state = {}

    state["active_pokemon_1"] = battle.p1.pokemon.index(battle.p1.active_pokemon[0])  # Index of active pokemon in battle.p1.active_pokemon
    state["active_pokemon_2"] = battle.p2.pokemon.index(battle.p2.active_pokemon[0])

    weather = ["clear","sunny", "rainy", "sandstorm", "hail"]
    state["weather"] = weather.index(battle.weather)
    state["weather_turns"] = battle.weather_n

    terrains = ["","grassy", "electric", "misty", "psychic"]
    state["terrain"] = terrains.index(battle.terrain)


    state["p1_spikes"] = int(battle.p1.spikes)
    state["p1_toxic_spikes"] = int(battle.p1.toxic_spikes)
    state["p1_stealth_rock"] = int(battle.p1.stealth_rock)
    state["p1_sticky_web"] = int(battle.p1.sticky_web)
    state["p1_tailwind"] = int(battle.p1.tailwind)
    state["p1_tailwind_n"] = battle.p1.tailwind_n

    state["p2_spikes"] = int(battle.p2.spikes)
    state["p2_toxic_spikes"] = int(battle.p2.toxic_spikes)
    state["p2_stealth_rock"] = int(battle.p2.stealth_rock)
    state["p2_sticky_web"] = int(battle.p2.sticky_web)
    state["p2_tailwind"] = int(battle.p2.tailwind)
    state["p2_tailwind_n"] = battle.p2.tailwind_n

    state["trick_room"] = int(battle.trickroom)
    state["trick_room_turns"] = battle.trickroom_n

    pokemon_statuses = ["burn", "poison", "paralysis", "freeze", "sleep", ""]

    active_pokemons = battle.p1.active_pokemon
    for p in active_pokemons:
        for field in dir(p):
            if not field.startswith('__'):
                if type(field) == int:
                    state[field] = getattr(p, field)
                elif type(field) == bool:
                    state[field] = int(getattr(p, field))

        state["pokemon_1_attack"] = Pokemon.get_attack(p, weather = battle.weather, crit = False)
        state["pokemon_1_defense"] = Pokemon.get_defense(p, terrain = battle.terrain, crit = False)
        state["pokemon_1_specialattack"] = Pokemon.get_specialattack(p, weather = battle.weather, crit = False)
        state["pokemon_1_specialdefense"] = Pokemon.get_specialdefense(p, weather = battle.weather, crit = False)
        state["pokemon_1_speed"] = Pokemon.get_speed(p, weather = battle.weather, terrain = battle.terrain, trickroom = battle.trickroom, tailwind = battle.p1.tailwind)
        state["pokemon_1_accuracy"] = Pokemon.get_accuracy(p)
        state["pokemon_1_evasion"] = Pokemon.get_evasion(p)

    active_pokemons = battle.p2.active_pokemon
    for p in active_pokemons:
        for field in dir(p):
            if not field.startswith('__'):
                if type(field) == int:
                    state[field] = getattr(p, field)
                elif type(field) == bool:
                    state[field] = int(getattr(p, field))

        state["pokemon_2_attack"] = Pokemon.get_attack(p, weather = battle.weather, crit = False)
        state["pokemon_2_defense"] = Pokemon.get_defense(p, terrain = battle.terrain, crit = False)
        state["pokemon_2_specialattack"] = Pokemon.get_specialattack(p, weather = battle.weather, crit = False)
        state["pokemon_2_specialdefense"] = Pokemon.get_specialdefense(p, weather = battle.weather, crit = False)
        state["pokemon_2_speed"] = Pokemon.get_speed(p, weather = battle.weather, terrain = battle.terrain, trickroom = battle.trickroom, tailwind = battle.p2.tailwind)
        state["pokemon_2_accuracy"] = Pokemon.get_accuracy(p)
        state["pokemon_2_evasion"] = Pokemon.get_evasion(p)

    for pokemon in battle.p1.bench:
        state["pokemon_1_bench_" + str(battle.p1.bench.index(pokemon))] = pokemon.hp
        state["pokemon_1_status"] = pokemon_statuses.index(pokemon.status)

    for pokemon in battle.p2.bench:
        state["pokemon_2_bench_" + str(battle.p2.bench.index(pokemon))] = pokemon.hp
        state["pokemon_2_status"] = pokemon_statuses.index(pokemon.status)

    print(state.values())
    print(len(state))
    
    return state

# Helper function to evaluate the battle outcome and return rewards
def evaluate_battle_outcome(battle):
    # Example: Evaluate based on remaining Pokémon, damage dealt, etc.
    hp1_heuristic = 0  # Replace with actual reward logic
    hp2_heuristic = 0
    done = battle.ended  # Replace with actual game over condition
    
    for pokemon in battle.p1.pokemon:
        if pokemon.hp > 0:
            hp1_heuristic += 1
            hp1_heuristic += pokemon.hp / pokemon.maxhp
            if pokemon.status != '':
                # TODO: this should have different rewards for different statuses
                hp2_heuristic += 0.5*pokemon.hp / pokemon.maxhp

    for pokemon in battle.p2.pokemon:
        if pokemon.hp > 0:
            hp2_heuristic += 1
            hp2_heuristic += pokemon.hp / pokemon.maxhp
            if pokemon.status != '':
                hp1_heuristic += 0.5*pokemon.hp / pokemon.maxhp

    reward_1 = hp1_heuristic - hp2_heuristic
    reward_2 = hp2_heuristic - hp1_heuristic
    if done:
        reward_1 += 10 if battle.winner == battle.name1 else -10
        reward_2 += 10 if battle.winner == battle.name2 else -10
    return reward_1, reward_2, done

# Helper function to create a decision (move/switch) based on action
def create_decision(action):
    # Example placeholder: action 0 is a move, action 1 is a switch, etc.

    pass

# pokemon_battle_training(Battle, episodes=1000, team_1=team_1, team_2=team_2)

teams = []
for i in range(2):
    teams.append(sim.dict_to_team_set(generate_team()))

battle = sim.Battle('single', 'Red', teams[0], 'Blue', teams[1], debug=True)

encode_battle_state(battle)