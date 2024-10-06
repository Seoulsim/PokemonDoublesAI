import sim.sim as sim
from tools.pick_six import generate_vgc_team, generate_team
from src.model import PokemonDQN
import numpy as np
import sim.pokemon as Pokemon
from sim.player import *

from util.parser import parse_pokemon_file

# Pokemon Battle AI training with Battle class integration


def start_battle(team_1=None, team_2=None, model_path   = None):

    agent_1 = PokemonDQN()
    agent_1.epsilon = 0
    # Create a battle object
    battle = sim.Battle('single', 'Red', team_1, 'Blue', team_2, debug=False)
    if model_path is not None:
        agent_1.model.load_weights(model_path)

    while True:
        # AI 1 makes a move
        p = input("Select Player\n")
        if p == "1":
            player = battle.p1
        else:
            player = battle.p2

        Choice = input("Set HP [0], Switch [2], Status [3], Terrain [4], Hazard [5], Weather [6], Trickroom [7], Incriment Turn [8], Change Stat [9], exit [e]\n")
        match Choice:
            case "0":
                print("Select Pokemon\n")
                print([p.name for p in player.pokemon])
                ind = input("")

                d = input("Enter Curr HP\n")
                damage( player.pokemon[int(ind)], -1000)
                damage( player.pokemon[int(ind)], int(d))

            case "2":
                ind = int(input("Select Pokemon\n"))
                print([p.name for p in player.pokemon])
                player.active_pokemon = [player.pokemon[ind]]

            case "3":
                ind = input("Select Pokemon\n")
                print([p.name for p in player.pokemon])
                s = input("Enter Status (brn, par, psn, slp, tox or nothing)")
                player.pokemon[int(ind)].status = s

            case "4":
                terrain = input("Select Terrain (grassy, electric, psychic, misty)\n")
                battle.terrain = terrain
            case "5":
                hazard = input("Select Hazard to Toggle: (spikes, toxic_spikes, stealth_rock, sticky_web, tailwind)\n")
                if hazard == "spikes":
                    player.spikes = not player.spikes
                elif hazard == "toxic_spikes":
                    player.toxic_spikes = not player.toxic_spikes
                elif hazard == "stealth_rock":
                    player.stealth_rock = not player.stealth_rock
                elif hazard == "sticky_web":    
                    player.sticky_web = not player.sticky_web
                elif hazard == "tailwind":
                    player.tailwind = not player.tailwind
                    turns_remaining = input("Enter number of turns remaining\n")
                    player.tailwind_n = int(turns_remaining)
            case "6":
                weather = input("Select Weather (sunny, rainy, sandstorm, hail, none)\n")
                battle.weather = weather
            case "7":
                turns = input("Enter number of turns remaining\n")
                player.trickroom_n = int(turns)
                if turns > 0:
                    player.trickroom = True
                else:
                    player.trickroom = False
            case "8":
                print("Incrementing Turn\n")
                battle.turn += 1
            case "9":
                print([p.name for p in player.pokemon])
                ind = input("Select Pokemon\n")
                
                stat = input("Select Stat to Change (atk, def, spa, spd, spe, evasion, accuracy)\n\n")
                mult = float(input("Enter Multiplier\n"))

                player.pokemon[int(ind)].boosts[stat] = mult
            case "e":
                print("Exiting")
                print("_______________________")
                break
            case _:
                print("Doing nothing")
                print("_______________________")

        state = encode_battle_state(battle)
        state = np.array(list(state.values()))
        print(state.shape)
        state = state.reshape(1, -1)
        action = agent_1.model.predict(state)
        print("_______________________")
        print("Moves")
        print(battle.p1.active_pokemon[0].moves)


        moves = 4
        # print(moves)
        # print(action)
        vec = action[0][:moves] - np.min(action[0][:moves])
        move_distribution = vec / np.sum(vec)

        rand_int_move = np.random.choice(moves, 1, p=move_distribution)

        vec = action[0][moves:] - np.min(action[0][moves:])
        switch_distribution = vec / np.sum(vec)

        rand_int_switch = np.random.choice(6, 1, p=switch_distribution)

        print("Model Prediction: Use move", rand_int_move)
        print("_______________________")
        print("Switches", [p.name for p in battle.p1.pokemon])
        print("Model Prediction: Switch", rand_int_switch)
        
        
    return


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

    pokemon_statuses = ["brn", "psn", "par", "frz", "slp", ""]

    active_pokemons = battle.p1.active_pokemon
    for p in active_pokemons:
        for field in dir(p):
            if not field.startswith('__'):
                if type(field) == int:
                    state[field] = getattr(p, field)
                elif type(field) == bool:
                    state[field] = int(getattr(p, field))

        state["pokemon_1_attack"] = get_attack(p, weather = battle.weather, crit = False)
        state["pokemon_1_defense"] = get_defense(p, terrain = battle.terrain, crit = False)
        state["pokemon_1_specialattack"] = get_specialattack(p, weather = battle.weather, crit = False)
        state["pokemon_1_specialdefense"] = get_specialdefense(p, weather = battle.weather, crit = False)
        state["pokemon_1_speed"] = get_speed(p, weather = battle.weather, terrain = battle.terrain, trickroom = battle.trickroom, tailwind = battle.p1.tailwind)
        state["pokemon_1_accuracy"] = get_accuracy(p)
        state["pokemon_1_evasion"] = get_evasion(p)

    active_pokemons = battle.p2.active_pokemon
    for p in active_pokemons:
        for field in dir(p):
            if not field.startswith('__'):
                if type(field) == int:
                    state[field] = getattr(p, field)
                elif type(field) == bool:
                    state[field] = int(getattr(p, field))

        state["pokemon_2_attack"] = get_attack(p, weather = battle.weather, crit = False)
        state["pokemon_2_defense"] = get_defense(p, terrain = battle.terrain, crit = False)
        state["pokemon_2_specialattack"] = get_specialattack(p, weather = battle.weather, crit = False)
        state["pokemon_2_specialdefense"] = get_specialdefense(p, weather = battle.weather, crit = False)
        state["pokemon_2_speed"] = get_speed(p, weather = battle.weather, terrain = battle.terrain, trickroom = battle.trickroom, tailwind = battle.p2.tailwind)
        state["pokemon_2_accuracy"] = get_accuracy(p)
        state["pokemon_2_evasion"] = get_evasion(p)

    for pokemon in battle.p1.bench:
        state["pokemon_1_bench_" + str(battle.p1.bench.index(pokemon))] = pokemon.hp
        state["pokemon_1_status"] = pokemon_statuses.index(pokemon.status)

    for pokemon in battle.p2.bench:
        state["pokemon_2_bench_" + str(battle.p2.bench.index(pokemon))] = pokemon.hp
        state["pokemon_2_status"] = pokemon_statuses.index(pokemon.status)
    
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
                # TODO: Update Para and Burn to be based off of the amount of speed and atk reduced
                if pokemon.status == "slp":
                    hp2_heuristic += 0.6 * pokemon.hp / pokemon.maxhp
                elif pokemon.status == "par":
                    hp2_heuristic += 0.3 * pokemon.hp / pokemon.maxhp
                elif pokemon.status == "psn":
                    hp2_heuristic += 0.1 * pokemon.hp / pokemon.maxhp
                elif pokemon.status == "brn":
                    hp2_heuristic += 0.3 * pokemon.hp / pokemon.maxhp
                elif pokemon.status == "frz":
                    hp2_heuristic += 0.6 * pokemon.hp / pokemon.maxhp

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
def create_decision(P:Player, action):
    # Action is a Q array of length 9 returned by the model
    # Example placeholder: action 0 is a move, action 1 is a switch, etc.
    # convert array of length 9 to a move where index 0 - 3 are moves and 4 - 8 are switches
    n = len(P.active_pokemon)

    for i in range(n):

        if P.request == 'switch':
            lst = [(P.pokemon[i], action[0][i + 4]) for i in range(len(P.pokemon))]
            sorted_pokemon = sorted(lst, key=lambda x: x[1], reverse=True)
            for pokemon, _ in sorted_pokemon:

                if pokemon.fainted == False:
                    # print(pokemon.position)
                    P.choice = Decision('switch', pokemon.position)
                    return 4 + pokemon.position

        elif P.request == 'pass':
            P.choice = Decision('pass', 0)
            return np.argmax(action[0])

        elif P.request == 'move':
            lst = [(P.active_pokemon[0].moves[i], action[i]) for i in range(n)]

            if len(P.active_pokemon[i].moves) > 0:
                moves = len(P.active_pokemon[i].moves)
                # print(moves)
                # print(action)
                vec = action[0][:moves] - np.min(action[0][:moves])
                move_distribution = vec / np.sum(vec)

                rand_int = np.random.choice(moves, 1, p=move_distribution)

                P.choice = Decision('move', rand_int[0])

                return rand_int[0]
            else:
                sys.exit('No moves!')
    return -1

if __name__ == "__main__":
    team_1 = parse_pokemon_file("examples/henry.txt")

    team_2 = parse_pokemon_file("examples/jasper.txt")
    args = sys.argv
    start_battle(team_1=team_1, team_2=team_2, model_path=args[1])



