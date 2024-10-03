from typing import List, Dict
import sim.sim as sim
from tools.pick_six import generate_vgc_team, generate_team
from sim.structs import PokemonSet, Stats, Pokemon
import numpy as np

def evs_to_stats(evs: dict) -> Stats:
    # Extract the EV values, using 0 if a specific stat is not in the dictionary
    hp = evs.get('hp', 0)
    attack = evs.get('atk', 0)  # Assuming 'atk' is used for Attack
    defense = evs.get('def', 0)  # Assuming 'def' is used for Defense
    specialattack = evs.get('spa', 0)  # Assuming 'spa' is used for Special Attack
    specialdefense = evs.get('spd', 0)  # Assuming 'spd' is used for Special Defense
    speed = evs.get('spe', 0)  # Assuming 'spe' is used for Speed

    # Create and return a Stats object
    return Stats(hp=hp, attack=attack, defense=defense,
                 specialattack=specialattack, specialdefense=specialdefense,
                 speed=speed)

def parse_evs(evs_str: str) -> List[int]:
    evs = [0, 0, 0, 0, 0, 0]  # Default values for HP, Attack, Defense, Special Attack, Special Defense, Speed
    ev_split = evs_str.split('/')
    for ev in ev_split:
        value, stat = ev.strip().split()
        index = {
            'hp': 0,
            'atk': 1,
            'def': 2,
            'spa': 3,
            'spd': 4,
            'spe': 5
        }.get(stat.lower(), -1)
        
        if index != -1:
            evs[index] = int(value)  # Set the corresponding index to the parsed value

    return evs

def parse_ivs(ivs_str: str) -> List[int]:
    ivs = [31, 31, 31, 31, 31, 31]  # Default values for HP, Attack, Defense, Special Attack, Special Defense, Speed
    iv_split = ivs_str.split('/')
    for iv in iv_split:
        value, stat = iv.strip().split()
        index = {
            'hp': 0,
            'atk': 1,
            'def': 2,
            'spa': 3,
            'spd': 4,
            'spe': 5
        }.get(stat.lower(), -1)

        if index != -1:
            ivs[index] = int(value)  # Set the corresponding index to the parsed value

    return ivs

def parse_moves(move_lines: List[str]) -> List[str]:
    moves = []
    for line in move_lines:
        moves.append(line.strip().replace('-', '').title().lower().replace(" ", ""))
    return moves

def parse_pokemon(pokemon_str: str) -> PokemonSet:
    lines = pokemon_str.strip().splitlines()
    
    name_line = lines[0].strip()
    name, item = name_line.split(' @ ')

    item = item.lower().replace(" ", "")
    name = name.lower()
    
    ability_line = lines[1].strip()
    ability = ability_line.split(': ')[1].lower().replace(" ", "")
    
    tera_type_line = lines[2].strip()
    tera_type = tera_type_line.split(': ')[1]
    
    evs_line = lines[3].strip()
    evs = parse_evs(evs_line.split(': ')[1])
    
    nature_line = lines[4].strip()
    nature = nature_line.split()[0].lower()  # Only take the nature itself (Timid, Calm, etc.)
    
    if "IVs" in lines[5]:
        ivs_line = lines[5].strip()
        ivs = parse_ivs(ivs_line.split(': ')[1])
        move_lines = lines[6:]  # Moves start from line 7
    else:
        ivs = [31, 31, 31, 31, 31, 31]
        move_lines = lines[5:]

    moves = parse_moves(move_lines)


    return PokemonSet(name, name, item, ability, moves, nature, evs, "", ivs)

def parse_team(team_str: str) -> List[PokemonSet]:
    pokemons = team_str.split('\n\n')[:6]

    return [parse_pokemon(pokemon) for pokemon in pokemons]

def parse_pokemon_file(filename: str) -> List[PokemonSet]:
    with open(filename, 'r') as file:
        content = file.read()
        return parse_team(content)

if __name__ == "__main__":


    team = parse_pokemon_file("examples/red.txt")

    teams = [team]
    for i in range(1):
        teams.append(sim.dict_to_team_set(generate_team()))

    # # print(teams[0][0])
    # # print(team[0])
    # print(teams[0][0])
    # pokemon = Pokemon(player_uid=1, position=1, poke=teams[0][0])
    # print(pokemon)

    battle = sim.Battle('single', "Red", teams[0], "Blue", teams[1], debug=True)


    sim.run(battle)