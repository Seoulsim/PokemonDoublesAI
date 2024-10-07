# PokemonDoublesAI
AI for pokemon double battles.

Credit to https://github.com/nicolaslindbloomairey/pokemon-python for the battle simulator!

Let $root = [root of repository] and $root_sim = [root of simulator]



## Usage

### Setup

  Copy $root/src into pokemon-python/src (may fix this later, having import issues)

### Usage

Training: (Run from $root_sim) (may fix this later, having import issues)

  python -m src.train team_path_1 team_path_2 number_of_episodes (team path defaults to the ones in $root/examples)

outputs go into $root/out where red_dqn.keras = player 1's model, blue_dqn.keras = player 2's model

Playing:

  python -m src.play team_path_1 team_path_2 team_1_AI (very very rough, you must manually input gamestate, sorry!)

Comparison:

  python -m src.compare team_path_1 team_path_2 team_1_model team_2_model games 

Runs [games] amount of games between the two teams and the two models
  
