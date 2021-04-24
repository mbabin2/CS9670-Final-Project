import numpy as np
import pandas as pd

class Move():
    def __init__(self, name, type, power):
        self.name = name
        self.type = type
        self.power = power
        
class Pokemon():
    def __init__(self, name, level, types, moves, base_stats):
        self.name = name
        self.level = level
        self.types = types
        self.moves = moves
        self.attack = base_stats["attack"]
        self.defense = base_stats["defense"]
        self.hp = base_stats["hp"]

class Battle():
    def __init__(self):
        self.struggle = Move('struggle', 'normal', 50)
        df = pd.read_csv('Data/type-chart.csv', index_col=0)
        types = df.columns.to_numpy()
        self.type_chart = {}
        for type1 in types:
            self.type_chart[type1] = {type2:df.loc[type1][type2] for type2 in types}
            
    def calculate_damage(self, attacker, defender, selected_move, deterministic):
        effectiveness = 1
        for defender_type in defender.types:
            if defender_type != "none":
                effectiveness *= self.type_chart[selected_move.type][defender_type]
        stab = 1.5 if selected_move.type in attacker.types else 1
        mod_rand = 1 if deterministic else np.random.uniform(low=0.85, high=1)
        modifier = stab*effectiveness*mod_rand
        damage = (((2*attacker.level/5+2)*selected_move.power*(attacker.attack/defender.defense))/50+2)*modifier
        return damage, effectiveness

    def simulate(self, pokemon, deterministic=False, render=False):
        pokemon_health = [pokemon[0].hp, pokemon[1].hp]   
        attacker_index = 0 if deterministic else np.random.randint(2)
        while True:
            attacker_index += 1
            attacker_index %= 2
            defender_index = 0 if attacker_index == 1 else 1
            attacker = pokemon[attacker_index]
            defender = pokemon[defender_index]
            max_dmg = 0
            max_effectiveness = ""
            max_move = attacker.moves[0]
            for move in attacker.moves:
                if move.name != "none":
                    dmg, effectiveness = self.calculate_damage(attacker, defender, move, deterministic)
                    if dmg > max_dmg:
                        max_dmg = dmg
                        max_effectiveness = effectiveness
                        max_move = move
            if max_dmg == 0:
                max_dmg = (((2*attacker.level/5+2)*self.struggle.power*(attacker.attack/defender.defense))/50+2)
                max_effectiveness = 1
                max_move = self.struggle
            if render:
                print(attacker.name + " used " + max_move.name)
                if max_effectiveness == 0:
                    print("It had no effect...")
                elif max_effectiveness == 0.5:
                    print("Its not very effective...")
                elif max_effectiveness == 1:
                    print("It did normal damage...")
                elif max_effectiveness >= 2:
                    print("Its super effective!...")
            pokemon_health[defender_index] -= max_dmg
            if pokemon_health[defender_index] < 0:
                return attacker.name, defender.name