import numpy as np
import itertools
import random
import pandas as pd
from pokemon import *

class PokemonEnv():
    """
    Description:
    A reinforcement learning environment where the agent's task is to generate a 
    Pokemon which balances those in a battle pool. Battles pools can be generated at
    random by sampling the Pokedex, or supplied by the user.
    
    Args:
        num_battles: The number of battles that should be simulated in a pool,
        the higher the value, the more accurate the advantage one Pokemon has
        over another.
        
        deterministic: A boolean which determines whether or not battles should be
        deterministic.
    """
    def __init__(self, num_battles=1000, deterministic=False):
        self.num_battles=num_battles
        self.deterministic = deterministic
        self.state = []
        self.random_pokemon = []
        self.build_pokedex()

    """
    Description: A helper function that Loads data and compiles it into a Pokedex.
    """
    def build_pokedex(self):
        #Types
        df_types = pd.read_csv('Data/type-chart.csv', index_col=0)
        self.types = np.append(df_types.columns.to_numpy(), 'none')
        
        #Moves
        df_moves = pd.read_csv('Data/moves_small.csv')
        self.moves = {}
        for i in range(df_moves.shape[0]):
            name, type, power = df_moves.loc[i].to_numpy()
            self.moves[name] = Move(name, type, power)
        
        #Pokemon
        df_pokemon = pd.read_csv('Data/pokemon_small.csv')
        df_pokemon = df_pokemon.drop(['hp','attack','defense','spattack', 'spdefense','speed'], axis=1)
        df_pokemon['type1'] = df_pokemon['type1'].str.lower()
        df_pokemon['type2'] = df_pokemon['type2'].str.lower()
        
        self.pokedex = {}
        for i in range(df_pokemon.shape[0]):
            name, type1, type2, move1, move2, move3, move4 = df_pokemon.loc[i].to_numpy()
            self.pokedex[name] = Pokemon(name, 1, 
                                              [type1, type2], 
                                              [self.moves[move1], self.moves[move2], self.moves[move3], self.moves[move4]], 
                                              {"hp": 12, "attack": 6, "defense": 6})
       
    """
    Description: A helper function which constructions a randomized battle pool.
    
    Args:
        size: The size of the randomized pool.
    Returns:
        pools: The randomized battle pool.
    """
    def random_pools(self, size):
        pools =[]
        for i in range(size):
            pool = []
            self.random_pokemon = random.sample(list(self.pokedex), 2)
            for pokemon in self.random_pokemon:
                pool.append(self.pokedex[pokemon].name)
            pools.append(pool)
        return pools
    
    """
    Description: Builds a user-defined battle pool.
    Args:
        pokemon_list: The names of desired Pokemon to be in the pool.
    """
    def build_pool(self, pokemon_list):
        self.pool = {}
        for pokemon in pokemon_list:
            self.pool[pokemon] = self.pokedex[pokemon]
      
    """
    Description: Resets the environment between episodes.

    Returns:
        state: The intial state s0.
    """
    def reset(self):
        _, pokemon_values = self.battle(False)
        self.state = []
        for i, pokemon in enumerate(self.pool.values()):
            self.state.append(pokemon_values[i])
        return self.state
       
    """
    Description: Simulates battle between all Pokemon in the pool.
    
    Args:
        render: whether or not text for the battle should be printed to the screen.
    
    Returns:
        reward: The reward for the episode.
        pokemon_values: The advantage score of each Pokemon in the pool.
    """
    def battle(self, render):
        #Simulate Battles
        winners=[]
        n=1 if self.deterministic else self.num_battles
        for i in range(n):
            if render: print("Round " + str(i))
            combinations = list(itertools.combinations(self.pool.values(), 2))
            battle = Battle()
            for combo in combinations:
                if render: print(combo[0].name + " vs. " + combo[1].name)
                winner, loser  = battle.simulate([combo[0],combo[1]], deterministic=self.deterministic, render=render)
                winners.append(winner)
                if render: 
                    print(winner + " won!")
                    print("")      
        target = 1/len(self.pool.keys())
        reward = 0
        pokemon_values = []
        for pokemon in self.pool.values():
            value = winners.count(pokemon.name)/len(winners)-target
            pokemon_values.append(value)
            reward -= np.abs(value)
            if render: print(pokemon.name + ": " + str(winners.count(pokemon.name)/len(winners)))
        return reward, pokemon_values
     
    """
    Description: Accepts an agent-defined Pokemon in the form of an action, and adds it to the pool
    before conducting battle.
    
    Args:
        action: The agent's action for this episode.
        render: Whether or not battle text should be rendered.
    Returns:
        reward: The episodes reward.
        arrelle: The Pokemon object generated using the agent's action vector.
    """
    def step(self, action, render=False):
        arrelle = self.vector_to_pokedex(action, "Arrelle")
        self.pool['Arrelle'] = arrelle
        
        balance, pokemon_values = self.battle(render)

        reward = balance
        if self.pool["Arrelle"].types[0] == self.pool["Arrelle"].types[1]:
            reward -= 0.02
        reward -= 0.02*(len(self.pool["Arrelle"].moves)-len(set(self.pool["Arrelle"].moves)))
        
        return reward, arrelle
    
    """
    Description: A helper function that converts an action vector into a Pokemon object.
    Args:
        vector: The action vector to be converted.
        name: The new Pokemon's name.
    Returns:
        Pokemon: The newly created Pokemon object.
    """
    def vector_to_pokedex(self, vector, name):
        type1 = self.types[vector[0]]
        type2 = self.types[vector[1]]
        move1 = list(self.moves.keys())[vector[2]]
        move2 = list(self.moves.keys())[vector[3]]
        move3 = list(self.moves.keys())[vector[4]]
        move4 = list(self.moves.keys())[vector[5]]
        return Pokemon(name, 1, [type1, type2], [self.moves[move1], self.moves[move2], self.moves[move3], self.moves[move4]],{"hp": 12, "attack": 6, "defense": 6})