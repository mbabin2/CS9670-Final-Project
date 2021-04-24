import torch
import numpy as np

from pokemon_env import *
from model import*

class Agent():
    def __init__(self, env, df):
        self.env = env
        self.df = df

    def get_pokemon_embedding(self, pokemon_name):
        embedding = np.delete(self.df.loc[self.df['Pokemon'] == pokemon_name].to_numpy()[0], 0).astype(np.float)
        return embedding

    def find_solution(self, npop, sigma, alpha, ES_accuracies, epochs_es, num_pools=10):
        pool = self.env.random_pool()
        self.env.build_pool(pool)

        num_pokemon = len(self.env.pool)
        embedding_size = len(self.get_pokemon_embedding(list(self.env.pool.keys())[0]))
        num_types = len(self.env.types)
        num_moves = len(self.env.moves)
        model = Model(num_pokemon*embedding_size,[128,64,32],[num_types-1,num_types,num_moves-1,num_moves])

        avg_reward = []
        
        w = utils.parameters_to_vector(model.parameters())
        for i in range(epochs_es):
            print("Episode: ", i)
            R = np.zeros(npop)
            N = torch.randn(npop, w.size()[0])

            #pools = []
            #for i in range(num_pools):
                #pools.append(env.random_pool())
            pools = [['Charmander','Bulbasaur'],['Bulbasaur','Charmander'],
                     ['Squirtle','Pikachu'],['Pikachu','Squirtle'],
                    ['Pidgey','Lapras'],['Lapras','Pidgey'],
                    ['Ekans','Diglett'],['Diglett','Ekans']]
            for j in range(npop):
                w_try = w + sigma*N[j]
                R[j], _ = self.f(pools, w_try)
            print(R)
            avg_reward.append(np.mean(R))
            A = (R - np.mean(R)) / (np.std(R)+0.00001)
            w = w + alpha/(npop*sigma) * torch.mm(torch.transpose(N, 0, 1), torch.from_numpy(A).view(npop,1).float()).view(w.size()[0])
        
        return w, model, avg_reward

    def f(self, pools, w):
        num_pokemon = 2
        embedding_size = len(self.get_pokemon_embedding(list(self.env.pool.keys())[0]))
        num_types = len(self.env.types)
        num_moves = len(self.env.moves)
        model = Model(num_pokemon*embedding_size,[128,64,32],[num_types-1,num_types,num_moves-1,num_moves])

        model.parameters = utils.vector_to_parameters(w, model.parameters())

        rewards = []
        arrelles = []

        for pool in pools:
            self.env.build_pool(pool)
            state = self.env.reset()
            state = []
            for pokemon in pool:
                embed = self.get_pokemon_embedding(pokemon)
                state = np.concatenate((state, embed))
            state_tensor = torch.FloatTensor(state).to(torch.float)
            with torch.no_grad():
                type0, type1, move0, move1, move2, move3 = model(state_tensor)
            action = [np.argmax(type0.numpy()),np.argmax(type1.numpy()),
                         np.argmax(move0.numpy()),np.argmax(move1.numpy()),
                         np.argmax(move2.numpy()),np.argmax(move3.numpy())]
            reward, arrelle = self.env.step(action, render=False)

            rewards.append(reward)
            arrelles.append(arrelle)
        return np.mean(rewards), arrelles