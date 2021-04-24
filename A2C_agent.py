import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


    """
    Description:
    The implemention of the A2C learning algorithm customized to the support the Pokemon environment.
    
    Args:
        env: The pokemon environment.
        df: A pandas dataframe containing Pokemon embeddings.
        actor_model: The actor's NN.
        critic_model: The critic's NN.
        actor_lr: The actor network's learning rate.
        critic_lr: The critic network's learning rate.
        gamma: Discount factor.
    """
class ActorCritic_OneStep():
    def __init__(self, env, df, actor_model, critic_model, actor_lr, critic_lr, gamma):
        self.env = env
        self.df = df
        self.actor = actor_model.to(device)
        self.critic = critic_model.to(device)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optim_actor = optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        self.gamma = gamma
    
    """
    Description:
    A helper function used for testing a trained model. Given a pool a new agent-defined Pokemon
    is generated and battle is simulated to check balance.
    
    Args:
        pool: The pool of Pokemon to simualte battle with.
        
    Returns:
        reward: The episode's reward (0 when fully balanced with no repeated moves or type on generated Pokemon)
        arrelle: The agent-generated Pokemon object.
    """
    def generate_arrelle(self, pool):
        self.env.build_pool(pool)
        state = self.env.reset()
        state = []
        for pokemon in pool:
            embed = self.get_pokemon_embedding(pokemon)
            state = np.concatenate((state, embed))
        state = torch.FloatTensor(state).to(device)

        dist_type0, dist_type1, dist_move0, dist_move1, dist_move2, dist_move3 = self.actor(state)

        type0 = dist_type0.sample()
        type1 = dist_type1.sample()
        move0 = dist_move0.sample()
        move1 = dist_move1.sample()
        move2 = dist_move2.sample()
        move3 = dist_move3.sample()
        action = [type0.cpu().numpy(), type1.cpu().numpy(), 
                  move0.cpu().numpy(), move1.cpu().numpy(), move2.cpu().numpy(), move3.cpu().numpy()]

        reward, arrelle = self.env.step(action, render=False)
        return reward, arrelle
    
    def get_pokemon_embedding(self, pokemon_name):
        embedding = np.delete(self.df.loc[self.df['Pokemon'] == pokemon_name].to_numpy()[0], 0).astype(np.float)
        return embedding
    
    def calculate_loss(self, action, reward, value, dist):
        log_prob = dist.log_prob(action).unsqueeze(0)
        delta = reward+self.gamma*0-value
        critic_loss = (delta)**2
        actor_loss = -log_prob * delta.detach()
        return actor_loss, critic_loss

    def update(self, actor_loss, critic_loss):
        
        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()
        
        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()
         
            
    """
    Description:
    The implemention of the A2C learning algorithm customized to the support the Pokemon environment.
    
    Args:
        pools: Pools on Pokemon to simulate battle with.
        n_episodes: The number of episdoes of training.
        use_random_pool: A boolean which determines whether or not random pools should be used.
        random_pool_size: The size of randomly generated pools.
        
    Returns:
        episode_rewards: The reward obtained for each episode.
        avg_rewards: The average of every 100 episodes of training.
    """
    def learn_task(self, pools, n_episodes, use_random_pool=False, random_pool_size=4):
        episode_rewards = []
        avg_rewards = []

        for episode in tqdm(range(n_episodes)):
            rewards = []
            if use_random_pool: pools = self.env.random_pools(random_pool_size)
            for pool in pools:
                self.env.build_pool(pool)
                state = self.env.reset()
                state = []
                for pokemon in pool:
                    embed = self.get_pokemon_embedding(pokemon)
                    state = np.concatenate((state, embed))
                state = torch.FloatTensor(state).to(device)

                dist_type0, dist_type1, dist_move0, dist_move1, dist_move2, dist_move3 = self.actor(state)
                value_type0, value_type1, value_move0, value_move1, value_move2, value_move3 = self.critic(state)
                
                type0 = dist_type0.sample()
                type1 = dist_type1.sample()
                move0 = dist_move0.sample()
                move1 = dist_move1.sample()
                move2 = dist_move2.sample()
                move3 = dist_move3.sample()
                action = [type0.cpu().numpy(), type1.cpu().numpy(), 
                          move0.cpu().numpy(), move1.cpu().numpy(), move2.cpu().numpy(), move3.cpu().numpy()]

                reward, _ = self.env.step(action, render=False)
                rewards.append(reward)

                actor_loss_type0, critic_loss_type0 = self.calculate_loss(type0, reward, value_type0, dist_type0)
                actor_loss_type1, critic_loss_type1 = self.calculate_loss(type1, reward, value_type1, dist_type1)
                actor_loss_move0, critic_loss_move0 = self.calculate_loss(move0, reward, value_move0, dist_move0)
                actor_loss_move1, critic_loss_move1 = self.calculate_loss(move1, reward, value_move1, dist_move1)
                actor_loss_move2, critic_loss_move2 = self.calculate_loss(move2, reward, value_move2, dist_move2)
                actor_loss_move3, critic_loss_move3 = self.calculate_loss(move3, reward, value_move3, dist_move3)

                actor_loss = actor_loss_type0+actor_loss_type1+actor_loss_move0+actor_loss_move1+actor_loss_move2+ actor_loss_move3
                critic_loss = critic_loss_type0+critic_loss_type1+critic_loss_move0+critic_loss_move1+critic_loss_move2+ critic_loss_move3     
                self.update(actor_loss, critic_loss)
                
            episode_rewards.append(np.mean(rewards))
           
            if episode >= 100:
                avg_rewards.append(np.mean(episode_rewards[-100:]))
        return episode_rewards, avg_rewards