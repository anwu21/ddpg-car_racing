import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from models_ac import (Actor, Critic)
from memory import SequentialMemory
from util import *

import matplotlib.pyplot as plt

criterion = nn.MSELoss()
criterion.cuda()

class DDPG(object):
    def __init__(self, args):
         
        self.actor = Actor()
        self.actor_target = Actor()
        self.actor.apply(init_weights)
        self.actor_target.apply(init_weights)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.rate)

        self.critic = Critic()
        self.critic_target = Critic()
        self.actor.apply(init_weights)
        self.actor_target.apply(init_weights)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)

        self.actor_target = self.actor
        self.critic_target = self.critic
        
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon
        self.epsilon = 1.0
        self.s_t = None 
        self.a_t = None 

        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

        torch.cuda.manual_seed(args.seed)
        
    def update_policy(self):
        # Sample batch
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        state_batch = state_batch/255.
        next_state_batch = next_state_batch/255.
        
        state_batch = state_batch.reshape(self.batch_size, 3, 96, 96)
        next_state_batch = next_state_batch.reshape(self.batch_size, 3, 96, 96)
        
        # Prepare for the target q
        next_q_values = self.critic_target(to_variable(next_state_batch, volatile=True), self.actor_target(to_variable(next_state_batch, volatile=True)))
        next_q_values.volatile=False

        target_q_batch = to_variable(reward_batch) + self.discount*to_variable(terminal_batch.astype(np.float))*next_q_values

        # Reset gradients
        self.critic.zero_grad()
        self.actor.zero_grad()

        # Update critic
        q_batch = self.critic(to_variable(state_batch), to_variable(action_batch))
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Update actor
        policy_loss = -self.critic(to_variable(state_batch), self.actor(to_variable(state_batch)))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Update target
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()
        

## For 3 dimension action space (i.e. steer, gas, brake)
    def random_action(self):
        steer = np.random.uniform(-1., 1.)
        gas = np.random.uniform(0., 1.)
        brake = np.random.uniform(0., 1.)
        action = np.array([steer, gas, brake])
        self.a_t = action
        return action
    
    def select_action(self, s_t, decay_epsilon=True):
        s_t = s_t/255.
        s_t = s_t.reshape(1, 3, 96, 96)
        s_t = to_variable(s_t)
        action = self.actor(s_t)
        action = to_numpy(action).squeeze(0)
        
        steer = action[0]
        gas = action[1]
        brake = action[2]
        
        steer = steer + max(self.epsilon, 0) * np.random.normal()
        gas = gas + max(self.epsilon, 0) * np.random.normal()
        brake = brake + max(self.epsilon, 0) * np.random.normal()

        steer = np.clip(steer, -1., 1.)
        gas = np.clip(gas, 0., 1.)
        brake = np.clip(brake, 0., 1.)
        
        action = np.concatenate([steer, gas, brake], axis=None)

        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        self.a_t = action
        return action
    
    def reset(self, obs):
        self.s_t = obs

    def load_weights(self, output):
        self.actor.load_state_dict(torch.load('{}/actor_3200.pkl'.format(output)))
        self.critic.load_state_dict(torch.load('{}/critic_3200.pkl'.format(output)))

    def save_model(self, output):
        torch.save(self.actor.state_dict(), '{}/actor.pkl'.format(output))
        torch.save(self.critic.state_dict(), '{}/critic.pkl'.format(output))

