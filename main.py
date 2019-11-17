#!/usr/bin/env python3 

import os
import argparse
import numpy as np
from copy import deepcopy
import torch
import gym

from agent import DDPG
from util import *

import matplotlib.pyplot as plt
from scipy.io import savemat

def train(num_episodes, agent, env, output, max_episode_length=None):
        
    step = 0
    episode = 0
    episode_steps = 0
    episode_reward = 0.
    observation = None

    rewards_array = []
    
    while episode < num_episodes:
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        if step <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation)
        
        observation2, reward, done, _ = env.step(action)
        observation2 = deepcopy(observation2)
        
        if done or episode_steps >= max_episode_length - 1:
            done = True

        agent.memory.append(observation2, agent.select_action(observation2), reward, done)
            
        if step > args.warmup :
            agent.update_policy()

        if episode % 200 == 0:
            agent.save_model(output)
            with open(output+'/train_reward.txt', 'w') as f:
                f.write(str(rewards_array))
        
        step = step + 1
        episode_steps = episode_steps + 1
        episode_reward = episode_reward + reward
        observation = deepcopy(observation2)

        if done: # end of episode
            print('#{}: episode_reward:{} steps:{}'.format(episode, episode_reward, step))
            rewards_array.append(episode_reward)
            agent.memory.append(observation, agent.select_action(observation), 0., False)
           
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1

def test(num_episodes, agent, env, model_path, max_episode_length, debug=False):

    agent.load_weights(model_path)
    agent.eval()

    observation = None
    result = []

    for episode in range(num_episodes):
        observation = env.reset()
        episode_steps = 0
        episode_reward = 0.    
        done = False
        
        while not done:
            action = agent.select_action(observation)
            observation, reward, done_, _ = env.step(action)
            
            if episode_steps >= max_episode_length - 1:
                done = True
            
            env.render(mode='human')

            episode_reward = episode_reward + reward
            episode_steps = episode_steps + 1

        print('[Evaluate] #Episode{}: episode_steps:{} episode_reward:{}'.format(episode, episode_steps, episode_reward))
        result.append(episode_reward)
        
    print('[Evaluate]: mean_reward:{}'.format(np.mean(result)))
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', default='train', type=str, help='train or test')
    parser.add_argument('--env', default='CarRacing-v0', type=str, help='open-ai gym environment')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--warmup', default=100, type=int, help='initialize replay memory (no policy updates)')
    parser.add_argument('--discount', default=0.99, type=float, help='discount factor')
    parser.add_argument('--batch_size', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=6000000, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--test_episodes', default=20, type=int, help='test episodes')
    parser.add_argument('--test_steps', default=2000, type=int, help='test steps')
    parser.add_argument('--max_episode_length', default=3000, type=int, help='')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--train_episodes', default=5000, type=int, help='training episodes')
    parser.add_argument('--epsilon', default=20000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=10, type=int, help='')
    parser.add_argument('--model_path', default='default', type=str, help='run #')

    args = parser.parse_args()
    if args.mode == 'train':
        args.output = get_output_folder(args.output, args.env, make_new=True)
    else:
        args.output = get_output_folder(args.output, args.env, make_new=False)
    
    if args.model_path == 'default':
        model_path = 'output/{}-run0'.format(args.env)
    else:
        model_path = 'output/{}-run{}'.format(args.env, args.model_path)
    
    env = gym.make(args.env)
    env = gym.wrappers.Monitor(env, "recording", force=True)
    
    np.random.seed(args.seed)
    env.seed(args.seed)

    agent = DDPG(args)

    if args.mode == 'train':
        train(args.train_episodes, agent, env, args.output, max_episode_length=args.max_episode_length)
    else:
        test(args.test_episodes, agent, env, model_path, max_episode_length=args.max_episode_length, debug=args.debug)
