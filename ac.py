import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import random
from collections import deque
from torch import Tensor
import os
from tqdm import tqdm
import gym_futures_trading
import math
from torch.utils.tensorboard import SummaryWriter
import rl_utils

K_LINE_NUM = 48
INPUT_SIZE = K_LINE_NUM * 5 + 4

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        # 呼叫nn.Module類的初始化方法
        super(PolicyNet, self).__init__()
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)     
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 500
        self.double()

        # calculate the observation space size
        self.observation_dim = INPUT_SIZE
        # for i in env.observation_space.shape:
        #     self.observation_dim *= i
        
        #self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        #self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.shared_layer1 = nn.Linear(self.observation_dim, self.hidden_size)
        self.shared_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.shared_layer3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.action_layer = nn.Linear(self.hidden_size, self.action_dim)
        # self.value_layer = nn.Linear(self.hidden_size, 1)
        
        # action & reward memory
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        x = self.shared_layer1(Tensor(state))
        x = F.relu(x)
        x = self.shared_layer2(x)
        x = F.relu(x)
        x = self.shared_layer3(x)
        x = F.relu(x)
        return F.softmax(self.action_layer(x), dim = 1)
    
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim = INPUT_SIZE, hidden_dim = 500):
        super(ValueNet, self).__init__()
        self.double()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device):
        state_dim = INPUT_SIZE
        hidden_dim = 500
        action_dim = env.action_space.n
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)  # 价值网络
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)  # 价值网络优化器
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(self.device)

        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -dones)
        td_delta = td_target - self.critic(states)  # 时序差分误差
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 均方误差损失函数
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数

if __name__ == "__main__":
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
    
    env = gym.make('futures4-v0') 
    env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)

    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)