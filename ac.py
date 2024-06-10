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
import time
import rl_utils
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

K_LINE_NUM = 48
INPUT_SIZE = K_LINE_NUM * 5 + 4


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        # 呼叫nn.Module類的初始化方法
        super(PolicyNet, self).__init__()
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.action_dim = (
            env.action_space.n if self.discrete else env.action_space.shape[0]
        )
        self.hidden_size = 500
        self.double()

        self.observation_dim = INPUT_SIZE

        self.shared_layer1 = nn.Linear(self.observation_dim, self.hidden_size)
        self.shared_layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.shared_layer3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.action_layer = nn.Linear(self.hidden_size, self.action_dim)

        # action & reward memory
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        x = self.shared_layer1(state)
        x = F.relu(x)
        x = self.shared_layer2(x)
        x = F.relu(x)
        x = self.shared_layer3(x)
        x = F.relu(x)
        return F.softmax(self.action_layer(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim=INPUT_SIZE, hidden_dim=500):
        super(ValueNet, self).__init__()
        self.double()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class ActorCritic:
    def __init__(
        self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device
    ):
        state_dim = INPUT_SIZE
        hidden_dim = 500
        action_dim = env.action_space.n
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)  # 价值网络
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )  # 价值网络优化器
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(
            self.device
        )
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float
        ).to(self.device)
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )

        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
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
        torch.save(self.actor.state_dict(), "./Tables/AC_actor.pt")
        torch.save(self.critic.state_dict(), "./Tables/AC_critic.pt")


def test(env, agent):
    """
    Test the agent on the given environment.
    Paramenters:
        env: the given environment.
    Returns:
        None (Don't need to return anything)
    """
    rewards = []
    # 创建一个新的智能体实例
    testing_agent = agent
    testing_agent.actor.eval()
    # 从保存的文件中加载预训练的目标网络参数
    testing_agent.actor.load_state_dict(
        torch.load("./Tables/AC_actor.pt", map_location=device)
    )
    # 创建目录（如果不存在）用于存储 TensorBoard 记录
    os.makedirs("./tb_record_1/comp_profit_train/DQN", exist_ok=True)
    # 初始化 TensorBoard 记录器
    w = SummaryWriter("./tb_record_1/comp_profit_train/DQN")
    start_tick = K_LINE_NUM
    profit_rate = []
    profit_rate_tick = []
    unrealized_profit = []
    while True:
        if start_tick == len(env.prices) - 4:
            break
        state = env.reset(start_tick=start_tick)
        t = 0
        while True:
            tempstate = state

            # 将状态转换为浮点张量，并移至 GPU（如果可用）
            # Q = testing_agent.target_net(torch.FloatTensor(tempstate).to(device)).squeeze(0).detach()
            # 选择具有最大 Q 值的动作
            # action = int(torch.argmax(Q).cpu().numpy())
            # action = agent.take_action(state)
            # 在环境中执行动作，获得下一状态、奖励、是否结束以及额外信息
            # print(action)
            # if t <=100:
            #     next_state, _, done, info = env.step(15)
            # else:
            #     next_state, _, done, info = env.step(9)
            next_state, _, done, info = env.step(15)

            # print(info)
            # 将当前总资产记录到 TensorBoard
            w.add_scalar("Profit", env.get_total_asset(), t)
            t += 1
            profit_rate.append(env.get_profit_rate())
            unrealized_profit.append(int(info["unrealized_profit"]))
            profit_rate_tick.append(info["done_tick"])
            # 如果回合结束，打印信息并退出循环
            if done:
                # env.render()
                # profit_rate.append(env.get_profit_rate())
                # profit_rate.append(int(info['unrealized_profit']))
                # profit_rate_tick.append(info["done_tick"])
                info["total_reward"] = int(info["total_reward"])
                info["total_asset"] = int(info["total_asset"])
                info["cash"] = int(info["cash"])
                info["long_position"] = int(info["long_position"])
                info["unrealized_profit"] = int(info["unrealized_profit"])
                print(info)
                start_tick = info["done_tick"]
                break
            state = next_state
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    # plt.figure()
    # plt.plot(env.prices, label='prices')
    ax1.plot(range(0, len(env.prices)), env.prices, label="prices", color="blue")
    ax1.set_ylabel("prices", color="blue")
    ax2.plot(profit_rate_tick, profit_rate, label="profit_rate", color="red")
    ax2.set_ylabel("profit_rate", color="red")
    # ax3.plot(profit_rate_tick, unrealized_profit, label='unrealized_profit', color='green')

    # plt.plot(profit_rate, label='profit_rate')
    # len(self.prices
    # 標註最後一個 profit_rate 的值
    last_tick = profit_rate_tick[-1]
    last_profit_rate = profit_rate[-1]
    ax2.annotate(
        f"{int(last_profit_rate)}",
        xy=(last_tick, last_profit_rate),
        xytext=(last_tick + 1, last_profit_rate),  # 調整文字標註的位置
        arrowprops=dict(facecolor="red", shrink=0.05),
    )
    plt.title("prices & profit_rate")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    # plt.xlabel('time')
    # plt.ylabel('value')
    # plt.legend()
    plt.show()


if __name__ == "__main__":
    actor_lr = 0.5
    critic_lr = 0.5
    num_episodes = 200
    hidden_dim = 2000
    gamma = 0.98
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = gym.make("futures4-v0")
    env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = ActorCritic(
        state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device
    )

    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)
