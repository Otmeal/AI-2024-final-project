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
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

K_LINE_NUM = 48
INPUT_SIZE = K_LINE_NUM * 5 + 4
def test(env):
    """
    Test on the given environment.
    average50 : the list of last 50 day's average
    average200 : the list of last 200 day's average
    """
    
    average50 = list()                  
    average200 = list()
    for i in range(env.frame_bound[1]):
        if(i>=50):
            average50.append(np.sum(env.signal_features[i-50:i,0:4]) / 50/4)
        if(i>=200):
            average200.append(np.sum(env.signal_features[i-200:i, 0:4]) / 200/4)
    count = 0                                   # count the number of iteration
    env.reset()                                 
    count1 = 0                                  # count the action buy
    count0 = 0                                  # count the action sell
    w = SummaryWriter('tb_record_1/comp_profit_train/baseline')
    t = 0
    start_tick = K_LINE_NUM
    profit_rate = []
    profit_rate_tick = []
    while True:
        if(start_tick == len(env.prices) - 4): 
            break
        state = env.reset(start_tick = start_tick)
        t = 0
        while True:
            if count<200:                           # skip to 201 th day
                env.step(9)
                count = count+1
                w.add_scalar('Profit', env.get_profit_rate(), t)
                t+=1
                continue
            # use the Moving Average Crossover method to decide whether to buy or not
            if average50[count-200+150] > average200[count-200]:
                action = 4
                count1 += 1
            else:
                action = 14
                count0 += 1

            next_state, _, done, info = env.step(action)
            w.add_scalar('Profit', env.get_profit_rate(), t)
            t+=1
            
            count = count + 1
            if done:
                profit_rate.append(env.get_profit_rate())
                profit_rate_tick.append(info["done_tick"])
                info['total_reward'] = int(info['total_reward'])
                info['total_asset'] = int(info['total_asset'])
                info['cash'] = int(info['cash'])
                info['long_position'] = int(info['long_position'])
                info['unrealized_profit'] = int(info['unrealized_profit'])
                print(info)
                start_tick = info["done_tick"]
                break
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(range(0, len(env.prices)), env.prices, label='prices', color='blue')
    ax1.set_ylabel('prices', color='blue')
    ax2.plot(profit_rate_tick, profit_rate, label='profit_rate', color='red')
    ax2.set_ylabel('profit_rate', color='red')
    # 標註最後一個 profit_rate 的值
    last_tick = profit_rate_tick[-1]
    last_profit_rate = profit_rate[-1]
    ax2.annotate(
        f'{int(last_profit_rate)}', 
        xy=(last_tick, last_profit_rate), 
        xytext=(last_tick + 1, last_profit_rate),  # 調整文字標註的位置
        arrowprops=dict(facecolor='red', shrink=0.05)
    )
    plt.title('prices & profit_rate')
    fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
    plt.show()

    print("count action - buy : " + str(count1))
    print("count action - sell : " + str(count0))
    print("total profit rate: " + str(env.get_profit_rate()))




if __name__ == "__main__":
    env = gym.make('futures1-v0')
    test(env)