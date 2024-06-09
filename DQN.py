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
total_rewards = []

K_LINE_NUM = 48
INPUT_SIZE = K_LINE_NUM * 5 + 4
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class replay_buffer:
    # store experience
    def __init__(self, capacity):
        self.capacity = capacity  # the size of the replay buffer
        self.memory = deque(maxlen=capacity)  # replay buffer itself

    def insert(self, state, action, reward, next_state, done):
        # Add experience when training
        self.memory.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        # fetch experience when learning in training
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, done = zip(*batch)
        return observations, actions, rewards, next_observations, done

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    """
    The structure of the Neural Network calculating Q values of each state.
    """

    def __init__(self,  num_actions, hidden_layer_size=888):
        super(Net, self).__init__()
        self.input_state = INPUT_SIZE  # the dimension of state space
        self.num_actions = num_actions  # the dimension of action space
        self.fc1 = nn.Linear(self.input_state, hidden_layer_size)  # input layer
        self.bn1 = nn.BatchNorm1d(hidden_layer_size)  # Batch normalization for input layer
        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size)  # hidden layer
        self.bn2 = nn.BatchNorm1d(hidden_layer_size)  # Batch normalization for hidden layer
        self.fc3 = nn.Linear(hidden_layer_size, hidden_layer_size)  # hidden layer
        self.bn3 = nn.BatchNorm1d(hidden_layer_size)  # Batch normalization for hidden layer
        self.fc4 = nn.Linear(hidden_layer_size, hidden_layer_size)  # hidden layer
        self.bn4 = nn.BatchNorm1d(hidden_layer_size)  # Batch normalization for hidden layer
        self.fc5 = nn.Linear(hidden_layer_size, hidden_layer_size)  # hidden layer
        self.bn5 = nn.BatchNorm1d(hidden_layer_size)  # Batch normalization for hidden layer
        self.fc6 = nn.Linear(hidden_layer_size, num_actions)  # output layer

    def forward(self, states):
        """
        Forward the state to the neural network.
        Parameter:
            states: a batch size of states
        Return:
            q_values: a batch size of q_values
        """
        if states.dim() == 1:
            states = states.unsqueeze(0)  # 添加 batch 維度
        
        x = F.relu(self.bn1(self.fc1(states)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        q_values = self.fc6(x)
        return q_values


class Agent:
    def __init__(
        self, env, epsilon = 0.9, learning_rate=0.5, GAMMA=0.999, batch_size=32, capacity=10000
    ):
        """
        The agent learning how to control the action of the cart pole.
        Hyperparameters:
            epsilon: Determines the explore/expliot rate of the agent
            learning_rate: Determines the step size while moving toward a minimum of a loss function
            GAMMA: the discount factor (tradeoff between immediate rewards and future rewards)
            batch_size: the number of samples which will be propagated through the neural network
            capacity: the size of the replay buffer/memory
        """
        self.env = env
        self.n_actions = 19  # the number of actions
        self.count = 0

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA
        self.batch_size = batch_size
        self.capacity = capacity

        self.buffer = replay_buffer(self.capacity)
        self.evaluate_net = Net(self.n_actions)  # the evaluate network
        self.evaluate_net.to(device)
        self.target_net = Net(self.n_actions)  # the target network
        self.target_net.to(device)

        self.optimizer = torch.optim.Adam(self.evaluate_net.parameters(), lr=self.learning_rate)  
        # Adam is a method using to optimize the neural network

    def learn(self):
        """
        - Implement the learning function.
        - Here are the hints to implement.
        Steps:
        -----
        1. Update target net by current net every 100 times. (we have done this for you)
        2. Sample trajectories of batch size from the replay buffer.
        3. Forward the data to the evaluate net and the target net.
        4. Compute the loss with MSE.
        5. Zero-out the gradients.
        6. Backpropagation.
        7. Optimize the loss function.
        -----
        Parameters:
            self: the agent itself.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)
        Returns:
            None (Don't need to return anything)
        """
        if self.count % 10 == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())
        
        # Begin your code
        """
        Sample trajectories of batch size from the replay buffer.
        Convert these sampled data into tensor.
        """
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.tensor(np.array(states), dtype=torch.float).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(-1).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(device)
        done = torch.tensor(np.array(dones), dtype=torch.float).to(device)


        self.evaluate_net.train()
        self.target_net.eval()
        """
        Forward the data to evaluate net and target net.
        q_eval is predicted values from evaluate network which is extracted based on action.
        q_next is actual values from target network. 
        q_target is the expected Q-values obtained from the formula "reward + gamma * max(q_next)".
        """
        q_eval = self.evaluate_net(states).gather(1, actions)
        q_next = self.target_net(next_states).detach() * (1 - done).unsqueeze(-1)
        q_target = rewards.unsqueeze(-1) + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)

        """
        Compute the loss of "q_eval" and "q_target" with nn.MSELoss().
        """
        loss_func = nn.MSELoss()
        loss = loss_func(q_eval, q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # End your code
        torch.save(self.target_net.state_dict(), "./Tables/DQN.pt")

    def choose_action(self, state):
        """
        - Implement the action-choosing function.
        - Choose the best action with given state and epsilon
        Parameters:
            self: the agent itself.
            state: the current state of the enviornment.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)
        Returns:
            action: the chosen action.
        """
        with torch.no_grad():
            # Begin your code
            """
            Generate a random number. If the number is bigger than epsilonreturn the index of the maximum Q of the given state in Q-table.
        Or return random action.
            """
            temp = np.random.random()
            if temp < self.epsilon:
                return np.random.randint(self.n_actions)
            # forward the state to nn and find the argmax of the actions
            
            # Ensure the state is 2D [batch_size, num_features]
        state_tensor = torch.tensor(state, dtype=torch.float).to(device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        self.evaluate_net.eval()

        action = torch.argmax(self.evaluate_net(state_tensor)).item()
            # End your code
        return action

def train(env, episode=500):
    """
    Train the agent on the given environment.
    Paramenters:
        env: the given environment.
    Returns:
        None (Don't need to return anything)
    """
    agent = Agent(env)
    rewards = []
    for i_episode in range(episode):
        state = env.reset()
        count = 0
        while True:
            count += 1
            agent.count += 1
            # env.render()
            tempstate1 = state
            state = state

            action = agent.choose_action(tempstate1)
            next_state, reward, done, info = env.step(action)
            tempstate2 = next_state
            next_state = next_state
            agent.buffer.insert(tempstate1, int(action), reward, tempstate2, int(done))

            if len(agent.buffer) >= 100:
                # every 100 step, learn from replay_buffer 
                agent.learn()
            if done:
                # end episode
                rewards.append(count)
                break
            state = next_state
        print(i_episode, info)
        if(i_episode % math.ceil(episode/500) == 0):
            agent.epsilon *= (0.05/agent.epsilon) ** (1/500)
            agent.learning_rate *= (0.002/agent.learning_rate) ** (1/500)
    total_rewards.append(rewards)


def test(env):
    """
    Test the agent on the given environment.
    Paramenters:
        env: the given environment.
    Returns:
        None (Don't need to return anything)
    """
    rewards = []
    testing_agent = Agent(env)
    testing_agent.target_net.load_state_dict(torch.load("./Tables/DQN.pt"))
    os.makedirs("./tb_record_1/comp_profit_train/DQN", exist_ok=True)
    w = SummaryWriter('./tb_record_1/comp_profit_train/DQN')
    for _ in range(1):
        state = env.reset()
        t = 0
        while True:
            tempstate = state
            Q = testing_agent.target_net(torch.FloatTensor(tempstate).to(device)).squeeze(0).detach()
            action = int(torch.argmax(Q).cpu().numpy())
            next_state, _, done, info = env.step(action)
            w.add_scalar('Profit', env._total_asset, t)
            t+=1
            if done:
                print(info)
                break
            state = next_state

    


if __name__ == "__main__":
    env = gym.make('futures1-v0') 
    os.makedirs("./Tables", exist_ok=True)

    # training section:
    for i in range(1):
        time0 = time.time()
        print(f"#{i + 1} training progress")
        train(env, 500)
        time1 = time.time()
        print(f"Training time: {time1 - time0} seconds")
        print ("Win rate: ", env.win_count ,"/", env.win_count + env.dead_count, f"({env.get_win_rate()})")
        [profit, loss] = env.get_cumulative_profit_loss_ratio()
        print("Profit Loss Ratio: ",f"{profit} : {loss}" )

    # testing section:
    test(env)
    env.close()

    os.makedirs("./Rewards", exist_ok=True)
    np.save("./Rewards/DQN_rewards.npy", np.array(total_rewards))