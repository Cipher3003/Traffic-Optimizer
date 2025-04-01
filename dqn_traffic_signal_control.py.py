import traci
import sumolib
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import os
import time

# SUMO Configuration (Edit paths as needed)
sumo_binary = "sumo-gui"  # or "sumo" for headless mode
sumo_config = "crossing.sumocfg"  # Your SUMO config file
sumo_cmd = [sumo_binary, "-c", sumo_config]

# DQN Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.95  # Discount factor
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
MEMORY_SIZE = 2000  # Replay buffer size
UPDATE_TARGET_EVERY = 100  # Steps to update target network
EPISODES = 100  # Training episodes

# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent with Target Network
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.target_update_counter = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([x[0] for x in minibatch])
        actions = torch.LongTensor([x[1] for x in minibatch])
        rewards = torch.FloatTensor([x[2] for x in minibatch])
        next_states = torch.FloatTensor([x[3] for x in minibatch])
        dones = torch.FloatTensor([x[4] for x in minibatch])

        # Current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values (using target network)
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * GAMMA * next_q

        # Compute loss and update
        loss = self.criterion(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

        # Update target network
        self.target_update_counter += 1
        if self.target_update_counter >= UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0

# Helper Functions
def get_state():
    """Extracts traffic state: queue lengths and speeds for all lanes"""
    lanes = traci.lane.getIDList()
    state = []
    for lane in lanes:
        state.append(traci.lane.getLastStepHaltingNumber(lane))
        state.append(traci.lane.getLastStepMeanSpeed(lane))
    return np.array(state)

def calculate_reward():
    """Reward = -(total waiting time + CO2 emissions penalty)"""
    lanes = traci.lane.getIDList()
    total_waiting = sum(traci.lane.getWaitingTime(lane) for lane in lanes)
    total_co2 = sum(traci.lane.getCO2Emission(lane) for lane in lanes)
    return -(total_waiting + 0.1 * total_co2)  # Adjust weights as needed

# Main Training Loop
def train_dqn():
    traci.start(sumo_cmd)
    lanes = traci.lane.getIDList()
    state_size = 2 * len(lanes)  # haltingNumber + meanSpeed per lane
    action_size = len(traci.trafficlight.getAllProgramLogics("TL1")[0].phases)
    agent = DQNAgent(state_size, action_size)
    rewards_history = []

    for episode in range(EPISODES):
        traci.load(["-c", sumo_config])
        state = get_state()
        total_reward = 0
        step = 0

        while traci.simulation.getMinExpectedNumber() > 0:
            action = agent.act(state)
            traci.trafficlight.setPhase("TL1", action)
            traci.simulationStep()
            next_state = get_state()
            reward = calculate_reward()
            done = traci.simulation.getMinExpectedNumber() == 0  # Episode end
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step += 1

            if len(agent.memory) > BATCH_SIZE:
                agent.replay(BATCH_SIZE)

        rewards_history.append(total_reward)
        print(f"Episode: {episode + 1}, Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    traci.close()
    return rewards_history

# Plot Results
def plot_results(rewards_history):
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Performance")
    plt.grid(True)
    plt.savefig("dqn_training.png")
    plt.show()

if __name__ == "__main__":
    rewards = train_dqn()
    plot_results(rewards)