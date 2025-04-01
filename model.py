import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import traci
import sumolib
import os
import time
from typing import Tuple, List, Dict

class TrafficLightDQN(nn.Module):
    """Deep Q-Network for traffic light optimization"""
    def __init__(self, state_size: int, action_size: int):
        super(TrafficLightDQN, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size + 1, 128)  # +1 for time
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, traffic_state, prev_action, time):
        x = torch.cat([traffic_state, prev_action, time], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        return self.fc4(x)

class TrafficLightAgent:
    """PyTorch DQN agent for traffic light optimization"""
    
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        
        # Main network
        self.model = TrafficLightDQN(state_size, action_size)
        
        # Target network
        self.target_model = TrafficLightDQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
    def act(self, state: Tuple) -> int:
        """Select action using ε-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        traffic_state, prev_action, time = state
        with torch.no_grad():
            q_values = self.model(
                torch.FloatTensor(np.array([traffic_state])),
                torch.FloatTensor(np.array([prev_action])),
                torch.FloatTensor(np.array([[time]]))
            )
        return torch.argmax(q_values).item()
    
    def remember(self, state: Tuple, action: int, reward: float, next_state: Tuple, done: bool):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train on past experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for state, action, reward, next_state, done in minibatch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        # Convert to tensors
        traffic_states = torch.FloatTensor(np.array([s[0] for s in states]))
        prev_actions = torch.FloatTensor(np.array([s[1] for s in states]))
        times = torch.FloatTensor(np.array([[s[2]] for s in states]))
        
        next_traffic_states = torch.FloatTensor(np.array([s[0] for s in next_states]))
        next_prev_actions = torch.FloatTensor(np.array([s[1] for s in next_states]))
        next_times = torch.FloatTensor(np.array([[s[2]] for s in next_states]))
        
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        # Current Q values
        current_q = self.model(traffic_states, prev_actions, times).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_model(next_traffic_states, next_prev_actions, next_times).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update
        loss = self.criterion(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        """Update target network weights"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def save(self, filename: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), filename)
    
    def load(self, filename: str):
        """Load model weights"""
        self.model.load_state_dict(torch.load(filename))
        self.target_model.load_state_dict(self.model.state_dict())

class SUMOTrafficLightController:
    """Controller for SUMO traffic lights using PyTorch DQN"""
    
    def __init__(self, net_file: str):
        self.net = sumolib.net.readNet(net_file)
        self.traffic_lights = {}
        self.agents = {}
        self.state_size = 8  # Number of lanes to monitor
        self.action_size = 4  # Number of possible phases
        self.vehicle_counts = {}
        self.waiting_times = {}
        
    def initialize_agents(self):
        """Initialize DQN agents for each traffic light"""
        tl_ids = traci.trafficlight.getIDList()
        
        for tl_id in tl_ids:
            self.agents[tl_id] = TrafficLightAgent(
                state_size=self.state_size,
                action_size=self.action_size
            )
            
            self.traffic_lights[tl_id] = {
                'incoming_lanes': traci.trafficlight.getControlledLanes(tl_id),
                'current_phase': 0,
                'last_change': 0,
                'phase_duration': 30
            }
    
    def get_state(self, tl_id: str) -> Tuple:
        """Get current state for a traffic light"""
        tl_info = self.traffic_lights[tl_id]
        
        # Get vehicle counts on incoming lanes
        vehicle_counts = []
        for lane in tl_info['incoming_lanes']:
            vehicle_counts.append(traci.lane.getLastStepVehicleNumber(lane))
        
        # Pad with zeros if needed
        while len(vehicle_counts) < self.state_size:
            vehicle_counts.append(0)
        
        # Normalize counts
        max_count = max(vehicle_counts) if max(vehicle_counts) > 0 else 1
        normalized_counts = [c/max_count for c in vehicle_counts]
        
        # Previous action (one-hot encoded)
        prev_action = np.zeros(self.action_size)
        prev_action[tl_info['current_phase']] = 1
        
        # Current simulation time (normalized to 0-1)
        current_time = traci.simulation.getTime() % 86400 / 86400
        
        return (np.array(normalized_counts), prev_action, current_time)
    
    def get_reward(self, tl_id: str) -> float:
        """Calculate reward for current state"""
        tl_info = self.traffic_lights[tl_id]
        reward = 0
        
        # Negative reward for waiting vehicles
        for lane in tl_info['incoming_lanes']:
            waiting_vehicles = traci.lane.getLastStepHaltingNumber(lane)
            reward -= waiting_vehicles * 0.1
            
            # Additional penalty for long waiting times
            for veh_id in traci.lane.getLastStepVehicleIDs(lane):
                waiting_time = traci.vehicle.getWaitingTime(veh_id)
                reward -= min(waiting_time, 10) * 0.01
        
        # Small positive reward for keeping traffic flowing
        reward += 0.01
        
        return reward
    
    def run_simulation(self, sumo_cfg: str, max_steps: int = 10000, target_update: int = 100):
        """Run the simulation and train agents"""
        sumo_binary = sumolib.checkBinary('sumo')
        traci.start([sumo_binary, '-c', sumo_cfg])
        
        self.initialize_agents()
        
        step = 0
        while step < max_steps:
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            
            # Control each traffic light
            for tl_id, agent in self.agents.items():
                # Get current state
                state = self.get_state(tl_id)
                
                # Get action from agent
                action = agent.act(state)
                
                # Change phase if needed (with minimum duration check)
                if (action != self.traffic_lights[tl_id]['current_phase'] and 
                    current_time - self.traffic_lights[tl_id]['last_change'] > 5):
                    traci.trafficlight.setPhase(tl_id, action)
                    self.traffic_lights[tl_id]['current_phase'] = action
                    self.traffic_lights[tl_id]['last_change'] = current_time
                
                # Get reward
                reward = self.get_reward(tl_id)
                
                # Get next state
                next_state = self.get_state(tl_id)
                
                # Store experience
                agent.remember(state, action, reward, next_state, False)
                
                # Train agent
                agent.replay()
                
                # Update target network periodically
                if step % target_update == 0:
                    agent.update_target_model()
            
            step += 1
        
        # Save trained models
        for tl_id, agent in self.agents.items():
            agent.save(f'traffic_light_{tl_id}.pth')
        
        traci.close()

if __name__ == "__main__":
    # Path to your SUMO configuration and network files
    sumo_cfg = "ggn\osm.sumocfg"
    net_file = "ggn\osm.net.xml"
    
    # Create and run the controller
    controller = SUMOTrafficLightController(net_file)
    controller.run_simulation(sumo_cfg, max_steps=10000)