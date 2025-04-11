import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import traci
import sumolib
import pandas as pd
from typing import Tuple, List, Dict
import json
import time


class TrafficLightDQN(nn.Module):
    """Enhanced Deep Q-Network with improved architecture"""

    def __init__(self, state_size: int, action_size: int):
        super(TrafficLightDQN, self).__init__()
        self.state_processor = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )
        self.time_processor = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )
        self.action_processor = nn.Sequential(
            nn.Linear(action_size, 16),
            nn.ReLU()
        )
        self.combined = nn.Sequential(
            nn.Linear(64 + 16 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )  # Fixed: Added missing closing parenthesis

    def forward(self, traffic_state, prev_action, time):
        state_out = self.state_processor(traffic_state)
        time_out = self.time_processor(time)
        action_out = self.action_processor(prev_action)
        combined = torch.cat([state_out, time_out, action_out], dim=1)
        return self.combined(combined)


class TrafficLightAgent:
    """Enhanced DQN Agent with evaluation capabilities"""

    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.97
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.batch_size = 64
        self.train_freq = 4

        self.model = TrafficLightDQN(state_size, action_size)
        self.target_model = TrafficLightDQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.criterion = nn.SmoothL1Loss()  # Huber loss

        self.train_step = 0

    def act(self, state: Tuple, eval_mode: bool = False) -> int:
        if not eval_mode and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        traffic_state, prev_action, current_time = state
        with torch.no_grad():
            q_values = self.model(
                torch.FloatTensor(np.array([traffic_state])),
                torch.FloatTensor(np.array([prev_action])),
                torch.FloatTensor(np.array([[current_time]]))
            )
        return torch.argmax(q_values).item()

    def remember(self, experience: Tuple):
        self.memory.append(experience)

    def replay(self):
        if len(self.memory) < self.batch_size or self.train_step % self.train_freq != 0:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

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

        # Compute loss
        loss = self.criterion(current_q.squeeze(), target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()

        # Update exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.train_step += 1

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, filename: str):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)

    def load(self, filename: str):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.target_model.load_state_dict(self.model.state_dict())


class SUMOTrafficLightController:
    """Enhanced controller with comprehensive monitoring and evaluation"""

    def __init__(self, net_file: str):
        self.net = sumolib.net.readNet(net_file)
        self.traffic_lights = {}
        self.agents = {}
        self.metrics = {
            'training': pd.DataFrame(),
            'evaluation': pd.DataFrame()
        }
        self.state_size = 10  # Increased state capacity
        self.action_size = 4
        self.previous_vehicles = {}

    def initialize_agents(self):
        tl_ids = traci.trafficlight.getIDList()
        for tl_id in tl_ids:
            self.agents[tl_id] = TrafficLightAgent(
                state_size=self.state_size,
                action_size=self.action_size
            )
            lanes = traci.trafficlight.getControlledLanes(tl_id)
            self.traffic_lights[tl_id] = {
                'incoming_lanes': lanes,
                'current_phase': 0,
                'last_change': 0,
                'phase_duration': 30
            }
            self.previous_vehicles[tl_id] = {
                lane: set() for lane in lanes
            }

    def get_state(self, tl_id: str) -> Tuple:
        tl_info = self.traffic_lights[tl_id]
        state_features = []

        # Lane metrics
        for lane in tl_info['incoming_lanes'][:self.state_size // 2]:
            count = traci.lane.getLastStepVehicleNumber(lane)
            speed = traci.lane.getLastStepMeanSpeed(lane)
            waiting = traci.lane.getLastStepHaltingNumber(lane)
            state_features.extend([
                count / 50,    # Normalized assuming max 50 vehicles
                speed / 13.89, # Normalized to 50 km/h
                waiting / 20   # Normalized
            ])

        # Fill missing features
        while len(state_features) < self.state_size:
            state_features.append(0.0)

        # Previous action (one-hot)
        prev_action = np.zeros(self.action_size)
        prev_action[tl_info['current_phase']] = 1

        # Time features
        current_sim_time = traci.simulation.getTime() % 86400 / 86400

        return (np.array(state_features[:self.state_size]), prev_action, current_sim_time)

    def get_reward(self, tl_id: str) -> float:
        reward = 0
        tl_info = self.traffic_lights[tl_id]

        for lane in tl_info['incoming_lanes']:
            current_vehicles = set(traci.lane.getLastStepVehicleIDs(lane))
            prev_vehicles = self.previous_vehicles[tl_id][lane]

            # Reward for departed vehicles
            reward += len(prev_vehicles - current_vehicles) * 0.2

            # Penalty for current state
            reward -= traci.lane.getLastStepHaltingNumber(lane) * 0.1
            for veh_id in current_vehicles:
                reward -= min(traci.vehicle.getWaitingTime(veh_id), 30) * 0.02

            self.previous_vehicles[tl_id][lane] = current_vehicles

        return reward

    def _collect_metrics(self, step: int, mode: str) -> Dict:
        metrics = {
            'step': step,
            'total_waiting': 0,
            'average_speed': 0,
            'vehicles_arrived': traci.simulation.getArrivedNumber(),
            'total_co2': 0,
            'queue_length': 0
        }

        vehicle_ids = traci.vehicle.getIDList()
        if vehicle_ids:
            speeds = [traci.vehicle.getSpeed(veh) for veh in vehicle_ids]
            metrics['average_speed'] = np.mean(speeds)
            metrics['total_co2'] = sum(traci.vehicle.getCO2Emission(veh) for veh in vehicle_ids)

        for tl_info in self.traffic_lights.values():
            for lane in tl_info['incoming_lanes']:
                metrics['queue_length'] += traci.lane.getLastStepHaltingNumber(lane)
                metrics['total_waiting'] += sum(
                    traci.vehicle.getWaitingTime(veh)
                    for veh in traci.lane.getLastStepVehicleIDs(lane)
                )

        return metrics

    def run_simulation(self, sumo_cfg: str, max_steps: int = 10000):
        sumo_binary = sumolib.checkBinary('sumo')
        traci.start([sumo_binary, '-c', sumo_cfg])
        self.initialize_agents()

        metrics = []
        start_time = time.time()

        try:
            for step in range(max_steps):
                traci.simulationStep()
                current_time = traci.simulation.getTime()

                # Agent decisions
                for tl_id, agent in self.agents.items():
                    state = self.get_state(tl_id)
                    action = agent.act(state)
                    tl_info = self.traffic_lights[tl_id]

                    if action != tl_info['current_phase'] and current_time - tl_info['last_change'] > 5:
                        traci.trafficlight.setPhase(tl_id, action)
                        tl_info['current_phase'] = action
                        tl_info['last_change'] = current_time

                    reward = self.get_reward(tl_id)
                    next_state = self.get_state(tl_id)
                    done = (step == max_steps - 1)
                    agent.remember((state, action, reward, next_state, done))
                    agent.replay()

                    if step % 1000 == 0:
                        agent.update_target_model()

                # Collect metrics every 100 steps
                if step % 100 == 0:
                    metrics.append(self._collect_metrics(step, 'training'))
        except Exception as e:
            print("An exception occurred during simulation:", e)
        finally:
            traci.close()
            print("Training simulation closed.")

        # Save training results
        self.metrics['training'] = pd.DataFrame(metrics)
        print(f"Training completed in {time.time() - start_time:.2f}s")

        # Save models and metrics
        for tl_id in self.agents:
            self.agents[tl_id].save(f'traffic_light_{tl_id}.pth')
        self.metrics['training'].to_csv('training_metrics.csv', index=False)

    def evaluate(self, sumo_cfg: str, max_steps: int = 3600):
        sumo_binary = sumolib.checkBinary('sumo')
        traci.start([sumo_binary, '-c', sumo_cfg])
        self.initialize_agents()

        # Load trained models
        for tl_id in self.agents:
            self.agents[tl_id].load(f'traffic_light_{tl_id}.pth')
            self.agents[tl_id].epsilon = 0.01  # Minimal exploration

        metrics = []
        start_time = time.time()

        try:
            for step in range(max_steps):
                traci.simulationStep()

                for tl_id, agent in self.agents.items():
                    state = self.get_state(tl_id)
                    action = agent.act(state, eval_mode=True)
                    tl_info = self.traffic_lights[tl_id]

                    if action != tl_info['current_phase'] and traci.simulation.getTime() - tl_info['last_change'] > 5:
                        traci.trafficlight.setPhase(tl_id, action)
                        tl_info['current_phase'] = action
                        tl_info['last_change'] = traci.simulation.getTime()

                if step % 100 == 0:
                    metrics.append(self._collect_metrics(step, 'evaluation'))
        except Exception as e:
            print("An exception occurred during evaluation:", e)
        finally:
            traci.close()
            print("Evaluation simulation closed.")

        # Save evaluation results
        self.metrics['evaluation'] = pd.DataFrame(metrics)
        print(f"Evaluation completed in {time.time() - start_time:.2f}s")
        self.metrics['evaluation'].to_csv('evaluation_metrics.csv', index=False)


if __name__ == "__main__":
    # Use raw strings for file paths to avoid backslash issues
    sumo_cfg = r"Traffic-Optimizer\2025-04-04-13-04-14\osm.sumocfg"
    net_file = r"Traffic-Optimizer\2025-04-04-13-04-14\osm.net.xml.gz"

    controller = SUMOTrafficLightController(net_file)

    # Training phase
    controller.run_simulation(sumo_cfg, max_steps=10000)

    # Evaluation phase
    controller.evaluate(sumo_cfg, max_steps=3600)
