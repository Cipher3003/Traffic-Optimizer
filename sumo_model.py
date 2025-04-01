import traci
import sumolib
import os
import time
from collections import defaultdict
from model_architecture import TrafficLightOptimizer

class SUMOTrafficLightController:
    def __init__(self, net_file):
        # Load the network
        self.net = sumolib.net.readNet(net_file)
        self.traffic_lights = {}
        
        # Initialize RL agent for each traffic light
        self.agents = {}
        
        # Define state and action sizes
        self.state_size = 8  # Number of lanes to monitor
        self.action_size = 4  # Number of possible phases
        
        # Initialize data structures
        self.vehicle_counts = defaultdict(int)
        self.waiting_times = defaultdict(float)
        
    def initialize_agents(self):
        # Get all traffic light IDs from the network
        tl_ids = traci.trafficlight.getIDList()
        
        for tl_id in tl_ids:
            # Initialize an agent for each traffic light
            self.agents[tl_id] = TrafficLightOptimizer(
                state_size=self.state_size,
                action_size=self.action_size
            )
            
            # Store traffic light information
            self.traffic_lights[tl_id] = {
                'incoming_lanes': traci.trafficlight.getControlledLanes(tl_id),
                'current_phase': 0,
                'last_change': 0,
                'phase_duration': 30  # Default phase duration
            }
    
    def get_state(self, tl_id):
        """Get the current state for a traffic light"""
        tl_info = self.traffic_lights[tl_id]
        
        # Get vehicle counts on incoming lanes
        vehicle_counts = []
        for lane in tl_info['incoming_lanes']:
            vehicle_counts.append(traci.lane.getLastStepVehicleNumber(lane))
        
        # Pad with zeros if we don't have enough lanes
        while len(vehicle_counts) < self.state_size:
            vehicle_counts.append(0)
        
        # Normalize counts
        max_count = max(vehicle_counts) if max(vehicle_counts) > 0 else 1
        normalized_counts = [c/max_count for c in vehicle_counts]
        
        # Get previous action (one-hot encoded)
        prev_action = np.zeros(self.action_size)
        prev_action[tl_info['current_phase']] = 1
        
        # Get current simulation time (normalized to 0-1 for 24h period)
        current_time = traci.simulation.getTime() % 86400 / 86400
        
        return (np.array(normalized_counts), prev_action, current_time)
    
    def get_reward(self, tl_id):
        """Calculate reward for the current state"""
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
    
    def run_simulation(self, sumo_cfg, max_steps=1000):
        # Start SUMO simulation
        sumo_binary = sumolib.checkBinary('sumo')
        traci.start([sumo_binary, '-c', sumo_cfg])
        
        # Initialize agents
        self.initialize_agents()
        
        step = 0
        while step < max_steps:
            traci.simulationStep()
            current_time = traci.simulation.getTime()
            
            # Control each traffic light
            for tl_id in self.agents.keys():
                # Get current state
                state = self.get_state(tl_id)
                
                # Get action from agent
                action = self.agents[tl_id].act(state)
                
                # Change phase if needed
                if action != self.traffic_lights[tl_id]['current_phase']:
                    # Minimum phase duration check
                    if current_time - self.traffic_lights[tl_id]['last_change'] > 5:
                        traci.trafficlight.setPhase(tl_id, action)
                        self.traffic_lights[tl_id]['current_phase'] = action
                        self.traffic_lights[tl_id]['last_change'] = current_time
                
                # Get reward
                reward = self.get_reward(tl_id)
                
                # Get next state
                next_state = self.get_state(tl_id)
                
                # Store experience in memory
                self.agents[tl_id].remember(state, action, reward, next_state, False)
                
                # Train agent
                self.agents[tl_id].replay(32)
            
            step += 1
        
        # Save trained models
        for tl_id, agent in self.agents.items():
            agent.save(f'traffic_light_{tl_id}.h5')
        
        traci.close()