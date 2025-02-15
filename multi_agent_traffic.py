import traci
import numpy as np
from pettingzoo import AECEnv
from gym import spaces

class MultiAgentTrafficEnv(AECEnv):
    def __init__(self, intersections):
        super().__init__()
        self.intersections = intersections
        self.agents = {i: f"agent_{i}" for i in range(len(intersections))}

        # Start SUMO
        sumo_binary = "sumo-gui"  # Use "sumo" for headless mode
        traci.start([sumo_binary, "-c", "traffic_simulation.sumocfg"])

        self.observation_spaces = {a: spaces.Box(low=0, high=100, shape=(3,), dtype=np.float32) for a in self.agents}
        self.action_spaces = {a: spaces.Discrete(3) for a in self.agents}

    def step(self, actions):
        rewards = {}
        observations = {}
        done = False

        for agent_id, action in actions.items():
            intersection_id = self.intersections[int(agent_id.split("_")[1])]

            if action == 0:  # Keep same
                pass
            elif action == 1:  # Extend Green
                traci.trafficlight.setPhase(intersection_id, 0)
            elif action == 2:  # Switch to Red
                traci.trafficlight.setPhase(intersection_id, 1)

            traci.simulationStep()

            queue_length = traci.edge.getLastStepVehicleNumber(f"{intersection_id}_in")
            avg_speed = traci.edge.getLastStepMeanSpeed(f"{intersection_id}_in")
            neighbor_signal = traci.trafficlight.getPhase(intersection_id)

            observations[agent_id] = np.array([queue_length, avg_speed, neighbor_signal], dtype=np.float32)
            rewards[agent_id] = -queue_length + avg_speed

        return observations, rewards, done, {}

    def reset(self):
        traci.load(["-c", "traffic_simulation.sumocfg"])
        return {agent: np.array([0, 0, 0], dtype=np.float32) for agent in self.agents}

    def render(self, mode="human"):
        pass

    def close(self):
        traci.close()
