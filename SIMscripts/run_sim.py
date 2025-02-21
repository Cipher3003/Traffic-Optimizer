import traci

sumo_cmd = ["sumo-gui", "-c", "config/trf_sim.sumocfg"]
traci.start(sumo_cmd)

for step in range(1000):  # Run for 1000 time steps
    traci.simulationStep()
    print("Step:", step)

traci.close()
