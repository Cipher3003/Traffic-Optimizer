from sumo_model import SUMOTrafficLightController

if __name__ == "__main__":
    # Path to your SUMO configuration file
    sumo_cfg = "ggn\osm.sumocfg"
    
    # Path to your network file
    net_file = "ggn\osm.net.xml"
    
    # Create and run the controller
    controller = SUMOTrafficLightController(net_file)
    controller.run_simulation(sumo_cfg, max_steps=10000)