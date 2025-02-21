# Traffic-Optimizer
Using SUMO to simulate the urban environment
SUMO guide-
The folder structure is made and is to be followed.
traffic.net.xml is a important file to be made by netconvert tool provided by sumo using command- netconvert --node-files=network/traffic.nod.xml --edge-files=network/traffic.edg.xml --connection-files=network/traffic.con.xml --output-file=network/traffic.net.xml
It and traffic.rou.xml file is to be saved in config/network/ directory. Then run the run_sim.py 
with command python SIMscripts/run_sim.py