# PV--hydrogen-hybrid-system
Code to reproduce the results of the work "Capacity optimization and economic analysis of PV--hydrogen hybrid system with physical solar power curve modeling."

# Requirement: 
If one wants to run the code in this github repository, he/she needs to install the software of Python(3.9.7), Gurobi (9.5.1), Anaconda3. Besides, the installation of the Python packages in the top of each .py file is also a necessity. 

# Code:
A total of eight Python scripts is offered to ensure the reproducibility. data_download.py is used to download the TMY data of the city with the specific longitude and latitude values; contour_map.py draws the countour map (Fig.5 in the paper); pvmodeling_model_chain.py provides the PV power simulated from the physical solar power curve modeling; pvmodeling_conventional.py simulates the PV power using PV conventional approach; The optimal equipment capacities and economics of the PV–hydrogen hybrid system can be acquired in Photo_hydrogen_coupling.py, whereas the optimal results under the different level of government surpoot is obtained in Lambda_air_environmental.py; heilongjiang_tmy.py finds the tilt and azimuth angles of the inclined surface with the maximum capacity factor; Heilongjiang_optimize.py provides the optimization results in different cities of Heilongjiang Province.

As for the other results, one can manually modify the related parameters. 

# Data
the file 
