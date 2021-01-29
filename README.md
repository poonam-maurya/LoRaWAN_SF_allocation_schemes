# LoRaWAN_SF_allocation_schemes
This Repository contaions all the simulation codes for following LoRaWAN SF-allocation schemes
## Fixed SF
End-nodes have a fixed predefined Spreading Factor.
## Random SF
End-nodes in any zone can have any Spreading Factor Randomly.
## EIB
End-nodes get SF based on their location in the zones. Zones are defined with equal cell spacing.
## EAB
End-nodes get SF based on their location in the zones. Zones are defined with equal area.

## Installing Dependencies
pip install -r requirements.txt

### Execution
Each folder contains 2 Files.
#### Main Script
The file name of form <allocation scheme>.py. 
Executing this script will provide us the Instantaneous packet success rate on a given number of nodes.
Running this file multiple times will provide us better results on the packet success.

#### Main_external_event.py
This engine file is to execute the main file multiple times in sub-processes. This provides us better results for the Main Script.

### SavedVars
When the scripts are executed, SavedVars folder is formed in the local directory.
This folder is the storage location for all the variables required in the simulator.
