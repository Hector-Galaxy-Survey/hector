import yaml
from pathlib import Path

import hector
hector_path = str(hector.__path__[0]) + '/'

# Load the file with all the constant values in. We need to do some work to get the paths correct...
# This file is 3 folders deep from the 'hop' folder, so we need to go three "parents" up
location_of_module = Path(__file__).parent.parent.parent
with open(f"{hector_path}/utils/HECTOR_CONSTANTS.yaml", 'r') as f:
    constants_dict = yaml.safe_load(f)

circular_magnet_radius                     = constants_dict['circular_magnet_radius']
rectangle_magnet_width                     = constants_dict['rectangle_magnet_width']
rectangle_magnet_length                    = constants_dict['rectangle_magnet_length']
circular_rectangle_magnet_distance         = constants_dict['circular_rectangle_magnet_distance']
circular_rectangle_magnet_center_distance  = (circular_magnet_radius + circular_rectangle_magnet_distance
                                              + rectangle_magnet_length / 2.0)
HECTOR_plate_center_coordinate             = constants_dict['HECTOR_plate_center_coordinate']
HECTOR_plate_radius                        = constants_dict['HECTOR_plate_radius'] # previous value=260.0
robot_arm_length                           = constants_dict['robot_arm_length']
robot_arm_width                            = constants_dict['robot_arm_width']
circular_magnet_pickuparea_length          = robot_arm_length
circular_magnet_pickuparea_width           = (3.0 * robot_arm_width / 2.0) + circular_magnet_radius
rectangle_magnet_pickuparea_length         = robot_arm_length

if (robot_arm_width < ((rectangle_magnet_length - robot_arm_width) / 2)):
    rectangle_magnet_pickuparea_width = (robot_arm_width + rectangle_magnet_length) / 2
elif (robot_arm_width >= ((rectangle_magnet_length - robot_arm_width) / 2)):
    rectangle_magnet_pickuparea_width = 2 * robot_arm_width
#rectangle_magnet_pickuparea_width          = 0.5 * (rectangle_magnet_length + 3.0 * robot_arm_width)

# robot file center coordinates, depending on the mechanical mounting of the plate
robot_center_x = constants_dict['robot_center_x']
robot_center_y = constants_dict['robot_center_y']
#print(f"Robot centre is at {robot_center_x}, {robot_center_y}")