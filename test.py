import numpy as np
import gcoordinator as gc

default_settings =  {
    "Print": {
        "nozzle": {
            "nozzle_diameter": 0.4,
            "filament_diameter": 1.75
        },
        "layer": {
            "layer_height": 0.2
        },
        "speed": {
            "print_speed": 5000,
            "travel_speed": 10000
        },
        "origin": {
            "x": 100,
            "y": 100
        },
        "fan_speed": {
            "fan_speed": 255
        },
        "temperature": {
            "nozzle_temperature": 999,
            "bed_temperature": 50
        },
        "travel_option": {
            "retraction": False,
            "retraction_distance": 2.0,
            "unretraction_distance": 2.0,
            "z_hop": False,
            "z_hop_distance": 3
        },
        "extrusion_option": {
            "extrusion_multiplier": 1.0
        }
    },
    "Hardware": {
        "kinematics": "Cartesian",
        "bed_size": {
            "bed_size_x": 200,
            "bed_size_y": 200,
            "bed_size_z": 205
        }
    },
    "Kinematics": {
        "NozzleTilt": {
            "tilt_code": "B",
            "rot_code": "A",
            "tilt_offset": 0.0,
            "rot_offset": 0
        },
        "BedTiltBC": {
            "tilt_code": "B",
            "rot_code": "C",
            "tilt_offset": 0.0,
            "rot_offset": 0,
            "div_distance": 0.5
        },
        "BedRotate": {
            "rot_code": "C",
            "rot_offset": 0.0,
            "div_distance": 0.5
        }
    }
}

gc.set_settings(default_settings)

radius = 15.0      
height = 10.0        
layer_height = 0.2   
points_per_layer = 100
bottom_layers = 3   
infill_distance = 0.2 

num_layers = int(height / layer_height)

full_object = []

for layer in range(num_layers):
    z = layer * layer_height
    
    theta = np.linspace(0, 2 * np.pi, points_per_layer + 1)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z_coords = np.full_like(theta, z)
    
    circle_path = gc.Path(x, y, z_coords)
    full_object.append(circle_path)
    
    if layer < bottom_layers:
        angle = np.pi / 4 if layer % 2 == 0 else -np.pi / 4
        infill = gc.line_infill(circle_path, infill_distance=infill_distance, angle=angle)
        full_object.append(infill)

gcode = gc.GCode(full_object)
gcode_text = gcode.generate()

