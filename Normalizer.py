import os
import json
import numpy as np

def process_sensor_data(sensor_data):
    canales = ['a', 'b', 'g', 'x', 'y', 'z']
    
    for canal in canales:
        values = np.array([entry[canal] for entry in sensor_data])
        average = np.mean(values)
        for entry in sensor_data:
            entry[field] -= average
    
    return sensor_data

def process_json_file(input_path, output_path):
    with open(input_path, 'r') as file:
        data = json.load(file)
    
    sensors = ['derecha', 'izquierda', 'espina_base']
    
    for sensor in sensors:
        if sensor in data:
            data[sensor] = process_sensor_data(data[sensor])
    
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=2)

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            process_json_file(input_path, output_path)

input_folder = './CTL-PSANO'
output_folder = './normalizadoSANO'
process_folder(input_folder, output_folder)