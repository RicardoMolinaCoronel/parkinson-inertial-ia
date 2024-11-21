import os
import json

# Function to load JSON files from a folder
def load_json_files(folder_path):
    data = {"derecha": [], "izquierda": [], "espina_base": []}
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            with open(os.path.join(folder_path, filename), 'r') as file:
                file_data = json.load(file)
                for sensor in data.keys():
                    if sensor in file_data:
                        data[sensor].extend(file_data[sensor])
    return data

def create_windows(data, window_size, overlap_ratio, sampling_rate, overlapping=True):
    windows = []
    window_samples = int(window_size * sampling_rate)
    step_size = int(window_samples * (1 - overlap_ratio)) if overlapping else window_samples
    
    for sensor, sensor_data in data.items():
        i = 0
        while i + window_samples <= len(sensor_data):
            window = sensor_data[i:i + window_samples]
            start_time = window[0]["t"]
            end_time = window[-1]["t"]
            if (end_time - start_time) >= (window_size * 1000):
                windows.append((sensor, window))
            i += step_size
    return windows

def process_imu_data(folder1, folder2, window_size, overlap_ratio, sampling_rate):
    data_sano = load_json_files(folder1)
    data_parkinson = load_json_files(folder2)
    windows_sano = create_windows(data_sano, window_size, overlap_ratio, sampling_rate, overlapping=True)
    windows_parkinson = create_windows(data_parkinson, window_size, overlap_ratio, sampling_rate, overlapping=False)

    return windows_sano, windows_parkinson

# Parameters
folder1 = "./SEGMENTED_DATA/CTL-PSANO/WALKING"  
folder2 = "./SEGMENTED_DATA/PAC-PARKINSON/WALKING"  
window_size = 2.5  
overlap_ratio = 0.75  
sampling_rate = 50  
windows_sano, windows_parkinson = process_imu_data(folder1, folder2, window_size, overlap_ratio, sampling_rate)

print(f"Ventanas con overlap (sano): {len(windows_sano)}")
print(f"Ventanas sin overlap (parkinson): {len(windows_parkinson)}")


