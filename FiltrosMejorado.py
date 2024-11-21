import os
import json
import numpy as np
from scipy.signal import butter, filtfilt
import Global.config as config

def low_pass_filter(data, cutoff=0.1, fs=100, order=4):
    """
    Aplicar filtro paso bajo a una señal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def calculate_vector_magnitude(data):
    """
    Calcular la magnitud del vector de las señales.
    """
    x = np.array([item["x"] for item in data])
    y = np.array([item["y"] for item in data])
    z = np.array([item["z"] for item in data])
    return np.sqrt(x ** 2 + y ** 2 + z ** 2)


def process_imu_data(input_file):
    """
    Procesar datos IMU desde un archivo JSON.
    """
    with open(input_file, 'r') as file:
        imu_data = json.load(file)

    results = {}
    for sensor, readings in imu_data.items():
        # Filtros paso bajo
        filtered_data = {
            "a": low_pass_filter([item["a"] for item in readings]),
            "b": low_pass_filter([item["b"] for item in readings]),
            "g": low_pass_filter([item["g"] for item in readings]),
            "x": low_pass_filter([item["x"] for item in readings]),
            "y": low_pass_filter([item["y"] for item in readings]),
            "z": low_pass_filter([item["z"] for item in readings]),
        }

        # Magnitudes
        gyro_magnitude = np.sqrt(
            np.array(filtered_data["a"]) ** 2 +
            np.array(filtered_data["b"]) ** 2 +
            np.array(filtered_data["g"]) ** 2
        )
        accel_magnitude = np.sqrt(
            np.array(filtered_data["x"]) ** 2 +
            np.array(filtered_data["y"]) ** 2 +
            np.array(filtered_data["z"]) ** 2
        )

        # Agregar "t" al resultado
        timestamps = [item["t"] for item in readings]

        results[sensor] = [
            {
                "t": timestamps[i],
                "gyro_magnitude": gyro_magnitude[i],
                "accel_magnitude": accel_magnitude[i]
            }
            for i in range(len(timestamps))
        ]

    return results


def process_multiple_files(input_folder, output_folder):
    """
    Procesar múltiples archivos JSON en una carpeta.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Obtener lista de archivos JSON en la carpeta de entrada
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

    for json_file in json_files:
        input_file_path = os.path.join(input_folder, json_file)
        output_file_path = os.path.join(output_folder, f"processed_{json_file}")

        # Procesar archivo
        processed_data = process_imu_data(input_file_path)

        # Guardar resultados en un nuevo archivo JSON
        with open(output_file_path, 'w') as output_file:
            json.dump(processed_data, output_file, indent=4)

        print(f"Procesado y guardado: {output_file_path}")


# Directorios de entrada y salida
input_folder = config.walking_data_path_no_parkinson # Cambia esto por la ruta de tu carpeta de entrada
output_folder = config.filtered_data_path_no_parkinson  # Cambia esto por la ruta de tu carpeta de salida

# Procesar archivos
process_multiple_files(input_folder, output_folder)
