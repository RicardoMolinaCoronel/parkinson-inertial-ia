import os
import json
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import Global.config as config
import pywt
from scipy.signal import butter, filtfilt, welch

# Función para aplicar el filtro Butterworth
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Filtrar y dividir en segmentos por brechas grandes
def split_by_large_gaps(sensor_data, max_gap=1000):
    df = pd.DataFrame(sensor_data)
    df['gap'] = df['t'].diff().fillna(0)  # Calcular la brecha entre lecturas consecutivas
    df['segment'] = (df['gap'] > max_gap).cumsum()  # Crear un segmento nuevo cuando hay una brecha
    segments = [group.drop(columns=['gap', 'segment']) for _, group in df.groupby('segment')]
    return segments

# Función personalizada para manejar tipos no serializables por JSON
def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)):  # Convertir tipos numpy a int
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):  # Convertir tipos numpy a float
        return float(obj)
    elif isinstance(obj, np.ndarray):  # Convertir numpy arrays a listas
        return obj.tolist()
    else:
        raise TypeError(f"Type {type(obj)} not serializable")

# Función para uniformizar datos
def uniformize_data(sensor_data, interval=20):
    min_time = sensor_data['t'].min()
    max_time = sensor_data['t'].max()
    uniform_time = np.arange(min_time, max_time + interval, interval)
    interpolated_data = {'t': uniform_time}
    for col in sensor_data.columns:
        if col != 't':
            f = interp1d(sensor_data['t'], sensor_data[col], kind='linear', fill_value='extrapolate')
            interpolated_data[col] = f(uniform_time)
    return pd.DataFrame(interpolated_data)


# Función para crear ventanas con la misma cantidad de lecturas
def create_windows_with_fixed_size(df, window_size, overlap=0.5):
    step_size = int(window_size * (1 - overlap))
    windows = []
    start = 0
    while start + window_size <= len(df):
        window = df.iloc[start:start + window_size]
        start_time = window['t'].iloc[0]
        end_time = window['t'].iloc[-1]
        window = window.copy()
        window['time_diff'] = window['t'] - start_time
        windows.append({
            "start_time": start_time,
            "end_time": end_time,
            "data": window.to_dict(orient='records')
        })
        start += step_size
    return windows





# Procesar datos de múltiples sensores
def process_sensor_data(json_data, interval=20, window_size=100, overlap=0.5, max_gap=1000):
    processed_data = {}
    cutoff = 12
    target_frequency = 50
    for sensor, readings in json_data.items():
        segments = split_by_large_gaps(readings, max_gap=max_gap)
        processed_segments = []
        for segment in segments:
            uniform_data = uniformize_data(segment, interval=interval)
            # Aplicación del filtro Butterworth a las columnas de interés
            #for col in ['a', 'b', 'g', 'x', 'y', 'z']:
            #   uniform_data[col] = butter_lowpass_filter(uniform_data[col], cutoff, target_frequency)

            windows = create_windows_with_fixed_size(uniform_data, window_size=window_size, overlap=overlap)
            #print("ENTRE BRO")
            # Filtrar ventanas con lecturas diferentes a 100
            filtered_windows = []
            for idx, window in enumerate(windows):
                num_readings = len(window['data'])  # Número de lecturas en la ventana
                if num_readings == window_size:
                    filtered_windows.append(window)  # Mantener solo ventanas válidas
                else:
                    print(f"⚠️ Eliminando ventana {idx} en el sensor {sensor} con {num_readings} lecturas (diferente a {window_size}).")

            processed_segments.append(filtered_windows)
            #processed_segments.append(windows)
        processed_data[sensor] = processed_segments
    return processed_data


# Leer y procesar todos los archivos JSON de una carpeta
def process_json_files(input_folder, output_folder, interval=20, window_size=75, overlap=0.01, max_gap=1000):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.json'):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, f"processed_{file_name}")

            with open(input_path, 'r') as file:
                json_data = json.load(file)

            # Procesar datos
            processed_data = process_sensor_data(json_data, interval=interval, window_size=window_size, overlap=overlap, max_gap=max_gap)

            with open(output_path, 'w') as outfile:
                json.dump(processed_data, outfile, indent=4, default=convert_to_serializable)
            print(f"Archivo procesado y guardado: {output_path}")



# Configuración
input_folder = config.rdced_walking_data_path_parkinson # Ruta a la carpeta de entrada
output_folder = config.p_walking_data_path_parkinson  # Ruta a la carpeta de salida

# Ejecutar el procesamiento
process_json_files(input_folder, output_folder, interval=20, window_size=100, overlap=0.5, max_gap=1000)

# Configuración
input_folder = config.walking_data_path_no_parkinson # Ruta a la carpeta de entrada
output_folder = config.p_walking_data_path_no_parkinson  # Ruta a la carpeta de salida

# Ejecutar el procesamiento
process_json_files(input_folder, output_folder, interval=20, window_size=100, overlap=0.5, max_gap=1000)
