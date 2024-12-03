import os
import json
import pandas as pd
import numpy as np
from tsfresh import extract_features, select_features

# Función para estructurar las ventanas en el formato requerido por tsfresh
def prepare_tsfresh_data(processed_data):
    """
    Convierte las ventanas procesadas en un DataFrame adecuado para tsfresh.
    """
    tsfresh_data = []
    for sensor, segments in processed_data.items():
        for segment_index, segment in enumerate(segments):
            for window_index, window in enumerate(segment):
                for reading in window['data']:
                    tsfresh_data.append({
                        "id": f"{sensor}_{segment_index}_{window_index}",  # ID único para cada ventana
                        "time": reading["t"],  # Tiempo de la lectura
                        "value": reading,  # Valores de las señales
                        "sensor": sensor,  # Sensor actual
                    })
    return pd.DataFrame(tsfresh_data)

# Función para extraer y filtrar características usando tsfresh
def extract_tsfresh_features(processed_data, labels, column_sort="time", column_value="value"):
    """
    Extrae características con tsfresh y selecciona las más relevantes.
    """
    # Prepara los datos para tsfresh
    tsfresh_data = prepare_tsfresh_data(processed_data)

    # Extrae características
    features = extract_features(
        tsfresh_data,
        column_id="id",
        column_sort=column_sort,
        column_value=column_value
    )

    # Selecciona las características relevantes basadas en las etiquetas
    relevant_features = select_features(features, labels)
    return relevant_features

# Función principal para procesar datos y usar tsfresh
def process_json_files_with_tsfresh(input_folder, output_folder, labels, interval=20, window_size=100, overlap=0.5, max_gap=1000):
    """
    Procesa archivos JSON y aplica tsfresh para extraer características relevantes.
    """
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

            # Extraer características relevantes
            relevant_features = extract_tsfresh_features(processed_data, labels)

            # Convertir a JSON
            relevant_features_json = relevant_features.to_dict(orient="index")

            # Guardar las características como JSON
            with open(output_path, "w") as json_file:
                json.dump(relevant_features_json, json_file, indent=4)

            print(f"Archivo procesado y guardado: {output_path}")

# Configuración
input_folder = config.walking_data_path_parkinson  # Ruta a la carpeta de datos con Parkinson
output_folder = config.p_walking_data_path_parkinson  # Ruta a la carpeta de salida
labels = pd.Series({  # Etiquetas para tsfresh
    "derecha_0_0": 1,  # Ejemplo de etiquetas: 1 para Parkinson
    "izquierda_0_0": 0,  # 0 para no Parkinson
    # Agrega más etiquetas según tus datos
})

# Procesar los archivos
process_json_files_with_tsfresh(input_folder, output_folder, labels, interval=20, window_size=100, overlap=0.5, max_gap=1000)
