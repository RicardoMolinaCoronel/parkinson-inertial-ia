import json
import pandas as pd
import numpy as np
import Global.config as config
def low_pass_filter(data, window_size=5):
    """Aplica un filtro paso bajo usando una media móvil y rellena valores NaN."""
    filtered_data = data.rolling(window=window_size, center=True).mean()
    # Rellenar los NaN generados en los bordes
    filtered_data.fillna(method='bfill', inplace=True)
    filtered_data.fillna(method='ffill', inplace=True)
    return filtered_data

def add_gaussian_noise(data, mean=0, std=1):
    """Añade ruido gaussiano a los datos."""
    noise = np.random.normal(mean, std, size=len(data))
    return data + noise

def temporal_transformation(data, method="normalize"):
    """Aplica transformaciones temporales a los datos."""
    if method == "normalize":
        return (data - data.min()) / (data.max() - data.min())
    elif method == "standardize":
        return (data - data.mean()) / data.std()
    else:
        return data

def process_imu_data(file_path, window_duration=2000, overlap=0.5, upsampling_interval=10, max_gap=100):
    """
    Procesa los datos de señales IMU aplicando filtros, ruido, transformaciones,
    upsampling y dividiéndolos en ventanas con solapamiento.

    Args:
        file_path (str): Ruta del archivo JSON con los datos.
        window_duration (int): Duración de cada ventana en milisegundos.
        overlap (float): Porcentaje de solapamiento entre ventanas (0.0 a 1.0).
        upsampling_interval (int): Intervalo para el upsampling en milisegundos.
        max_gap (int): Máximo intervalo permitido entre puntos consecutivos antes de considerarlo como una brecha.

    Returns:
        dict: Diccionario con las ventanas procesadas por cada sensor.
    """
    # Cargar los datos del archivo JSON
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Crear un DataFrame para cada sensor
    sensors = {}
    for sensor_name, sensor_data in data.items():
        df = pd.DataFrame(sensor_data)
        #df['t'] = df['millis']  # Usar millis como referencia de tiempo
        sensors[sensor_name] = df

    processed_data = {}

    for sensor_name, df in sensors.items():
        # Verificar si hay datos en el sensor
        if df.empty:
            continue

        # Detectar rangos válidos donde no hay brechas mayores al max_gap
        df['time_diff'] = df['t'].diff().fillna(0)
        df['segment'] = (df['time_diff'] > max_gap).cumsum()

        # Procesar cada segmento por separado
        segments = []
        for _, segment_data in df.groupby('segment'):
            if segment_data['t'].size > 1:  # Solo considerar segmentos con más de un punto
                start_time = segment_data['t'].min()
                end_time = segment_data['t'].max()

                # Crear un rango uniforme para el upsampling dentro del segmento
                upsampled_time = np.arange(start_time, end_time + upsampling_interval, upsampling_interval)
                upsampled_df = pd.DataFrame({'t': upsampled_time})

                # Interpolar los datos
                segment_data = pd.merge(upsampled_df, segment_data, on='t', how='left')
                segment_data.interpolate(method='linear', inplace=True)
                segment_data.fillna(method='bfill', inplace=True)
                segment_data.fillna(method='ffill', inplace=True)

                # Aplicar el filtro paso bajo a cada eje
                for col in ['a', 'b', 'g', 'x', 'y', 'z']:
                    segment_data[col].interpolate(method='linear', inplace=True)
                    segment_data[col].fillna(method='bfill', inplace=True)
                    segment_data[col].fillna(method='ffill', inplace=True)
                    segment_data[col] = low_pass_filter(segment_data[col])

                # Añadir ruido gaussiano
                for col in ['a', 'b', 'g', 'x', 'y', 'z']:
                    segment_data[col] = add_gaussian_noise(segment_data[col])

                # Aplicar transformaciones temporales
                for col in ['a', 'b', 'g', 'x', 'y', 'z']:
                    segment_data[col] = temporal_transformation(segment_data[col])

                segments.append(segment_data)

        # Concatenar los segmentos procesados
        interpolated_df = pd.concat(segments, ignore_index=True)


        # Dividir los datos en ventanas con solapamiento
        windows = []
        step = int(window_duration * (1 - overlap))  # Paso entre ventanas considerando el solapamiento
        start_time = interpolated_df['t'].min()
        end_time = interpolated_df['t'].max()

        while start_time <= end_time:
            window_end = start_time + window_duration
            window_data = interpolated_df[(interpolated_df['t'] >= start_time) & (interpolated_df['t'] < window_end)].copy()
            if not window_data.empty:
                windows.append({
                    "start_time": start_time,
                    "end_time": window_end,
                    "data": window_data.to_dict(orient='records')
                })
            start_time += step

        processed_data[sensor_name] = windows

    return processed_data

# Ruta al archivo JSON
file_path = config.walking_data_path_no_parkinson+'/8764c21b-dd9c-4c96-a99b-d72f781137ad.json'

# Procesar los datos
processed_data = process_imu_data(file_path)

# Convertir los datos a un formato serializable
def convert_to_serializable(data):
    if isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, (np.int64, np.int32)):
        return int(data)
    elif isinstance(data, (np.float64, np.float32)):
        return float(data)
    else:
        return data

processed_data_serializable = convert_to_serializable(processed_data)

# Guardar todas las ventanas en un único archivo JSON
output_file = config.p_walking_data_path_no_parkinson+'/8764c21b-dd9c-4c96-a99b-d72f781137ad.json'
with open(output_file, 'w') as f:
    json.dump(processed_data_serializable, f, indent=4)

print(f"Datos procesados guardados en {output_file}")
