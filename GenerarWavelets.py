import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import resample
from scipy.ndimage import gaussian_filter1d
from PIL import Image
import Global.config as config
def cwt_to_image(signal, wavelet='cmor2.0-0.8', scale_factor=1):
    """
    Aplica la transformada wavelet continua (CWT) y genera una imagen.

    Args:
        signal (array): Señal de entrada.
        wavelet (str): Nombre de la wavelet con parámetros especificados (ejemplo: 'cmor2.0-1.0').
        scale_factor (float): Factor para ajustar el rango de las escalas.

    Returns:
        np.ndarray: Imagen generada a partir de la transformada wavelet.
    """
    # Ajustar las escalas al rango adecuado
    scales = np.arange(1, 128 * scale_factor)  # Frecuencias bajas dominantes
    coefficients, freqs = pywt.cwt(signal, scales, wavelet)
    return np.abs(coefficients)

def process_windows_to_wavelet(input_folder, output_folder, sampling_interval=21, window_duration=1950):
    """
    Procesa ventanas de señales para generar transformadas wavelet.

    Args:
        input_folder (str): Carpeta que contiene los archivos JSON con ventanas.
        output_folder (str): Carpeta donde se guardarán las imágenes generadas.
        sampling_interval (int): Intervalo de muestreo en ms (default: 10 ms).
        window_duration (int): Duración de cada ventana en ms (default: 2000 ms).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Leer todos los archivos JSON en la carpeta de entrada
    files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

    for file_name in files:
        input_path = os.path.join(input_folder, file_name)
        with open(input_path, 'r') as f:
            data = json.load(f)

        # Iterar por cada sensor en el archivo
        for sensor_name, windows in data.items():
            sensor_folder = os.path.join(output_folder, sensor_name)
            if not os.path.exists(sensor_folder):
                os.makedirs(sensor_folder)

            # Procesar cada ventana
            for i, window in enumerate(windows):
                t = np.arange(0, window_duration, sampling_interval)  # Vector de tiempo

                # Procesar cada señal específica
                for signal_name in ['a', 'b', 'g', 'x', 'y', 'z']:
                    signal = np.array([entry[signal_name] for entry in window["data"]])

                    # Reescalar la señal si es necesario
                    if len(signal) != len(t):
                        signal = resample(signal, len(t))

                    # Normalizar la señal
                    signal = (signal - np.mean(signal)) / np.std(signal)

                    # Suavizar la señal con un filtro Gaussiano
                    #signal = gaussian_filter1d(signal, sigma=2)

                    # Generar la transformada wavelet continua (CWT)
                    wavelet_image = cwt_to_image(signal)

                    # Guardar la imagen sin decoraciones
                    output_image_path = os.path.join(
                        sensor_folder, f"{file_name}_window_{i}_{signal_name}.png"
                    )
                    plt.figure(figsize=(6, 6))
                    plt.imshow(wavelet_image, cmap='viridis', aspect='auto')
                    plt.axis('off')  # Elimina los ejes y etiquetas
                    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)  # Sin espacios extra
                    plt.close()

                    # Redimensionar la imagen a 224x224 para VGG-16
                    img = Image.open(output_image_path)
                    img_resized = img.resize((224, 224), Image.BICUBIC)  # Mantener la calidad
                    img_resized.save(output_image_path)

        print(f"Procesado archivo: {file_name}")

# Carpetas de entrada y salida
input_folder = config.p_walking_data_path_no_parkinson  # Carpeta con archivos JSON de ventanas
output_folder = config.wavelets1_data_path_no_parkinson  # Carpeta para guardar imágenes wavelet

# Procesar los archivos
process_windows_to_wavelet(input_folder, output_folder)


# Carpetas de entrada y salida
input_folder = config.p_walking_data_path_parkinson  # Carpeta con archivos JSON de ventanas
output_folder = config.wavelets1_data_path_parkinson  # Carpeta para guardar imágenes wavelet

# Procesar los archivos
process_windows_to_wavelet(input_folder, output_folder)