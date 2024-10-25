import os
import json
import numpy as np
import scipy.signal as signal
from scipy import interpolate
import Global.config as config

# Función para aplicar el filtro de paso bajo
def aplicar_filtro_paso_bajo(data, frecuencia_corte, frecuencia_muestreo, orden=2):
    nyquist = 0.5 * frecuencia_muestreo
    normalizada = frecuencia_corte / nyquist
    b, a = signal.butter(orden, normalizada, btype='low', analog=False)
    return signal.filtfilt(b, a, data)


# Función para realizar la interpolación lineal
def interpolar_datos(tiempos, valores):
    tiempo_interpolado = np.linspace(tiempos.min(), tiempos.max(), len(tiempos))
    interpolador = interpolate.interp1d(tiempos, valores, kind='linear', fill_value='extrapolate')
    return tiempo_interpolado, interpolador(tiempo_interpolado)


# Procesar datos de un sensor
def procesar_sensor(datos_sensor, frecuencia_corte, frecuencia_muestreo):
    tiempos = np.array([dato['millis'] for dato in datos_sensor])
    eje_x = np.array([dato['x'] for dato in datos_sensor])
    eje_y = np.array([dato['y'] for dato in datos_sensor])
    eje_z = np.array([dato['z'] for dato in datos_sensor])

    # Aplicar filtro de paso bajo
    eje_x_filtrado = aplicar_filtro_paso_bajo(eje_x, frecuencia_corte, frecuencia_muestreo)
    eje_y_filtrado = aplicar_filtro_paso_bajo(eje_y, frecuencia_corte, frecuencia_muestreo)
    eje_z_filtrado = aplicar_filtro_paso_bajo(eje_z, frecuencia_corte, frecuencia_muestreo)

    # Realizar la interpolación
    tiempos_interpolados, eje_x_interpolado = interpolar_datos(tiempos, eje_x_filtrado)
    _, eje_y_interpolado = interpolar_datos(tiempos, eje_y_filtrado)
    _, eje_z_interpolado = interpolar_datos(tiempos, eje_z_filtrado)

    # Actualizar los datos con los valores filtrados e interpolados
    datos_procesados = []
    for i, dato in enumerate(datos_sensor):
        datos_procesados.append({
            'a': dato['a'],
            'b': dato['b'],
            'g': dato['g'],
            'millis': tiempos_interpolados[i],
            't': dato['t'],
            'x': eje_x_interpolado[i],
            'y': eje_y_interpolado[i],
            'z': eje_z_interpolado[i]
        })
    return datos_procesados


# Procesar archivos en una carpeta
def procesar_archivos_en_carpeta(carpeta, frecuencia_corte=1.0, frecuencia_muestreo=50.0):
    archivos = [f for f in os.listdir(carpeta) if f.endswith('.json')]

    for archivo in archivos:
        ruta_archivo = os.path.join(carpeta, archivo)
        with open(ruta_archivo, 'r') as f:
            datos = json.load(f)

        # Procesar cada sensor
        sensores = ['derecha', 'izquierda', 'espina_base']
        for sensor in sensores:
            if sensor in datos:
                datos[sensor] = procesar_sensor(datos[sensor], frecuencia_corte, frecuencia_muestreo)

        # Guardar los datos procesados en un nuevo archivo JSON
        nuevo_archivo = os.path.join(carpeta, f'procesado_{archivo}')
        with open(nuevo_archivo, 'w') as f:
            json.dump(datos, f, indent=4)
        print(f'Archivo procesado guardado como: {nuevo_archivo}')


# Directorio donde están los archivos JSON
carpeta =  config.walking_data_path_no_parkinson

# Ejecutar el procesamiento
procesar_archivos_en_carpeta(carpeta)
