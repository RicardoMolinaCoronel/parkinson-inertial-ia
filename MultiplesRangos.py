import json
import os
import Global.config as config

variableTiempo = 't'
# Función para leer los rangos de un archivo txt que contiene varios archivos con sus respectivos rangos
def leer_rangos_de_txt(archivo_txt):
    rangos_por_uuid = {}
    uuid_actual = None

    with open(archivo_txt, 'r') as archivo:
        for linea in archivo:
            linea = linea.strip()
            guiones = linea.count('-')# Quitar saltos de línea y espacios en blanco
            if guiones>1 and len(linea) > 0:  # Detecta el UUID
                uuid_actual = linea
                rangos_por_uuid[uuid_actual] = []
            elif '-' in linea and uuid_actual:  # Detecta un rango
                rango_min, rango_max = map(int, linea.split('-'))
                rangos_por_uuid[uuid_actual].append((rango_min, rango_max))
    print(rangos_por_uuid)
    return rangos_por_uuid


# Función para filtrar los datos según múltiples rangos de "millis"
def filtrar_por_multiples_rangos(data, rangos):
    # Diccionarios para almacenar los datos dentro de los rangos y fuera de ellos
    datos_dentro_de_rangos = {f'rango_{i + 1}': {} for i in range(len(rangos))}
    datos_fuera_de_todos_los_rangos = {}

    # Inicializar listas vacías para cada sección dentro de los rangos y fuera
    for clave in data.keys():
        for i in range(len(rangos)):
            datos_dentro_de_rangos[f'rango_{i + 1}'][clave] = []
        datos_fuera_de_todos_los_rangos[clave] = []

    # Filtrar los datos
    for clave, valores in data.items():
        for elemento in valores:
            agregado_a_rango = False
            # Revisar si el elemento pertenece a alguno de los rangos
            for i, (rango_min, rango_max) in enumerate(rangos):
                if rango_min <= elemento[variableTiempo] <= rango_max:
                    datos_dentro_de_rangos[f'rango_{i + 1}'][clave].append(elemento)
                    agregado_a_rango = True
                    break
            # Si no pertenece a ningún rango, se agrega a los datos fuera de todos los rangos
            if not agregado_a_rango:
                datos_fuera_de_todos_los_rangos[clave].append(elemento)

    return datos_dentro_de_rangos, datos_fuera_de_todos_los_rangos


# Función para procesar los archivos .json utilizando los rangos correspondientes
def procesar_archivos_json_y_rangos(carpeta_json, archivo_rangos_txt):
    # Leer los rangos de tiempo desde el archivo .txt
    rangos_por_archivo = leer_rangos_de_txt(archivo_rangos_txt)

    # Procesar cada archivo .json que coincida con los nombres en el archivo .txt
    for nombre_archivo, rangos in rangos_por_archivo.items():
        archivo_json = f'{nombre_archivo}.json'
        ruta_json = os.path.join(carpeta_json, archivo_json)

        if os.path.exists(ruta_json):
            # Leer los datos del archivo JSON
            with open(ruta_json, 'r') as archivo_j:
                datos = json.load(archivo_j)

            # Filtrar los datos por los múltiples rangos
            datos_dentro_rangos, datos_fuera_rangos = filtrar_por_multiples_rangos(datos, rangos)

            # Exportar los datos dentro de cada rango a archivos separados
            for i in range(len(rangos)):
                with open(config.turns_data_path_no_parkinson + '/' + f'{nombre_archivo}_t{i + 1}.json', 'w') as archivo_dentro:
                    json.dump(datos_dentro_rangos[f'rango_{i + 1}'], archivo_dentro, indent=4)

            # Exportar los datos que no pertenecen a ningún rango a un solo archivo
            with open(config.walking_data_path_no_parkinson + '/' + f'{nombre_archivo}.json', 'w') as archivo_fuera:
                json.dump(datos_fuera_rangos, archivo_fuera, indent=4)

            print(f"Procesado archivo {archivo_json} con los rangos especificados.")
        else:
            print(f"El archivo {archivo_json} no se encuentra en la carpeta.")


# Directorio donde están los archivos JSON
carpeta_json = config.normalized_data_path_no_parkinson

# Ruta al archivo .txt con los rangos de tiempo
archivo_rangos_txt = 'C:/Users/Ricardo/Desktop/Espol/SFINAL/Materia Integradora/RangosTiempo.txt'

# Procesar los archivos
procesar_archivos_json_y_rangos(carpeta_json, archivo_rangos_txt)
