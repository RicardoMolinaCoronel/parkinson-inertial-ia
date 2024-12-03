import json
import Global.config as config
# Función para filtrar los datos según múltiples rangos de "millis"
nombre_archivo='f18c0cd7-1238-4a86-a3f5-29b46a684e9c'
variableTiempo = 't'
rangos = [(10000, 11500), (20450, 22000), (29600, 31500)]
#rangos = [(8200, 10400)]
def filtrar_por_multiples_rangos(data, rangos):
    # Diccionarios para almacenar los datos dentro de los rangos y fuera de ellos
    datos_dentro_de_rangos = {f'rango_{i+1}': {} for i in range(len(rangos))}
    datos_fuera_de_todos_los_rangos = {}

    # Inicializar listas vacías para cada sección dentro de los rangos y fuera
    for clave in data.keys():
        for i in range(len(rangos)):
            datos_dentro_de_rangos[f'rango_{i+1}'][clave] = []
        datos_fuera_de_todos_los_rangos[clave] = []

    # Filtrar los datos
    for clave, valores in data.items():
        for elemento in valores:
            agregado_a_rango = False
            # Revisar si el elemento pertenece a alguno de los rangos
            for i, (rango_min, rango_max) in enumerate(rangos):
                if rango_min <= elemento[variableTiempo] <= rango_max:
                    datos_dentro_de_rangos[f'rango_{i+1}'][clave].append(elemento)
                    agregado_a_rango = True
                    break
            # Si no pertenece a ningún rango, se agrega a los datos fuera de todos los rangos
            if not agregado_a_rango:
                datos_fuera_de_todos_los_rangos[clave].append(elemento)

    return datos_dentro_de_rangos, datos_fuera_de_todos_los_rangos

# Leer el archivo JSON
with open(config.raw_data_path_no_parkinson + '/' +nombre_archivo+'.json', 'r') as archivo_json:
    datos = json.load(archivo_json)

# Definir múltiples rangos de tiempo (cada tupla es un rango: (rango_min, rango_max))
 # Ejemplo de tres rangos

# Filtrar los datos por los múltiples rangos
datos_dentro_rangos, datos_fuera_rangos = filtrar_por_multiples_rangos(datos, rangos)

# Exportar los datos dentro de cada rango a archivos separados
for i in range(len(rangos)):
    with open(config.turns_data_path_no_parkinson + '/' + nombre_archivo + f'_t{i+1}.json', 'w') as archivo_dentro:
        json.dump(datos_dentro_rangos[f'rango_{i+1}'], archivo_dentro, indent=4)

# Exportar los datos que no pertenecen a ningún rango a un solo archivo
with open(config.walking_data_path_no_parkinson + '/' + nombre_archivo + '.json', 'w') as archivo_fuera:
    json.dump(datos_fuera_rangos, archivo_fuera, indent=4)

print(f"Datos exportados correctamente a archivos para cada rango y un archivo fuera de rangos.")
