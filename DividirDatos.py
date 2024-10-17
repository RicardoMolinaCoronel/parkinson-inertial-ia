import json
import Global.config as config

nombre_archivo = "ff27e220-c663-4d9c-9c01-57a59c51c2a3"

# Función para filtrar los datos según el rango de "millis"
def filtrar_por_millis(data, rango_min, rango_max):
    dentro_del_rango = {}
    fuera_del_rango = {}

    for clave, valores in data.items():
        dentro_del_rango[clave] = []
        fuera_del_rango[clave] = []
        for elemento in valores:
            if rango_min <= elemento['t'] <= rango_max:
                dentro_del_rango[clave].append(elemento)
            else:
                fuera_del_rango[clave].append(elemento)

    return dentro_del_rango, fuera_del_rango


# Leer el archivo JSON
with open(config.raw_data_path + '/' + nombre_archivo + '.json', 'r') as archivo_json:
    datos = json.load(archivo_json)

# Definir el rango de tiempo (en este ejemplo, entre 50000 y 53000)
rango_min = 5000
rango_max = 5400

# Filtrar los datos
dentro_rango, fuera_rango = filtrar_por_millis(datos, rango_min, rango_max)

# Guardar los datos dentro del rango en un archivo JSON
with open(
        config.turns_data_path_parkinson + '/' + nombre_archivo + '.json',
        'w') as archivo_dentro:
    json.dump(dentro_rango, archivo_dentro, indent=4)

# Guardar los datos fuera del rango en otro archivo JSON
with open(
        config.walking_data_path_parkinson + '/' + nombre_archivo + '.json',
        'w') as archivo_fuera:
    json.dump(fuera_rango, archivo_fuera, indent=4)

print("Datos exportados correctamente a 'dentro_del_rango.json' y 'fuera_del_rango.json'")
