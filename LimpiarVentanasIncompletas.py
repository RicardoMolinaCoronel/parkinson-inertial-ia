import json
import Global.config as config
#IMPORTANTE!: APLICAR SOLO UNA VEZ ESTE SCRIPT O SE PERDERAN VENTANAS IMPORTANTES
# Función para verificar si un valor está en un rango
def is_in_range(value, ranges):
    for start, end in ranges:
        if start <= value <= end:
            return True
    return False

def have_commons(start1, end1, ranges):
    for start2, end2 in ranges:
        if max(start1, start2) <= min(end1, end2):
            return True
    return False

# Leer el archivo con identificadores y rangos
def parse_ranges(file_path):
    ranges_dict = {}
    with open(file_path, "r") as f:
        content = f.read().strip().split("\n")

    current_id = None
    for line in content:

        if len(line) > 20:  # Es un identificador
            current_id = line.strip()
            ranges_dict[current_id] = []
        elif '-' in line:  # Es un rango
            start, end = map(int, line.split("-"))
            ranges_dict[current_id].append((start, end))

    return ranges_dict


# Procesar los archivos JSON
def clean_json_data(json_folder, ranges_dict):
    for identifier, ranges in ranges_dict.items():
        json_file_path = f"{json_folder}/{identifier}.json"  # Ajusta si los archivos están en otra carpeta
        try:
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)

            # Filtrar ventanas por start_time y end_time
            for key in ["derecha", "izquierda", "espina_base"]:
                if key in data:
                    data[key] = [
                        window for window in data[key]
                        if not have_commons(window["start_time"], window["end_time"], ranges)
                    ]

                    if data[key]:  # Asegurarse de que la lista no está vacía
                        data[key].pop(-1)  # Elimina la última ventana

            # Guardar el archivo modificado
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file, indent=4)
            print(f"Archivo procesado: {identifier}.json")

        except FileNotFoundError:
            print(f"Archivo {identifier}.json no encontrado, saltando...")
        except Exception as e:
            print(f"Error procesando {identifier}.json: {e}")


# Ruta de los archivos
ranges_file = 'C:/Users/Ricardo/Desktop/Espol/SFINAL/Materia Integradora/RangosTiempoParkinson.txt'  # Archivo con los identificadores y rangos
json_folder = config.p_walking_data_path_parkinson  # Carpeta donde están los archivos JSON

# Leer los rangos y procesar los JSON
ranges_dict = parse_ranges(ranges_file)
clean_json_data(json_folder, ranges_dict)

# Ruta de los archivos
ranges_file = 'C:/Users/Ricardo/Desktop/Espol/SFINAL/Materia Integradora/RangosTiempoNoParkinson.txt'  # Archivo con los identificadores y rangos
json_folder = config.p_walking_data_path_no_parkinson  # Carpeta donde están los archivos JSON

# Leer los rangos y procesar los JSON
ranges_dict = parse_ranges(ranges_file)
clean_json_data(json_folder, ranges_dict)

# Ruta de los archivos
ranges_file = 'C:/Users/Ricardo/Desktop/Espol/SFINAL/Materia Integradora/RangosNoPPosibleRuido.txt'  # Archivo con los identificadores y rangos
json_folder = config.p_walking_data_path_no_parkinson  # Carpeta donde están los archivos JSON

# Leer los rangos y procesar los JSON
ranges_dict = parse_ranges(ranges_file)
clean_json_data(json_folder, ranges_dict)