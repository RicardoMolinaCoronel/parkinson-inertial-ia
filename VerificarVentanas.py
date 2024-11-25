import os
import json
import Global.config as config

def analyze_and_clean_output_files(output_folder, cleaned_output_folder):
    # Verifica si las carpetas existen
    if not os.path.exists(output_folder):
        print(f"La carpeta {output_folder} no existe.")
        return
    if not os.path.exists(cleaned_output_folder):
        os.makedirs(cleaned_output_folder)

    # Itera por todos los archivos JSON en la carpeta de salida
    for file_name in os.listdir(output_folder):
        if file_name.endswith('.json'):
            file_path = os.path.join(output_folder, file_name)
            cleaned_file_path = os.path.join(cleaned_output_folder, file_name)

            # Carga el contenido del archivo JSON
            with open(file_path, 'r') as file:
                data = json.load(file)

            cleaned_data = {}
            print(f"\nArchivo: {file_name}")

            for sensor, segments in data.items():
                cleaned_segments = []
                print(f"  Sensor: {sensor}")

                for segment_index, segment in enumerate(segments):
                    cleaned_segment = []
                    print(f"    Segmento {segment_index + 1}: {len(segment)} ventanas antes de limpieza")

                    for window_index, window in enumerate(segment):
                        num_readings = len(window['data'])

                        if num_readings == 100:
                            cleaned_segment.append(window)
                        else:
                            print(f"        ⚠️ Ventana {window_index + 1} eliminada (tiene {num_readings} lecturas).")

                    cleaned_segments.append(cleaned_segment)
                    print(f"    Segmento {segment_index + 1}: {len(cleaned_segment)} ventanas después de limpieza")

                cleaned_data[sensor] = cleaned_segments

            # Guardar el archivo limpio
            with open(cleaned_file_path, 'w') as outfile:
                json.dump(cleaned_data, outfile, indent=4)
            print(f"Archivo limpio guardado: {cleaned_file_path}")


import os
import json


def analyze_output_files(output_folder):
    # Verifica si la carpeta existe
    if not os.path.exists(output_folder):
        print(f"La carpeta {output_folder} no existe.")
        return

    # Itera por todos los archivos JSON en la carpeta de salida
    for file_name in os.listdir(output_folder):
        if file_name.endswith('.json'):
            file_path = os.path.join(output_folder, file_name)

            # Carga el contenido del archivo JSON
            with open(file_path, 'r') as file:
                data = json.load(file)

            print(f"\nArchivo: {file_name}")
            for sensor, segments in data.items():
                total_windows = 0
                windows_with_less_than_100 = 0
                print(f"  Sensor: {sensor}")

                for segment_index, segment in enumerate(segments):
                    num_windows = len(segment)
                    total_windows += num_windows

                    print(f"    Segmento {segment_index + 1}: {num_windows} ventanas")

                    for window_index, window in enumerate(segment):
                        num_readings = len(window['data'])
                        print(f"      Ventana {window_index + 1}: {num_readings} lecturas")

                        # Verificar si la ventana tiene menos de 100 lecturas
                        if num_readings != 100:
                            print(f"        ⚠️ Ventana {window_index + 1} tiene menos de 100 lecturas.")
                            windows_with_less_than_100 += 1

                print(f"  Total de ventanas para {sensor}: {total_windows}")
                if windows_with_less_than_100 > 0:
                    print(f"  ⚠️ {sensor} tiene {windows_with_less_than_100} ventanas con menos de 100 lecturas.")



# Configuración
output_folder = config.p_walking_data_path_parkinson  # Ruta a la carpeta de salida original
cleaned_output_folder = config.p_walking_data_path_parkinson  # Ruta a la carpeta de salida para los archivos limpios

# Ejecutar el análisis y limpieza
analyze_and_clean_output_files(output_folder, cleaned_output_folder)
print("TERMINOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO LA LIMPIEZA MI BRO- VERIFICACION MOMENT")
analyze_output_files(output_folder)


# Configuración
output_folder = config.p_walking_data_path_no_parkinson  # Ruta a la carpeta de salida original
cleaned_output_folder = config.p_walking_data_path_no_parkinson  # Ruta a la carpeta de salida para los archivos limpios

# Ejecutar el análisis y limpieza
analyze_and_clean_output_files(output_folder, cleaned_output_folder)
print("ACABOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO LA LIMPIEZA MI BRO- VERIFICACION MOMENT")
analyze_output_files(output_folder)