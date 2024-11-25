import json
import matplotlib.pyplot as plt
import pandas as pd
import Global.config as config

def load_and_visualize_json(file_path):
    """
    Carga y visualiza las ventanas y datos del archivo JSON generado.

    Args:
        file_path (str): Ruta del archivo JSON generado.
    """
    # Cargar el archivo JSON
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Iterar por cada sensor
    for sensor_name, windows in data.items():
        print(f"Sensor: {sensor_name}, Total de Ventanas: {len(windows)}")

        # Iterar por cada ventana
        for i, window in enumerate(windows):
            print(f"Ventana {i + 1}: {window['start_time']} - {window['end_time']}")
            df = pd.DataFrame(window['data'])

            # Visualizar las señales
            plt.figure(figsize=(12, 6))
            for col in ['a', 'b', 'g', 'x', 'y', 'z']:
                if col in df.columns:
                    plt.plot(df['t'], df[col], label=col)

            plt.title(f"Sensor: {sensor_name}, Ventana {i + 1}")
            plt.xlabel("Tiempo (ms)")
            plt.ylabel("Valor")
            plt.legend()
            plt.show()

            # Limitar la cantidad de ventanas mostradas
            if i >= 29:  # Cambia este valor para visualizar más ventanas
                break


# Ruta del archivo JSON generado
file_path = config.p_walking_data_path_no_parkinson+'/c2cad4cd-9857-4493-9954-82567e2d185a.json'

# Cargar y visualizar
load_and_visualize_json(file_path)
