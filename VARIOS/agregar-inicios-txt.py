def agregar_rango_a_identificadores(input_file, output_file):
    """
    Agrega el rango '0-1800' después de cada identificador en un archivo de texto.
    :param input_file: Ruta del archivo de entrada.
    :param output_file: Ruta del archivo de salida.
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if len(line) > 20 and ',' not in line and line:  # Es un identificador
                outfile.write(f"{line}\n0-1800\n")  # Escribir el identificador y el nuevo rango
            elif line:  # Es un rango
                outfile.write(f"{line}\n")
            elif line == "":  # Línea en blanco
                outfile.write("\n")


# Rutas de los archivos
input_file = "C:/Users/Ricardo/Desktop/Espol/SFINAL/Materia Integradora/RangosTiempoParkinson.txt"  # Cambia esta ruta por la correcta si es necesario
output_file = "C:/Users/Ricardo/Desktop/Espol/SFINAL/Materia Integradora/RangosCInicioTiempoParkinson.txt"

# Ejecutar la función
agregar_rango_a_identificadores(input_file, output_file)

print(f"Archivo procesado guardado en {output_file}")
