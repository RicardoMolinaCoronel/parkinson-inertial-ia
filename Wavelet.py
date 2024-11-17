import json
import numpy as np
import matplotlib.pyplot as plt
import pywt
import pywt.data
import Global.config as config

# Load JSON data
with open(config.walking_data_path_parkinson+"/7f536dbc-dc98-4ac3-aa18-4ac769e7bee3.json", 'r') as f:
    data = json.load(f)


# Define a function to apply wavelet transform and plot scalograms
def plot_scalogram(axis_data, title):
    # Convert axis data to a numpy array
    axis_array = np.array(axis_data)

    # Apply Continuous Wavelet Transform
    scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(axis_array, scales, 'morl')

    # Plot the scalogram
    plt.figure(figsize=(10, 5))
    plt.imshow(np.abs(coefficients), extent=[0, len(axis_array), 1, 128], cmap='jet', aspect='auto',
               vmax=abs(coefficients).max(), vmin=-abs(coefficients).max())
    plt.colorbar(label='Coefficient Magnitude')
    plt.title(f'Scalogram of {title}')
    plt.xlabel('Time')
    plt.ylabel('Scale')
    plt.show()


# Process each category and each axis
for category, measurements in data.items():
    # Separate each axis data
    a_data = [entry["a"] for entry in measurements]
    b_data = [entry["b"] for entry in measurements]
    g_data = [entry["g"] for entry in measurements]
    x_data = [entry["x"] for entry in measurements]
    y_data = [entry["y"] for entry in measurements]
    z_data = [entry["z"] for entry in measurements]

    # Plot scalograms for each axis in the current category
    plot_scalogram(a_data, f'{category} - Axis a')
    plot_scalogram(b_data, f'{category} - Axis b')
    plot_scalogram(g_data, f'{category} - Axis g')
    plot_scalogram(x_data, f'{category} - Axis x')
    plot_scalogram(y_data, f'{category} - Axis y')
    plot_scalogram(z_data, f'{category} - Axis z')
