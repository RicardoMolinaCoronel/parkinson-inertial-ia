import json
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import Global.config as config

with open(config.normalized_data_path_parkinson + '/a535059c-6318-4b2f-9c35-3fc836988106' + '.json', 'r') as file:
    data = json.load(file)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.75, top=0.9, hspace=0.3)
variables = ['a', 'b', 'g', 'x', 'y', 'z']
#sensors = ['derecha', 'izquierda', 'espina_base']
sensors = ['derecha', 'izquierda', 'espina_base']
sensor_colors = {
    'derecha': {'a': 'red', 'b': 'green', 'g': 'blue', 'x': 'red', 'y': 'green', 'z': 'blue'},
    'izquierda': {'a': 'darkred', 'b': 'darkgreen', 'g': 'darkblue', 'x': 'darkred', 'y': 'darkgreen', 'z': 'darkblue'},
    'espina_base': {'a': 'salmon', 'b': 'lightgreen', 'g': 'lightskyblue', 'x': 'salmon', 'y': 'lightgreen',
                    'z': 'lightskyblue'}
}
sensor_styles = {
    'derecha': '-', 'izquierda': '--', 'espina_base': ':'
}

lines = {sensor: {var: None for var in variables} for sensor in sensors}
var_visible = {var: True for var in variables}
sensor_visible = {sensor: True for sensor in sensors}

for sensor in sensors:
    if sensor in data:
        sensor_data = data[sensor]
        time = [entry['t'] for entry in sensor_data]

        for var in variables:
            ax = ax1 if var in ['a', 'b', 'g'] else ax2
            ylabel = 'Acceleration' if var in ['a', 'b', 'g'] else 'Position/Rotation'

            values = [entry[var] for entry in sensor_data]
            line, = ax.plot(time, values, label=f'{sensor} - {var}',
                            color=sensor_colors[sensor][var], linestyle=sensor_styles[sensor])
            lines[sensor][var] = line

        ax1.set_ylabel(ylabel)
        ax2.set_ylabel(ylabel)

ax1.set_title('Giroscopio (a, b, g)')
ax2.set_title('Accelerometro (x, y, z)')
ax2.set_xlabel('Time (ms)')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

rax_vars = plt.axes([0.8, 0.3, 0.15, 0.3])
check_vars = CheckButtons(rax_vars, variables, [True] * len(variables))
rax_sensors = plt.axes([0.8, 0.7, 0.15, 0.2])
check_sensors = CheckButtons(rax_sensors, sensors, [True] * len(sensors))


def update_visibility():
    for sensor in sensors:
        for var in variables:
            if sensor in lines and var in lines[sensor]:
                lines[sensor][var].set_visible(sensor_visible[sensor] and var_visible[var])
    plt.draw()


def toggle_vars(label):
    var_visible[label] = not var_visible[label]
    update_visibility()


def toggle_sensors(label):
    sensor_visible[label] = not sensor_visible[label]
    update_visibility()


check_vars.on_clicked(toggle_vars)
check_sensors.on_clicked(toggle_sensors)

plt.show()
