import pandas as pd
import  matplotlib.pyplot as plt
import numpy as np
import sys

def read_and_plot_energy_from_file(path):
    f = open(path, 'r')
    for i in range(2):
        f.readline()
    l3 = f.readline()
    sample_time = float(l3[1:-2].split(':')[1].replace(' ', ''))
    l4 = f.readline()
    trigger_index = int(l4[1:-2].split(':')[1].replace(' ', ''))
    f.close()

    # Importe les mesures et aligne les données
    data = pd.read_csv(path, skiprows=5, header=0)
    start_index = data.iloc[0, 0]
    sample = data.iloc[:, 0] - start_index
    trigger_index -= start_index
    time = sample * sample_time
    voltage = data.iloc[:, 1]
    current = data.iloc[:, 2]
    power = voltage * current

    # Calcul la fin de l'inférence
    select = np.zeros(len(voltage))
    min_current = min(current)
    print(min_current)
    thrld =1.05 * min_current
    index = max(trigger_index, start_index)

    print(thrld)
    #index =
    while current[index] > thrld:  #
        # while index < 142: #

        select[index] = 1
        index += 1

    print(index)
    print(current[index])
    # Calcul l'energie consommée
    energy_cmpt = sample_time * np.sum(power[select == 1])
    print("Energy consumption during inference: %f J" % energy_cmpt)

    # Affiche les différentes mesures
    plt.plot(time, voltage)
    plt.plot(time, select*max(voltage))
    plt.legend(['voltage', 'selected part'])
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title('Voltage from:  %s' % path)
    plt.show()

    plt.plot(time, current)
    plt.plot(time, select*max(current))
    plt.legend(['current', 'selected part'])
    plt.xlabel('Time (s)')
    plt.ylabel('Current (W)')
    plt.title('Current from:  %s' % path)
    plt.show()

    plt.plot(time, power)
    plt.plot(time, select * max(power))
    plt.legend(['power', 'selected part'])
    plt.xlabel('Time (s)')
    plt.ylabel('Instantaneous power (W)')
    plt.title('Instantaneous power from:  %s' % path)
    plt.show()

