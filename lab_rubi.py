import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import warnings
import json
from math import sqrt
from scipy import constants
from matplotlib.patches import Circle

# there seems to be some random error due to matplotlib
warnings.filterwarnings("ignore", message="Blended transforms not yet supported.")

warnings.filterwarnings("ignore", message="divide by zero encountered in log10")

color_dict = mcol.TABLEAU_COLORS
colors = []
color_name = []
for i, (name, color) in enumerate(color_dict.items()):
    colors.append(color)
    color_name.append(name)

results_folder = ("C:\\Documents\\LAB_OON\\lab3-network-Gersi15\\results\\")
main_folder = ("C:\\Documents\\lab3-network-Gersi15\\main\\")


def lin2db(x):
    return 10 * np.log10(x)


def db2lin(x):
    return 10 ** (x / 10)


# class ADC:
#     fs_step = 2.75625e3
#
#     def __init__(self, n_bit):
#         self._n_bit = np.array(n_bit, dtype=np.int64)
#
#     @property
#     def n_bit(self):
#         return self._n_bit
#
#     @n_bit.setter
#     def n_bit(self, n_bit):
#         self._n_bit = n_bit
#
#     def snr(self):
#         return db2lin(6 * self.n_bit)
#
#     def sampling_freq_coefficient(self, analog_bandwidth):
#         fs_step = self.fs_step
#         mult = np.ceil(2 * analog_bandwidth / fs_step)
#         return mult


# class BSC:
#
#     def __init__(self, error_probability):
#         self._error_probability = error_probability
#
#     @property
#     def error_probability(self):
#         return self._error_probability
#
#     @error_probability.setter
#     def error_probability(self, error_probability):
#         self._error_probability = error_probability
#
#     def snr(self):
#         return 1 / (4 * self.error_probability)


# class PCM:
#     analog_bandwidth = 22e3
#
#     def __init__(self, adc, dsi, line):  # writing digital system information as DSI
#         self._adc = adc
#         self._dsi = dsi
#         self._line = line
#
#     @property
#     def adc(self):
#         return self._adc
#
#     @adc.setter
#     def adc(self, adc):
#         self._adc = adc
#
#     @property
#     def dsi(self):
#         return self._dsi
#
#     @dsi.setter
#     def dsi(self, dsi):
#         self._dsi = dsi
#
#     @property
#     def line(self):
#         return self._line
#
#     @line.setter
#     def line(self, line):
#         self._line = line
#
#     def crit_pe(self):
#         m = 2 ** self.adc.n_bit
#         return 1 / (4 * (m ** 2 - 1))
#
#     def snr(self):
#         return db2lin(self.line.snr_digital(self.dsi.signal_power))
#
#     def ber_evaluation(self):
#         match self.dsi.n_bit_mod:
#             case 1:
#                 return (1 / 2) * np.float64(erfc(np.sqrt(self.snr())))
#             case 2:
#                 return (1 / 2) * np.float64(erfc(np.sqrt(self.snr() / 2)))
#             case 3:
#                 return (2 / 3) * np.float64(erfc(np.sqrt((3 / 14) * self.snr())))
#             case 4:
#                 return (3 / 8) * np.float64(erfc(np.sqrt(self.snr() / 10)))


class Signal_information:
    def __init__(self, signal_power, path):
        if not isinstance(signal_power, float or int):
            raise TypeError('Signal power must be a float')
        self._signal_power = signal_power
        self._noise_power = 0.0
        self._latency = 0.0
        self._path = path

    @property
    def latency(self):
        return self._latency

    def latency_update(self, latency):
        if not isinstance(latency, float or int):
            raise TypeError('latency must be a float')
        self._latency += latency

    @property
    def path(self):
        return self._path

    @property
    def signal_power(self):
        return self._signal_power

    def signal_power_update(self, signal_power):
        if not isinstance(signal_power, float or int):
            raise TypeError("Signal power must be a floating point number")
        self._signal_power += signal_power

    @property
    def noise_power(self):
        return self._noise_power

    def noise_power_update(self, noise_power):
        if not isinstance(noise_power, float or int):
            raise TypeError("Noise power must be a floating point number")
        self._noise_power += noise_power

    def update_path(self):
        if len(self._path) > 0:
            return self.path.pop(0)
        else:
            print("Reached end of path")


class Line:
    def __init__(self, label_val):
        self._label = str(label_val)
        self._length = 0
        self._successive = dict()

    @property
    def label(self):
        return self._label

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, length):
        self._length = length

    def length_calc(self, x1, x2, y1, y2):
        self.length = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def latency_generation(self):
        c = constants.c
        return 3 * self.length / (2 * c)

    def noise_generation(self, signal_power):
        return 1e-9 * signal_power * self.length

    @property
    def successive(self):
        return self._successive

    def successive_update(self, successive):
        self._successive.update(successive)

    def propagate(self, signal_information):
        signal_information.latency_update(self.latency_generation())
        signal_information.noise_power_update(self.noise_generation(signal_information.signal_power))
        self.successive[signal_information.path[0]].propagate(signal_information)


class Node:
    def __init__(self, label_val, x, y, connections):
        self._label = str(label_val)
        if not (isinstance(x, float or int) and isinstance(y, float or int)):
            raise TypeError("position must be made up of floating point numbers")
        self._position = (x, y)
        self._connections = list(connections)
        self._successive = dict()

    @property
    def label(self):
        return self._label

    @property
    def position(self):
        return self._position

    @property
    def connections(self):
        return self._connections

    @property
    def successive(self):
        return self._successive

    def successive_update(self, successive):
        self._successive.update(successive)

    def propagate(self, signal_information):
        if len(signal_information.path) >= 2:
            signal_information.update_path()
            self.successive[str(self.label) + str(signal_information.path[0])].propagate(signal_information)


class Network:
    def __init__(self, json_data):
        self.nodes = dict()
        self.lines = dict()

        for key in json_data:
            sub_keys = list(json_data[key].keys()).copy()
            self.nodes[key] = Node(key, float(json_data[key][sub_keys[1]][0]), float(json_data[key][sub_keys[1]][1]),
                                   json_data[key][sub_keys[0]])

        for key in self.nodes:
            for sub_key in self.nodes[key].connections:
                self.lines[str(key) + str(sub_key)] = Line(str(key) + str(sub_key))
                self.lines[str(key) + str(sub_key)].length_calc(self.nodes[key].position[0],
                                                                self.nodes[sub_key].position[0],
                                                                self.nodes[key].position[1],
                                                                self.nodes[sub_key].position[1])

    def connect(self):
        for key in self.nodes:
            for sub_key in self.nodes[key].connections:
                self.nodes[str(key)].successive_update({str(key) + str(sub_key): self.lines[str(key) + str(sub_key)]})
                self.lines[str(key) + str(sub_key)].successive_update({key: self.nodes[key]})
                self.lines[str(key) + str(sub_key)].successive_update({sub_key: self.nodes[sub_key]})

    def find_paths(self, start_node, end_node):
        if start_node == end_node:
            all_paths = [[end_node]]
        else:
            stack = list()
            path = list()
            all_paths = list()
            index = 0
            stack.append([start_node, index])
            while stack:
                flag = True
                current = stack.pop()
                if current[0] in path:
                    if current[1] <= path.index(current[0]):
                        path = path[:current[1]]
                        path.append(current[0])
                    else:
                        flag = False
                elif current[1] < len(path):
                    path = path[:current[1]]
                    path.append(current[0])
                else:
                    path.append(current[0])
                if current[0] == end_node:
                    all_paths.append(path.copy())
                    path.pop()
                elif flag:
                    index = len(path)
                    for node_dud in self.nodes[current[0]].connections:
                        stack.append([node_dud, index])
        return all_paths.copy()

    def propagate(self, signal_information):
        path = signal_information.path.copy()
        self.nodes[str(path[0])].propagate(signal_information)
        signal_power = signal_information.signal_power
        noise_power = signal_information.noise_power
        latency = signal_information.latency
        if noise_power == 0:
            snr = np.inf
        else:
            snr = lin2db(signal_power / noise_power)

        path_string = path.pop(0)
        while path:
            node_dud = path.pop(0)
            path_string = path_string + "->" + str(node_dud)
        return {path_string: [latency, noise_power, snr]}

    def draw(self):
        x = list()
        y = list()
        labels = list(self.nodes.keys())
        for node_dud in self.nodes:
            x.append(self.nodes[node_dud].position[0])
            y.append(self.nodes[node_dud].position[1])
            for sub_node_dud in self.nodes[node_dud].connections:
                x_dud = [self.nodes[node_dud].position[0], self.nodes[sub_node_dud].position[0]]
                y_dud = [self.nodes[node_dud].position[1], self.nodes[sub_node_dud].position[1]]
                plt.plot(x_dud, y_dud, c='blue', linewidth=5)

        minimum_number = min(min(y), min(x))
        maximum_number = max(max(y), max(x))

        limiter = max(abs(maximum_number), abs(minimum_number))

        exp = int(np.floor(np.log10(abs(minimum_number))))

        for ii, label in enumerate(labels):
            circle = Circle((x[ii], y[ii]), 0.3*10**exp, color='blue', fill=True, zorder=10)
            plt.gca().add_patch(circle)
            plt.text(x[ii], y[ii], label, c='white', fontsize=10, ha='center', va='center', zorder=20)

        x_tick_range = [ii * (10 ** exp) for ii in range(int(np.ceil(-1.3 * limiter / 10 ** exp)),
                                                         int(np.ceil(1.3 * limiter / 10 ** exp)))]
        y_tick_range = [ii * (10 ** exp) for ii in range(int(np.ceil(-1.3 * limiter / 10 ** exp)),
                                                         int(np.ceil(1.3 * limiter / 10 ** exp)))]

        plt.xlim([-1.3 * limiter, 1.3 * limiter])
        plt.ylim([-1.3 * limiter, 1.3 * limiter])
        plt.title('Nodes')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.xticks(x_tick_range)
        plt.yticks(y_tick_range)
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        plt.gca().set_aspect('equal', adjustable='box')

        plt.grid()
        plt.savefig(results_folder + 'network.png', dpi=600)
        plt.show()


if __name__ == "__main__":
    print("Laboratory 5 - Stream\n")
    input_power = 1e-3
    json_file = main_folder + "nodes.json"
    panda_data = dict()
    try:
        with open(json_file, 'r') as jason:
            data = json.load(jason)
            my_network = Network(data)
            my_network.connect()
            for node in my_network.nodes:
                for sub_node in my_network.nodes:
                    if node != sub_node:
                        possible_paths = my_network.find_paths(str(node), str(sub_node))
                        for pth in possible_paths:
                            panda_data.update(my_network.propagate(Signal_information(input_power, pth)))
            propagation = pd.DataFrame(panda_data, ["Latency (s)", "Noise Power (W)", "SNR (dB)"]).transpose()
            propagation.reset_index(inplace=True)
            propagation.rename(columns={'index': 'Path followed'}, inplace=True)
            propagation.set_index('Path followed', inplace=True)
            propagation.to_csv(results_folder + "propagation.csv")
            my_network.draw()
    except FileNotFoundError:
        print("File not found!")