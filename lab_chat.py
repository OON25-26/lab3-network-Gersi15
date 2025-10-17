# lab3_network.py

import json
import math
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt

C = 3e8
FIBER_SPEED = (2/3) * C  # m/s


# ---------------- Exercise 1 ----------------
class SignalInformation:
    def __init__(self, signal_power, path):
        self._signal_power = float(signal_power)
        self._noise_power = 0.0
        self._latency = 0.0
        self._path = list(path)

    # getters
    def get_signal_power(self): return self._signal_power
    def get_noise_power(self): return self._noise_power
    def get_latency(self): return self._latency
    def get_path(self): return self._path

    # setters
    def set_signal_power(self, v): self._signal_power = float(v)
    def set_noise_power(self, v): self._noise_power = float(v)
    def set_latency(self, v): self._latency = float(v)
    def set_path(self, p): self._path = list(p)

    # updates
    def update_noise_power(self, inc): self._noise_power += float(inc)
    def update_latency(self, inc): self._latency += float(inc)

    def update_path(self):
        if len(self._path) > 1:
            self._path = self._path[1:]
        else:
            self._path = []

    def __repr__(self):
        return (f"SignalInformation(P={self._signal_power}, "
                f"N={self._noise_power}, L={self._latency}, path={self._path})")


# ---------------- Exercise 2 ----------------
class Node:
    def __init__(self, label, position, connected_nodes):
        self._label = label
        self._position = tuple(position)
        self._connected_nodes = list(connected_nodes)
        self._successive = {}  # next_node_label -> Line

    # getters/setters (minimal set used by the lab)
    def get_label(self): return self._label
    def get_position(self): return self._position
    def get_connected_nodes(self): return self._connected_nodes
    def get_successive(self): return self._successive
    def set_successive(self, succ): self._successive = dict(succ)

    def propagate(self, signal_information: SignalInformation):
        # stop if this is the last node in path
        if len(signal_information.get_path()) <= 1:
            return signal_information
        # consume current node
        signal_information.update_path()
        next_label = signal_information.get_path()[0]
        line = self._successive[next_label]
        return line.propagate(signal_information)


# ---------------- Exercise 3 ----------------
class Line:
    def __init__(self, label, length):
        self._label = label
        self._length = float(length)
        self._successive = {}  # next_node_label -> Node

    def get_label(self): return self._label
    def get_length(self): return self._length
    def get_successive(self): return self._successive
    def set_successive(self, succ): self._successive = dict(succ)

    # latency and noise models
    def latency_generation(self) -> float:
        return self._length / FIBER_SPEED

    def noise_generation(self, signal_power: float) -> float:
        return 1e-9 * float(signal_power) * self._length

    def propagate(self, signal_information: SignalInformation):
        # add impairments
        signal_information.update_latency(self.latency_generation())
        signal_information.update_noise_power(
            self.noise_generation(signal_information.get_signal_power())
        )
        # forward to the next node in the path
        next_label = signal_information.get_path()[0]
        next_node = self._successive[next_label]
        return next_node.propagate(signal_information)


# ---------------- Exercise 4 ----------------
class Network:
    def __init__(self, json_path="nodes.json"):
        with open(json_path, "r") as f:
            data = json.load(f)

        # build nodes
        self.nodes = {}
        for label, nd in data["nodes"].items():
            self.nodes[label] = Node(
                label=label,
                position=nd["position"],
                connected_nodes=nd["connected_nodes"],
            )

        # build directed lines between connected nodes
        self.lines = {}
        for a, na in self.nodes.items():
            xa, ya = na.get_position()
            for b in na.get_connected_nodes():
                xb, yb = self.nodes[b].get_position()
                length = math.hypot(xa - xb, ya - yb)  # minimum distance
                lbl = f"{a}{b}"
                self.lines[lbl] = Line(lbl, length)

        # connect elements
        self.connect()

    def connect(self):
        # node -> line
        for a, node in self.nodes.items():
            succ = {}
            for b in node.get_connected_nodes():
                succ[b] = self.lines[f"{a}{b}"]
            node.set_successive(succ)

        # line -> next node
        for lbl, line in self.lines.items():
            a, b = lbl[0], lbl[1]
            line.set_successive({b: self.nodes[b]})

    def find_paths(self, start: str, end: str):
        # all simple paths (no node repeated)
        paths = []
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if node == end:
                paths.append(path)
                continue
            for nxt in self.nodes[node].get_connected_nodes():
                if nxt not in path:
                    stack.append((nxt, path + [nxt]))
        return paths

    def propagate(self, signal_information: SignalInformation):
        start = signal_information.get_path()[0]
        return self.nodes[start].propagate(signal_information)

    def draw(self):
        plt.figure()
        # nodes
        for label, node in self.nodes.items():
            x, y = node.get_position()
            plt.scatter([x], [y])
            plt.text(x, y, label)
        # lines
        for lbl in self.lines:
            a, b = lbl[0], lbl[1]
            xa, ya = self.nodes[a].get_position()
            xb, yb = self.nodes[b].get_position()
            plt.plot([xa, xb], [ya, yb])
        plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("Network")
        plt.show()

    # --------------- Exercise 5 ---------------
    def all_paths_dataframe(self, signal_power_w=1e-3):
        rows = []
        labels = sorted(self.nodes.keys())
        for s, t in combinations(labels, 2):
            for path in self.find_paths(s, t):
                sig = SignalInformation(signal_power_w, path)
                res = self.propagate(sig)
                snr_db = 10.0 * math.log10(
                    res.get_signal_power() / max(res.get_noise_power(), 1e-30)
                )
                rows.append(
                    {
                        "path": "->".join(path),
                        "latency_s": res.get_latency(),
                        "noise_W": res.get_noise_power(),
                        "signal_W": res.get_signal_power(),
                        "snr_dB": snr_db,
                    }
                )
        return pd.DataFrame(rows)


# ---------- Helper to build a sample nodes.json if needed ----------
def write_sample_nodes_json(path="nodes.json"):
    # matches the figure topology; coordinates are arbitrary meters
    data = {
        "nodes": {
            "A": {"position": [0.0, 1.0], "connected_nodes": ["B", "C", "D"]},
            "B": {"position": [2.0, 2.0], "connected_nodes": ["A", "F", "D"]},
            "C": {"position": [0.0, -1.0], "connected_nodes": ["A", "D", "E"]},
            "D": {"position": [1.5, 0.0], "connected_nodes": ["A", "B", "C", "E", "F"]},
            "E": {"position": [3.0, -1.0], "connected_nodes": ["C", "D", "F"]},
            "F": {"position": [3.0, 1.0], "connected_nodes": ["B", "D", "E"]},
        }
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# -------------- Example run --------------
if __name__ == "__main__":
    # write_sample_nodes_json()  # uncomment to generate a test topology

    net = Network("nodes.json")
    df = net.all_paths_dataframe(signal_power_w=1e-3)
    print(df.sort_values(["snr_dB", "latency_s"], ascending=[False, True]).head())
    # net.draw()  # optional: visualize
