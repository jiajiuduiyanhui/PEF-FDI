import os
import numpy as np

def read_block(lines, start_key):
    data = []
    recording = False

    for line in lines:
        line = line.strip()

        if start_key in line:
            recording = True
            continue

        if recording:
            if "];" in line:
                break
            if line == "" or line.startswith("%"):
                continue

            line = line.replace(";", "")
            nums = [float(x) for x in line.split()]
            data.append(nums)

    return data


def print_array(name, data, cols):
    print(f"{name} = np.array([")
    for row in data:
        selected = [row[i] for i in cols]
        row_str = ", ".join(f"{x:.6g}" for x in selected)
        print(f"    [{row_str}],")
    print("])\n")


current_dir = os.path.dirname(os.path.abspath(__file__))
txt_file = os.path.join(current_dir, "ieee118.txt")

with open(txt_file, "r") as f:
    lines = f.readlines()

bus = read_block(lines, "mpc.bus")
branch = read_block(lines, "mpc.branch")


# BUS: bus_id, type, Pd, Qd, Vm, Va
print_array(
    "BUS_DATA",
    bus,
    cols=[0, 1, 2, 3, 7, 8]
)

# BRANCH: from, to, r, x, b, tap
print_array(
    "BRANCH_DATA",
    branch,
    cols=[0, 1, 2, 3, 4, 8]
)