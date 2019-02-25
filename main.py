from quantum import Circuit
from quantum.ConstGates import *


if __name__ == '__main__':
    out = Circuit(3).add_column([
        H, I2, H
    ]).add_column([
        CNOT, I2
    ]).calculate_probabilities(np.array([
        1, 0, 0, 0, 0, 0, 0, 0
    ]).reshape(-1, 1))

    print(out)