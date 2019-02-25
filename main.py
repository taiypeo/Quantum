from quantum import Circuit
from quantum.ConstGates import *


if __name__ == '__main__':
    # Random 3-qubit circuit demonstation

    out = Circuit(3).add_column([
        H, I2, H
    ]).add_column([
        CNOT, I2
    ]).calculate_probabilities(np.array([
        1, 0, 0, 0, 0, 0, 0, 0
    ]).reshape(-1, 1))

    print(out)
    '''
    [[0.5]
    [0.5]
    [0. ]
    [0. ]
    [0. ]
    [0. ]
    [0.5]
    [0.5]]
    '''

    print()
    #=======================================================
    # Bell state simulation demonstration

    out = Circuit(2).add_column([
        H, I2
    ]).add_column([
        CNOT
    ]).calculate_probabilities(np.array([
        1, 0, 0, 0
    ]).reshape(-1, 1))

    print(out)
    '''
    [[0.70710678]
    [0.        ]
    [0.        ]
    [0.70710678]]
    '''