import numpy as np


'''
A collection of widely used pre-defined quantum gates
'''

X = np.array([[0, 1],
              [1, 0]])
Y = np.array([[0, -1j],
              [1j, 0]])
Z = np.array([[1, 0],
              [0, -1]])
H = 1/(2**.5) * np.array([[1, 1],
                          [1, -1]])
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])
I2 = np.eye(2)