import numpy as np
from .helpers import _kron, _check_list_unitary, _get_col_kronecker_size
from .Exceptions import QuantumException


class Circuit:
    '''
    Class that represents a quantum circuit
    '''

    def __init__(self, qubit_num):
        '''
        Circuit constructor

        :param qubit_num: number of qubits in this circuit
        '''

        self.qubit_num = qubit_num
        self.circuit = []
        self.matrix = None
    
    def add_column(self, col):
        '''
        Adds a column to the circuit

        :param col: list of unitary matrices
        :returns: self
        :raises QuantumException: if not all matrices in col are unitary or
                                  if the size of a Kronecker matrix product of all elements of col is not 2 ** self.qubit_num;
                                  see also _check_list_unitary, _get_col_kronecker_size and _kron
        '''

        if not _check_list_unitary(col):
            raise QuantumException('Not all matrices in col are unitary')
        elif _get_col_kronecker_size(col) != 2 ** self.qubit_num:
            raise QuantumException('Column size is not right for this circuit')

        self.matrix = _kron(col) if self.matrix is None else _kron(col).dot(self.matrix)
        self.circuit.append(col)

        return self
    
    def calculate_probabilities(self, inp):
        '''
        Calculates probabilities of end states

        :param inp: numpy.ndarray of size 2 ** self.qubits with a 1 in the required state index
        :returns: numpy.ndarray of size 2 ** self.qubits of end state probabilities
        :raises QuantumException: if inp is not a numpy.ndarray or
                                  if the size of inp is not 2 ** self.qubits
        '''

        if not isinstance(inp, np.ndarray):
            raise QuantumException('inp should be a numpy.ndarray')
        elif not len(inp) == 2 ** self.qubit_num:
            raise QuantumException('Input size is not right for this circuit')
        
        return self.matrix.dot(inp)