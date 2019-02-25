import operator
import functools
import numpy as np
from .Exceptions import QuantumException


_DELTA = 1e-5

def _kron(lst):
    '''
    Kronecker product for all items in the list
    
    :param lst: list of matrices
    :returns: resulting Kronecker product matrix or the first item of lst if len(lst) == 1
    :raises QuantumException: if lst is not a list or
                              if lst is empty      or
                              if not all items of lst are numpy.ndarray
    '''

    if not isinstance(lst, list):
        raise QuantumException('lst should be a list')
    elif len(lst) < 1:
        raise QuantumException('Length of lst should not be 0')
    elif not all([isinstance(x, np.ndarray) for x in lst]):
        raise QuantumException('Every member of lst should be a numpy.ndarray')           

    if len(lst) == 1:
        return lst[0]
    
    res = np.kron(lst[0], lst[1])
    for i in range(2, len(lst)):
        res = np.kron(res, lst[i])
    
    return res

def _check_unitary(mat):
    '''
    Checks if a matrix is unitary

    :param mat: matrix to check
    :returns: True if mat is a unitary matrix, False otherwise
    :raises QuantumException: if mat is not numpy.ndarray
    '''

    if not isinstance(mat, np.ndarray):
        raise QuantumException('mat should be a numpy.ndarray')

    rows, cols = mat.shape
    product = mat.conj().T.dot(mat)
    return (rows == cols) and (product - np.eye(rows) < _DELTA).all()

def _check_list_unitary(lst):
    '''
    Checks if all items of lst are unitary matrices

    :param lst: list of matrices
    :returns: True if all items of lst are unitary matrices, False otherwise
    :raises QuantumException: if lst is not a list or
                              if lst is empty;
                              see also _check_unitary
    '''

    if not isinstance(lst, list):
        raise QuantumException('lst should be a list')
    elif len(lst) == 0:
        raise QuantumException('Length of lst should not be 0')
    
    return all([_check_unitary(x) for x in lst])

def _get_col_kronecker_size(col):
    '''
    Calculates the size of a Kronecker product matrix of all elements of col

    :param col: list of matrices
    :returns: size of a Kronecker product matrix
    :raises QuantumException: if some items of col are not unitary matrices;
                              see also _check_list_unitary
    '''

    if not _check_list_unitary(col):
        raise QuantumException('All items of col should be unitary matrices')
    
    return functools.reduce(operator.mul, [x.shape[0] for x in col])