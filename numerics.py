"""This module is responsible for assisting with various generic numerical calculations on scalars and NumPy arrays."""
import numpy as np
from math import isclose

diffeq_steps = 500  #number of steps to take when numerically solving differential equations
atol = 0.0001  #absolute tolerance for determining when a number is close to zero
rtol = 0.001   #releative tolerance for determining when a number is close to zero

def is_near_zero(x, buffer=0.0):
    """Returns whether of not a scalar is sufficiently close to zero.

    Args:
        x: any scalar quantity, like int, double, or float
        buffer (:obj:`float`, optional): adds this to absolute tolerance, for some applications

    Returns:
        bool: True if close to zero with absolute tol. atol and relative tol. rtol.
    """
    return isclose(0, x, atol=(atol + buffer), rtol=rtol)

def is_near_positive(x, buffer=0.0):
    """Returns whether of not a scalar is sufficiently close to zero.

    Args:
        x: any scalar quantity, like int, double, or float
        buffer (:obj:`float`, optional): adds this to absolute tolerance, for some applications

    Returns:
        bool: True if positive or close to zero with absolute tol. atol and relative tol. rtol.
    """
    return x > 0 or isclose(0, x, abs_tol=(atol + buffer), rel_tol=rtol)

def is_near_zero_array(array, buffer=0.0):
    """Returns whether all entries within array are sufficiently close to zero

    Args:
        array: numpy array of scalar quantities
        buffer (:obj:`float`, optional): adds this to absolute tolerance, for some applications

    Returns:
        is_zero (bool): whether all entries within array are close to zero
    """
    return np.allclose(0, array, atol=(atol + buffer), rtol=rtol)

def is_near_positive_array(array, buffer=0.0):
    """Returns whether all entries within array are sufficiently close to positive

    Args:
        array (:obj:`np.ndarray`): numpy array of real-valued quantities 
        buffer (:obj:`float`, optional): adds this to absolute tolerance, for some applications

    Returns:
        is_positive (bool): whether all entires within array are closed to positive
    """
    return np.all(which_near_positive(array, buffer=buffer))

def which_near_zero(array, buffer=0.0):
    """Returns which elements of the array are near zero

    Args:
        array (:obj:`np.ndarray`): numpy array of real-valued quantities 
        buffer (:obj:`float`, optional): adds this to absolute tolerance, for some applications

    Returns:
        which_zero (:obj:`np.ndarray` of `bool`): True wherever an element is close to zero
    """
    return np.isclose(0.0, array, atol=(atol + buffer), rtol=rtol)

def which_near_positive(array, buffer=0.0):
    """Returns which elements of the array are near positive

    Args:
        array (:obj:`np.ndarray`): numpy array of real-valued quantities 
        buffer (:obj:`float`, optional): adds this to absolute tolerance, for some applications

    Return - self.amount_array - change_array, buffer=1.0)

        #computes net expenditures for each country
 - self.amount_array - change_array, buffer=1.0)

        #computes net expenditures for each country
s:
        which_zero (:obj:`np.ndarray` of `bool`): True wherever an element is close to zero or positive
    """
    return np.logical_or(array > 0, np.isclose(0, array, atol=(atol + buffer), rtol=rtol))

def is_close(array1, array2, buffer=0.0):
    """Given two arrays, determines if they are sufficiently close in each entry

    Args:
        array1: numpy array of scalar quantities
        array2: numpy array of same shape as array1
        buffer (:obj:`float`, optional): adds this to absolute tolerance, for some applications

    Returns:
        bool: True if each component is close according to module level atol, rtol
    """
    return np.allclose(array1, array2, atol=(atol + buffer), rtol=rtol)
    
def safe_divide(array1, array2, zero_substitute=1.0):
    """Given two arrays of the same shape, returns their element-wise ratio array1 / array2.
        Wherever array2 is approximately zero (see is_near_zero_array()), replaces fraction with zero_substitute instead

    Args:
        array1 (:obj:`np.ndarray` of :obj:`float`): any array 
        array2 (:obj:`np.ndarray` of :obj:`float`): array of same size as array1
        zero_substitute (:obj:`float`, optional):
    """

    #if array2 is not the same shape as array1, assumes array2 is 1D array whose size is row coutn of array1
    #computationally very slow, should change ASAP once rest of tests begin to pass
    if np.shape(array2) != np.shape(array1):
        array2 = np.expand_dims(array2, axis=1)
        array2 = np.broadcast_to(array2, np.shape(array1))
    
    where_zero = which_near_zero(array2)
    ratio = array1 / (array2 + where_zero)
        
    np.place(ratio, where_zero, zero_substitute)

    return ratio