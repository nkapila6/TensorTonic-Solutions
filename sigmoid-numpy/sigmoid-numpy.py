# 30.01.2026, 12:00 PM
# Nikhil Kapila
# https://www.tensortonic.com/problems/sigmoid-numpy

import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # constaints: scalars, lists, numpy arrays
    x = np.array(x)
    return 1/(1+np.exp(-x))
