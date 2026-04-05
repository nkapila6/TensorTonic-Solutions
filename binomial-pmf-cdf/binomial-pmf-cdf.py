# 30.01.2026, 12:05 PM
# Nikhil Kapila
# https://www.tensortonic.com/problems/binomial-pmf-cdf

import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    # Write code here

    pmf = comb(n,k)*(p**k)*((1-p)**(n-k))
    cdf = np.sum([comb(n,i)*(p**i)*(1-p)**(n-i) for i in range(0,k+1)])

    # return (pmf, cdf)
    return (float(pmf), float(cdf))
