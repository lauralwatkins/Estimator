#!/usr/bin/env python
# -----------------------------------------------------------------------------
# KINEMATICS.MAXLH
# Laura L Watkins [lauralwatkins@gmail.com]
# -----------------------------------------------------------------------------

import math, numpy as np
from scipy.optimize import fmin
from ll import ll


def maxlh(values, errors, weights=None, guess=None, bias=True):
    
    """
    Finds the mean and dispersion that maximises the likelihood for a given
    set of values and their uncertainties.  The routine also performs a
    correction for the bias inherent in maximum likelihood estimators,
    following the description in van de Ven et al. 2006 (A&A 445 513).
    
    INPUTS
        values : data values
        errors : data uncertainties
    
    OPTIONS
        weights : weights on data points (default: None)
        guess : initial guesses for parameters (default: None, will use mean
                and dispersion of values)
        bias : if set, perform bias correction (default: True)
    """
    
    # intialise with mean and sigma of inputs unless given seeds
    if guess is None:
        wmean = np.average(values, weights=weights)
        wstdv = np.sqrt(np.average((values-wmean)**2, weights=weights))
        dmean = np.average(errors, weights=weights)
        guess = np.array([wmean, np.sqrt(wstdv**2 + dmean**2)])
    
    # do the maximum likelihood fitting (find minimum of -logL)
    def llpart(pp): return -ll(pp, values, errors, weights=weights)
    mean, dispersion = fmin(llpart, guess, disp=False)
    
    # calculate bias in ML estimator and correct dispersion
    # (see appendix A1 of van de Ven et al. 2006 A&A 445 513)
    if bias:
        b = np.exp(math.lgamma(values.size/2.)-math.lgamma((values.size-1)/2.))\
            * np.sqrt(2./values.size)
        dispersion = np.sqrt(dispersion**2 + (1.-b**2) * errors.mean()**2)/b
    
    return mean, dispersion
