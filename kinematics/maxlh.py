#!/usr/bin/env python

from __future__ import division, print_function
import math, numpy as np
from scipy.optimize import fmin
from .ll import ll


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
    
    # check for units and unit consistency
    if getattr(values, "unit", None) is not None \
        and getattr(errors, "unit", None) is not None:
        errors = errors.to(values.unit)
        unit = values.unit
    else: unit = 1
    
    # use equal weights (==1) if none are set
    if weights is None: weights = np.ones(values.size)
    
    # intialise with mean and sigma of inputs unless given seeds
    if guess is None:
        wmean = np.mean(values*weights)/np.mean(weights)
        wstdv = np.sqrt(np.mean((values-wmean)**2*weights)/np.mean(weights))
        dmean = np.mean(errors*weights)/np.mean(weights)
        guess = (wmean/unit, np.sqrt(wstdv**2 + dmean**2)/unit)
    else:
        guess = [g/unit for g in guess]
    
    # do the maximum likelihood fitting (find minimum of -logL)
    def llpart(pp): return -ll(pp*unit, values, errors, weights=weights)
    mean, dispersion = fmin(llpart, guess, disp=False)*unit
    
    # calculate bias in ML estimator and correct dispersion
    # (see appendix A1 of van de Ven et al. 2006 A&A 445 513)
    if bias:
        b = np.exp(math.lgamma(values.size/2.)-math.lgamma((values.size-1)/2.))\
            * np.sqrt(2./values.size)
        dispersion = np.sqrt(dispersion**2 + (1.-b**2) * errors.mean()**2)/b
    
    return mean, dispersion