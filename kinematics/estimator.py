#!/usr/bin/env python
# -----------------------------------------------------------------------------
# KINEMATICS.ESTIMATOR
# Laura L Watkins [lauralwatkins@gmail.com]
# -----------------------------------------------------------------------------

from __future__ import division, print_function
import numpy as np
from .maxlh import maxlh
from .mcerror import mcerror


def estimator(values, errors, num_mcerrors=1000, weights=None, guess=None,
    bias=True):
    
    """
    A maximum likelihood estimator that returns the Gaussian mean and
    dispersion that best describes a given data set. Errors on the
    best-fitting parameters are calculated using Monte Carlo sampling, but
    this can be turned off to speed up run time.
    
    INPUTS
        values : data values
        errors : data uncertainties
    
    OPTIONS
        num_samples : number of Monte Carlo samples to generate
        weights : weights on data points (default: None)
        guess : initial guesses for parameters (default: None, will use mean
                and dispersion of values)
        bias : if set, perform bias correction (default: True)
    """
    
    # estimate mean and disperison using maximum likelihood estimator
    mean, dispersion = maxlh(values, errors, weights=weights, guess=guess,
        bias=bias)
    
    # if no errors required then return mean and dispersion
    if num_mcerrors<=0: return mean, dispersion
    
    # monte carlo to get errors and corrected dispersion
    error_mean, error_dispersion, dispersion = mcerror(num_mcerrors, mean,
        dispersion, errors, weights=weights, guess=guess, lhbias=bias)
    
    return mean, dispersion, error_mean, error_dispersion
