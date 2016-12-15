#!/usr/bin/env python
# -----------------------------------------------------------------------------
# KINEMATICS.LL
# Laura L Watkins [lauralwatkins@gmail.com]
# -----------------------------------------------------------------------------

import numpy as np
from scipy.stats import norm


def ll((mean, dispersion), values, errors, weights=None):
    
    """
    Calculates the total log-likelihood of an ensemble of values, with
    uncertainties, given a model mean and dispersion.
    
    INPUTS
        mean : model mean
        dispersion : model dispersion
        values : data values
        errors : data uncertainties
    
    OPTIONS
        weights : weights on each data point [default: None, ie unweighted]
    """
    
    # require positive dispersion
    dispersion = np.abs(dispersion)
    
    # log likelihood of each data point
    loglike = norm.logpdf(values, mean, np.sqrt(dispersion**2+errors**2))
    
    # multiply by weights:
    if weights is not None: loglike *= weights
    
    # remove -infinities
    loglike[loglike==-np.inf] = loglike[loglike>-np.inf].min()
    
    # total likelihood
    loglike_total = np.sum(loglike)
    
    # renormalise by weights
    if weights is not None: loglike_total *= np.size(loglike)/np.sum(weights)
    
    return loglike_total
