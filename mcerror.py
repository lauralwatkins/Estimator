#!/usr/bin/env python
# -----------------------------------------------------------------------------
# KINEMATICS.MCERROR
# Laura L Watkins [lauralwatkins@gmail.com]
# -----------------------------------------------------------------------------

import numpy as np
from maxlh import maxlh


def mcerror(num_samples, mean, sigma, errs, weights=None, guess=None, lhbias=True):
    
    """
    Calculates errors on estimates of a gaussian mean and standard deviation
    via a Monte Carlo method.
    
    INPUTS
        num_samples : number of Monte Carlo samples to generate
        mean : estimated mean
        dispersion : estimated dispersion
        errors : uncertainties on measurements
    
    OPTIONS
        weights : weights on data points (default: None)
        guess : initial guesses for parameters (default: None, will use mean
                and dispersion of values)
        bias : if set, perform bias correction (default: True)
    """
    
    # create arrays for sample means and dispersion
    sample_means = np.zeros(num_samples)
    sample_disps = np.zeros(num_samples)
    
    # draw Monte Carlo samples and get maximum likelihood parameter estimates
    for k in range(num_samples):
        
        # draw sample from Gaussian, broadened with uncertainties
        sample = np.random.normal(mean, dispersion, errors.size) \
            + np.random.normal(scale=errors)
        
        # get mean and dispersion of Monte Carlo samples
        sample_means[k], sample_disps[k] = maxlh(sample, errors,
            weights=weights, guess=guess, bias=lhbias)
    
    # error on mean is dispersion of Monte Carlo'd sample means
    error_mean = sample_means.std(ddof=1)
    
    # error on dispersion is dispersion of Monte Carlo'd sample dispersions
    error_dispersion = sample_disps.std(ddof=1)
    
    # ratio of average dispersion in Monte Carlo samples to input dispersion
    bias = sample_disps.mean()/dispersion
    
    # apply correction to dispersion and error to correct for bias
    corrected_dispersion = dispersion / bias
    error_dispersion /= bias
    
    return error_mean, error_dispersion, corrected_dispersion
