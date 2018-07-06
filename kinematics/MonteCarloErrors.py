#!/usr/bin/env python

from __future__ import division, print_function
import numpy as np
from .maxlh import maxlh


def MonteCarloErrors(num_samples, mean, dispersion, errors, weights=None, guess=None, lhbias=True):
    
    """
    Calculates errors on estimates of a Gaussian mean and standard deviation
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
    
    # check for units and unit consistency
    if getattr(mean, "unit", None) is not None \
        and getattr(dispersion, "unit", None) is not None \
        and getattr(errors, "unit", None) is not None:
        dispersion = dispersion.to(mean.unit)
        errors = errors.to(mean.unit)
        unit = mean.unit
    else: unit = 1
    
    # create arrays for sample means and dispersion
    sample_means = np.zeros(num_samples)*unit
    sample_disps = np.zeros(num_samples)*unit
    
    # draw Monte Carlo samples and get maximum likelihood parameter estimates
    for k in range(num_samples):
        
        # draw sample from Gaussian, broadened with uncertainties
        sample = np.random.normal(mean, dispersion, errors.size)*unit \
            + np.random.normal(scale=errors)*unit
        
        # get mean and dispersion of Monte Carlo samples
        sample_means[k], sample_disps[k] = maxlh(sample, errors,
            weights=weights, guess=guess, bias=lhbias)
    
    # error on mean is dispersion of Monte Carlo'd sample means
    error_mean = sample_means.std(ddof=min(num_samples-1,1))
    
    # error on dispersion is dispersion of Monte Carlo'd sample dispersions
    error_dispersion = sample_disps.std(ddof=min(num_samples-1,1))
    
    # ratio of average dispersion in Monte Carlo samples to input dispersion
    bias = sample_disps.mean()/dispersion
    
    # apply correction to dispersion and error to correct for bias
    corrected_dispersion = dispersion / bias
    error_dispersion /= bias
    
    return error_mean, error_dispersion, corrected_dispersion
