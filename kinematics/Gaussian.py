#!/usr/bin/env python

import numpy as np
from scipy import optimize, stats
import math


def lnLikelihoodGaussian(parameters, values, errors, weights=None):
    
    """
    Calculates the total log-likelihood of an ensemble of values, with
    uncertainties, for a Gaussian distribution.
    
    INPUTS
        parameters : model parameters (see below)
        values : data values
        errors : data uncertainties
    
    OPTIONS
        weights : weights on each data point [default: None, ie unweighted]
    
    PARAMETERS
        mean : model mean
        dispersion : model dispersion
    """
    
    mean, dispersion = parameters
    
    # check for unit consistency
    if getattr(mean, "unit", None) is not None \
        and getattr(dispersion, "unit", None) is not None \
        and getattr(values, "unit", None) is not None \
        and getattr(errors, "unit", None) is not None:
        mean = mean.to(values.unit)
        dispersion = dispersion.to(values.unit)
        errors = errors.to(values.unit)
    
    # require positive dispersion
    dispersion = np.abs(dispersion)
    
    # log likelihood of each data point
    ln_likelihoods = stats.norm.logpdf(values, mean, np.sqrt(dispersion**2+errors**2))
    
    # multiply by weights:
    if weights is not None:
        ln_likelihoods *= weights
    
    # remove -infinities
    ln_likelihoods[ln_likelihoods==-np.inf] \
        = ln_likelihoods[ln_likelihoods>-np.inf].min()
    
    # total likelihood
    total_ln_likelihood = np.sum(ln_likelihoods)
    
    # renormalise by weights
    if weights is not None:
        total_ln_likelihood *= np.size(ln_likelihoods)/np.sum(weights)
    
    return total_ln_likelihood

def FitGaussian(values, errors, weights=None, guess=None, bias=True):
    
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
        unit = values.unit
        values = values.to(unit).value
        errors = errors.to(unit).value
    else: unit = 1
    
    # use equal weights (==1) if none are set
    if weights is None:
        weights = np.ones(values.size)
    
    # intialise with mean and sigma of inputs unless given seeds
    if guess is None:
        wmean = np.mean(values*weights)/np.mean(weights)
        wstdv = np.sqrt(np.mean((values-wmean)**2*weights)/np.mean(weights))
        dmean = np.mean(errors*weights)/np.mean(weights)
        guess = (wmean, np.sqrt(wstdv**2 + dmean**2))
    else:
        guess = [g for g in guess]
    
    # do the maximum likelihood fitting (find minimum of -logL)
    def partialFunction(parameters):
        return -lnLikelihoodGaussian(parameters, values, errors,
            weights=weights)
    mean, dispersion = optimize.fmin(partialFunction, guess, disp=False)
    
    # calculate bias in ML estimator and correct dispersion
    # (see appendix A1 of van de Ven et al. 2006 A&A 445 513)
    if bias:
        b = np.exp(math.lgamma(values.size/2)-math.lgamma((values.size-1)/2))\
            * np.sqrt(2/values.size)
        dispersion = np.sqrt(dispersion**2 + (1-b**2) * errors.mean()**2)/b
    
    # re-set units of outputs
    mean *= unit
    dispersion *= unit
    
    return mean, dispersion

def MonteCarloGaussian(Nsamples, mean, dispersion, errors, weights=None, guess=None, bias=True):
    
    """
    Calculates uncertainties on estimates for a Gaussian distribution
    via a Monte Carlo method.
    
    INPUTS
        Nsamples : number of Monte Carlo samples to generate
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
        unit = errors.unit
        mean = mean.to(unit).value
        dispersion = dispersion.to(unit).value
        errors = errors.value
    else: unit = 1
    
    # create arrays for sample means and dispersion
    sample_means = np.zeros(Nsamples)
    sample_dispersions = np.zeros(Nsamples)
    
    # draw Monte Carlo samples and get maximum likelihood parameter estimates
    for k in range(Nsamples):
        
        # draw sample from Gaussian, broadened with uncertainties
        sample = np.random.normal(mean, np.sqrt(dispersion**2+errors**2))
        
        # get mean and dispersion of Monte Carlo samples
        sample_means[k], sample_dispersions[k] = FitGaussian(sample, errors,
            weights=weights, guess=guess, bias=bias)
    
    # uncertainty on parameters are dispersions of Monte Carlo'd samples
    error_mean = sample_means.std(ddof=min(Nsamples-1,1))
    error_dispersion = sample_dispersions.std(ddof=min(Nsamples-1,1))
    
    # ratio of average dispersion in Monte Carlo samples to input dispersion
    ratio = sample_dispersions.mean()/dispersion
    
    # apply correction to dispersion and error
    corrected_dispersion = dispersion / ratio
    error_dispersion /= ratio
    
    # re-set units of outputs
    error_mean *= unit
    error_dispersion *= unit
    corrected_dispersion *= unit
    
    return error_mean, error_dispersion, corrected_dispersion

def Gaussian(values, errors, Nsamples=100, weights=None, guess=None, bias=True):
    
    """
    A maximum likelihood estimator that returns the parameters for a Gaussian
    that best describes a given data set. Uncertainties on the
    best-fitting parameters are calculated using Monte Carlo sampling, but
    this can be turned off to speed up run time.
    
    INPUTS
        values : data values
        errors : data uncertainties
    
    OPTIONS
        Nsamples : number of Monte Carlo samples to generate
        weights : weights on data points (default: None)
        guess : initial guesses for parameters (default: None, will use mean
                and dispersion of values)
        bias : if set, perform bias correction (default: True)
    """
    
    # estimate mean and disperison using maximum likelihood estimator
    mean, dispersion = FitGaussian(values, errors, weights=weights,
        guess=guess, bias=bias)
    
    # if no errors required then return mean and dispersion
    if Nsamples<=0: return mean, dispersion
    
    # monte carlo to get errors and corrected dispersion
    error_mean, error_dispersion, dispersion = MonteCarloGaussian(Nsamples,
        mean, dispersion, errors, weights=weights, guess=guess, bias=bias)
    
    return mean, dispersion, error_mean, error_dispersion
