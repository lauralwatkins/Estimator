#!/usr/bin/env python

import numpy as np
from scipy import optimize, stats
import math


def lnLikelihoodDouble(parameters, values, errors, weights=None):
    
    """
    Calculates the total log-likelihood of an ensemble of values, with
    uncertainties, for a double Gaussian distribution (two means and
    two dispersions).
    
    INPUTS
        parameters : model parameters (see below)
        values : data values
        errors : data uncertainties
    
    OPTIONS
        weights : weights on each data point [default: None, ie unweighted]
    
    PARAMETERS
        mean1 : model mean 1
        dipsersion1 : model dispersion 1
        mean2 : model mean 2
        dipsersion2 : model dispersion 2
        f : fraction of component 1
    """
    
    mean1, dispersion1, mean2, dispersion2, f = parameters
    
    # insist that mean1 is less than mean2 or solution is degenerate
    if mean1>mean2:
        return -np.inf
    
    # check for unit consistency
    if getattr(mean1, "unit", None) is not None \
        and getattr(dispersion1, "unit", None) is not None \
        and getattr(mean2, "unit", None) is not None \
        and getattr(dispersion2, "unit", None) is not None \
        and getattr(values, "unit", None) is not None \
        and getattr(errors, "unit", None) is not None:
        mean1 = mean1.to(values.unit)
        dispersion1 = dispersion1.to(values.unit)
        mean2 = mean2.to(values.unit)
        dispersion2 = dispersion2.to(values.unit)
        errors = errors.to(values.unit)
    
    # require positive dispersions
    dispersion1 = np.abs(dispersion1)
    dispersion2 = np.abs(dispersion2)
    
    # likelihood of each data point
    conv_dispersion1 = np.sqrt(dispersion1**2+errors**2)
    conv_dispersion2 = np.sqrt(dispersion2**2+errors**2)
    likelihoods = f*stats.norm.pdf(values, mean1, conv_dispersion1) \
        + (1-f)*stats.norm.pdf(values, mean2, conv_dispersion2)
    
    # check that all are positive (should be!) and non-zero
    if np.all(likelihoods<=0):
        return -np.inf
    
    # set zeros (or negatives) to the lowest non-zero value
    likelihoods[likelihoods<=0] = likelihoods[likelihoods>0].min()*1e-5
    
    # and take the log
    ln_likelihoods = np.log(likelihoods)
    
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

def FitDouble(values, errors, weights=None, guess=None, bias=True):
    
    """
    Finds the mean and upper and lower dispersions that maximise the
    likelihood for a given set of values and their uncertainties.  The
    routine also performs a correction for the bias inherent in maximum
    likelihood estimators, following the description in van de Ven et al.
    2006 (A&A 445 513).
    
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
        guess = (wmean-wstdv, np.sqrt(wstdv**2 + dmean**2),
            wmean+wstdv, np.sqrt(wstdv**2 + dmean**2), 0.5)
    else:
        guess = [g/unit for g in guess]
    
    # do the maximum likelihood fitting (find minimum of -logL)
    def partialFunction(parameters):
        return -lnLikelihoodDouble(parameters, values, errors,
            weights=weights)
    mean1, dispersion1, mean2, dispersion2, f \
        = optimize.fmin(partialFunction, guess, disp=False)
    
    # calculate bias in ML estimator and correct dispersion
    # (see appendix A1 of van de Ven et al. 2006 A&A 445 513)
    if bias:
        b = np.exp(math.lgamma(values.size/2.)-math.lgamma((values.size-1)/2))\
            * np.sqrt(2/values.size)
        dispersion1 = np.sqrt(dispersion1**2 + (1-b**2) * errors.mean()**2)/b
        dispersion2 = np.sqrt(dispersion2**2 + (1-b**2) * errors.mean()**2)/b
    
    # re-set units of outputs
    mean1 *= unit
    dispersion1 *= unit
    mean2 *= unit
    dispersion2 *= unit
    
    return mean1, dispersion1, mean2, dispersion2, f

def MonteCarloDouble(Nsamples, mean1, dispersion1, mean2, dispersion2, f, errors, weights=None, guess=None, bias=True):
    
    """
    Calculates uncertainties on estimates for a Double Gaussian distribution
    via a Monte Carlo method.
    
    INPUTS
        Nsamples : number of Monte Carlo samples to generate
        mean1 : estimated mean component 1
        dispersion1 : estimated dispersion component 1
        mean2 : estimated mean component 2
        dispersion2 : estimated dispersion component 2
        f : fraction of component 1
        errors : uncertainties on measurements
    
    OPTIONS
        weights : weights on data points (default: None)
        guess : initial guesses for parameters (default: None, will use mean
                and dispersion of values)
        bias : if set, perform bias correction (default: True)
    """
    
    # check for units and unit consistency
    if getattr(mean1, "unit", None) is not None \
        and getattr(dispersion1, "unit", None) is not None \
        and getattr(mean2, "unit", None) is not None \
        and getattr(dispersion2, "unit", None) is not None \
        and getattr(errors, "unit", None) is not None:
        unit = errors.unit
        mean1 = mean1.to(unit).value
        dispersion1 = dispersion1.to(unit).value
        mean2 = mean2.to(unit).value
        dispersion2 = dispersion2.to(unit).value
        errors = errors.value
    else: unit = 1
    
    # create arrays for sample means and dispersion
    sample_mean1 = np.zeros(Nsamples)
    sample_dispersion1 = np.zeros(Nsamples)
    sample_mean2 = np.zeros(Nsamples)
    sample_dispersion2 = np.zeros(Nsamples)
    sample_f = np.zeros(Nsamples)
    
    # draw Monte Carlo samples and get maximum likelihood parameter estimates
    for k in range(Nsamples):
        
        # array to randomise samples
        randomise = np.random.rand(len(errors)).argsort()
        
        # draw sample from Gaussian, broadened with uncertainties
        sample1 = np.random.normal(mean1,
            np.sqrt(dispersion1**2+errors**2))[randomise]
        sample2 = np.random.normal(mean2,
            np.sqrt(dispersion2**2+errors**2))[randomise]
        
        # create combined sample
        Nsplit = int(np.round(f*len(errors)))
        sample = np.concatenate((sample1[:Nsplit], sample2[Nsplit:]))
        
        # get mean and dispersion of Monte Carlo samples
        sample_mean1[k], sample_dispersion1[k], sample_mean2[k], \
            sample_dispersion2[k], sample_f[k] = FitDouble(sample,
            errors[randomise], weights=weights, guess=guess, bias=bias)
    
    # uncertainty on parameters are dispersions of Monte Carlo'd samples
    error_mean1 = sample_mean1.std(ddof=min(Nsamples-1,1))
    error_mean2 = sample_mean2.std(ddof=min(Nsamples-1,1))
    error_dispersion1 = sample_dispersion1.std(ddof=min(Nsamples-1,1))
    error_dispersion2 = sample_dispersion2.std(ddof=min(Nsamples-1,1))
    error_f = sample_f.std(ddof=min(Nsamples-1,1))
    
    # ratio of average dispersion in Monte Carlo samples to input dispersion
    ratio1 = sample_dispersion1.mean()/dispersion1
    ratio2 = sample_dispersion2.mean()/dispersion2
    
    # apply correction to dispersion and error
    corrected_dispersion1 = dispersion1/ratio1
    corrected_dispersion2 = dispersion2/ratio2
    error_dispersion1 /= ratio1
    error_dispersion2 /= ratio2
    
    # re-set units of outputs
    error_mean1 *= unit
    error_dispersion1 *= unit
    corrected_dispersion1 *= unit
    error_mean2 *= unit
    error_dispersion2 *= unit
    corrected_dispersion2 *= unit
    
    return error_mean1, error_dispersion1, error_mean2, error_dispersion2, error_f, corrected_dispersion1, corrected_dispersion2

def EstimatorDouble(values, errors, Nsamples=100, weights=None, guess=None, bias=True):
    
    """
    A maximum likelihood estimator that returns the parameters for a Double
    Gaussian that best describes a given data set. Errors on the best-fitting
    parameters are calculated using Monte Carlo sampling, but this can be
    turned off to speed up run time.
    
    INPUTS
        values : data values
        errors : data uncertainties
    
    OPTIONS
        Nsamples : number of Monte Carlo samples to generate (default: 100)
        weights : weights on data points (default: None)
        guess : initial guesses for parameters (default: None, will use mean
                and dispersion of values)
        bias : if set, perform bias correction (default: True)
    """
    
    # estimate mean and disperison using maximum likelihood estimator
    mean1, dispersion1, mean2, dispersion2, f = FitDouble(values, errors,
        weights=weights, guess=guess, bias=bias)
    
    # if no errors required then return mean and dispersion
    if Nsamples<=0: return mean1, dispersion1, mean2, dispersion2, f
    
    # monte carlo to get errors and corrected dispersion
    error_mean1, error_dispersion1, error_mean2, error_dispersion2, error_f, \
        dispersion1, dispersion2 = MonteCarloDouble(Nsamples, mean1,
        dispersion1, mean2, dispersion2, f, errors, weights=weights,
        guess=guess, bias=bias)
    
    return mean1, dispersion1, mean2, dispersion2, f, error_mean1, error_dispersion1, error_mean2, error_dispersion2, error_f
