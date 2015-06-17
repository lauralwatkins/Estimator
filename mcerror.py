#!/usr/bin/env python
# -----------------------------------------------------------------------------
# KINEMATICS.MCERROR
# Laura L Watkins [lauralwatkins@gmail.com]
# -----------------------------------------------------------------------------

from numpy import *
from maxlh import maxlh


def mcerror(nmc, mean, sigma, errs, w=None, p0=None, lhbias=True):
    
    """
    Calculates errors on estimates of a gaussian mean and standard deviation
    via a Monte Carlo method.
    
    INPUTS
      nmc   : number of Monte Carlo samples to use to estimate errors
      mean  : estimated mean
      sigma : estimated dispersion
      errs  : uncertainties on measurements
    
    OPTIONS
      p0 : initial guesses for parameters (default: None)
    """
    
    vm = zeros(nmc)
    vs = zeros(nmc)
    
    # monte carlo errors
    for k in range(nmc):
        
        # draw velocities from gaussian, broadened with uncertainties
        v = random.normal(mean, sigma, errs.size) + random.normal(scale=errs)
        
        # get mean and dispersion of monte-carlo velocities
        vm[k], vs[k] = maxlh(v, errs, w=w, p0=p0, bias=lhbias)
    
    # error on mean is dispersion of mc means
    dm = vm.std(ddof=1)
    
    # error on dispersion is dispersion of mc dispersions
    ds = vs.std(ddof=1)
    
    # ratio of average dispersion in mc drawings to input dispersion
    bias = vs.mean() / sigma
    
    # apply correction to dispersion and error to correct for bias
    corrsigma = sigma / bias
    ds /= bias
    
    return dm, ds, corrsigma
