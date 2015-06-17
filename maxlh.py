#!/usr/bin/env python
# -----------------------------------------------------------------------------
# KINEMATICS.MAXLH
# Laura L Watkins [lauralwatkins@gmail.com]
# -----------------------------------------------------------------------------

import math
from numpy import *
from scipy.optimize import fmin
from ll import ll


def maxlh(v, dv, w=None, p0=False, bias=True):
    
    """
    Finds the mean and dispersion that maximises the likelihood for a given
    set of velocities and velocity uncertainties.  The routine also performs
    a correction for the bias inherent in maximum likelihood estimators,
    following the description in van de Ven et al. 2006 (A&A 445 513).
    
    INPUTS
      v  : stellar velocities
      dv : stellar velocity uncertainties
    
    OPTIONS
      p0 : initial guesses for parameters (default: mean and dispersion of v)
      bias : if set, perform bias correction (default: True)
    """
    
    # intialise with mean and sigma of inputs unless given seeds
    if not any(p0):
        wmean = average(v, weights=w)
        wstdv = sqrt(average((v-wmean)**2, weights=w))
        dmean = average(dv, weights=w)
        p0 = array([wmean, sqrt(wstdv**2 + dmean**2)])
    
    # do the maximum likelihood fitting (find minimum of -logL)
    llpart = lambda pp : -ll(pp, v=v, dv=dv, w=w)
    p = fmin(llpart, p0, disp=False)
    
    # calculate bias in ML estimator and correct dispersion
    # (see appendix A1 of van de Ven et al. 2006 A&A 445 513)
    if bias:
        b = exp(math.lgamma(v.size/2.) - math.lgamma((v.size-1)/2.)) \
            * sqrt(2./v.size)
        p[1] = sqrt(p[1]**2 + (1.-b**2) * dv.mean()**2)/b
    
    return p
