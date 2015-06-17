#!/usr/bin/env python
# -----------------------------------------------------------------------------
# KINEMATICS.LL
# Laura L Watkins [lauralwatkins@gmail.com]
# -----------------------------------------------------------------------------

from numpy import *
from scipy.stats import norm


def ll(p, v, dv, w=None):
    
    """
    Calculates the total log-likelihood of an ensemble of velocites, with
    uncertainties, given a model mean and dispersion.
    
    INPUTS
      p  : model mean and dispersion
      v  : velocities
      dv : velocity uncertainties
    
    OPTIONS
      w  : weights [default: 1, ie unweighted]
    """
    
    # require positive sigma
    p[1] = abs(p[1])
    
    # likelihood of each star
    ll = norm.logpdf(v, p[0], sqrt(p[1]**2+dv**2))
    
    # multiply by weights:
    if any(w): ll *= w
    
    # fix zeros which will give logs of -inf
    ll[ll==-inf] = ll[ll>-inf].min()
    
    # total likelihood
    lltot = sum(ll)
    
    # renormalise by weights
    if any(w): lltot *= size(ll) / sum(w)
    
    return lltot
