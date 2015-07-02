#!/usr/bin/env python
# -----------------------------------------------------------------------------
# KINEMATICS.LL
# Laura L Watkins [lauralwatkins@gmail.com]
# -----------------------------------------------------------------------------

import numpy as np
from scipy.stats import norm


def ll(p, v, dv, w=None):
    
    """
    Calculates the total log-likelihood of an ensemble of velocities, with
    uncertainties, given a model mean and dispersion.
    
    INPUTS
      p  : model mean and dispersion
      v  : velocities
      dv : velocity uncertainties
    
    OPTIONS
      w  : weights [default: 1, ie unweighted]
    """
    
    # require positive sigma
    p[1] = np.abs(p[1])
    
    # likelihood of each star
    ll = norm.logpdf(v, p[0], np.sqrt(p[1]**2+dv**2))
    
    # multiply by weights:
    if np.any(w): ll *= w
    
    # fix zeros which will give logs of -inf
    ll[ll==-np.inf] = ll[ll>-np.inf].min()
    
    # total likelihood
    lltot = np.sum(ll)
    
    # renormalise by weights
    if np.any(w): lltot *= np.size(ll) / np.sum(w)
    
    return lltot
