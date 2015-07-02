#!/usr/bin/env python
# -----------------------------------------------------------------------------
# KINEMATICS.BIN_STATS
# Laura L Watkins [lauralwatkins@gmail.com]
# -----------------------------------------------------------------------------

import numpy as np
import kinematics


def bin_stats(x, y, dy, w=None, nmc=0, quiet=False):
    
    """
    Estimate the properties of all stars in a single bin.
    
    INPUTS
      x  : coordinate
      y  : quantity
      dy : quantity errors
    
    OPTIONS
      w     : individual weights for measurements [default: None, will use 1]
      nmc   : number of monte carlo errors to generate [default:0]
      quiet : supress read out [default:False]
    """
    
    
    # number of members in bin
    b_n = x.size
    
    # weighted mean coordinate
    b_x = np.average(x, weights=w)
    b_dx = np.sqrt(np.average((x-b_x)**2, weights=w))
    
    # mean velocities and dispersions
    b_m, b_s = kinematics.maxlh(y, dy, w=w)
    
    # monte carlo errors
    if nmc > 0: b_dm, b_ds, b_s = kinematics.mcerror(nmc, b_m, b_s, dy, w=w)
    else: b_dm, b_ds = 0., 0.
    
    if not quiet: print "{:7.3f} {:7.3f}   {:7.3f} {:7.3f}   "\
        "{:7.3f} {:7.3f}   ".format(b_x, b_dx, b_m, b_dm, b_s, b_ds)
    
    return b_n, b_x, b_dx, b_m, b_dm, b_s, b_ds
