#!/usr/bin/env python
# -----------------------------------------------------------------------------
# KINEMATICS.BIN_STATS
# Laura L Watkins [lauralwatkins@gmail.com]
# -----------------------------------------------------------------------------

from numpy import *
import kinematics


def bin_stats(x, y, dy, w=None, nmc=0, quiet=False):
    
    # number of members in bin
    b_n = x.size
    
    # weighted mean coordinate
    b_x = average(x, weights=w)
    b_dx = sqrt(average((x-b_x)**2, weights=w))
    
    # mean velocities and dispersions
    b_m, b_s = kinematics.maxlh(y, dy, w=w)
    
    # monte carlo errors
    if nmc > 0: b_dm, b_ds, b_s = kinematics.mcerror(nmc, b_m, b_s, dy, w=w)
    else: b_dm, b_ds = 0., 0.
    
    if not quiet: print "{:7.3f} {:7.3f}   {:7.3f} {:7.3f}   "\
        "{:7.3f} {:7.3f}   ".format(b_x, b_dx, b_m, b_dm, b_s, b_ds)
    
    return b_n, b_x, b_dx, b_m, b_dm, b_s, b_ds
