#!/usr/bin/env python

import numpy as np
from astropy import table
from .BinStats import BinStats


def Bin1d(x, v, dv, weights=None, limits=None, split=[], nbin=10, binby="pop",
    nmc=0, coord="x", mean="v", disp="s", quiet=True):
    
    """
    Generate a 1-dimensional kinematic profile for a given data set.  The data
    can be sorted into bins of equal spacing or equal population using the
    'binby' keyword.  The data can also be split (using the 'split' keyword)
    so that different subsets use different binning schemes.
    
    INPUTS
      x  : coordinate
      v  : velocity
      dv : velocity errors
    
    OPTIONS
      weights: individual weights for measurements [default: None, will use 1]
      limits : exclude data outside limits [default:None - uses min/max of x]
      split  : split dataset at these boundaries [default:[]]
      nbin   : number of bins to use (per subset) [default:10]
      binby  : binning scheme ['space'|'pop'] (per subset) [default:'pop']
      nmc    : number of monte carlo errors to generate [default:0]
      coord  : name of coordinate [default:'x']
      mean   : name of means [default:'v']
      disp   : name of dispersions [default:'s']
      quiet  : supress read out [default:False]
    
    EXAMPLES
      1) Generate a profile with 10 equally-populated bins:
      $ bins = bin1d(x, v, dv)
      
      2) Generate a profile with 20 equally-spaced bins and change
      coordinate name to 'radius':
      $ bins = bin1d(x, v, dv, nbin=20, binby="space", coord="radius")
      
      3) Generate a profile split into 3 subsets (with splits occuring at 40
      and 80 units); use 10 bins for each subset and generate 50 Monte-Carlo
      errors.  For x<40 and x>80, bins are equally populated; for 40<x<80,
      bins are equally spaced:
      $ bins = bin1d(x, v, dv, split=[40,80], nbin=10, nmc=50,
                     binby=["pop","space","pop"])
    """
    
    
    # if not specified, bin limits are data limits
    if not limits: limits = [x.min(), x.max()]
    
    # add bin limits to split to set subdivision boundaries
    split = [np.min(limits)] + split + [np.max(limits)]
    nsub = np.size(split) - 1
    
    # make sure that nbin and binby are lists
    if type(nbin) != list: nbin = [nbin]*nsub
    if type(binby) != list: binby = [binby]*nsub
    
    # total number of bins requested
    ntot = np.sum(nbin)
    
    
    if not quiet:
        print("\nmaximum likelihood fit\n")
        print("  subdivisions:", nsub)
        print("  total bins:", ntot)
        print("  number of monte carlo errors:", nmc)
        print("")
    
    
    # --------------------------------------
    
    
    # set up array for bin properties
    bins = table.Table(
        names=("id","n", coord, "d"+coord, mean, "d"+mean, disp, "d"+disp),
        dtype=(int, int, float, float, float, float, float, float))
    
    
    # bin in specified coordinate
    for j in range(nsub):
        
        # select stars in subdivision
        if j == 0: sub = np.where( (x>=split[j]) & (x<=split[j+1]) )[0]
        else: sub = np.where( (x>split[j]) & (x<=split[j+1]) )[0]
        
        if not quiet:
            print("  subset ", j+1)
            print("    bins:", nbin[j])
            print("    bin by:", binby[j])
            print("    members:", np.size(sub))
        
        # sort by x and get population required for ~equal numbers
        if binby[j] == "pop":
            pcs = np.linspace(0,100,nbin[j]+1)
            sortx = x[sub].argsort()
            nsbin = np.size(sub) / nbin[j]
            if not quiet: print("    bin pop: ", nsbin)
        
        # bin space for equal width bins
        if binby[j] == "space":
            if nbin[j] == 1: bs = split[j+1] - split[j]
            else: bs = (split[j+1] - split[j]) / nbin[j]
            if not quiet: print("    bin space: ", bs)
        
        for i in range(nbin[j]):
            
            nfill = np.int(np.sum(nbin[:j]))
            
            # select stars in radial bin for equal population
            if binby[j] == "pop":
                lo = np.percentile(x[sub], pcs[i])
                hi = np.percentile(x[sub], pcs[i+1])
                if i == 0: d = sub[x[sub]<hi]
                if i == nbin[j]-1: d = sub[x[sub]>=lo]
                else: d = sub[(x[sub]>=lo) & (x[sub]<hi)]
            
            if binby[j] == "space":
                # select stars in radial bin for equal width
                d = sub[(x[sub]>=split[j]+i*bs) & (x[sub]<split[j]+(i+1)*bs)]
            
            if not quiet: print("    {:2}  {:4}  ".format(i+nfill,np.size(d)),)
            if np.size(d) > 0:
                if np.any(weights): bins.add_row((i+nfill,) \
                    + bin_stats(x[d], v[d].reshape(v[d].size),
                    dv[d].reshape(v[d].size),
                    weights=weights[d].reshape(v[d].size),
                    nmc=nmc, quiet=quiet))
                else: bins.add_row((i+nfill,) + bin_stats(x[d],
                    v[d].reshape(v[d].size), dv[d].reshape(v[d].size),
                    nmc=nmc, quiet=quiet))
        
        if not quiet: print("")
    
    
    # remove empty bins
    bins = bins[bins["n"]>0]
    
    
    return bins
