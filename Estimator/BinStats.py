#!/usr/bin/env python

import numpy as np
from .Gaussian import FitGaussian, MonteCarloGaussian


def BinStats(coords, values, errors, weights=None, nmc=0, quiet=False):
    
    """
    Estimate the properties of all stars in a single bin.
    
    INPUTS
      coords : data coordinates
      values : data values
      errors : data uncertainties on values
    
    OPTIONS
      weights : individual weights for measurements [default: None, will use 1]
      num_mcerrors : number of monte carlo errors to generate [default:0]
      quiet : supress read out [default:False]
    """
    
    
    # number of members in bin
    num_points = coords.size
    
    # weighted mean coordinate
    mean_coords = np.average(coords, weights=weights)
    disp_coords = np.sqrt(np.average((coords-mean_coords)**2, weights=weights))
    
    # mean velocities and dispersions
    mean_values, disp_values = FitGaussian(values, errors, weights=weights)
    
    # monte carlo errors
    if nmc>0: error_mean, error_disp, disp_values \
        = MonteCarloGaussian(nmc, mean_values, disp_values, errors,
        weights=weights)
    else: error_mean, error_disp = 0., 0.
    
    if not quiet: print("{:7.3f} {:7.3f}   {:7.3f} {:7.3f}   "\
        "{:7.3f} {:7.3f}   ".format(mean_coords, disp_coords, mean_values,
        error_means, disp_values, error_disp))
    
    return num_points, mean_coords, disp_coords, mean_values, error_mean, disp_values, error_disp
