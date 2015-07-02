TOOLBOX
=======

> **AUTHORS**
Laura L Watkins (STScI), <lauralwatkins@gmail.com>


-------------------------------------------------------------------------------


CONTENTS
--------

* license and referencing
* code description
* requirements


-------------------------------------------------------------------------------


LICENSE AND REFERENCING
-----------------------

This code is released under a BSD 2-clause license.

If you find this code useful for your research, please mention it in your acknowledgements.


-------------------------------------------------------------------------------


CODE DESCRIPTION
----------------

This code bins a dataset along a given axis and calculates the mean and dispersion of a given quantity in each bin.

* **bin1d**: Generate a 1-dimensional kinematic profile for a given data set.  The data can be sorted into bins of equal spacing or equal population, and can also be split such that different subsets use different binning schemes.
* **bin_stats**: Estimate the properties of all stars in a single bin.
* **ll**: Calculates the total log-likelihood of an ensemble of velocities, with
uncertainties, given a model mean and dispersion.
* **maxlh**: Finds the mean and dispersion that maximises the likelihood for a given set of velocities and velocity uncertainties.  The routine also performs a correction for the bias inherent in maximum likelihood estimators, following the description in van de Ven et al. 2006 (A&A 445 513).
* **mcerror**: Calculates errors on estimates of a gaussian mean and standard deviation via a Monte Carlo method.


-------------------------------------------------------------------------------


REQUIREMENTS
----------------------------------------

This code uses the standard python libraries numpy and scipy, and also makes use of astropy.
