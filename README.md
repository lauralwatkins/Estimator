ESTIMATOR
=========

> **AUTHORS**
Laura L Watkins (ESA Office, STScI), <lauralwatkins@gmail.com>


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

* **Bin1d**: Generate a 1-dimensional kinematic profile for a given data set.  The data can be sorted into bins of equal spacing or equal population, and can also be split such that different subsets use different binning schemes.
* **BinStats**: Estimate the properties of all stars in a single bin.
* **Gaussian**: Fits a Gaussian to a given set of values (with uncertainties). Uncertainties on the parameters are calculated via Monte Carlo sampling.
* **Double**: Fits a Double Gaussian to a given set of values (with uncertainties). Uncertainties on the parameters are calculated via Monte Carlo sampling.


-------------------------------------------------------------------------------


REQUIREMENTS
----------------------------------------

This code uses the standard python libraries numpy and scipy, and also makes use of astropy.
