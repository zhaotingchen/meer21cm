Releases
========

dev
---


v0.2.0
------

Features
++++++++
* MeerKLASS map i/o functionalities consistent with meerpower
* Basic cosmological calculator
* plot functionalities as a separate module
* telescope-related functions including beam size and convolution

Enhancements
++++++++++++
* consistent mean and covariance calculation in PCA

Bugfixes
++++++++
* Fixed a los-axis tranpose back issue

v0.1.1
------

Enhancements
++++++++++++
* allow fixed RA and Dec in lognormal simulations
* add calculation of angles between coorindates on the sphere


v0.1.0
------

Features
++++++++
* Generation of colored noise for simulating systematics
* A bit more docs

Enhancements
++++++++++++
* RSD effect in lognormal mocks based on Kaiser effects

Bugfixes
++++++++
* Fixed a mismatch of h unit in the lognormal simulation

v0.0.1
------
This is the first version

Features
++++++++
* Log-Normal and Poisson generation of HI galaxy signals based on HIMF and velocity dispersion
* Stacking in 3D space
* Calculating effective weights for correcting signal loss for PCA
* Simulation of synchrotron foreground emission using Haslam template
* Docs with API summary
* Unit test coverage
