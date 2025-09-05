Releases
========

v0.7.0
------
Features
++++++++
* transfer function class
* parallelization for transfer function
* numerical transfer function calculation from cross and auto mock
* parameter fitting sampler class
* parallelization for parameter fitting
* support emcee and nautilus for sampling

Enhancements
++++++++++++
* allow k-mode cut in 1D power spectrum
* add validation tests 00 and 01 in paper

Bugfixes
++++++++
* fix a bug in FPS initialization
* fix a bug in incorrect sigma_z usage in model power calculation

v0.6.1
------
Features
++++++++
* allow dndz input for mock galaxy simulation
* allow flat-sky approximation in mock and ps calculation
* allow read in UHF pickle file

Bugfixes
++++++++
* fix a bug in get_enclosing_box

v0.6.0
------
Features
++++++++
* mock tracer positions
* new HI galaxy simulation class
* new cosmology parameter class

Enhancements
++++++++++++
* better RSD routine for lognormal simulation
* allow using baccoemu instead of camb

v0.5.0
------
Features
++++++++
* mock simulation
* consistent model power to mock to field power
* explicit dependency checks and cache
* different beam models

Enhancements
++++++++++++
* gridding now part of PS class
* consistent tests for sky map to power spectrum
* remove hiimtool dependency
* end-to-end test from input ps to final power spectrum estimation

v0.4.0
------
Features
++++++++
* a base class for better structure
* model power spectrum
* power spectrum weights and convolution

Enhancements
++++++++++++
* allows more flexibility in PCA
* more precise HI average temp

v0.3.0
------
Features
++++++++
* Gridding functionalities to grid sky map to regular grids
* Basic power spectrum estimation functionalities

Enhancements
++++++++++++
* find enclosing box functions migrated to grid module

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
