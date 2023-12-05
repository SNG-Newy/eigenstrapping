
|

.. image:: https://img.shields.io/pypi/v/eigenstrapping
   :target: https://pypi.python.org/pypi/eigenstrapping/
   :alt: Latest PyPI version
   
.. image:: https://zenodo.org/badge/710175458.svg
  :target: https://zenodo.org/doi/10.5281/zenodo.10218973
  :alt: Zenodo DOI

.. image:: https://github.com/SNG-newy/eigenstrapping/actions/workflows/tests.yml/badge.svg
  :target: https://github.com/SNG-newy/eigenstrapping/actions/workflows/tests.yml
  :alt: run-tests status
   
.. image:: https://github.com/SNG-newy/eigenstrapping/actions/workflows/docs.yml/badge.svg
  :target: https://eigenstrapping.readthedocs.io/en/latest/
  :alt: deploy-docs status

|

The ``eigenstrapping`` toolbox is designed to help researchers generate
statistically-rigorous models for null hypothesis testing between brain maps using
non-local spectral shape descriptors - or geometric eigenmodes.
Documentation can be found `here <https://eigenstrapping.readthedocs.io/en/latest/>`_.

Features
--------

- A growing library of eigenmodes of standard surfaces and surface densities (fsaverage, fsLR)
- Cortical and subcortical null models for assessing statistical correspondence between brain maps
- Generation of geometric eigenmodes on user-derived surfaces

Installation Guide
------------------

Eigenstrapping is available in Python 3.7+. MATLAB version coming soon!

Installation
============

eigenstrapping can be installed using ``pip``: ::

    pip install eigenstrapping

Alternatively, you can install the package from the Github repository: ::

    git clone https://github.com/SNG-newy/eigenstrapping.git
    cd eigenstrapping
    python3 -m pip install .

Dependencies
============

To run eigenstrapping, the following Python packages are required (these should be automatically installed with the above command(s)):

* `numpy <https://numpy.org/>`_
* `scipy <https://scipy.org/scipylib/index.html>`_
* `scikit-learn <https://scikit-learn.org/stable/>`_
* `lapy <https://github.com/DeepMI/lapy>`_
* `matplotlib <https://matplotlib.org/>`_
* `nibabel <https://nipy.org/nibabel/index.html>`_
* `nilearn <https://nilearn.github.io/>`_
* `pandas <https://pandas.pydata.org/>`_
* `brainspace <https://github.com/MICA-MNI/BrainSpace/tree/master/>`_
* netneurotools - note: the latest version must be installed directly from this `link here <https://github.com/nikitas-k/netneurotools_scipyfix>`_

``nibabel`` and ``nilearn`` are required for surfaces and volumes. ``matplotlib`` 
is only required for fitting plots in :mod:`eigenstrapping.fit` and some of the surface 
plotting functions. Future improvements will reduce the number of dependencies
needed.

Important additional dependencies
=================================

In order to avail the user of the volumetric pipeline, the following packages must be installed:

* `Connectome Workbench <https://www.humanconnectome.org/software/connectome-workbench>`_
* `FreeSurfer 6.0.0+ <https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall>`_

Citing
------

When using ``eigenstrapping``, please cite the following manuscript:

* null

And please also cite the papers for the method that we use to calculate eigenmodes
on the surface:

* Laplace-Beltrami spectra as 'Shape-DNA' of surfaces and solids. Reuter M, Wolter F-E, Peinecke N. Computer-Aided Design. 2006;38(4):342-366. `<http://dx.doi.org/10.1016/j.cad.2005.10.011>`_

* BrainPrint: a discriminative characterization of brain morphology. Wachinger C, Golland P, Kremen W, Fischl B, Reuter M. Neuroimage. 2015;109:232-48. `<http://dx.doi.org/10.1016/j.neuroimage.2015.01.032>`_ `<http://www.ncbi.nlm.nih.gov/pubmed/25613439>`_

License information
-------------------

This work is licensed under a
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License ``cc-by-nc-sa``.
The full license can be found in the
`LICENSE <https://github.com/SNG-newy/eigenstrapping/blob/main/
/LICENSE>`_ file in the ``eigenstrapping`` distribution.
