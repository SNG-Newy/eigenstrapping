.. _installation:

Installation Guide
==================

Eigenstrapping is available in Python 3.7+. MATLAB version coming soon!

.. _installation_requirements:

Dependencies
------------

To install eigenstrapping, the following Python packages are required:

* `numpy <https://numpy.org/>`_
* `scipy <https://scipy.org/scipylib/index.html>`_
* `scikit-learn <https://scikit-learn.org/stable/>`_
* `lapy <https://github.com/DeepMI/lapy>`_
* `matplotlib <https://matplotlib.org/>`_
* `nibabel <https://nipy.org/nibabel/index.html>`_
* `nilearn <https://nilearn.github.io/>`_
* `ANTsPy <https://github.com/ANTsX/ANTsPy>`_
* `pandas <https://pandas.pydata.org/>`_
* `netneurotools <https://netneurotools.readthedocs.io/en/latest/installation.html/>`_

``nibabel`` and ``nilearn`` are required for surfaces and volumes. ``matplotlib`` 
is only required for fitting plots in ``fit.py`` and some of the surface 
plotting functions.

Additional dependencies
-----------------------

* `Connectome Workbench <https://www.humanconnectome.org/software/connectome-workbench>`_
* `FreeSurfer 6.0.0+ <https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall>`_

Installation
^^^^^^^^^^^^

eigenstrapping can be installed using ``pip``: ::

    pip install eigenstrapping

Alternatively, you can install the package from the Github repository: ::

    git clone https://github.com/SNG-newy/eigenstrapping.git
    cd eigenstrapping
    python setup.py install

Once that's done, get started with a quick example dataset 
(:ref:`gettingstarted`).
