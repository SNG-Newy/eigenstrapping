.. _references:
    
References
----------

When using ``eigenstrapping``, please cite the following manuscript:

* Generation of surrogate brain maps preserving spatial autocorrelation through random rotation of geometric eigenmodes. Koussis, N.C., Pang, J.C., Jeganathan, J., Paton, B., Fornito, A., Robinson, P.A., Misic, B., Breakspear, M. (2024). bioRxiv 2024.02.07.579070 [Preprint]. `<https://dx.doi.org/10.1101/2024.02.07.579070>`_

The core functions expand on code from `BrainEigenmodes <https://github.com/NSBLab/BrainEigenmodes>`_, please cite their work if you use our code:

* Geometric constraints on human brain function. Pang, J. C., Aquino, K. M., Oldehinkel, M., Robinson, P. A., Fulcher, B. D., Breakspear, M., & Fornito, A. (2023). Nature, 1-9. `<https://doi.org/10.1038/s41586-023-06098-1>`_

Please also cite the papers for the method that we use to calculate eigenmodes on the surface:

* Laplace-Beltrami spectra as 'Shape-DNA' of surfaces and solids. Reuter M, Wolter F-E, Peinecke N. Computer-Aided Design. 2006;38(4):342-366. `<http://dx.doi.org/10.1016/j.cad.2005.10.011>`_

* BrainPrint: a discriminative characterization of brain morphology. Wachinger C, Golland P, Kremen W, Fischl B, Reuter M. Neuroimage. 2015;109:232-48. `<http://dx.doi.org/10.1016/j.neuroimage.2015.01.032>`_ `<http://www.ncbi.nlm.nih.gov/pubmed/25613439>`_

And if you use the Heat Kernel for Geodesics method in :func:`eigenstrapping.geometry.geodesic_distmat`:

* Crane, K., Weischedel, C., & Wardetzky, M. (2013). Geodesics in heat: A new approach to computing distance based on heat flow. ACM Transactions on Graphics (TOG), 32(5), 1-11. `<https://arxiv.org/pdf/1204.6216>`_