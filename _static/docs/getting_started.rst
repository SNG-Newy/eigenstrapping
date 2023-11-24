.. _gettingstarted:

Getting Started
===============


Quick introduction to brain maps and eigenmodes
-----------------------------------------------

Patterns on the cortical surface or in the subcortex ("brain maps") more often
than not express with smoothness - spatial autocorrelation. Accounting
for this in statistical testing of the associations between or the influence
that one brain map has on another is a non-trivial problem. More recently,
methods that account for smoothness have been developed, but these carry their own
issues (see our `paper <https://null.null>`_ for an in-depth discussion on this topic).

Eigenmodes of a surface encode all pairwise (auto)correlations (i.e., smoothness).
The reason for this is too maths heavy for a quick intro, but suffice to say
that this has been discussed many times by people smarter than the people who came up with
this method (see :ref:`references`).

Eigenstrapping offers a solution by leveraging this fact, and another property of
eigenmodes: they are orthogonal. By taking random rotations of them, one can
create new brain maps with the same smoothness but randomized topology. The steps
below will help you get started and build your own surrogate maps.

First run
---------

The package comes with example surfaces (and their eigenmodes) and cortical gradient data 
from Margulies' 2015 paper (see here: :ref:`references`). Let's start by importing the data (this may take
a few moments):

.. code-tab:: py

    >>> from eigenstrapping.datasets import load_fslr()
    
    >>> # load left and right hemispheres
    >>> data_lh, data_rh, emodes_lh, emodes_rh, evals_lh, evals_rh = load_fsaverage()
    >>> surf_lh.shape
    (10242,)
    
    >>> emodes_lh.shape
    (10242, 1000)
    
    >>> evals_lh.shape
    (1000,)
    
    

Now let's make a surrogate brain map and plot it on the surface.

.. code-tab:: py


    >>> from eigenstrapping.plotting import csplot
    >>> from eigenstrapping import SurfaceEigenstrapping
    
    
    >>> eigen = SurfaceEigenstrapping(
                    data=data_lh,
                    emodes=emodes_lh,
                    evals=evals_lh,
                    num_modes=6000,
                    resample=True,
                    )      
    >>> surr = eigen.generate()
    
    >>> csplot(surr)
    
.. image:: ./python_doc/examples/example_figs/gettingstarted1.png
   :scale: 70%
   :align: center
   
   
   
Let's calculate the connectivity matrices of the original data and the surrogate:

.. code-tab:: py
    
    
    >>> from eigenstrapping.utils import calc_surface_fc
    
    >>> fc_matrix_orig = calc_surface_fc(data_lh, atlas='schaefer', scale=400)
    >>> fc_matrix_surr = calc_surface_fc(surr, atlas='schaefer', scale=400)
    >>> fc_matrix_orig.shape, fc_matrix_surr.shape
    >>> (400, 400) (400, 400)
    


Now let's plot the matrices and compare them:

.. code-tab:: py


    >>> import matplotlib.pyplot as plt
    
    >>> plt.imshow(fc_matrix_orig, cmap='bwr')
    >>> plt.imshow(fc_matrix_surr, cmap='bwr')
    

.. image:: ./python_doc/examples/example_figs/gettingstarted2.png
   :scale: 70%
   :align: center
   
.. image:: ./python_doc/examples/example_figs/gettingstarted3.png
   :scale: 70%
   :align: center
   
As we can see...



Next steps
----------

* Creating a cortical surrogate distribution :ref:`tutorialcortex`

* Working with subcortical data :ref:`tutorialsubcortex`

* Deriving eigenmodes on your own surface :ref:`tutorialownsurface`





