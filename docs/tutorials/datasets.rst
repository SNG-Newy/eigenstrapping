.. _usage_datasets:

Eigenmodes and eigenvalues
==========================

``eigenstrapping`` offers a selection of standard surfaces and their
eigenmodes and eigenvalues for use with the core functions. This way,
the and its base classes :class:`eigenstrapping.SurfaceEigenstrapping`
and :class:`eigenstrapping.VolumetricEigenstrapping` has several uses and inputs 
for working with surface and volume data. This guide will address the general
use of these classes, as well as information on the types of surfaces and volumes
that can be imported.

.. _usage_formats_supported:

Surface file formats
--------------------

The class :class:`eigenstrapping.SurfaceEigenstrapping` supports data in the 
following file formats:

  1. Delimited `*.txt` files
  2. Neuroimaging files such as `*.surf.gii`, or `*.shape.gii`
  3. Numpy arrays and array-like objects
  4. ``nibabel.GiftiImage`` classes
  
At minimum, :class:`eigenstrapping.SurfaceEigenstrapping` expects two things:

  1. A 1D surface vector of ``data`` (a brain map) with a length `N`. This can be a 
  vector in array format (e.g., a numpy array of shape `(N,)`), and other array-like
  objects, Connectome Workbench GIFTI format files, i.e., a `*.func.gii` 
  (as long as the brain map is the first index) or a `.shape.gii` file, FreeSurfer 
  triangle files (e.g., ?h.curv, etc.), or a `*.txt` file.
  
  2. A ``surface`` mesh and/or a set of ``emodes`` eigenmodes and their 
  corresponding ``evals`` eigenvalues.
  
It is important that the surface mesh or the eigenmodes+eigenvalues (if one is given, the other
need not be given) have the same number of vertices as the data array, i.e., they
must have the same length `(N,)`. The eigenmodes array should have the shape
`(N, M)` when loaded by ``numpy``, though the method should recognize the modes
aren't in the right order and reshape the array accordingly.

If given as a *.gii file, the surface mesh must be composed of two arrays: a 
vertex coordinate array, and a triangular face array (i.e., 3 vertices per face 
index). There should be no duplicate faces. FreeSurfer format files (e.g., ?h.pial),
as long as they're produced by the standard ``recon-all`` pipeline, are read by
``nilearn.interfaces.freesurfer`` subroutines and shouldn't have any issues.

CIFTI surface formats (such as `*.nii` or `*scalar.nii`) are currently not accepted,
but we are currently working on implementing them.

Volumetric file formats
-----------------------

Unlike the class :class:`eigenstrapping.SurfaceEigenstrapping`, 
:class:`eigenstrapping.VolumetricEigenstrapping` supports slightly different data in the 
following formats:

  1. Delimited `*.txt` files (as vectors)
  2. Neuroimaging `*.nii` volume files
  3. Numpy arrays and array-like objects
  4. ``nibabel.Nifti1Image`` classes
  
At minimum, :class:`eigenstrapping.VolumetricEigenstrapping` expects a volumetric
brain map (in 3D, 4D inputs such as for fMRI data will be implemented in the future)
and an ROI mask. It is important that the ROI be contiguous (no separate structures),
otherwise the ``mri_mc`` routine will only generate the surface for one of the
structures. Unfortunately, there is no way around this - so do check if your
ROI is one whole structure.

Currently, the volumetric method is only recommended for subcortical volumes, as it
generates a tetrahedral mesh by default (and calculates eigenmodes on it). A source
of future research is trying to understand whether the cortex and the subcortex
can or should be combined into a composite tetrahedral mesh to calculate eigenmodes
on.

General usage
-------------

Both :class:`eigenstrapping.SurfaceEigenstrapping` and :class:`eigenstrapping.VolumetricEigenstrapping`
can be called in similar ways, but accept slightly different inputs:

.. code-block:: py
    
    >>> from eigenstrapping import SurfaceEigenstrapping
    
    >>> surf_eigen = SurfaceEigenstrapping(data='path/to/your/data/file',
                                           surface='path/to/your/surface.surf.gii')
    Computing eigenmodes on surface using N=200 modes
    TriaMesh with regular Laplace-Beltrami
    Solver: spsolve (LU decomposition) ...
    IMPORTANT: EIGENMODES MUST BE TRUNCATED AT FIRST NON-ZERO MODE FOR THIS FUNCTION TO WORK
    
By default, the function will calculate 200 eigenmodes on the input surface, and
remove the first mode (the constant mode). If you do not wish this to happen, then
pass ``remove_zero=False`` to :class:`eigenstrapping.SurfaceEigenstrapping`.

The eigenmodes and eigenvalues can then be accessed by (the following is just an example
on the 32k mid-thickness mesh, your actual values will differ unless you use the same
surface):

.. code-block:: py

    >>> surf_eigen.emodes
    array([[-0.00104721, -0.00644147,  0.00180854, ..., -0.00173471,
             0.00086994, -0.00478536],
           [ 0.0002369 , -0.00577345, -0.0035953 , ...,  0.00077048,
             0.00826997,  0.00210746],
           [ 0.00310662, -0.00214299, -0.00517964, ..., -0.00379839,
            -0.00049509, -0.00039869],
           ...,
           [-0.0043459 ,  0.00412747, -0.0012873 , ..., -0.00181644,
             0.00402434, -0.00313128],
           [-0.00429347,  0.00412993, -0.00143325, ..., -0.00169964,
             0.00355123, -0.00360749],
           [-0.00434974,  0.00403133, -0.00125077, ..., -0.00123351,
             0.00468343, -0.00315979]])
    
    >>> surf_eigen.emodes.shape
    (32492, 200)
    
    >>> surf_eigen.evals
    array([0.00015674, 0.00034542, 0.00049198, ..., 0.03862564, 0.03873034, 0.03883246]
    
    >>> surf_eigen.evals.shape
    (200,)

You can specify how many modes you want to calculate for the mesh by setting
``num_modes=<num_modes>``. For example, if I want to solve 5000 modes on the cortical
mesh:

.. code-block:: py

    >>> surf_eigen = SurfaceEigenstrapping(data='path/to/your/data/file',
                                           surface='path/to/your/surface.surf.gii',
                                           num_modes=5000)
    Computing eigenmodes on surface using N=5000 modes
    TriaMesh with regular Laplace-Beltrami
    Solver: spsolve (LU decomposition) ...
    IMPORTANT: EIGENMODES MUST BE TRUNCATED AT FIRST NON-ZERO MODE FOR THIS FUNCTION TO WORK
    
You may have noticed the line "IMPORTANT: EIGENMODES MUST BE TRUNCATED AT 
FIRST NON-ZERO MODE FOR THIS FUNCTION TO WORK". This is non-optional, though
the routines will let you specify ``remove_zero=False`` as before, if only the eigenmodes
of the surface are needed. The behavior of the nulls if ``remove_zero=False`` is passed,
or if the pre-calculated eigenmodes have the first (constant) mode, then the group
indexing WILL NOT work. 

*To reiterate: do not pass the eigenmode array with the constant mode to the null generator.*

To generate surface map nulls (100, for example):

.. code-block:: py

    >>> nulls = surf_eigen(n=10)
    >>> nulls
    array([[-0.74663688,  1.23662583, -1.39049185, ...,  0.14000489,
             0.08279563,  0.16406706],
           [ 0.48178048, -0.07707572, -0.67461382, ..., -0.21409802,
            -0.09778982, -0.25223958],
           [ 0.58768244, -0.93810351,  0.21764879, ..., -1.81227513,
            -1.86579597, -1.7101046 ],
           ...,
           [ 0.50142442,  0.00672288,  1.70244843, ..., -1.53966179,
            -1.59164348, -1.49195656],
           [ 0.04231571, -1.84780828, -0.54451441, ...,  0.34094984,
             0.20542123,  0.29713696],
           [ 3.3444511 ,  0.22457945, -1.41373573, ...,  1.20156473,
             1.1260775 ,  1.09572322]])
    
    >>> nulls.shape
    (100, 32492)
    
These can then be used for comparison to another brain map through the
:mod:`eigenstrapping.stats` module.

The use of the :class:`eigenstrapping.VolumetricEigenstrapping` class is
very similar, with a slight difference:

.. code-block:: py
    
    >>> vol_eigen = VolumetricEigenstrapping(data='/path/to/your/data/volume.nii',
                                             volume='/path/to/your/ROI/or/Mask.nii')

If ``gmsh``, FreeSurfer, and Connectome Workbench are installed and sourced
correctly, then you should get something like the following output:

.. code-block::

    preprocessing...done
    starting generation of surface...
        slice nb 30...
        slice nb 40...
    constructing final surface...
    (surface with 2056 faces and 1030
    vertices)...done
    computing the maximum edge length...2.828427 mm
    reversing orientation of faces...
    checking orientation of surface...
    0.000 % of the vertices (0 vertices) exhibit an orientation change
    
    counting number of connected components...
        1030 voxel in cpt #1: X=2
    [v=1030,e=3084,f=2056] located at (-23.000000, -31.722330, 7.277670)
    For the whole surface: X=2 [v=1030,e=3084,f=2056]
    One single component has been found
    nothing to do
    writing out surface...done
    --> VTK format         ...
     --> DONE ( V: 1030 , T: 2056 )
    
    --> VTK format         ...
     --> DONE ( V: 1449 , T: 5168 )
     
    TetMesh with regular Laplace
    Solver: spsolve (LU decomposition) ...
    
The above is simply a readout of the ``mri_mc`` command and the ``lapy`` reading
the `*.vtk` surface that is made from the output of ``mri_mc``. If you wish to
suppress these messages, pass ``verbose=False`` to :class:`eigenstrapping.VolumetricEigenstrapping`.
Be aware that any errors may be suppressed as well.