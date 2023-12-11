.. _usage_geometry:

Meshes and mesh operations
==========================

.. _usage_geometry_remesh:

Remeshing
---------

Sometimes it is useful to remesh a triangular surface as a tetrahedral mesh and
vice-versa - say, if one wants to derive 3D surfaces within subcortex. This routine
is implemented within the :func:`eigenstrapping.geometry.remesh` function, which
automatically recognizes the surface type (triangles or tetrahedra) and maps the
surface to the other type.

For example, you have a hippocampal volume in MNI152 space (we can load this
from the :func:`eigenstrapping.datasets.fetch_data` function), you can remesh
it from a volume to a triangular mesh. By default these will be generated in the
same folder that the original volume/mesh is in.

.. code-block:: py

    >>> from eigenstrapping.datasets import fetch_data
    
    >>> hipp_lh = fetch_data(
                    name='brainmaps',
                    hemi='lh',
                    format='hippocampus'
                    )
    >>> hipp_lh
    '/mnt/eigenstrapping_data/brainmaps/space-MNI152_res-2mm_hemi-lh_hippocampus.nii.gz'
    
    >>> from eigenstrapping.geometry import remesh
    
    >>> hipp_tria_lh = remesh(hipp_lh)
    >>> hipp_tria_lh
    '/mnt/eigenstrapping_data/brainmaps/space-MNI152_res-2mm_hemi-lh_hippocampus.tria.vtk'
    
You can also remesh from a triangular mesh to a tetrahedral one:

.. code-block:: py

    >>> hipp_tetra_lh = remesh(hipp_tria_lh)
    >>> hipp_tetra_lh
    '/mnt/eigenstrapping_data/brainmaps/space-MNI152_res-2mm_hemi-lh_hippocampus.tria.tetra.vtk'

.. _usage_geometry_eigenmodes:

Deriving eigenmodes from a mesh
-------------------------------

Deriving LBO vector solutions to the Helmholtz equation, or eigenmodes as we refer to them, 
is a fairly trivial process once a mesh has been defined. As graph representations of meshes
are sparse by their very nature (on a triangular mesh, a single vertex
only ever has three edges at the most; four in a tetrahedral mesh), we can use 
sparse methods of deriving these modes.

We use the finite element method as implemented in ``ShapeDNA``, the details
of which can be found in the requisite papers in :ref:`References <references>`.
The form matrices, A and B, are derived either through Cholesky decomposition
using the ``scikit-sparse`` libraries, or LU decomposition in ``scipy.sparse.splu``
if the former is not installed. We recommend, if possible, to install the ``scikit-sparse``
libraries as Cholesky decomposition is much faster than LU.

Let's take our remeshed tetrahedral hippocampus and use :func:`eigenstrapping.geometry.calc_eig`:

.. code-block:: py

    >>> from eigenstrapping import geometry
    >>> tetra = geometry.load_mesh(hipp_tetra_lh)
    >>> # in this case, we'll use eigenstrapping.geometry.calc_eig without sksparse
    >>> emodes, evals = geometry.calc_eig(tetra, num_modes=20) 
    TetMesh with regular Laplace
    Solver: spsolve (LU decomposition) ...
    
    >>> # by default, this function will remove the first (constant) mode
    >>> # this behavior can be changed by setting `return_zero=True` in
    >>> # the calc_eig function
    
    >>> emodes
    array([[ 0.01210057,  0.01249517,  0.00465954, ..., -0.00146934,
            -0.00544099,  0.0271714 ],
           [ 0.01209385,  0.01253431,  0.00548657, ..., -0.00434336,
            -0.00504929,  0.0242422 ],
           [ 0.01198019,  0.01202892,  0.00418164, ..., -0.00095343,
            -0.0038022 ,  0.02745249],
           ...,
           [-0.00412068, -0.01866016,  0.00250916, ...,  0.00476944,
            -0.00805506, -0.00263126],
           [ 0.00052041, -0.01061745, -0.00131715, ...,  0.00542975,
            -0.00041589,  0.01133534],
           [-0.01403812, -0.0024371 ,  0.00991763, ...,  0.02579329,
            -0.02810065,  0.00869789]], dtype=float32)
    
    >>> evals
    array([0.00365112, 0.01341067, 0.02674828, 0.03218102, 0.04489195,
           0.05093223, 0.06653866, 0.07089102, 0.09374754, 0.09695608,
           0.10020526, 0.11701339, 0.12567881, 0.13450024, 0.14623912,
           0.15450524, 0.15965985, 0.16787478, 0.17781733, 0.18272817],
          dtype=float32)
          
    >>> # as you can see, the first column of `emodes` is not constant.

Now, let's plot the first non-constant eigenmode on the surface of the mesh.

.. code-block:: py

    >>> from eigenstrapping import plotting
    
    >>> plotting.meshplot(hipp_tetra_lh, emodes[0], vrange=0.01, colorbar=True)
    
.. image:: ./_static/examples/example_figs/geometry1.png
   :scale: 70%
   :align: center

.. _usage_geometry_transform:

Transformations and co-registration of meshes
---------------------------------------------

Beyond operations on a single mesh, ``eigenstrapping`` has implemented transformations
and resampling of meshes to standard spaces (fsLR, fsaverage, MNI152).
Additionally, you can resample one mesh in one space to another mesh-space,
or co-register two meshes to the same space.

We use Connectome Workbench tools to perform this task, wrapped in the
:mod:`eigenstrapping.geometry` module.

.. code-block:: py

    >>> 
    
.. _usage_geometry_distance:

Mesh distance calculations
--------------------------

We also provide geodesic (surface) and Euclidean (volumetric) distance calculations
for meshes
    

Other mesh operations
---------------------

Alongside the above functions, there are several useful tools within
the :mod:`eigenstrapping.geometry` module, including inflation of
surfaces (using Connectome Workbench tools):

.. code-block:: py

    >>>
    
    
There is also 