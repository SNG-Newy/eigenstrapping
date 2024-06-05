.. _api:

.. currentmodule:: eigenstrapping

Reference API
=============

.. contents:: **List of modules**
   :local:
   
.. _ref_eigenstrapping::

:mod:`eigenstrapping` - Null models
-----------------------------------
.. automodule:: eigenstrapping
   :no-members:
   :no-inherited-members:

.. currentmodule:: eigenstrapping

.. autosummary::
   :template: class.rst
   :toctree: generated/

   eigenstrapping.SurfaceEigenstrapping
   eigenstrapping.VolumetricEigenstrapping
   
.. _ref_dataio::
   
:mod:`eigenstrapping.dataio` - Data helpers
-------------------------------------------
.. automodule:: eigenstrapping.dataio
   :no-members:
   :no-inherited-members:
   
.. currentmodule:: eigenstrapping.dataio

.. autosummary::
   :template: function.rst
   :toctree: generated/

   eigenstrapping.dataio.dataio
   eigenstrapping.dataio.load
   eigenstrapping.dataio.export_cifti_mapping
   eigenstrapping.dataio.load_surface
   
.. _ref_datasets::

:mod:`eigenstrapping.datasets` - Dataset fetchers
-------------------------------------------------
.. automodule:: eigenstrapping.datasets
   :no-members:
   :no-inherited-members:
   
.. currentmodule:: eigenstrapping.datasets

.. autosummary::
   :template: function.rst
   :toctree: generated/

   eigenstrapping.datasets.load_surface_examples
   eigenstrapping.datasets.load_genepc
   eigenstrapping.datasets.load_distmat
   eigenstrapping.datasets.load_subcort
   eigenstrapping.datasets.load_native
   
   eigenstrapping.datasets.fetch_data
   
   eigenstrapping.datasets.txt2memmap
   eigenstrapping.datasets.load_memmap
   
.. _ref_fit::
   
:mod:`eigenstrapping.fit` - Variogram fits
------------------------------------------
.. automodule:: eigenstrapping.fit
   :no-members:
   :no-inherited-members:
   
.. currentmodule:: eigenstrapping.fit

.. autosummary::
   :template: function.rst
   :toctree: generated/

   eigenstrapping.fit.surface_fit
   eigenstrapping.fit.volumetric_fit

.. _ref_geometry::

:mod:`eigenstrapping.geometry` - Mesh and eigenmode functions
-------------------------------------------------------------
.. automodule:: eigenstrapping.geometry
   :no-members:
   :no-inherited-members:
   
.. currentmodule:: eigenstrapping.geometry

.. autosummary::
   :template: function.rst
   :toctree: generated/

   eigenstrapping.geometry.get_tkrvox2ras
   eigenstrapping.geometry.make_tetra_file
   eigenstrapping.geometry.make_tria_file
   eigenstrapping.geometry.create_temp_surface
   eigenstrapping.geometry.calc_eig
   eigenstrapping.geometry.calc_surface_eigenmodes
   eigenstrapping.geometry.load_mesh
   eigenstrapping.geometry.compute_normals
   eigenstrapping.geometry.inflate_mesh
   eigenstrapping.geometry.nearest_neighbor
   eigenstrapping.geometry.calc_volume
   eigenstrapping.geometry.normalize_vtk
   eigenstrapping.geometry.read_annot
   eigenstrapping.geometry.read_geometry
   eigenstrapping.geometry.read_label
   eigenstrapping.geometry.write_label
   eigenstrapping.geometry.read_morph_data
   eigenstrapping.geometry.geodesic_distmat
   eigenstrapping.geometry.euclidean_distmat
   
.. _ref_plotting::

:mod:`eigenstrapping.plotting` - Plotting functions
---------------------------------------------------
.. automodule:: eigenstrapping.plotting
   :no-members:
   :no-inherited-members:
   
.. currentmodule:: eigenstrapping.plotting

.. autosummary::
   :template: function.rst
   :toctree: generated/

   eigenstrapping.plotting.vrange
   
   eigenstrapping.plotting.csplot
   eigenstrapping.plotting.flatplot
   eigenstrapping.plotting.meshplot

.. _ref_rotations::

:mod:`eigenstrapping.rotations` - Random rotation functions
-----------------------------------------------------------
.. automodule:: eigenstrapping.rotations
   :no-members:
   :no-inherited-members:
   
.. currentmodule:: eigenstrapping.rotations

.. autosummary::
   :template: function.rst
   :toctree: generated/

   eigenstrapping.rotations.direct_method
   eigenstrapping.rotations.indirect_method
   eigenstrapping.rotations.rotate_matrix
   
.. _ref_stats::

:mod:`eigenstrapping.stats` - Statistical functions
---------------------------------------------------
.. automodule:: eigenstrapping.stats
   :no-members:
   :no-inherited-members:
   
.. currentmodule:: eigenstrapping.stats

.. autosummary::
   :template: function.rst
   :toctree: generated/

   eigenstrapping.stats.compare_maps
   
   eigenstrapping.stats.gpd_inference
   eigenstrapping.stats.msle
   eigenstrapping.stats.joint_differential_entropy
   eigenstrapping.stats.conditional_entropy
   eigenstrapping.stats.kl
   eigenstrapping.stats.distance_correlation
   eigenstrapping.stats.ks
   eigenstrapping.stats.spearmanr
   eigenstrapping.stats.pearsonr
   eigenstrapping.stats.pairwise_r
   eigenstrapping.stats.nonparp
   
   eigenstrapping.stats.normalize
   
.. _ref_utils::

:mod:`eigenstrapping.utils` - Utility functions
-----------------------------------------------
.. automodule:: eigenstrapping.utils
   :no-members:
   :no-inherited-members:
   
.. currentmodule:: eigenstrapping.utils

.. autosummary::
   :template: function.rst
   :toctree: generated/

   eigenstrapping.utils.is_string_like
   
   eigenstrapping.utils.parcify
   eigenstrapping.utils.get_schaefer
   eigenstrapping.utils.calc_parcellate
   eigenstrapping.utils.parcellate_distmat
   
   eigenstrapping.utils.reconstruct_data
   eigenstrapping.utils.eigen_decomposition
   eigenstrapping.utils.calc_surface_connectivity
   eigenstrapping.utils.compute_axes_ellipsoid
   eigenstrapping.utils.transform_to_spheroid
   eigenstrapping.utils.transform_to_ellipsoid