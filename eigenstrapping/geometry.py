#from nipype.interfaces.freesurfer import MRIMarchingCubes
from lapy import TriaMesh, TriaIO, Solver
from lapy.DiffGeo import tria_compute_gradient, tria_compute_divergence
import warnings
from collections import OrderedDict
import scipy.optimize as optimize
import subprocess
import os
import nibabel as nib
#from nilearn import image
import numpy as np
#from neuromaps.datasets.atlases import fetch_mni152
#from ants import image_read, registration, apply_transforms
from scipy.spatial import Delaunay, KDTree
from brainspace import mesh
from joblib import Parallel, delayed
from scipy.sparse.linalg import splu

try:
    from sksparse.cholmod import cholesky
except:
    print("Scikit-sparse libraries not found, using LU decomposition for eigenmodes (slower)")


"""
Helper utilities for geometry and registration

    - Runs mri_mc from Freesurfer to create 2d surface
    - Projects 2d surface using gmsh
    - Writes out geometry in tetrahedral format, triangular format, or in Freesurfer binary
    - Writes out label files
    - Registers from native space to MNI152 and fsaverage space

read_geometry() was taken from nibabel.freesurfer package (https://github.com/nipy/nibabel/blob/master/nibabel/freesurfer/io.py).
get_tkrvox2ras(), make_tetra_file(), normalize_vtk(), and calc_volume() taken
from <https://github.com/BMHLab/BrainEigenmodes>, authors J. Pang and K. Aquino.

"""

_ANNOT_DT = ">i4"
"""Data type for Freesurfer `.annot` files.

Used by :func:`read_annot` and :func:`write_annot`.  All data (apart from
strings) in an `.annot` file is stored as big-endian int32.
"""

# mni152_2mm = np.asarray(
#                  [[  2.,    0.,   0.,    -90,],
#                   [ -0.,    2.,   0.,   -126,],
#                   [ -0.,    0.,   2.,    -72,],
#                   [  0.,    0.,   0.,      1.]])

# mni152_1mm = np.asarray(
#                 [[  -1.,    0.,    0.,   90.],
#                  [   0.,    1.,    0., -126.],
#                  [   0.,    0.,    1.,  -72.],
#                  [   0.,    0.,    0.,    1.]])

# def _check_mni(in_file):
#     """
#     Checks if input image is in MNI152 space
#     """
    
#     img = image.load_img(in_file)
    
#     if img.affine != mni152_2mm:
#         if img.affine != mni152_1mm:
#             return False
        
#     else:
#         return True
    
# def native_to_mni152(in_file, nonlinear=True):
#     """
#     Linear or nonlinear registration of native volumetric image to MNI152 space
#     Uses ANTsPy
#     """
    
#     img = image_read(in_file)
    
#     # get template image
#     mni_file = fetch_mni152(density='1mm').get('2009cAsym_T1w')
#     mni = image_read(mni_file)

#     if nonlinear is True:
#         transform_type='SyN'
        
#     else:
#         transform_type='Affine'
        
#     # do transform
#     fixed_image = mni
#     moving_image = img
    
#     mytx = registration(fixed=fixed_image, moving=moving_image, type_of_transform=transform_type)
    
#     warped_moving_image = apply_transforms(fixed=fixed_image, moving=moving_image,
#                                            transformlist=mytx['fwdtransforms'])
    
#     # rebuild as nib.Nifti1Image
#     transformed_image = warped_moving_image.to_nibabel()
    
#     return transformed_image

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(- ((x - mean) ** 2) / (2 * (stddev ** 2)))

def estimate_fwhm(image):
    # Find the maximum value and its position in the image
    max_val = np.max(image)
    max_pos = np.argmax(image)
    max_row, max_col = np.unravel_index(max_pos, image.shape)
    
    # Create a 1d profile along the maximum row and column
    row_profile = image[max_row, :]
    col_profile = image[:, max_col]
    
    options = {'maxfev' : np.prod(image.shape)}
    
    # Fit a gaussian curve to the row profile
    row_x = np.arange(len(row_profile))
    row_params, _ = optimize.curve_fit(gaussian, row_x, row_profile, p0=[max_val, max_col, 1], **options)
    
    # Fit a gaussian curve to the column profile
    col_x = np.arange(len(col_profile))
    col_params, _ = optimize.curve_fit(gaussian, col_x, col_profile, p0=[max_val, max_row, 1], **options)
    
    # Calculate the FWHM as 2.355 times the std dev of the Gaussian
    row_fwhm = 2.355 * row_params[2]
    col_fwhm = 2.355 * col_params[2]
    
    # take the average
    fwhm = (row_fwhm + col_fwhm) / 2
    
    return fwhm
    
def resel_count(image, fwhm):
    # returns resel count of image given FWHM
    return np.prod(image.shape / fwhm)
        
def get_tkrvox2ras(voldim, voxres):
    """Generate transformation matrix to switch between tetrahedral and volume space.

    Parameters
    ----------
    voldim : array (1x3)
        Dimension of the volume (number of voxels in each of the 3 dimensions)
    voxres : array (!x3)
        Voxel resolution (resolution in each of the 3 dimensions)

    Returns
    ------
    T : array (4x4)
        Transformation matrix
    """

    T = np.zeros([4,4]);
    T[3,3] = 1;

    T[0,0] = -voxres[0];
    T[0,3] = voxres[0]*voldim[0]/2;

    T[1,2] = voxres[2];
    T[1,3] = -voxres[2]*voldim[2]/2;


    T[2,1] = -voxres[1];
    T[2,3] = voxres[1]*voldim[1]/2;

    return T

def make_tetra_file(nifti_input_filename):
    """Generate tetrahedral version of the ROI in the nifti file.

    Parameters
    ----------
    nifti_input_filename : str
        Filename of input volume where the relevant ROI have voxel values = 1

    Returns
    ------
    tetra_file : str
        Filename of output tetrahedral vtk file
    """

    nifti_input_file_head, nifti_input_file_tail = os.path.split(nifti_input_filename)
    nifti_input_file_main, nifti_input_file_ext = os.path.splitext(nifti_input_file_tail)

    os.system('mri_mc ' + nifti_input_filename + ' 1 ' + nifti_input_file_head + '/rh.tmp_surface.vtk')
    os.system('mv -f ' + nifti_input_file_head + '/rh.tmp_surface.vtk ' + nifti_input_filename + '.vtk')

    geo_file = nifti_input_filename + '.geo'
    tria_file = nifti_input_filename + '.vtk'
    tetra_file = nifti_input_filename + '.tetra.vtk'

    file = tria_file.rsplit('/')
    inputGeo = file[len(file)-1]
    
    with open(geo_file, 'w') as writer:
        writer.write('Mesh.Algorithm3D=4;\n')
        writer.write('Mesh.Optimize=1;\n')
        writer.write('Mesh.OptimizeNetgen=1;\n')
        writer.write('Merge "'+inputGeo+'";\n')
        writer.write('Surface Loop(1) = {1};\n')
        writer.write('Volume(1) = {1};\n')
        writer.write('Physical Volume(1) = {1};\n')

    cmd = 'gmsh -3 -o ' + tetra_file + ' ' + geo_file
    output = subprocess.check_output(cmd,shell="True")
    output = output.splitlines()

    cmd = "sed 's/double/float/g;s/UNSTRUCTURED_GRID/POLYDATA/g;s/CELLS/POLYGONS/g;/CELL_TYPES/,$d' " + tetra_file + " > " + tetra_file + "'_fixed'"
    os.system(cmd)
    os.system('mv -f ' + tetra_file + '_fixed ' + tetra_file)
    
    return tetra_file

def make_tria_file(nifti_input_filename):
    """
    Generate triangular mesh of the ROI in the nifti file.

    Parameters
    ----------
    nifti_input_filename : str
        Filename of input volume where the relevant ROI have voxel values = 1

    Returns
    -------
    tria_file : str
        Filename of output triangular vtk file

    """
    nifti_input_file_head, nifti_input_file_tail = os.path.split(nifti_input_filename)
    nifti_input_file_main, nifti_input_file_ext = os.path.splitext(nifti_input_file_tail)

    os.system('mri_mc ' + nifti_input_filename + ' 1 ' + nifti_input_file_head + '/rh.tmp_surface.vtk')
    os.system('mv -f ' + nifti_input_file_head + '/rh.tmp_surface.vtk ' + nifti_input_filename + '.vtk')

    geo_file = nifti_input_filename + '.geo'
    tria_file = nifti_input_filename + '.vtk'

    file = tria_file.rsplit('/')
    inputGeo = file[len(file)-1]
    
    with open(geo_file, 'w') as writer:
        writer.write('Mesh.Algorithm3D=4;\n')
        writer.write('Mesh.Optimize=1;\n')
        writer.write('Mesh.OptimizeNetgen=1;\n')
        writer.write('Merge "'+inputGeo+'";\n')
        writer.write('Surface Loop(1) = {1};\n')
        writer.write('Volume(1) = {1};\n')
        writer.write('Physical Volume(1) = {1};\n')

    cmd = 'gmsh -2 -o ' + tria_file + ' ' + geo_file
    output = subprocess.check_output(cmd,shell="True")
    output = output.splitlines()

    cmd = "sed 's/double/float/g;s/UNSTRUCTURED_GRID/POLYDATA/g;s/CELLS/POLYGONS/g;/CELL_TYPES/,$d' " + tria_file + " > " + tria_file + "'_fixed'"
    os.system(cmd)
    os.system('mv -f ' + tria_file + '_fixed ' + tria_file)
    
    return tria_file

def create_temp_surface(surface_input, surface_output_filename):
    """Write surface to a new vtk file.

    Parameters
    ----------
    surface_input : brainspace compatible object
        Loaded vtk object corresponding to a surface triangular mesh
    surface_output_filename : str
        Filename of surface to be saved
    """

    f = open(surface_output_filename, 'w')
    f.write('# vtk DataFile Version 2.0\n')
    f.write(surface_output_filename + '\n')
    f.write('ASCII\n')
    f.write('DATASET POLYDATA\n')
    f.write('POINTS ' + str(np.shape(surface_input.Points)[0]) + ' float\n')
    for i in range(np.shape(surface_input.Points)[0]):
        f.write(' '.join(map(str, np.array(surface_input.Points[i, :]))))
        f.write('\n')
    f.write('\n')
    f.write('POLYGONS ' + str(np.shape(surface_input.polys2D)[0]) + ' ' + str(4* np.shape(surface_input.polys2D)[0]) + '\n')
    for i in range(np.shape(surface_input.polys2D)[0]):
        f.write(' '.join(map(str, np.append(3, np.array(surface_input.polys2D[i, :])))))
        f.write('\n')
    f.close()

def get_indices(surface_original, surface_new):
    """Extract indices of vertices of the two surfaces that match.

    Parameters
    ----------
    surface_original : brainspace compatible object
        Loaded vtk object corresponding to a surface triangular mesh
    surface_new : brainspace compatible object
        Loaded vtk object corresponding to a surface triangular mesh

    Returns
    ------
    indices : array
        indices of vertices
    """

    indices = np.zeros([np.shape(surface_new.Points)[0],1])
    for i in range(np.shape(surface_new.Points)[0]):
        indices[i] = np.where(np.all(np.equal(surface_new.Points[i,:],surface_original.Points), axis=1))[0][0]
    indices = indices.astype(int)
    
    return indices

def calc_eig(tria, num_modes, use_cholmod=False):
    """Calculate the eigenvalues and eigenmodes of a surface.

    Parameters
    ----------
    tria : lapy compatible object
        Loaded vtk object corresponding to a surface triangular mesh
    num_modes : int
        Number of eigenmodes to be calculated

    Returns
    ------
    evals : array (num_modes x 1)
        Eigenvalues
    emodes : array (number of surface points x num_modes)
        Eigenmodes
    """
    
    fem = Solver(tria, use_cholmod=use_cholmod)
    evals, emodes = fem.eigs(k=num_modes)
    
    return evals, emodes
    
def calc_surface_eigenmodes(surface_input_filename, mask_input, save_cut=False, num_modes=200, use_cholmod=False, remove_zero=True):
    """Main function to calculate the eigenmodes of a cortical surface with 
    application of a mask (e.g., to remove the medial wall).

    Parameters
    ----------
    surface_input_filename : str
        Filename of input surface with ``n_vertices``
    mask_input : str or 1D ndarray
        Filename of mask to be applied on the surface (e.g., cortex without 
        medial wall, values = 1 for mask and 0 elsewhere)
    save_cut : boolean 
        Boolean to decide if the new surface with mask applied will be saved to 
        a new surface file
    num_modes : int
        Number of eigenmodes to be calculated      
    use_cholmod : bool, optional
        Spec to use cholesky libraries of ``scikit-sparse``. Requires installation
        of ``scikit-sparse`` and ``libsuitesparse-dev``.
        
    Returns
    -------
    np.ndarray of shape=(num_modes,)
        Eigenvalues computed on masked surface.
    np.ndarray of shape=(n_vertices, num_modes)
        Eigenmodes corresponding to eigenvalues computed on masked surface
        padded with zeros to match size of original surface (``n_vertices``)
    """

    # load surface (as a brainspace object)
    surface_orig = mesh.mesh_io.read_surface(surface_input_filename)
    
    # load mask
    # can be any ROI (even whole cortex)
    if isinstance(mask_input, np.ndarray):
        mask = mask_input
    else:
        mask_input_file_main, mask_input_file_ext = os.path.splitext(mask_input)
        if mask_input_file_ext == '.txt':
            mask = np.loadtxt(mask_input)
        elif mask_input_file_ext == '.gii':
            mask = nib.load(mask_input).darrays[0].data
        else:
            mask = mask_input
    
    # create temporary suface based on mask
    surface_cut = mesh.mesh_operations.mask_points(surface_orig, mask)

    if save_cut is True:
        # old method: save vtk of surface_cut and open via lapy TriaIO 
        # The writing phase of this process is very slow especially for large surfaces
        temp_cut_filename='temp_cut.gii'
        temp_cut = nib.GiftiImage()
        temp_cut.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(surface_cut.GetPoints().astype('float32')))
        temp_cut.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(mesh.mesh_elements.get_cells(surface_cut).astype('int32')))
        nib.save(temp_cut, temp_cut_filename)
        # load surface (as a lapy object)
        vertices, faces = temp_cut.agg_data()
        tria = TriaMesh(vertices, faces)
    else:
        # new method: replace v and t of surface_orig with v and t of surface_cut
        # faster version without the need to write the vtk file
        # load surface (as a lapy object)
        tria = TriaIO.import_vtk(surface_input_filename)
        tria.v = surface_cut.Points
        tria.t = np.reshape(surface_cut.Polygons, [surface_cut.n_cells, 4])[:,1:4]

    # calculate eigenvalues and eigenmodes
    evals, emodes = calc_eig(tria, num_modes, use_cholmod)
    
    if remove_zero:
        # remove zeroth eigenmode
        evals = evals[1:]
        emodes = emodes[:, 1:]
    
    # get indices of vertices of surface_orig that match surface_cut
    indices = get_indices(surface_orig, surface_cut)
    
    # reshape emodes to match vertices of original surface
    emodes_reshaped = np.zeros([surface_orig.n_points,np.shape(emodes)[1]])
    for mode in range(np.shape(emodes)[1]):
        emodes_reshaped[indices,mode] = np.expand_dims(emodes[:,mode], axis=1)
        
    return evals, emodes_reshaped

def compute_normals(coords, faces):
    """
    Compute the normal for each vertex in the mesh.
    """

    normals = np.zeros(coords.shape, dtype=coords.dtype)
    for i in range(faces.shape[0]):
        fn = np.cross(coords[faces[i, 1]] - coords[faces[i, 0]], coords[faces[i, 2]] - coords[faces[i, 0]])
        normals[faces[i, 0]] += fn
        normals[faces[i, 1]] += fn
        normals[faces[i, 2]] += fn

    norms = np.sqrt((normals ** 2).sum(axis=1))
    normals /= norms[:, np.newaxis]

    return normals

def inflate_mesh(vertices, faces, iterations=10, inflation_rate=0.02):
    """
    Inflate the mesh using basic inflation algorithm.
    """
    new_vertices = vertices
    for _ in range(iterations):

        normals = compute_normals(vertices, faces)

        new_vertices += normals * inflation_rate

    return new_vertices   

def nearest_neighbor(P, X, radius=None):
    """
    Find the one-nearest neighbors of vertices in points `P` on another 
    surface `X` using Delaunay triangulation and KDTree query.

    Parameters
    ----------
    P : np.ndarray of shape (N,3)
        Points to search for within the coordinate set of `X`. `P` can
        be a single point
    X : np.ndarray of shape (M,3)
        Vertices of the surface to search within
    radius : float
        Radius to search for nearest neighbors within

    Returns
    -------
    nearest_indexes : int
        Indexes of one-nearest neighbors of vertices in `P`. Note that
        if two vertices in `X` are the same distance away from a point in `P`,
        function returns only the first one.

    """
    
    # Create Delaunay triangulation for first surface
    tri = Delaunay(X)
    
    # Create tree of vertices to query on
    kdtree = KDTree(X)

    indices = np.empty(P.shape[0], dtype=int)
    for i, p in enumerate(P):
        simplex_index = tri.find_simplex(p)
        if simplex_index == -1 or (radius is not None and not _is_point_within_radius(p, X[tri.simplices[simplex_index]], radius)):
            _, nearest_neighbor_index = kdtree.query(p)
        else:
            simplex_vertices = X[tri.simplices[simplex_index]]
            dist = np.linalg.norm(simplex_vertices - p, axis=1)
            if radius is not None:
                valid_indices = np.where(dist <= radius)[0]
                if valid_indices.size == 0:
                    _, nearest_neighbor_index = kdtree.query(p)
                else:
                    nearest_neighbor_index = tri.simplices[simplex_index][valid_indices[np.argmin(dist[valid_indices])]]
            else:
                nearest_neighbor_index = tri.simplices[simplex_index][np.argmin(dist)]
        indices[i] = nearest_neighbor_index

    return indices


def _is_point_within_radius(p, vertices, radius):
    """
    Check if a point is within a given radius of any vertex in a set of vertices.
    """
    return np.any(np.linalg.norm(vertices - p, axis=1) <= radius)


def calc_volume(nifti_input_filename):
    """Calculate the physical volume of the ROI in the nifti file.

    Parameters
    ----------
    nifti_input_filename : str
        Filename of input volume where the relevant ROI have voxel values = 1

    Returns
    ------
    ROI_number : int
        Total number of non-zero voxels
    ROI_volume : float
        Total volume of non-zero voxels in physical dimensions   
    """

    # Load data
    ROI_data = nib.load(nifti_input_filename)
    roi_data = ROI_data.get_fdata()

    # Get voxel dimensions in mm
    voxel_dims = (ROI_data.header["pixdim"])[1:4]
    voxel_vol = np.prod(voxel_dims)

    # Compute volume
    ROI_number = np.count_nonzero(roi_data)
    ROI_volume = ROI_number * voxel_vol

    # print("Number of non-zero voxels = {}".format(ROI_number))
    # print("Volume of non-zero voxels = {} mm^3".format(ROI_volume))

    return ROI_number, ROI_volume

def normalize_vtk(tet, nifti_input_filename, normalization_type='none', normalization_factor=1):
    """Normalize tetrahedral surface.

    Parameters
    ----------
    tet : lapy compatible object
        Loaded vtk object corresponding to a surface tetrahedral mesh
    nifti_input_filename : str
        Filename of input volume where the relevant ROI have voxel values = 1
    normalization_type : str (default: 'none')
        Type of normalization
        number - normalization with respect to the total number of non-zero voxels
        volume - normalization with respect to the total volume of non-zero voxels in physical dimensions   
        constant - normalization with respect to a chosen constant
        others - no normalization
    normalization_factor : float (default: 1)
        Factor to be used in a constant normalization     

    Returns
    ------
    tet_norm : lapy compatible object
        Loaded vtk object corresponding to the normalized surface tetrahedral mesh
    """

    nifti_input_file_head, nifti_input_file_tail = os.path.split(nifti_input_filename)
    nifti_input_file_main, nifti_input_file_ext = os.path.splitext(nifti_input_file_tail)

    ROI_number, ROI_volume = calc_volume(nifti_input_filename)

    # normalization process
    tet_norm = tet
    if normalization_type == 'number':
        tet_norm.v = tet.v/(ROI_number**(1/3))
    elif normalization_type == 'volume':
        tet_norm.v = tet.v/(ROI_volume**(1/3))
    elif normalization_type == 'constant':
        tet_norm.v = tet.v/(normalization_factor**(1/3))
    else:
        pass

    # writing normalized surface to a vtk file
    if normalization_type == 'number' or normalization_type == 'volume' or normalization_type == 'constant':
        surface_output_filename = nifti_input_filename + '_norm=' + normalization_type + '.tetra.vtk'

        f = open(surface_output_filename, 'w')
        f.write('# vtk DataFile Version 2.0\n')
        f.write(nifti_input_file_tail + '\n')
        f.write('ASCII\n')
        f.write('DATASET POLYDATA\n')
        f.write('POINTS ' + str(np.shape(tet.v)[0]) + ' float\n')
        for i in range(np.shape(tet.v)[0]):
            f.write(' '.join(map(str, tet_norm.v[i, :])))
            f.write('\n')
        f.write('\n')
        f.write('POLYGONS ' + str(np.shape(tet.t)[0]) + ' ' + str(5 * np.shape(tet.t)[0]) + '\n')
        for i in range(np.shape(tet.t)[0]):
            f.write(' '.join(map(str, np.append(4, tet.t[i, :]))))
            f.write('\n')
        f.close()

    return tet_norm

def read_annot(filepath, orig_ids=False):
    """Read in a Freesurfer annotation from a ``.annot`` file.

    An ``.annot`` file contains a sequence of vertices with a label (also known
    as an "annotation value") associated with each vertex, and then a sequence
    of colors corresponding to each label.

    Annotation file format versions 1 and 2 are supported, corresponding to
    the "old-style" and "new-style" color table layout.

    Note that the output color table ``ctab`` is in RGBT form, where T
    (transparency) is 255 - alpha.

    See:
     * https://surfer.nmr.mgh.harvard.edu/fswiki/LabelsClutsAnnotationFiles#Annotation
     * https://github.com/freesurfer/freesurfer/blob/dev/matlab/read_annotation.m
     * https://github.com/freesurfer/freesurfer/blob/8b88b34/utils/colortab.c

    Parameters
    ----------
    filepath : str
        Path to annotation file.
    orig_ids : bool
        Whether to return the vertex ids as stored in the annotation
        file or the positional colortable ids. With orig_ids=False
        vertices with no id have an id set to -1.

    Returns
    -------
    labels : ndarray, shape (n_vertices,)
        Annotation id at each vertex. If a vertex does not belong
        to any label and orig_ids=False, its id will be set to -1.
    ctab : ndarray, shape (n_labels, 5)
        RGBT + label id colortable array.
    names : list of bytes
        The names of the labels. The length of the list is n_labels.
    """
    with open(filepath, "rb") as fobj:
        dt = _ANNOT_DT

        # number of vertices
        vnum = np.fromfile(fobj, dt, 1)[0]

        # vertex ids + annotation values
        data = np.fromfile(fobj, dt, vnum * 2).reshape(vnum, 2)
        labels = data[:, 1]

        # is there a color table?
        ctab_exists = np.fromfile(fobj, dt, 1)[0]
        if not ctab_exists:
            raise Exception('Color table not found in annotation file')

        # in old-format files, the next field will contain the number of
        # entries in the color table. In new-format files, this must be
        # equal to -2
        n_entries = np.fromfile(fobj, dt, 1)[0]

        # We've got an old-format .annot file.
        if n_entries > 0:
            ctab, names = _read_annot_ctab_old_format(fobj, n_entries)
        # We've got a new-format .annot file
        else:
            ctab, names = _read_annot_ctab_new_format(fobj, -n_entries)

    # generate annotation values for each LUT entry
    ctab[:, [4]] = _pack_rgb(ctab[:, :3])

    if not orig_ids:
        ord = np.argsort(ctab[:, -1])
        mask = labels != 0
        labels[~mask] = -1
        labels[mask] = ord[np.searchsorted(ctab[ord, -1], labels[mask])]
    return labels, ctab, names

def _pack_rgb(rgb):
    """Pack an RGB sequence into a single integer.

    Used by :func:`read_annot` and :func:`write_annot` to generate
    "annotation values" for a Freesurfer ``.annot`` file.

    Parameters
    ----------
    rgb : ndarray, shape (n, 3)
        RGB colors

    Returns
    -------
    out : ndarray, shape (n, 1)
        Annotation values for each color.
    """
    bitshifts = 2 ** np.array([[0], [8], [16]], dtype=rgb.dtype)
    return rgb.dot(bitshifts)

def _read_annot_ctab_old_format(fobj, n_entries):
    """Read in an old-style Freesurfer color table from `fobj`.

    Note that the output color table ``ctab`` is in RGBT form, where T
    (transparency) is 255 - alpha.

    This function is used by :func:`read_annot`.

    Parameters
    ----------

    fobj : file-like
        Open file handle to a Freesurfer `.annot` file, with seek point
        at the beginning of the color table data.
    n_entries : int
        Number of entries in the color table.

    Returns
    -------

    ctab : ndarray, shape (n_entries, 5)
        RGBT colortable array - the last column contains all zeros.
    names : list of str
        The names of the labels. The length of the list is n_entries.
    """
    assert hasattr(fobj, 'read')

    dt = _ANNOT_DT
    # orig_tab string length + string
    length = np.fromfile(fobj, dt, 1)[0]
    orig_tab = np.fromfile(fobj, '>c', length)
    orig_tab = orig_tab[:-1]
    names = list()
    ctab = np.zeros((n_entries, 5), dt)
    for i in range(n_entries):
        # structure name length + string
        name_length = np.fromfile(fobj, dt, 1)[0]
        name = np.fromfile(fobj, "|S%d" % name_length, 1)[0]
        names.append(name)
        # read RGBT for this entry
        ctab[i, :4] = np.fromfile(fobj, dt, 4)

    return ctab, names


def _read_annot_ctab_new_format(fobj, ctab_version):
    """Read in a new-style Freesurfer color table from `fobj`.

    Note that the output color table ``ctab`` is in RGBT form, where T
    (transparency) is 255 - alpha.

    This function is used by :func:`read_annot`.

    Parameters
    ----------

    fobj : file-like
        Open file handle to a Freesurfer `.annot` file, with seek point
        at the beginning of the color table data.
    ctab_version : int
        Color table format version - must be equal to 2

    Returns
    -------

    ctab : ndarray, shape (n_labels, 5)
        RGBT colortable array - the last column contains all zeros.
    names : list of str
        The names of the labels. The length of the list is n_labels.
    """
    assert hasattr(fobj, 'read')

    dt = _ANNOT_DT
    # This code works with a file version == 2, nothing else
    if ctab_version != 2:
        raise Exception('Unrecognised .annot file version (%i)', ctab_version)
    # maximum LUT index present in the file
    max_index = np.fromfile(fobj, dt, 1)[0]
    ctab = np.zeros((max_index, 5), dt)
    # orig_tab string length + string
    length = np.fromfile(fobj, dt, 1)[0]
    np.fromfile(fobj, "|S%d" % length, 1)[0]  # Orig table path
    # number of LUT entries present in the file
    entries_to_read = np.fromfile(fobj, dt, 1)[0]
    names = list()
    for _ in range(entries_to_read):
        # index of this entry
        idx = np.fromfile(fobj, dt, 1)[0]
        # structure name length + string
        name_length = np.fromfile(fobj, dt, 1)[0]
        name = np.fromfile(fobj, "|S%d" % name_length, 1)[0]
        names.append(name)
        # RGBT
        ctab[idx, :4] = np.fromfile(fobj, dt, 4)

    return ctab, names
    
def _fread3(fobj):
    """Read a 3-byte int from an open binary file object
    Parameters
    ----------
    fobj : file
        File descriptor
    Returns
    -------
    n : int
        A 3 byte int
    """
    b1, b2, b3 = np.fromfile(fobj, ">u1", 3)
    return (b1 << 16) + (b2 << 8) + b3


def _read_volume_info(fobj):
    """Helper for reading the footer from a surface file."""
    volume_info = OrderedDict()
    head = np.fromfile(fobj, ">i4", 1)
    if not np.array_equal(head, [20]):  # Read two bytes more
        head = np.concatenate([head, np.fromfile(fobj, ">i4", 2)])
        if not np.array_equal(head, [2, 0, 20]) and not np.array_equal(
            head, [2, 1, 20]
        ):
            warnings.warn("Unknown extension code.")
            return volume_info
        head = [2, 0, 20]

    volume_info["head"] = head
    for key in [
        "valid",
        "filename",
        "volume",
        "voxelsize",
        "xras",
        "yras",
        "zras",
        "cras",
    ]:
        pair = fobj.readline().decode("utf-8").split("=")
        if pair[0].strip() != key or len(pair) != 2:
            raise IOError("Error parsing volume info.")
        if key in ("valid", "filename"):
            volume_info[key] = pair[1].strip()
        elif key == "volume":
            volume_info[key] = np.array(pair[1].split()).astype(int)
        else:
            volume_info[key] = np.array(pair[1].split()).astype(float)
    # Ignore the rest
    return volume_info    
    
def read_geometry(filepath, read_metadata=False, read_stamp=False):
    """Read a triangular format Freesurfer surface mesh.
    Parameters
    ----------
    filepath : str
        Path to surface file.
    read_metadata : bool, optional
        If True, read and return metadata as key-value pairs.
        Valid keys:
        * 'head' : array of int
        * 'valid' : str
        * 'filename' : str
        * 'volume' : array of int, shape (3,)
        * 'voxelsize' : array of float, shape (3,)
        * 'xras' : array of float, shape (3,)
        * 'yras' : array of float, shape (3,)
        * 'zras' : array of float, shape (3,)
        * 'cras' : array of float, shape (3,)
    read_stamp : bool, optional
        Return the comment from the file
    Returns
    -------
    coords : numpy array
        nvtx x 3 array of vertex (x, y, z) coordinates.
    faces : numpy array
        nfaces x 3 array of defining mesh triangles.
    volume_info : OrderedDict
        Returned only if `read_metadata` is True.  Key-value pairs found in the
        geometry file.
    create_stamp : str
        Returned only if `read_stamp` is True.  The comment added by the
        program that saved the file.
    """
    volume_info = OrderedDict()

    TRIANGLE_MAGIC = 16777214

    with open(filepath, "rb") as fobj:
        magic = _fread3(fobj)

        if magic == TRIANGLE_MAGIC:  # Triangle file
            create_stamp = fobj.readline().rstrip(b"\n").decode("utf-8")
            test_dev = fobj.peek(1)[:1]
            if test_dev == b"\n":
                fobj.readline()
            vnum = np.fromfile(fobj, ">i4", 1)[0]
            fnum = np.fromfile(fobj, ">i4", 1)[0]
            coords = np.fromfile(fobj, ">f4", vnum * 3).reshape(vnum, 3)
            faces = np.fromfile(fobj, ">i4", fnum * 3).reshape(fnum, 3)

            if read_metadata:
                volume_info = _read_volume_info(fobj)
        else:
            raise ValueError(
                "File does not appear to be a Freesurfer surface (triangle file)"
            )

    coords = coords.astype(float)  # XXX: due to mayavi bug on mac 32bits

    ret = (coords, faces)
    if read_metadata:
        if len(volume_info) == 0:
            warnings.warn("No volume information contained in the file")
        ret += (volume_info,)
    if read_stamp:
        ret += (create_stamp,)

    return ret

def read_label(filepath, read_scalars=False):
    """Load in a Freesurfer .label file.

    Parameters
    ----------
    filepath : str
        Path to label file.
    read_scalars : bool, optional
        If True, read and return scalars associated with each vertex.

    Returns
    -------
    label_array : numpy array
        Array with indices of vertices included in label.
    scalar_array : numpy array (floats)
        Only returned if `read_scalars` is True.  Array of scalar data for each
        vertex.
    """
    label_array = np.loadtxt(filepath, dtype=int, skiprows=2, usecols=[0])
    if read_scalars:
        scalar_array = np.loadtxt(filepath, skiprows=2, usecols=[-1])
        return label_array, scalar_array
    return label_array

def write_label(filepath, vertices, values=None):
    """
    Write Freesurfer label data `values` to filepath `filepath`

    Parameters
    ----------
    filepath : str
        String containing path to label file to be written
    vertices : ndarray, shape (n_vertices, 3)
        Coordinates of each vertex
    values : optional, shape (n_vertices,)
        Array of scalar data for each vertex. The default is None.
    """
    
    if values is not None:
        vector = np.asarray(values)
        vnum = np.prod(vector.shape)
        if vector.shape not in ((vnum,), (vnum, 1), (1, vnum), (vnum, 1, 1)):
            raise ValueError('Invalid shape: argument values must be a vector')
            
    else:
        vector = np.zeros(vnum, dtype=np.float32)
    
    start_line = '#!ascii label  , from subject  vox2ras=TkReg\n'
    magic_number = vnum + 6000
    
    with open(filepath, 'w') as fobj:
        fobj.write(start_line)
        fobj.write(f'{magic_number}\n')
        
        #now write vertex and label array
        label_array = np.vstack((np.array(range(vnum)), vertices.T, vector.T)).T.astype('>f4')
        np.savetxt(fobj, label_array, fmt=['%i','%f','%f','%f','%f'])
        
def read_morph_data(filepath):
    """Read a Freesurfer morphometry data file.

    This function reads in what Freesurfer internally calls "curv" file types,
    (e.g. ?h. curv, ?h.thickness), but as that has the potential to cause
    confusion where "curv" also refers to the surface curvature values,
    we refer to these files as "morphometry" files with PySurfer.

    Parameters
    ----------
    filepath : str
        Path to morphometry file

    Returns
    -------
    curv : numpy array
        Vector representation of surface morpometry values
    """
    with open(filepath, "rb") as fobj:
        magic = _fread3(fobj)
        if magic == 16777215:
            vnum = np.fromfile(fobj, ">i4", 3)[0]
            curv = np.fromfile(fobj, ">f4", vnum)
        else:
            vnum = magic
            _fread3(fobj)
            curv = np.fromfile(fobj, ">i2", vnum) / 100
    return curv

def geodesic_distmat(surf, parc=None, n_jobs=1, use_cholmod=True):
    """
    Compute geodesic distance using the heat diffusion method built into LaPy
        Based on: doi/10.1145/2516971.2516977
        Crane et al., "Geodesics in heat: A new approach to computing distance 
        based on heat flow"
    
    Parameters
    ----------
    surf : Shape class
        Input surface
    n_jobs : int, optional
        Number of workers to use for parallel calls to ._thread_method(),
        default is 1.
    
    Returns
    -------
    D : (N,N) np.ndarray
        Distance matrix of every vertex to every other vertex
    
    """
    surf = nib.load(surf).agg_data()
    tria = TriaMesh(surf[0], surf[1])
    try:
        import_optional_dependency('sksparse')
        use_cholmod=True
    except:
        use_cholmod=False
    
    fem = Solver(tria, lump=True, use_cholmod=use_cholmod)
        
    D = __distance_threading__(tria, fem, n_jobs=n_jobs)
    
    if parc:
        D = np.row_stack([
            D[parc == lab].mean(axis=0) for lab in np.unique(parc)
        ])
        D[np.diag_indices_from(D)] = 0
        D = D[1:, 1:]
    
    return D

def __distance_threading__(tria, fem, n_jobs=1):

    D = np.column_stack(
        Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(_distance_thread_method)(tria, fem, bvert=bvert) for bvert in range(tria.v.shape[0])
            )
        )
    
    return np.asarray(D.squeeze())

def _distance_thread_method(tria, fem, bvert):    
    u = diffusion(tria, fem, vids=bvert, m=1.0)
    
    tfunc = tria_compute_gradient(tria, u)
    
    X = -tfunc / np.sqrt((tfunc**2).sum(1))[:,np.newaxis]
    X = np.nan_to_num(X)
    
    b0 = tria_compute_divergence(tria, X)
    
    if fem.use_cholmod:
        chol = cholesky(-fem.stiffness)
        d = chol(b0)
    else:
        A = fem.stiffness
        H = -A
        lu = splu(H)
        d = lu.solve(b0.astype(np.float32))
    
    d = d - min(d)
        
    return d

def diffusion(geometry, fem, vids, m=1.0, use_cholmod=True):
    """
    Computes heat diffusion from initial vertices in vids using
    backward Euler solution for time t [MO2]:
    
      t = m * avg_edge_length^2
    
    Parameters
    ----------
      geometry      TriaMesh or TetMesh, on which to run diffusion
      vids          vertex index or indices where initial heat is applied
      m             factor (default 1) to compute time of heat evolution:
                    t = m * avg_edge_length^2
      use_cholmod   (default True), if Cholmod is not found
                    revert to LU decomposition (slower)
    
    Returns
    -------
      vfunc         heat diffusion at vertices
    """
    sksparse = import_optional_dependency("sksparse", raise_error=use_cholmod)
    
    nv = len(geometry.v)
    fem = fem
    # time of heat evolution:
    t = m * geometry.avg_edge_length() ** 2
    # backward Euler matrix:
    hmat = fem.mass + t * fem.stiffness
    # set initial heat
    b0 = np.zeros((nv,))
    b0[np.array(vids)] = 1.0
    # solve H x = b0
    #print("Matrix Format now:  " + hmat.getformat())
    if sksparse is not None:
        print("Solver: Cholesky decomposition from scikit-sparse cholmod ...")
        chol = sksparse.cholmod.cholesky(hmat)
        vfunc = chol(b0)
    else:
        from scipy.sparse.linalg import splu
    
        print("Solver: spsolve (LU decomposition) ...")
        lu = splu(hmat)
        vfunc = lu.solve(np.float32(b0))
    return vfunc

"""Handle optional dependency imports.

Inspired from pandas: https://pandas.pydata.org/
"""

import importlib

# A mapping from import name to package name (on PyPI) when the package name
# is different.
INSTALL_MAPPING = {
    "sksparse": "scikit-sparse",
}


def import_optional_dependency(
    name: str,
    extra: str = "",
    raise_error: bool = True,
):
    """Import an optional dependency.

    By default, if a dependency is missing an ImportError with a nice message
    will be raised.

    Parameters
    ----------
    name : str
        The module name.
    extra : str, default=""
        Additional text to include in the ImportError message.
    raise_error : bool, default=True
        What to do when a dependency is not found.
        * True : Raise an ImportError.
        * False: Return None.

    Returns
    -------
    module : Optional[ModuleType]
        The imported module when found.
        None is returned when the package is not found and raise_error is
        False.

    Raises
    -------
    ImportError
        dependency not found; see raise_error
    """

    package_name = INSTALL_MAPPING.get(name)
    install_name = package_name if package_name is not None else name

    try:
        module = importlib.import_module(name)
    except ImportError:
        if raise_error:
            raise ImportError(
                f"Missing optional dependency '{install_name}'. {extra} "
                f"Use pip or conda to install {install_name}."
            )
        else:
            return None

    return module