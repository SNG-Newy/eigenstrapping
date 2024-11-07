from lapy import TetMesh, TriaMesh, Solver
from lapy.diffgeo import compute_geodesic_f

import warnings
import subprocess
import tempfile
import os

import nibabel as nib
import numpy as np
from scipy.spatial import Delaunay, KDTree
from scipy.spatial.distance import squareform, pdist
from brainspace import mesh as me
from joblib import Parallel, delayed
from scipy.sparse.linalg import splu
import scipy.optimize as optimize

from brainspace.vtk_interface import wrap_vtk, serial_connect
from vtk import (vtkThreshold, vtkDataObject, vtkGeometryFilter)
from brainspace.utils.parcellation import relabel_consecutive
from brainspace.mesh.mesh_creation import build_polydata
from eigenstrapping.utils import (_suppress, _print, is_string_like,
                    enablePrint, blockPrint, eigen_decomposition)

from eigenstrapping.dataio import dataio
from sklearn.utils.validation import check_random_state
from eigenstrapping.rotations import indirect_method
from eigenstrapping.utils import _get_eigengroups
from scipy.spatial import distance_matrix
from sklearn.neighbors import BallTree
from scipy.optimize import linear_sum_assignment
import copy
from parspin import utils as putils

from tqdm import tqdm

ASSOC_CELLS = vtkDataObject.FIELD_ASSOCIATION_CELLS
ASSOC_POINTS = vtkDataObject.FIELD_ASSOCIATION_POINTS

#blockPrint()

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

def geodesic_distmat(surface, mask_input=None, use_cholmod=False, n_jobs=1, m=1.0):
    """
    Compute geodesic distance using the heat diffusion method built into LaPy
        Based on: doi/10.1145/2516971.2516977
        Crane et al., "Geodesics in heat: A new approach to computing distance 
        based on heat flow"
    
    Parameters
    ----------
    surf : str
        Path to input surface
    mask_input : array_like or str
        Mask to remove from `surf` (i.e. medial wall) during the geodesic distance calculations
    use_cholmod : bool, optional
        Specify whether to use ``scikit-sparse`` cholmod libraries (much faster), if False then
        uses ``scipy.sparse`` libraries instead (much slower). Default False.
    n_jobs : int, optional
        Number of workers to use for parallel calls to ._thread_method(),
        default is 1.
    m : float, optional
        Scaling factor for time in the heat kernel calculation. As `m` decreases,
        distances increase in accuracy (if you care) but increases computation time. Best put between
        1.0 and 0.1. Default 1.0.
    
    Returns
    -------
    D : (N,N) np.ndarray
        Distance matrix of every vertex to every other vertex
    
    """
    # load surface (as a brainspace object)
    surface_orig = me.mesh_io.read_surface(surface)
    
    # load mask
    # can be any ROI (even whole cortex)
    if isinstance(mask_input, np.ndarray):
        mask = mask_input
    elif mask_input is None:
        mask = np.ones(len(surface_orig.GetPoints().astype('float32')))
    else:
        mask_input_file_main, mask_input_file_ext = os.path.splitext(mask_input)
        if mask_input_file_ext == '.txt':
            mask = np.loadtxt(mask_input)
        elif mask_input_file_ext == '.gii':
            mask = nib.load(mask_input).darrays[0].data
        else:
            mask = mask_input

    # create temporary suface based on mask
    surface_cut = _surface_mask(surface_orig, mask)
    # load surface (as a lapy object)
    vertices = surface_cut.GetPoints().astype('float32')
    faces = me.mesh_elements.get_cells(surface_cut).astype('int32')
    tria = TriaMesh(vertices, faces)

    fem = Solver(tria, lump=True, use_cholmod=use_cholmod)
    # time of heat evolution:
    t = m * tria.avg_edge_length() ** 2
    # backward Euler matrix:
    factor = fem.mass + t * fem.stiffness # keep factor, dummy. the slowest part is the cholmod factorization :grug:
        
    D = __distance__(tria, factor, n_jobs=n_jobs)
    
    # if parc:
    #     D = np.row_stack([
    #         D[parc == lab].mean(axis=0) for lab in np.unique(parc)
    #     ])
    #     D[np.diag_indices_from(D)] = 0
    #     D = D[1:, 1:]
    
    return D

def __distance__(tria, factor, n_jobs=1, use_cholmod=False):
    D = np.column_stack(
        Parallel(n_jobs=n_jobs, prefer='processes')(
            delayed(_distance_method)(tria, factor, use_cholmod=use_cholmod, v=v) for v in tqdm(range(tria.v.shape[0]))
            )
        )
    
    return np.asarray(D.squeeze())

def _distance_method(tria, factor, use_cholmod=False, v=0):
    b0 = np.zeros((tria.v.shape[0]))
    b0[np.array(v)] = 1.0

    if use_cholmod:
        sksparse = import_optional_dependency("sksparse", raise_error=True)
        importlib.import_module(".cholmod", sksparse.__name__)
        #print("Solver: Cholesky decomposition from scikit-sparse cholmod ...")
        factor = sksparse.cholmod.cholesky(factor)
        u = factor(b0)
    else:
        sksparse=None
        from scipy.sparse.linalg import splu
        #print("Solver: spsolve (LU decomposition) ...")
        factor = splu(factor)
        u = factor.solve(b0)
    d = compute_geodesic_f(tria, u)

    return d

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

def make_tetra(volume, label=None, aseg=False, norm=None, verbose=True):
    """
    Generate tetrahedral version of the ROI in the nifti file. Can
    specify label value.

    Parameters
    ----------
    volume : str
        Filename of input volume where the relevant ROI has voxel 
        values = 1 or ``label``
    label : int or list of ints, optional
        Label value(s) of input volume. Extracts surface from voxels that have
        this(ese) intensity value(s). If None, defaults to 1
    aseg : bool, optional
        Specify whether input volume and label is from FreeSurfer
        aseg.nii.gz (post-FS preprocessing).

    Returns
    -------
    tetra_file : str
        Filename of output tetrahedral vtk file
        
    Raises
    ------
    RuntimeError
        Multiple labels given but aseg not passed.

    """
    if verbose:
        enablePrint()
        func = _print
    else:
        blockPrint()
        func = _suppress
    
    voldir = os.path.dirname(volume)
    tmpf = tempfile.NamedTemporaryFile(suffix='.mgz')
    tmpf = tmpf.name
    
    multiple = False
    
    if label is None:
        label = ' '.join('1')
    elif type(label) == list:
        if len(label) > 1:
            multiple = True
        tmp_label = ' '
        for idx in range(len(label)):
            tmp_label.join(str(label[idx]))
        
        label = tmp_label
    else:
        label = ' '.join(str(label))
        
    if aseg is False:
        if multiple is True:
            raise RuntimeError('Multiple label values given, aseg must be passed')
    
    # binarize first
    cmd = 'mri_binarize --i ' + volume + ' --match ' + label + ' --o ' + tmpf
    output = func(cmd, shell="True")
    
    # pass norm for pretess
    if aseg is True and norm is not None:
        cmd = 'mri_pretess ' + tmpf + ' 1 ' + norm + ' ' + tmpf
        output = func(cmd, shell="True")
    
    # run marching cubes
    cmd = 'mri_mc ' + tmpf + ' 1 ' + voldir + '/tmp_surface.vtk'
    output = func(cmd, shell="True")
    
    geo_file = volume + '.geo'
    tria_file = volume + '.vtk'
    tetra_file = volume + '.tetra.vtk'
    
    cmd = 'mv -f ' + voldir + '/tmp_surface.vtk ' + tria_file
    output = func(cmd, shell='True')
    
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
    output = func(cmd, shell="True")
    
    cmd = "sed 's/double/float/g;s/UNSTRUCTURED_GRID/POLYDATA/g;s/CELLS/POLYGONS/g;/CELL_TYPES/,$d' " + tetra_file + " > " + tetra_file + "'_fixed'"            
    output = func(cmd, shell="True")
    
    cmd = 'mv -f ' + tetra_file + '_fixed ' + tetra_file
    output = func(cmd, shell="True")
    
    # remove auxiliary files
    os.remove(geo_file)
    os.remove(tria_file)
    
    return tetra_file

def remesh(m, outfile=None, oformat='vtk'):
    """
    Generate (re)meshed version of the volume/mesh in `m`. Goes from volume to 
    triangular mesh, or goes from triangular to tetrahedral mesh (and vice-versa).
    Function does not go back from triangular or tetrahedral mesh to volume.

    Parameters
    ----------
    m : str
        The filename of the volume/mesh.
    outfile : str, optional
        The filename of the output mesh, default None.
    oformat : str, optional
        Output format of mesh, default "vtk"

    Returns
    -------
    file : str
        Filename of output tetra or triangular mesh.

    """
    _, ext = os.path.splitext(m)
    if ext == '.nii':
        return vol_to_tria(m, outfile=outfile, oformat=oformat)
    
    mesh = me.mesh_io.read_surface(m)
    if mesh.GetCells2D().shape[-1] == 4:
        func = to_tria
    elif mesh.GetCells2D().shape[-1] == 3:
        func = to_tetra
    else:
        raise RuntimeError('Unknown file structure, check input')
    
    file = func(m, outfile=outfile, oformat=oformat)
    
    return file

def vol_to_tria(volume, outfile=None, oformat='vtk', verbose=False):
    """
    Converts the volume in `volume` to a triangular surface using
    FreeSurfer `mri_mc` and other libraries.

    Parameters
    ----------
    volume : str
        Path to file to convert to triangular mesh.
    outfile : str, optional
        The filename of the output mesh, default None.
    oformat : str, optional
        Output format of mesh, default "vtk"
    verbose : bool, optional
        Print output messages, default False

    Returns
    -------
    tria_file : str
        Filename of saved triangular mesh.

    """
    tmpf = tempfile.NamedTemporaryFile(suffix='.mgz')
    tmpf = tmpf.name
    voldir = os.path.dirname(volume)
    volname = os.splitext(volume)[0]
    
    if outfile is None:
        outfile = volname + '.tria.' + oformat
    
    if verbose is True:
        enablePrint()
        func = _print
    else:
        blockPrint()
        func = _suppress
    
    cmd = 'mri_binarize --i ' + volume + ' --match 1' + ' --o ' + tmpf
    output = func(cmd, shell="True")
    
    # run marching cubes
    tmp_tria = voldir + '/tmp_surface.' + oformat
    cmd = 'mri_mc ' + tmpf + ' 1 ' + tmp_tria
    output = func(cmd, shell="True")
    
    cmd = 'mv -f ' + voldir + '/tmp_surface.' + oformat + ' ' + outfile
    output = func(cmd, shell='True')
    
    os.unlink(tmpf)
    os.unlink(tmp_tria)
    
    return outfile

def to_tetra(tria_file, outfile=None, oformat='vtk'):
    """
    Converts the triangular surface contained in `tria_file` into
    a tetrahedral mesh by using `gmsh` routines.

    Parameters
    ----------
    tria_file : str
        Filename of triangular surface.
    outfile : str, optional
        Filename of output tetrahedral mesh. If None, adds the suffix .tetra.<oformat>
        to the `tria_file` filename.
    oformat : str, optional
        Format of output mesh, accepts `vtk`, `gii`, `ply`, `obj`

    Returns
    -------
    outfile : str
        Filename of output triangular mesh.

    """
    if oformat not in ['vtk', 'gii', 'ply', 'obj']:
        raise ValueError('Output format not valid: {}'.format(str(oformat)))
    
    if outfile is None:
        outfile = os.path.splitext(tria_file)[0] + '.tetra.' + oformat
        
    geo_file = tria_file + '.geo'
    tetra_file = outfile
    
    inputGeo = os.path.split(tria_file)[-1]
    print(inputGeo)
    
    with open(geo_file, 'w') as writer:
        writer.write('Mesh.Algorithm3D=4;\n')
        writer.write('Mesh.Optimize=1;\n')
        writer.write('Mesh.OptimizeNetgen=1;\n')
        writer.write('Merge "'+inputGeo+'";\n')
        writer.write('Surface Loop(1) = {1};\n')
        writer.write('Volume(1) = {1};\n')
        writer.write('Physical Volume(1) = {1};\n')
        
    cmd = 'gmsh -3 -o ' + tetra_file + ' ' + geo_file
    output = _print(cmd, shell="True")
    
    cmd = "sed 's/double/float/g;s/UNSTRUCTURED_GRID/POLYDATA/g;s/CELLS/POLYGONS/g;/CELL_TYPES/,$d' " + tetra_file + " > " + tetra_file + "'_fixed'"            
    output = _print(cmd, shell="True")
    
    cmd = 'mv -f ' + tetra_file + '_fixed ' + tetra_file
    output = _print(cmd, shell="True")
    
    # remove auxiliary files
    os.remove(geo_file)

    return outfile

def to_tria(tetra_file, outfile=None, oformat='vtk', verbose=True):
    """
    Converts the tetrahedral mesh contained in `tetra_file` into
    a triangular surface by bounding the convex hull of the tetrahedra.

    Parameters
    ----------
    tetra_file : str
        Filename of tetrahedral mesh.
    outfile : str, optional
        Filename of output triangular mesh. If None, adds the suffix .tria.<oformat>
        to the `tetra_file` filename.
    oformat : str, optional
        Format of output mesh, accepts `vtk`, `gii`, `ply`, `obj`

    Returns
    -------
    outfile : str
        Filename of output triangular mesh.

    """
    if oformat not in ['vtk', 'gii', 'ply', 'obj']:
        raise ValueError('Output format not valid: {}'.format(str(oformat)))
    
    if outfile is None:
        outfile = os.path.splitext(tetra_file)[0] + '.tria.' + oformat
        
    m = me.mesh_io.read_surface(tetra_file)
    tetra = TetMesh(m.GetPoints(), m.GetCells2D())
    # get all triangles
    allt = np.vstack(
        (
            tetra.t[:, np.array([3, 1, 2])],
            tetra.t[:, np.array([2, 0, 3])],
            tetra.t[:, np.array([1, 3, 0])],
            tetra.t[:, np.array([0, 2, 1])],
        )
    )
    # sort rows so that faces are reorder in ascending order of indices
    allts = np.sort(allt, axis=1)
    # find unique trias without a neighbor
    tria, indices, count = np.unique(
        allts, axis=0, return_index=True, return_counts=True
    )
    tria = allt[indices[count == 1]]
    blockPrint()
    if verbose:
        enablePrint()
        print("Found " + str(np.size(tria, 0)) + " triangles on boundary.")
    # if we have tetra function, map these to the boundary triangles
        
    surf = TriaMesh(tetra.v, tria)
    surf.write_vtk(outfile)
    
    return outfile

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

def load_mesh(mesh):
    """
    Mesh loader function.

    Parameters
    ----------
    mesh : str or array-like
        Filename 'vtk'-like, or tuple of surface (expecting two arrays of `vertices`
        and `faces`)

    Returns
    -------
    ``lapy`` compatible object : `TetMesh` or `TriaMesh`
        Loaded mesh in ``lapy`` format

    """
    
    if isinstance(mesh, tuple) or isinstance(mesh, list):
        m = build_polydata(mesh[0], mesh[1])
    elif is_string_like(mesh):
        m = me.mesh_io.read_surface(mesh)
    else:
        raise ValueError('Unknown file type, check input')
        
    if m.GetCells2D().shape[-1] == 4:
        func = TetMesh
    elif m.GetCells2D().shape[-1] == 3:
        func = TriaMesh
    else:
        raise RuntimeError('Unknown file structure, check input')
    
    return func(m.GetPoints(), m.GetCells2D())
    

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

    # potential target for improved runtime
    # takes several minutes for a big mesh
    
    return indices

def calc_eig(mesh, num_modes, use_cholmod=False, remove_zero=False):
    """Calculate the eigenvalues and eigenmodes of a surface.

    Parameters
    ----------
    mesh : lapy compatible object
        Loaded vtk object corresponding to a surface mesh
    num_modes : int
        Number of eigenmodes to be calculated

    Returns
    ------
    evals : array (num_modes x 1)
        Eigenvalues
    emodes : array (number of surface points x num_modes)
        Eigenmodes
    """
    
    fem = Solver(mesh, use_cholmod=use_cholmod)
    evals, emodes = fem.eigs(k=num_modes)
    
    if remove_zero:
        return evals[1:], emodes[:, 1:]
    
    return evals[:-1], emodes[:, :-1]
    
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
    surface_orig = me.mesh_io.read_surface(surface_input_filename)
    
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
    surface_cut = _surface_mask(surface_orig, mask)

    # load surface (as a lapy object)
    vertices = surface_cut.GetPoints().astype('float32')
    faces = me.mesh_elements.get_cells(surface_cut).astype('int32')
    tria = TriaMesh(vertices, faces)
    
    # calculate eigenvalues and eigenmodes
    evals, emodes = calc_eig(tria, num_modes+1, use_cholmod, remove_zero=remove_zero)
    
    # get indices of vertices of surface_orig that match surface_cut
    indices = get_indices(surface_orig, surface_cut)
    
    # reshape emodes to match vertices of original surface
    emodes_reshaped = np.zeros([surface_orig.n_points,np.shape(emodes)[1]])
    for mode in range(np.shape(emodes)[1]):
        emodes_reshaped[indices,mode] = np.expand_dims(emodes[:,mode], axis=1)
        
    return evals, emodes_reshaped

def _surface_mask(surf, mask, use_cell=False):
    """Selection fo points or cells meeting some criteria.

    Parameters
    ----------
    surf : vtkPolyData or BSPolyData
        Input surface.
    mask : str or ndarray
        Binary boolean or integer array. Zero or False elements are
        discarded.
    use_cell : bool, optional
        If True, apply selection to cells. Otherwise, use points.
        Default is False.

    Returns
    -------
    surf_masked : BSPolyData
        PolyData after masking.

    """

    if isinstance(mask, np.ndarray):
        if np.issubdtype(mask.dtype, np.bool_):
            mask = mask.astype(np.uint8)
    else:
        mask = surf.get_array(name=mask, at='c' if use_cell else 'p')

    if np.any(np.unique(mask) > 1):
        raise ValueError('Cannot work with non-binary mask.')

    return _surface_selection(surf, mask, low=1, upp=1, use_cell=use_cell)

def _surface_selection(surf, array, low=-np.inf, upp=np.inf, use_cell=False):
    """Selection of points or cells meeting some thresholding criteria.

    Parameters
    ----------
    surf : vtkPolyData or BSPolyData
        Input surface.
    array : str or ndarray
        Array used to perform selection.
    low : float or -np.inf
        Lower threshold. Default is -np.inf.
    upp : float or np.inf
        Upper threshold. Default is +np.inf.
    use_cell : bool, optional
        If True, apply selection to cells. Otherwise, use points.
        Default is False.

    Returns
    -------
    surf_selected : BSPolyData
        Surface after thresholding.

    """
    
    if low > upp:
        raise ValueError('Threshold not valid: [{},{}]'.format(low, upp))

    at = 'c' if use_cell else 'p'
    if isinstance(array, np.ndarray):
        drop_array = True
        array_name = surf.append_array(array, at=at)
    else:
        drop_array = False
        array_name = array
        array = surf.get_array(name=array, at=at, return_name=False)

    if array.ndim > 1:
        raise ValueError('Arrays has more than one dimension.')

    if not use_cell:
        order_name = surf.append_array(np.arange(surf.n_points), at='p')

    if low == -np.inf:
        low = array.min()
    if upp == np.inf:
        upp = array.max()

    tf = wrap_vtk(vtkThreshold, allScalars=True)
    tf.SetUpperThreshold(upp)
    tf.SetLowerThreshold(low)
    if use_cell:
        tf.SetInputArrayToProcess(0, 0, 0, ASSOC_CELLS, array_name)
    else:
        tf.SetInputArrayToProcess(0, 0, 0, ASSOC_POINTS, array_name)

    gf = wrap_vtk(vtkGeometryFilter(), merging=False)
    surf_sel = serial_connect(surf, tf, gf)

    # Check results
    n_exp = np.logical_and(array >= low, array <= upp).sum()
    n_sel = surf_sel.n_cells if use_cell else surf_sel.n_points
    if n_exp != n_sel:
        element = 'cells' if use_cell else 'points'
        warnings.warn('Number of selected {}={}. Expected {}.'
                      'This may be due to the topology after selection.'.
                      format(element, n_exp, n_sel))

    if drop_array:
        surf.remove_array(name=array_name, at=at)
        surf_sel.remove_array(name=array_name, at=at)

    if not use_cell:
        surf_sel = _sort_polydata_points(surf_sel, order_name)
        surf_sel.remove_array(name=order_name, at='p')

        surf.remove_array(name=order_name, at='p')

    return surf_sel

def _sort_polydata_points(surf, labeling, append_data=True):

    if isinstance(labeling, str):
        labeling = surf.get_array(labeling, at='p')

    lab_con = relabel_consecutive(labeling)

    idx_sorted = np.argsort(lab_con)
    new_pts = surf.Points[idx_sorted]
    # new_cells = relabel(surf.GetCells2D().ravel(), lab_con).reshape(-1, 3)
    new_cells = lab_con[surf.GetCells2D()]
    s = build_polydata(new_pts, cells=new_cells)

    if append_data is None or append_data is False:
        return s

    if append_data is True:
        append_data = {'p', 'c', 'f'}
    elif isinstance(append_data, str):
        append_data = {append_data}

    for at in append_data:
        for v, k in zip(*surf.get_array(at=at, return_name=True)):
            if at in {'p', 'point'}:
                v = v[idx_sorted]
            s.append_array(v, name=k, at=at)
    return s

def find_wavelength(surface, data, roi=None, column=None, demean=True):
    """
    Compute the characteristic wavelength of the data in `data` based
    on the full-width half-maximum (FWHM). This is equal to 2*FWHM.
    This function simply wraps ``wb_command -metric-estimate-fwhm``,
    converting stdout to a Python float.

    Can be used to estimate the number of eigengroups to include
    in the reconstruction of surrogates in 
    :class:`eigenstrapping.SurfaceEigenstrapping.generate()`
    based on the relationship of eigenvalue to spatial scale.

    Parameters
    ----------
    surface : str to filepath
        Surface for `data`.
    data : np.ndarray or str to filepath (N,)
        Data on `surface` to find FWHM
    roi : str to filepath, optional
        Region-of-interest mask. Default None
    column
    
    Returns
    -------
    cost : float
        If `return_cost`, returns maximum cost of kde fit.
    wavelength : float
        Value for full-width half-maximum of `data`.

    Notes
    -----
    This function uses Euclidean distance to measure nearest
    neighbors as an approximation, and assumes a Gaussianity 
    to the smoothness in `data`. It may have unpredictable 
    effects if the data is highly non-Gaussian. Use with caution. 
    If the fit is bad (cost is high) then we recommend the 
    variogram calculation in :func:`eigenstrapping.fit.surface_fit()`

    """
    data = dataio(data)
    tmpf = tempfile.NamedTemporaryFile(suffix='.func.gii').name
    metric = nib.GiftiImage()
    metric.add_gifti_data_array(nib.gifti.gifti.GiftiDataArray(data.astype(np.float32)))
    nib.save(metric, tmpf)

    if roi is not None:
        roistr = '-roi {roi}'
    else:
        roistr = ''

    if column is not None:
        colstr = '-column {column}'
        wholestr = ''
    else:
        colstr = ''
        wholestr = '-whole-file'
    
    if demean is True:
        destr = '-demean'
    else:
        destr = ''

    cmd = f'wb_command -metric-estimate-fwhm {surface} {tmpf} {roistr} {colstr} {wholestr} {destr}'
    output = subprocess.check_output(cmd, shell="True")
    output = output.splitlines()

    os.unlink(tmpf)

    return 2*np.pi*float(output[0].decode('ascii').split(' ')[1])

def truncate_emodes(surface, data, emodes, evals, 
                    mask=None, ret_fwhm=False, find_group_only=False,
                    per_spectrum=None):
    """
    Uses the estimation of full-width half-maximum to truncate the
    number of eigenmodes to use for surrogate generation. Provides a
    heuristic to assist the end-user with number of mode selection.
    How it works is it finds the first eigenvalue such that:

                2 * pi
        argmin(--------) <=  2*fwhm
                 lam
    
    Actually truncates at the number of *whole* groups that include
    this eigenvalue or smaller, corresponding to larger wavelengths.
    Either "rounds" up or down, depending on whether the eigenvalue
    index of `lam` is greater or lesser than the total number of
    eigenvalues in the group G, where G is the group number.
    
    In other words,

        floor(G(lam)) if ||lam|| < np.sum(||lam_G||),
            ceil(G(lam)) if ||lam|| >= np.sum(||lam_G||)

    Also has the ability to truncate modes that account for `per_spectrum` 
    percentage of the power spectra in `data`, which is quicker
    and may suit your end-goals better, particularly if small-scale
    spatial structure doesn't matter to your map. For example,
    per_spectra=0.9 would truncate the eigenmodes and eigenvalues
    to the total number of groups that contain 90% or more of the
    modal power spectrum in `data`.

    Parameters
    ----------
    data : np.ndarray or str to filepath (N,)
        Surface data to find FWHM
    surface : nib.GiftiImage like or filepath
        Surface for `data`. Must have coordinates of shape (N, 3)
    mask : np.ndarray or str to filepath (N,), optional
        Mask for values in `data` to include e.g. non-medial wall
        vertices. Default None
    ns : int, optional
        Number of samples for random sampling. Default 100
    knn : int, optional
        Number of neighbors to calculate FWHM. Default 100
    seed : None or int or np.random.RandomState, optional
        Seed for random number generation. Default None
    ret_fwhm : bool, optional
        If True, returns estimated `fwhm`.
        Default False
    find_group_only : bool, optional
        If True, this function does not truncate `emodes` and `evals`,
        and only returns the group number that contains the eigenvalue
        where the above equation is true.
    per_spectra : None or float, optional
        If True, truncates `emodes` and `evals` at whole number of groups
        with `per_spectrum`
    
    Returns
    -------
    emodes_truncated, evals_truncated : np.ndarrays
        Eigenmodes and corresponding eigenvalues as whole groups
        with higher or equal wavelengths to the smallest `fwhm` in `data`.
        Group number to truncate `emodes` and `evals` at is rounded to nearest
        group with full membership.
    fwhm : float
        Value for full-width half-maximum of `data`. Only returned
        if `ret_fwhm` is True. Exclusive with `per_spectra`.
    group : int
        Eigengroup number (zero-indexed)

    Notes
    -----
    This function assumes a Gaussianity to the smoothness in
    `data`. It may have unpredictable effects if the data is highly
    non-Gaussian. Use with caution. If the data is highly non-Gaussian
    then we recommend the variogram calculation in 
    :func:`eigenstrapping.fit.surface_fit()`

    """
    groups = _get_eigengroups(emodes)
    if per_spectrum is not None:
        coeffs = eigen_decomposition(data[mask], emodes[mask])
        power = np.abs(coeffs)**2
        normed_power = power/np.sum(power)
        total_power = 0.0
        j = 0
        while total_power <= per_spectrum*np.sum(normed_power):
            total_power += np.sum(normed_power[groups[j]])
            j += 1

        if find_group_only:
            return j
        else:
            emodes_truncated, evals_truncated = (emodes[:, :groups[j][-1] + 1], evals[:groups[j][-1] + 1])
            return evals_truncated, emodes_truncated
        

    fwhm = find_wavelength(surface, data, demean=False)
    
    j = 0
    while j <= len(groups):
        for k in range(j):
            l = 2*np.pi/np.sqrt(evals[groups[j]][k])
            if l <= fwhm:
                if find_group_only:
                    if ret_fwhm:
                        return fwhm, j if (groups[j][k] - groups[j][0]) >= (groups[j][-1] - groups[j][0]) else j-1
                    return j if (groups[j][k] - groups[j][0]) >= (groups[j][-1] - groups[j][0]) else j-1
                emodes_truncated, evals_truncated = (emodes[:, :groups[j][-1] + 1], evals[:groups[j][-1] + 1]) if (groups[j][k] - groups[j][0]) >= (groups[j][-1] - groups[j][0]) else (emodes[:, :groups[j-1][-1] + 1], evals[:groups[j-1][-1] + 1])
                if ret_fwhm:
                    return fwhm, evals_truncated, emodes_truncated
                else:
                    return evals_truncated, emodes_truncated
        j += 1
    
    raise RuntimeError('Could not find eigenvalue for fwhm\n'
                       'consider using more modes.')


def gen_eigensamples(emodes, evals, mask=None, n_rotate=1000, num_modes=100, check_duplicates=True,
                     method='original', seed=None, n_jobs=1):
    """
    compute a permutation matrix for a set of eigenmodes, based on the concept
    of rotating the eigenmodes in the eigenspace per group. Produces a sparse
    permutation matrix of (`n_rotate`, N, M).
    
    Parameters
    ----------
    emodes : np.ndarray of shape (N, M)
        eigenmodes to resample
    evals : np.ndarray of shape (M,)
        eigenvalues corresponding to `emodes`
    n_rotate : int, optional
        number of permutations to compute. Default 1000
    check_duplicates: bool, optional
        Whether to check for and attempt to avoid duplicate resamplings. A
        warnings will be raised if duplicates cannot be avoided. Setting to
        True may increase the runtime of this function! Default: True    
    seed : None or int or np.random.RandomState, optional
        seed for random number generation. Default None
    method : {'original', 'vasa', 'hungarian'}, optional
        Method by which to match non- and rotated coordinates. Default: 'original'
    verbose : bool, optional
        Whether to print occasional status messages. Default: False
    return_cost : bool, optional
        Whether to return cost array (specified as Euclidean distance) for each
        coordinate for each rotation Default: True
        
    Returns
    -------
    samples : np.ndarray of ints of shape (`n_rotate`, N, M)
        nearest neighbor remapping matrix

    """
    from multiprocessing import Pool

    seed = check_random_state(seed)

    # mask emodes
    emodes_masked = copy.deepcopy(emodes[:, :num_modes])
    emodes_masked = emodes_masked[mask]
    evals = copy.deepcopy(evals)
    evals = evals[:num_modes]
    groups = _get_eigengroups(emodes_masked)

    # TODO implement costs
    #cost = np.zeros((n_rotate, len(emodes_masked), len(groups)))
    inds = np.arange(len(emodes_masked), dtype='int32')

    # empty array to store resampling indices
    #eigen_spins = np.zeros((n_rotate, len(emodes_masked), len(evals)), dtype='int32')

    # generate resampled modes and fill indexing array
    def _gen_method(emodes_masked, evals, groups, method='original', seed=0):
        resampled = np.zeros((len(emodes_masked), len(groups)), dtype='int32')
        for j in range(len(groups)):
            group = emodes_masked[:, groups[j]] / np.sqrt(evals[groups[j]])
            rot = indirect_method(len(groups[j]), seed=seed)
            new_group = group @ rot * np.sqrt(evals[groups[j]])
            if method == 'original':
                dist, col = BallTree(new_group).query(emodes_masked[:, groups[j]], 1)
                col = col.reshape(-1,)
                #cols = np.stack(cols, axis=-1)
                #cost[:, j] = dist
            elif method == 'hungarian':
                dist = distance_matrix(group, new_group)
                row, col = optimize.linear_sum_assignment(dist)
                #cost[hinds, n] = dist[row, col]
            elif method == 'vasa':
                dist = distance_matrix(group, new_group)
                # min of max a la Vasa et al., 2018
                col = np.zeros(len(group), dtype='int32')
                for _ in range(len(dist)):
                    # find parcel whose closest neighbor is farthest away
                    # overall; assign to that
                    row = dist.min(axis=1).argmax()
                    col[row] = dist[row].argmin()
                    #cost[inds[hinds][row], n] = dist[row, col[row]]
                    # set to -inf and inf so they can't be assigned again
                    dist[row] = -np.inf
                    dist[:, col[row]] = np.inf
            else:
                raise ValueError('unrecognized method {}\n'
                                 'choose from "original", "hungarian" or "vasa"'.format(method))

            resampled[:, j] = inds[col]

        return resampled

    eigen_spins = Parallel(n_jobs=n_jobs)(
        delayed(_gen_method)(
            emodes_masked, evals, groups, method, seed=n) for n in putils.trange(n_rotate)
    )

    return np.asarray(eigen_spins).squeeze()

def spin_modes(emodes, evals, mask=None, n_rotate=1000, eigen_spins=None, num_modes=100, **kwargs):
    """
    compute a permutation matrix for a set of eigenmodes, based on the concept
    of rotating the eigenmodes in the eigenspace per group. Produces a sparse
    permutation matrix of (`n_rotate`, N, M).
    
    Parameters
    ----------
    emodes : np.ndarray of shape (N, M)
        eigenmodes to resample
    evals : np.ndarray of shape (M,)
        eigenvalues corresponding to `emodes`
    n_rotate : int, optional
        number of permutations to compute. Default 1000
    check_duplicates: bool, optional
        Whether to check for and attempt to avoid duplicate resamplings. A
        warnings will be raised if duplicates cannot be avoided. Setting to
        True may increase the runtime of this function! Default: True    
    seed : None or int or np.random.RandomState, optional
        seed for random number generation. Default None
    method : {'original', 'vasa', 'hungarian'}, optional
        Method by which to match non- and rotated coordinates. Default: 'original'
    verbose : bool, optional
        Whether to print occasional status messages. Default: False
    return_cost : bool, optional
        Whether to return cost array (specified as Euclidean distance) for each
        coordinate for each rotation Default: True
        
    Returns
    -------
    resampled_emodes : np.ndarray of shape (`n_rotate`, N, M)
        new eigenmodes matrix after resampling procedure

    Notes
    -----
    While applicable in most circumstances, resampled modes, especially with
    ``method='original'`` set, may result in some inaccuracies of reconstructions
    of surrogates, particularly in the presence of near-white maps or 

    """
    
    if eigen_spins is None:
        eigen_spins = gen_eigensamples(emodes, evals, mask=mask, n_rotate=n_rotate, num_modes=num_modes,
                                       **kwargs)

    emodes_masked = copy.deepcopy(emodes[:, :num_modes])
    emodes_masked = emodes_masked[mask]
    evals = copy.deepcopy(evals[:num_modes])
    resampled_emodes = np.zeros((n_rotate, len(emodes), len(evals)), dtype=emodes.dtype)
    groups = _get_eigengroups(emodes_masked)
    for n in range(n_rotate):
        resampled_masked = np.zeros((len(emodes_masked), len(evals)), dtype=emodes.dtype)
        for j in range(len(groups)):
            resampled_masked[:, groups[j]] = emodes_masked[:, groups[j]][eigen_spins[n, :, j]]
        
        resampled_emodes[n, mask] = resampled_masked
    
    return resampled_emodes

def spin_single(emodes, evals, mask=None, eigen_spins=None, num_modes=100, return_masked=False, **kwargs):
    if eigen_spins is None:
        eigen_spins = gen_eigensamples(emodes, evals, mask=mask, n_rotate=1, num_modes=num_modes,
                                       **kwargs)[0]
        
    emodes_masked = copy.deepcopy(emodes[:, :num_modes])
    emodes_masked = emodes_masked[mask]
    evals = copy.deepcopy(evals[:num_modes])
    resampled_emodes = np.zeros((len(emodes), len(evals)), dtype=emodes.dtype)
    groups = _get_eigengroups(emodes_masked)
    resampled_masked = np.zeros((len(emodes_masked), len(evals)), dtype=emodes.dtype)
    for j in range(len(groups)):
        resampled_masked[:, groups[j]] = emodes_masked[:, groups[j]][eigen_spins[:, j]]
    
    if return_masked is True:
        return resampled_masked
    
    resampled_emodes[mask] = resampled_masked
    
    return resampled_emodes

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

def inflate_mesh(*, vertices=None, faces=None, iterations=10, smoothing_factor=1.,
                 inflation_factor=0.1, normalize=False):
    """
    Inflate the mesh using Connectome Workbench `surface-inflation`
    
    Parameters
    ----------
    (vertices, faces) : mesh-like tuple or str
        Surface to inflate. Can also be a filename to a surface.
    iterations : int, optional
        Number of smoothing iterations, default 10.
    smoothing_factor : float, optional
        Sigma of Gaussian smoothing kernel, default 1.
    inflation_factor : float, optional
        How much to inflate the surface, used for soft 
        correction of surface area distortions, default 0.1.
    normalize : bool, optional
        Normalize surface area of inflated surface to original
        surface. Hard correction of surface area distortions
        (forces surface area of inflated surface to be the same
        as the original - may induce distortions of its own,
        needs more testing). Default False.
    """
    new_vertices = vertices
    for _ in range(iterations):

        normals = compute_normals(vertices, faces)

        new_vertices += normals * inflation_factor

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

def euclidean_distmat(mesh):
    """
    Calculates Euclidean distance matrix of the mesh in `mesh`

    Parameters
    ----------
    mesh : lapy compatible object
        TriaMesh or TetMesh

    Returns
    -------
    ndarray of shape (N,N)
        Euclidean distance matrix of mesh with `N` vertices.

    """
    verts = mesh.v
    
    return squareform(pdist(verts))

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
