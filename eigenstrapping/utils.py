import numpy as np
from . import stats
from netneurotools import datasets as nndata
import os.path as op
from nilearn import image
import sys
import os

import warnings
from collections import OrderedDict
import subprocess

"""
Utility functions (C) Systems Neuroscience Newcastle &
Nikitas C. Koussis 2023
"""

DROP = [  # regions that should always be dropped from analyses
    'lh_unknown', 'rh_unknown',
    'lh_corpuscallosum', 'rh_corpuscallosum',
    'lh_Background+FreeSurfer_Defined_Medial_Wall',
    'rh_Background+FreeSurfer_Defined_Medial_Wall',
]

_ANNOT_DT = ">i4"
"""Data type for Freesurfer `.annot` files.

Used by :func:`read_annot` and :func:`write_annot`.  All data (apart from
strings) in an `.annot` file is stored as big-endian int32.
"""

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
    
# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def _suppress(cmd, shell='True'):
    return subprocess.call(cmd, shell=shell,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.STDOUT)

def _print(cmd, shell='True'):
    output = subprocess.check_output(cmd, shell=shell)
    return output.splitlines()

def is_string_like(obj):
    """ Check whether `obj` behaves like a string. """
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True

def get_schaefer(vers='fsaverage5', data_dir=None, networks='7', scale=400):
    """
    Returns CSchaefer 2018 atlases as dictionary

    Parameters
    ----------
    vers : str, optional
        Which version of the atlases to get. Default: 'fsaverage5'
    data_dir : str or os.PathLike, optional
        Data directory where downloaded atlases should be stored. If not
        specified will default to $NNT_DATA or ~/nnt-data
    networks : {'7', '17'}, optional
        Which networks to get for Schaefer 2018 atlas. Default: '7'

    Returns
    -------
    list of filenames
        Filenames where left [0] and right [1] hemisphere labels are kept
    """
    schaefer = nndata.fetch_schaefer2018(vers, data_dir=data_dir)
    schaefer = {k: schaefer.get(k) for k in schaefer.keys()
                if f'{scale}Parcels7Networks' in k}

    return schaefer[f'{scale}Parcels7Networks']

def parcify(annotation, drop=True):
    """
    Returns an array of parcel labels for every vertex in the surface
    of `annotation`

    Parameters
    ----------
    annotation : str
        FreeSurfer annotation file
    drop : bool, optional
        Whether to drop the medial wall labels. Default True.

    Returns
    -------
    parc : np.ndarray of shape (N,)
        Parcel labels for every vertex

    """
    
    labels, ctab, names = read_annot(annotation)
    parc = labels
    
    if drop:
        drop = np.intersect1d(names, DROP)
        parc = np.delete(labels, [names.index(f) for f in drop])
        
    return parc


def _get_eigengroups(eigs, suppress_message=False):
    """
    Helper function to find eigengroups
    """
    if suppress_message is False:
        print("IMPORTANT: EIGENMODES MUST BE TRUNCATED AT FIRST NON-ZERO MODE FOR THIS FUNCTION TO WORK")
    lam = eigs.shape[1] # number of eigenmodes, accounting for discarded non-zero mode
    l = np.floor((lam-1)/2).astype(int)    
    # Handle cases where lam is from 4 to 7
    if lam == 3:
        return [np.arange(0, 3)]
    elif 4 <= lam < 8:
        return [np.arange(0, 3), np.arange(3, lam)]
    elif lam < 3:
        raise ValueError('Number of modes to resample cannot be less than 3')
    
    groups = []#[np.array([0])]
    ii = 0
    i = 0
    for g in range(1, l):
        ii += 2*g+1
        if ii >= lam:
            groups.append(np.arange(i,lam))
            return groups
        groups.append(np.arange(i,ii))
        i = ii
    

def reconstruct_data(coeffs, eigenmodes):
    """
    Reconstruct a dataset of `n_vertices` given a set of eigenmodes and coeffs
    conditioned on data using ordinary least squares (OLS)

    Parameters
    ----------
    coeffs : np.ndarray of shape (M,)
        Coefficients output from fitting OLS
    eigenmodes : np.ndarray of shape (n_vertices, M)
        Eigenmodes of `n_vertices` by number of eigenvalues M
    pv : float (default 100), rounds up to nearest index
        Percentage of `eigenmodes` to use for reconstruction (default 100%)

    Returns
    -------
    new_data : np.ndarray of (n_vertices,)
        Reconstructed data

    """
    
    coeffs = coeffs.reshape(-1, 1)
    
    new_data = eigenmodes @ coeffs
    
    return new_data.squeeze()
    
def eigen_decomposition(data, eigenmodes, method='matrix'):
    """
    Decompose data using eigenmodes and calculate the coefficient of 
    contribution of each vector
    
    Parameters:
    -----------
    data : np.ndarray of shape (n_vertices, 3)
        N = number of vertices, P = columns of independent data
    eigenmodes : np.ndarray of shape (n_vertices, M)
        N = number of vertices, M = number of eigenmodes
    method : string
        method of calculation of coefficients: 'matrix', 'matrix_separate', 
        'regression'
    
    Returns:
    -------
    coeffs : numpy array of shape (N, 3)
     coefficient values
    
    """
    
    if data.ndim > 1:
        N, P = data.shape
    else:
        P = 1
    
    _, M = eigenmodes.shape
    
    if method == 'matrix':
        #print("Using matrix decomposition to reconstruct data")
        coeffs = np.linalg.solve((eigenmodes.T @ eigenmodes), (eigenmodes.T @ data))
    elif method == 'matrix_separate':
        coeffs = np.zeros((M, P))
        if P > 1:
            for p in range(P):
                coeffs[:, p] = np.linalg.solve((eigenmodes.T @ eigenmodes), (eigenmodes.T @ data[:, p]))
    elif method == 'regression':
        #print("Using regression method to reconstruct data")
        coeffs = np.zeros((M, P))
        if P > 1:
            for p in range(P):
                coeffs[:, p] = np.linalg.lstsq(eigenmodes, data[:, p], rcond=None)[0]
        else:
            coeffs = np.linalg.lstsq(eigenmodes, data, rcond=None)[0]
            
    else:
        raise ValueError("Accepted methods for decomposition are 'matrix', and 'regression'")
                
    return coeffs

def calc_surface_connectivity(data, parc):
    """
    Calculate surface-based connectivity matrix.    

    Parameters
    ----------
    data : ndarray of shape (N,)
        Surface data to calculate connectivity matrix
    parc : ndarray of shape (N,)
        Surface labels for each parcel that corresponds to an atlas
    
    Returns
    -------
    surface_connectivity : np.ndarray of shape (num_parcels, num_parcels)
        Surface connectivity matrix

    """

    parc_data = calc_parcellate(parc, data)
    matrix = stats.pairwise_r(parc_data)

    return matrix
    
    
def compute_axes_ellipsoid(eigenvalues):
    """
    Compute the axes of an ellipsoid given the eigenmodes.
    """    
    return np.sqrt(eigenvalues)
    

def transform_to_spheroid(eigenvalues, eigenmodes):
    """
    Transform the eigenmodes to a spheroid space
    """
    ellipsoid_axes = compute_axes_ellipsoid(eigenvalues)
    #ellipsoid_axes = ellipsoid_axes.reshape(-1, 1)
    
    spheroid_eigenmodes = np.divide(eigenmodes, np.sqrt(eigenvalues))
    
    return spheroid_eigenmodes
    
    
def transform_to_ellipsoid(eigenvalues, eigenmodes):
    """
    Transform the eigenmodes in spheroid space back to ellipsoid by stretching
    """
    
    ellipsoid_axes = compute_axes_ellipsoid(eigenvalues)
    
    ellipsoid_eigenmodes = np.multiply(eigenmodes, np.sqrt(eigenvalues))
    
    return ellipsoid_eigenmodes

def calc_parcellate(parc, data_input):
    """
    Calculate parcellated data given (N,) parcellation for every vertex of shape 
    (N,)

    Parameters
    ----------
    parc : str or np.ndarray (N,)
        Parcellation (np.ndarray of n_labels)
    data_input : np.ndarray (N, M)
        Data to parcellate

    Returns
    -------
    data_parcellated : (n_labels, M)
        Parcellated data

    """

    if is_string_like(parc):
        _, ext = op.splitext(parc)
        if ext == '.annot':
            parc = parcify(parc)
        elif ext == '.txt':
            parc = np.loadtxt(parc)
        elif ext == '.gii':
            parc = image.load_img(parc).agg_data()
        else:
            raise ValueError('Cannot read input parcellation')
            
    num_vertices = parc.shape[0]
    parcels = np.unique(parc[parc > 0])
    num_parcels = len(parcels)
    
    if data_input.ndim == 1:
        data_input = data_input.reshape(-1,1)

    if data_input.shape[0] != num_vertices:
        data_input = data_input.T

    data_parcellated = np.zeros((num_parcels, data_input.shape[1]))

    for parcel_ind in range(num_parcels):
        parcel_interest = parcels[parcel_ind]

        ind_parcel = np.where(parc == parcel_interest)[0]

        data_parcellated[parcel_ind] = np.nanmean(data_input[ind_parcel])

    return data_parcellated.squeeze()

def parcellate_distmat(D, parc):
    """
    Parcellate a dense distance matrix

    Parameters
    ----------
    D : np.ndarray of shape (N, N)
        Distance matrix of surface
    parc : np.ndarray of int of shape (N,)
        Vector of labels (ints) for each vertex on surface with `num_parcels`
        number of labels

    Returns
    -------
    D : ndarray of shape (num_parcels, num_parcels)
        Parcellated distance matrix

    """
    
    D = np.row_stack([
        calc_parcellate(parc, D[row]) for row in range(D.shape[0])
    ])
    D[np.diag_indices_from(D)] = 0
    D = D[1:, 1:] + D[1:, 1:].T
    
    return D

def normalize_data(data):
    data_normalized = np.subtract(data, np.nanmean(data, axis=0))
    data_normalized = np.divide(data_normalized, np.nanstd(data_normalized, axis=0))
    
    return data_normalized

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