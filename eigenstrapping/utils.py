import numpy as np
from . import spatial, parutils, stats
from .geometry import read_annot
from joblib import Parallel, delayed, dump, load
from pathlib import Path
from netneurotools import datasets as nndata
import os.path as op
from nilearn import image
from scipy.sparse import csr_matrix


"""
Eigenmode helper functions (C) Systems Neuroscience Newcastle &
Nikitas C. Koussis 2023
"""

DROP = [  # regions that should always be dropped from analyses
    'lh_unknown', 'rh_unknown',
    'lh_corpuscallosum', 'rh_corpuscallosum',
    'lh_Background+FreeSurfer_Defined_Medial_Wall',
    'rh_Background+FreeSurfer_Defined_Medial_Wall',
]

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


def resample_spheroid(spheroid_eigenmodes, angle):
    """
    Resample the N-D hypersphere generated by the N orthogonal unit modes

    """    
    # initialize the new points p
    p = spheroid_eigenmodes * np.cos(angle)
    
    new_modes = p
    
    return new_modes

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

def calc_moran(dist, nulls, medmask, n_jobs=1):
    """
    Calculates Moran's I for every column of `nulls`

    Parameters
    ----------
    dist : (N, N) array_like
        Hemispheric distance matrix
    nulls : (N, P) array_like
        Null brain maps for which to compute Moran's I
    medmask : (N,) array_like
        Medial wall mask array, where medial wall values are 0
    n_jobs : int, optional
        Number of parallel workers to use for calculating Moran's I. Default: 1

    Returns
    -------
    moran : (P,) np.ndarray
        Moran's I for `P` null maps
    """

    # calculate moran's I, masking out NaN values for each null (i.e., the
    # rotated medial wall)
    fn = dump(dist, spatial.make_tmpname('.mmap', prefix='/tmp/'))[0]
    dist = load(fn, mmap_mode='r')
    
    #do some pre-calculation on our distance matrix to reduce computation time
    with np.errstate(divide='ignore', invalid='ignore'):
        dist = 1 / dist
        np.fill_diagonal(dist, 0)
        dist /= dist.sum(axis=-1, keepdims=True)
        
    if medmask is None:
        medmask = np.isnan(nulls[:, 0])
        
    moran = np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(_moran)(dist, nulls[:, n], medmask)
            for n in parutils.trange(nulls.shape[-1], desc="Running Moran's I")
        )
    )

    Path(fn).unlink()
    return moran

def _moran(dist, sim, medmask):
    mask = np.logical_and(medmask, np.logical_not(np.isnan(sim)))
    return spatial.morans_i(dist[np.ix_(mask, mask)], sim[mask],
                            normalize=False, invert_dist=False)
