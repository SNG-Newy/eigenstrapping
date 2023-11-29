# -*- coding: utf-8 -*-
"""
Functions for calculating and manipulating spatial autocorrelation
Author: Ross Markello, netneurolab
"""

import os
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt

import nibabel as nib
import numpy as np
from scipy import fftpack, stats as sstats
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.special import erfc

from netneurotools.datasets import make_correlated_xy
from netneurotools.freesurfer import check_fs_subjid
from netneurotools.utils import run
from neuromaps import transforms, datasets

from brainspace import mesh

VOL2SURF = 'mri_vol2surf --src {} --out {} --hemi {} --mni152reg ' \
           '--trgsubject fsaverage5 --projfrac 0.5 --interp nearest'
MSEED = 4294967295

def estimate_ex_params(x, initial_params=[0., 1., 1.], plot=False):
    """
    Estimate the parameters of an ex-Gaussian distribution from given `data`
    where the PDF of the distribution is
    
                          tau       tau
    p(x|mu, sigma, tau) = --- exp ( --- (2 * mu + tau * sigma**2 - 2 * x) ) *
                           2         2
                                  mu + tau * sigma ** 2 - x
                          erfc( -------------------------- ),
                                      sqrt(2) * sigma
                        
                        where erfc = 1 - erf.
                        
    Uses negative log-likelihood minimization.
    
    Parameters
    ----------
    x : np.ndarray of shape (N,)
        Data array to estimate parameters, must be 1D
    initial_params : list or tuple, optional
        Initial guesses for parameters mu, sigma, and tau. Default is 
        [0., 1., 1.].
    plot : bool, optional
        Flag to plot data with overlay curve estimated by parameters. Default 
        is False.

    Returns
    -------
    np.ndarray of shape (3,)
        Parameter estimates for mu, sigma, and tau.

    """
    
    # Mask invalid values
    masked_data = np.ma.masked_invalid(x)
    
    result = minimize(_neg_log_likelihood,
                      x0=initial_params,
                      method='Nelder-Mead',
                      args=(masked_data)
            )
    
    mu, sigma, tau = result.x
    
    if plot:
        x_values = np.linspace(np.min(masked_data), np.max(masked_data), 1000)
        pdf = ex_gaussian_pdf(x_values, mu, sigma, tau)
        fig = plt.figure(figsize=(10, 6), constrained_layout=False)
        ax = fig.add_axes([0.12, 0.15, 0.8, 0.77])
        
        ax.plot(x_values, pdf, label=f'ex-Gaussian PDF ($\mu$={mu:.3f},\n$\sigma$={sigma:.3f}, $\tau$={tau:.3f})', color='blue')
        ax.hist(masked_data.compressed(), bins=50, density=True, alpha=0.6, color='gray', label='Data')
        ax.set_xlabel('Value', fontsize=20)
        ax.set_ylabel('Density', fontsize=20)
        leg = ax.legend(loc=0, fontsize=20)
        leg.get_frame().set_linewidth(0.0)
        plt.show()        
    
    return np.array((mu, sigma, tau))
    
def _neg_log_likelihood(params, data):
    mu, sigma, tau = params
    return -np.sum(np.ma.log(ex_gaussian_pdf(data, mu, sigma, tau)))    

def ex_gaussian_pdf(x, mu, sigma, tau):
    return (tau /2) * np.exp((tau / 2) * (2 * mu + tau * sigma**2 - 2 * x)) * erfc((mu + tau * sigma**2 - x) / (np.sqrt(2) * sigma))

def bump_function(coords, vertex, radius=4., value=1.):
    """
    Given a list of 3D coordinates, this bump function sets the value at 
    `vertex` in a 1D array to `value`. It also sets the value in the 1D array 
    to `value` for any vertex that is within the specified `radius` from the 
    vertex at initial `vertex`.

    Parameters
    ----------
    coords : np.ndarray of shape (N, 3)
        List of 3D coordinates
    vertex : int
        Index of centroid of bump function
    distance : float, optional
        Radius of bump function. Default is 4
    value : float, optional
        Value to substitute for bump function

    Returns
    -------
    bump_array : np.ndarray of shape (N,)
        Output bump function.

    """
    
    num_vertices = coords.shape[1]
    bump_array = np.zeros(num_vertices)
    
    # set initial vertex to value
    bump_array[vertex] = value
    vertex_coords = coords[vertex]
    
    for i, vert in enumerate(coords):
        # Calculate the Euclidean distance between the two points
        dist = np.linalg.norm(vertex_coords - vert)
        if dist <= radius:
            bump_array[i] = value
    
    return bump_array

def _objective(alpha, x, y_obs):
    y_pred = x ** -alpha
    res = y_obs - y_pred
    return np.sum(res**2)

def fit_alpha(p, coeffs, fmin=5.):
    """
    Fit power spectrum with a model specified in `model` to estimate
    exponent of spatial autocorrelation (by Wiener-Khinchin theorem).
    Minimizes the objective function (sum of squared residuals).

    Parameters
    ----------
    p : np.ndarray of shape (num_modes,)
        Modal power spectrum
    coeffs : np.ndarray of shape (num_modes,)
        Mode coefficients
    model : method
        Function to fit data. e.g., np.exp(-alpha * p)
    fmin : float
        Minimum spatial frequency to fit (in mm). Default 5.0

    Returns
    -------
    est_alpha : float
        Estimated alpha
    cov : np.ndarray
        Covariance for goodness of fit

    """
    # spatial frequencies
    num_modes = p.shape[0]
    x = np.linspace(fmin, np.max(np.abs(coeffs)), num_modes)
    
    params = minimize(_objective, x0=[1.0], args=(x, p))
    est_alpha = params.x[0]
    
    return est_alpha
    

def map_data_to_grid(points, faces, data, mask=None, spacing=0.5, k=1, 
                     avg_method='mean', eps=0, n_jobs=1):
    """
    Map data from a (cortical or subcortical) mesh of elements ``points``
    and ``faces`` to a square 2D matrix. Uses k-nearest neighbors tree query
    approach.

    Parameters
    ----------
    points : np.ndarray of shape (N,3)
        Mesh coordinates of ``data``
    faces : np.ndarray of shape (F,3)
        Array of face connectivity of mesh
    data : np.ndarray of shape (N,) or (N,M)
        Data to resample to grid, must have the same number of rows as
        ``points`` but can have arbitrarily many columns (data points)
    mask : np.ndarray of bool of shape (N,)
        Mask of surface mesh to exclude from final resampling. Must be an array
        of np.ndarray of bool of True (to keep) and False (to exclude) for each
        index in ``points``
    spacing : non-negative float, optional
        Spacing in mm of grid to resample data. Determines density of resulting
        grid.
    k : int, optional
        Number of neighbors to resample. Default 1
    avg_method : str, optional
        If k is larger than 1, use ``avg_method`` to resample reweighted data
        to grid. Accepts "mean", "weighted_mean", "mode", or "weighted_mode".
    eps : non-negative float, optional
        Error tolerance between k nearest distances on grid and points. Default 0
    n_jobs : int, optional
        Number of workers for tree search. Default 1        

    Returns
    -------
    grid_values : np.ndarray of shape (size, size) if data was vector or
        np.ndarray of shape (M, size, size) if data was array
        
        Interpolated values of ``data`` on 2D grid for each column in ``data``

    """
    m = mesh.mesh_creation.build_polydata(points, cells=faces)
    
    if mask is not None:
        masked = mesh.mesh_operations.mask_points(m, mask)
        data = mesh.array_operations.resample_pointdata(m, masked, data=data)
        m = masked
    
    # get boundary
    bp = mesh.mesh_elements.get_boundary_points(m)
    
    # use min and max to define grid
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    x = np.arange(min_x, max_x, spacing)
    y = np.arange(min_y, max_y, spacing)
    grid_x, grid_y = np.meshgrid(x, y)
    
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    
    # use query tree to find k-nearest neighbors
    tree = cKDTree(points, leafsize=20, compact_nodes=False,
                   copy_data=False, balanced_tree=False)
    d, idx = tree.query(grid_points, k=k, eps=eps, workers=n_jobs,
                        distance_upper_bound=eps+np.finfo(np.float64).eps)

    weights = 1.0 / d
    
    new_data = np.zeros(data.shape)
    
    for i in range(data.shape[1]):
        feat = data[:, i]
        
        if k == 1:
            f = feat[idx]
        elif avg_method == 'mean':
            f = np.nanmean(feat[idx], axis=1)
        elif avg_method == 'weighted_mean':
            f = np.average(feat[idx], weights=weights, axis=1)
        elif avg_method == 'mode':
            f = sstats.mode(feat[idx], axis=1)[0].squeeze()
        elif avg_method == 'weighted_mode':
            f = sstats.weighted_mode(feat[idx], weights=weights, axis=1)[0].squeeze()
            f = f.astype(data.dtype)
        else:
            raise ValueError('Unknown avg_method: {0}'.format(str(avg_method)))
           
    if data.ndim > 1:
        grid_values = new_data.reshape(i, grid_x.shape[0], grid_x.shape[1])
        return grid_values
    
    grid_values = new_data.reshape(grid_x.shape)
    
    return grid_values

def morans_i(dist, y, normalize=False, local=False, invert_dist=True):
    """
    Calculates Moran's I from distance matrix `dist` and brain map `y`

    Parameters
    ----------
    dist : (N, N) array_like
        Distance matrix between `N` regions / vertices / voxels / whatever
    y : (N,) array_like
        Brain map variable of interest
    normalize : bool, optional
        Whether to normalize rows of distance matrix prior to calculation.
        Default: False
    local : bool, optional
        Whether to calculate local Moran's I instead of global. Default: False
    invert_dist : bool, optional
        Whether to invert the distance matrix to generate a weight matrix.
        Default: True

    Returns
    -------
    i : float
        Moran's I, measure of spatial autocorrelation
    """

    # convert distance matrix to weights
    if invert_dist:
        with np.errstate(divide='ignore'):
            dist = 1 / dist
    np.fill_diagonal(dist, 0)

    # normalize rows, if desired
    if normalize:
        dist /= dist.sum(axis=-1, keepdims=True)

    # calculate Moran's I
    z = y - y.mean()
    if local:
        with np.errstate(all='ignore'):
            z /= y.std()

    zl = np.squeeze(dist @ z[:, None])
    den = (z * z).sum()

    if local:
        return (len(y) - 1) * z * zl / den

    return len(y) / dist.sum() * (z * zl).sum() / den


def _fftind(x, y, z):
    """
    Return 3D shifted Fourier coordinates

    Returned coordinates are shifted such that zero-frequency component of the
    square grid with shape (x, y, z) is at the center of the spectrum

    Parameters
    ----------
    {x,y,z} : int
        Size of array to be generated

    Returns
    -------
    k_ind : (3, x, y, z) np.ndarray
        Shifted Fourier coordinates, where:
            k_ind[0] : k_x components
            k_ind[1] : k_y components
            k_ind[2] : k_z components

    Notes
    -----
    See scipy.fftpack.fftshift

    References
    ----------
    Based on code from Burt et al., 2020, NeuroImage
    """

    k_ind = np.mgrid[:x, :y, :z]
    zero = np.array([int((n + 1) / 2) for n in [x, y, z]])
    while zero.ndim < k_ind.ndim:
        zero = np.expand_dims(zero, -1)
    k_ind = fftpack.fftshift(k_ind - zero)

    return k_ind


def gaussian_random_field(x, y, z, noise=None, alpha=3.0, normalize=True,
                          seed=None):
    """
    Generate a Gaussian random field with k-space power law |k|^(-alpha/2).

    Parameters
    ----------
    {x,y,z} : int
        Grid size of generated field
    noise : (x, y, z) array_like, optional
        Noise array to which gaussian smoothing is added. If not provided an
        array will be created by drawing from the standard normal distribution.
        Default: None
    alpha : float (positive), optional
        Power (exponent) of the power-law distribution. Default: 3.0
    normalize : bool, optional
        Normalize the returned field to unit variance. Default: True
    seed : None, int, default_rng, optional
        Random state to seed `noise` generation. Default: None

    Returns
    -------
    gfield : (x, y, z) np.ndarray
        Realization of Gaussian random field

    References
    ----------
    Based on code from Burt et al., 2020, NeuroImage
    """

    rs = np.random.default_rng(seed)

    if not alpha:
        return rs.normal(size=(x, y, z))

    assert alpha > 0

    # k-space indices
    k_idx = _fftind(x, y, z)

    # define k-space amplitude as a power law 1/|k|^(alpha/2)
    amplitude = np.power(np.sum([k ** 2 for k in k_idx], axis=0) + 1e-10,
                         -alpha / 2.0)
    amplitude[0, 0, 0] = 0  # remove zero-freq mean shit

    # generate a complex gaussian random field where phi = phi_1 + i*phi_2
    if noise is None:
        noise = rs.normal(size=(x, y, z))
    elif noise.shape != (x, y, z):
        try:
            noise = noise.reshape(x, y, z)
        except ValueError:
            raise ValueError('Provided noise cannot be reshape to target: '
                             f'({x}, {y}, {z})')

    # transform back to real space
    gfield = np.fft.ifftn(np.fft.fftn(noise) * amplitude).real

    if normalize:
        return (gfield - gfield.mean()) / gfield.std()

    return gfield


def make_tmpname(suffix, prefix=None):
    """
    Stupid helper function because :man_shrugging:

    Parameters
    ----------
    suffix : str
        Suffix of created filename

    Returns
    -------
    fn : str
        Temporary filename; user is responsible for deletion
    """

    # I don't want to deal with a bunch of nested tempfile.NameTemporaryFile
    # in the create_surface_grf() function so this is the easiest way to do
    # things that's safe from race conditions :man_shrugging:

    fd, fn = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(fd)

    return fn

def make_decimated_surfaces(initial_surface, per=0.1, medial=None, 
                            outdir="datasets/decimated", prefix='decsurf', n_jobs=1):
    """
    Creates decimated surfaces based on ``initial_mesh`` using brainspace.

    Parameters
    ----------
    initial_surface : str
        Path to surface to decimate.
    medial : str
        Path to medial wall mask of .txt, where 1 corresponds to non-medial wall
        and 0 corresponds to medial wall.
    per : float, optional
        "Percentage" of vertices to remove each step, where 0.0 < ``per`` < 1.0.
        The default is 0.1.
    outdir : str, optional
        Output directory for surfaces. Default is "datasets/decimated"
    prefix : str, optional
        Prefix of the decimated surfaces. Default is "decsurf".
    n_jobs : int, optional
        Number of workers to use for decimation. Default is 1.

    Returns
    -------
    surface_fnames : list of str
        List of absolute paths of decimated surfaces
    medial_fnames : list of str
        List of absolute paths of resampled medial wall masks 
        (if ``medial`` is given)

    """
    
    outdir = Path(outdir)
    prefix = str(prefix)
    surface = mesh.mesh_io.read_surface(initial_surface)
    surface_fnames = []
    if medial:
        medial = np.loadtxt(medial)
        medial_fnames = []
    
    if per <= 0.0 or per >= 1.0:
        raise ValueError('Percentage of vertices to remove must be between 0 and 1')
    
    n_steps = np.int(np.floor(1.0 / per))
    
    for step in range(n_steps):
        factor = 1.0 - per*step
        if factor == 1.0:
            n_vertices = surface.GetNumberOfPoints()
            fn = os.path.join(outdir, f'{prefix}_verts={n_vertices}.gii')
            surface_fnames.append(os.path.abspath(fn))
            mesh.mesh_io.write_surface(surface, fn, otype='gii')
            if medial is not None:
                medial_fn = os.path.join(outdir, f'{prefix}_medial_verts={n_vertices}.txt')
                np.savetxt(medial_fn, medial, fmt='%d')
                medial_fnames.append(os.path.abspath(medial_fn))
            continue
                
        # mask surface
        new_surface_mask = mesh.mesh_cluster.sample_points_decimation(surface,
                                keep=factor, n_jobs=n_jobs)
        new_surface = mesh.mesh_operations.mask_points(surface, new_surface_mask)
        n_vertices = new_surface.GetNumberOfPoints()
        fn = os.path.join(outdir, f'{prefix}_verts={n_vertices}.gii')
        surface_fnames.append(os.path.abspath(fn))
        mesh.mesh_io.write_surface(new_surface, fn, otype='gii')
        
        if medial is not None:
            medial_out = mesh.array_operations.resample_pointdata(surface, new_surface,
                                data=medial, append=False)
            medial_fn = os.path.join(outdir, f'{prefix}_medial_verts={n_vertices}.txt')
            np.savetxt(medial_fn, medial_out, fmt='%d')
            medial_fnames.append(os.path.abspath(medial_fn))
            
    
    if medial is not None:
        return surface_fnames, medial_fnames
    
    return surface_fnames

def make_correlated_exgaussian(corr=0.15, size=10000, seed=None, tol=0.005, mu=0, sigma=1, tau=0.3):
    """
    Generates two exGaussian random vectors that are correlated at approximately the specified correlation value.

    Parameters
    ----------
    corr : float, optional
        The approximate correlation desired between the two generated exGaussian vectors. Default: 0.15
    size : int, optional
        Desired size of the generated vectors. Default: 10000
    seed : int or np.random.RandomState instance or None, optional
        Seed for random number generation. Default: None
    tol : float, optional
        Tolerance of correlation between generated vectors and specified `corr`. Default: 0.005
    mu : float, optional
        Mean of the Gaussian component for the exGaussian distribution. Default: 0
    sigma : float, optional
        Standard deviation of the Gaussian component for the exGaussian distribution. Default: 1
    tau : float, optional
        Mean and standard deviation of the exponential component for the exGaussian distribution. Default: 0.3

    Returns
    -------
    x_exgaussian, y_exgaussian : numpy.ndarray
        Two random vectors of size `size` with exGaussian distributions and correlation specified by `corr`.

    Examples
    --------
    >>> x, y = make_correlated_exgaussian(corr=0.2, seed=42)
    >>> np.corrcoef(x, y)[0, 1]
    0.2003...
    """
    
    # Generate two correlated vectors using make_correlated_xy
    rs = np.random.RandomState(seed)
    x, y = make_correlated_xy(corr=corr, size=size, seed=seed, tol=tol)
    
    # Transform each vector from normal to exgaussian using quantile transformation
    exgauss_samples = [exgaussian(rs, mu, sigma, tau) for _ in range(size)]
    exgauss_samples.sort()
    
    x_exgaussian = [exgauss_samples[int(rank)] for rank in sstats.rankdata(x) - 1]
    y_exgaussian = [exgauss_samples[int(rank)] for rank in sstats.rankdata(y) - 1]
    
    return np.array(x_exgaussian), np.array(y_exgaussian)

def exgaussian(rs, mu=0, sigma=1, tau=0.3):
    "Generate an exGaussian random variable"
    gaussian_component = rs.normal(mu, sigma)
    exponential_component = rs.exponential(tau)
    return gaussian_component + exponential_component

def exgaussian_field(x, y, z, alpha, noise=None, normalize=True, seed=None, 
                     mu=0., sigma=1., tau=0.3):
    """
    Generate a ex-Gaussian random field with k-space power law |k|^(-alpha/2).

    Parameters
    ----------
    {x,y,z} : int
        Grid size of generated field
    noise : (x, y, z) array_like, optional
        Noise array to which gaussian smoothing is added. If not provided an
        array will be created by drawing from the standard normal distribution.
        Default: None
    alpha : float (positive), optional
        Power (exponent) of the power-law distribution. Default: 3.0
    normalize : bool, optional
        Normalize the returned field to unit variance. Default: True
    seed : None, int, default_rng, optional
        Random state to seed `noise` generation. Default: None

    Returns
    -------
    gfield : (x, y, z) np.ndarray
        Realization of Gaussian random field

    References
    ----------
    Based on code from Burt et al., 2020, NeuroImage
    """
    
    rs = np.random.default_rng(seed)
    # k-space indices
    k_idx = _fftind(x, y, z)

    # define k-space amplitude as a power law 1/|k|^(alpha/2)
    amplitude = np.power(np.sum([k ** 2 for k in k_idx], axis=0) + 1e-10,
                         -alpha / 2.0)
    amplitude[0, 0, 0] = 0  # remove zero-freq mean shift

    # generate a complex ex-Gaussian random field where phi = phi_1 + i*phi_2
    if noise is None:
        noise = np.array([exgaussian(rs, mu=mu, sigma=sigma, tau=tau) for _ in range(x*y*z)]).reshape(x, y, z)
    elif noise.shape != (x, y, z):
        try:
            noise = noise.reshape(x, y, z)
        except ValueError:
            raise ValueError('Provided noise cannot be reshaped to target: '
                             f'({x}, {y}, {z})')

    # transform back to real space
    gfield = np.fft.ifftn(np.fft.fftn(noise) * amplitude).real
    
    if normalize:
        return (gfield - gfield.mean()) / gfield.std()

    return gfield

def create_surface_exgaussian(noise=None, alpha=3.0, normalize=True, seed=None,
                       medial_val=None, mu=0., sigma=1., tau=0.3, hemis=['lh']):
    """
    Generates ex-GRF on surface (fsaverage5)

    Uses exgaussian_field() and mri_vol2surf to generate GRF

    Parameters
    ----------
    noise : (x, y, z) array_like, optional
        Noise array to which gaussian smoothing is added. If not provided an
        array will be created by drawing from the standard normal distribution.
        Default: None
    alpha : float (positive), optional
        Power (exponent) of the power-law distribution. Default: 3.0
    normalize : bool, optional
        Normalize the returned field to unit variance. Default: True
    seed : None, int, default_rng, optional
        Random state to seed GRF. Default: None

    Returns
    -------
    data : (20484,) np.ndarray
        Surface representation of GRF
    """

    affine = np.eye(4) * 2
    affine[:, -1] = [-90, -90, -72, 1]

    gfield = exgaussian_field(91, 109, 91, noise=noise, alpha=alpha,
                                   normalize=normalize, seed=seed,
                                   mu=mu, sigma=sigma, tau=tau)
    fn = make_tmpname(suffix='.nii.gz')
    nib.save(nib.nifti1.Nifti1Image(gfield, affine), fn)
    
    if len(hemis) < 2:
        outname = make_tmpname(suffix='.mgh')
        run(VOL2SURF.format(fn, outname, hemis[0]), quiet=True)
        data = nib.load(outname).get_fdata().squeeze()
        os.remove(outname)
    else:
        data = np.zeros((20484,))
        for n, hemi in enumerate(hemis):
            outname = make_tmpname(suffix='.mgh')
            run(VOL2SURF.format(fn, outname, hemi), quiet=True)
            sl = slice(len(data) // 2 * n, len(data) // 2 * (n + 1))
            data[sl] = nib.load(outname).get_fdata().squeeze()
            os.remove(outname)

    os.remove(fn)

    if medial_val is not None:
        data = _mod_medial(_mod_medial(data, True), False, medial_val)

    return data

def create_surface_grf(noise=None, alpha=3.0, normalize=True, seed=None,
                       medial_val=None):
    """
    Generates GRF on surface (fsaverage5)

    Uses gaussian_random_field() and mri_vol2surf to generate GRF

    Parameters
    ----------
    noise : (x, y, z) array_like, optional
        Noise array to which gaussian smoothing is added. If not provided an
        array will be created by drawing from the standard normal distribution.
        Default: None
    alpha : float (positive), optional
        Power (exponent) of the power-law distribution. Default: 3.0
    normalize : bool, optional
        Normalize the returned field to unit variance. Default: True
    seed : None, int, default_rng, optional
        Random state to seed GRF. Default: None

    Returns
    -------
    data : (20484,) np.ndarray
        Surface representation of GRF
    """

    affine = np.eye(4) * 2
    affine[:, -1] = [-90, -90, -72, 1]

    gfield = gaussian_random_field(91, 109, 91, noise=noise, alpha=alpha,
                                   normalize=normalize, seed=seed)
    fn = make_tmpname(suffix='.nii.gz')
    nib.save(nib.nifti1.Nifti1Image(gfield, affine), fn)

    data = np.zeros((20484,))
    for n, hemi in enumerate(('lh', 'rh')):
        outname = make_tmpname(suffix='.mgh')
        run(VOL2SURF.format(fn, outname, hemi), quiet=True)
        sl = slice(len(data) // 2 * n, len(data) // 2 * (n + 1))
        data[sl] = nib.load(outname).get_fdata().squeeze()
        os.remove(outname)

    os.remove(fn)

    if medial_val is not None:
        data = _mod_medial(_mod_medial(data, True), False, medial_val)

    return data

def create_decimated_grf(surface, hemi='L', noise=None, alpha=3.0, normalize=True, seed=None,
                         medial=None):
    """
    Generates GRF on surface given by brainspace decimation. (1) resamples
    from MNI152 2mm space to fsLR 164k standard space, then (2) resamples GRF 
    to decimated surface in ``surface``. Makes medial wall vertices in ``data`` 
    equal to NaN if not None.

    Parameters
    ----------
    surface : str
        Decimated surface, must be single hemisphere
    hemi : str
        Hemisphere that decimated surface is in. Must be 'L' or 'R', default is 'L'
    noise : (x, y, z) array_like, optional
        Noise array to which gaussian smoothing is added. If not provided an
        array will be created by drawing from the standard normal distribution.
        Default: None
    alpha : float (positive), optional
        Power (exponent) of the power-law distribution. Default: 3.0
    normalize : bool, optional
        Normalize the returned field to unit variance. Default: True
    seed : None, int, default_rng, optional
        Random state to seed GRF. Default: None
    medial : str
        Medial wall mask where medial wall is 0 and everything else is 1

    Returns
    -------
    data : (n_vertices,) np.ndarray
        Surface representation of GRF where ``n_vertices`` is the number of
        vertices in the input ``surface``. If ``medial`` is given, np.nan values
        are inserted into corresponding indices.

    """
    
    affine = np.eye(4) * 2
    affine[:, -1] = [-90, -90, -72, 1]
    
    gfield = gaussian_random_field(91, 109, 91, noise=noise, alpha=alpha,
                                   normalize=normalize, seed=seed)
    fn = make_tmpname(suffix='.nii.gz')
    nib.save(nib.nifti1.Nifti1Image(gfield, affine), fn)
    
    surface = mesh.mesh_io.read_surface(surface)
    
    if hemi == 'L':
        gfield = transforms.mni152_to_fslr(fn, fslr_density='164k')[0].agg_data()
        standard = str(datasets.fetch_fslr(density='164k', verbose=0)['midthickness'][0])
    elif hemi == 'R':
        gfield = transforms.mni152_to_fslr(fn, fslr_density='164k')[1].agg_data()
        standard = str(datasets.fetch_fslr(density='164k', verbose=0)['midthickness'][1])
    else:
        raise ValueError("Hemi must be 'L' or 'R'")
    
    standard = mesh.mesh_io.read_surface(standard)
    data = mesh.array_operations.resample_pointdata(surface, standard,
                        data=gfield, append=False)
    
    if medial:
        medial = np.loadtxt(medial).astype(np.bool_)
        medial_mask = np.logical_not(medial)
        data[medial_mask] = np.nan
    
    os.remove(fn)
    
    return data


def matching_multinorm_decimated_grfs(surface, corr, hemi='L', medial=None, 
                                      tol=0.005, alpha=3.0, normalize=True, seed=None):
    """
    Generates two surface GRFs that correlate at r = `corr`

    Starts by generating two random variables from a multivariate normal
    distribution with correlation `corr`, adds spatial autocorrelation with
    specified `alpha`, and projects to the surface. Continues this procedure
    until two variables are generated that have correlation `corr` on the
    surface.

    Parameters
    ----------
    surface : str
        Decimated surface, must be single hemisphere
    corr : float
        Desired correlation of generated GRFs
    hemi : str
        Hemisphere that decimated surface is in. Must be 'L' or 'R', default is 'L'
    tol : float
        Tolerance for correlation between generated GRFs
    alpha : float (positive), optional
        Exponent of the power-law distribution. Only used if `use_gstools` is
        set to False. Default: 3.0
    normalize : bool, optional
        Whether to normalize the returned field to unit variance. Default: True
    seed : None, int, default_rng, optional
        Random state to seed GRF generation. Default: None

    Returns
    -------
    xs, ys : (n_vertices)

    """
    
    rs = np.random.default_rng(seed)
    acorr = np.inf
    
    surface_mesh = mesh.mesh_io.read_surface(surface)
    
    while np.abs(np.abs(acorr) - corr) > tol:
        if alpha > 0:
            x, y = make_correlated_xy(corr, size=902629,
                                      seed=rs.integers(MSEED))
            # smooth and project
            xs = create_decimated_grf(surface, hemi=hemi, noise=x, 
                                      alpha=alpha, normalize=normalize,
                                      medial=medial)
            ys = create_decimated_grf(surface, hemi=hemi, noise=y, 
                                      alpha=alpha, normalize=normalize,
                                      medial=medial)
            
            if medial is not None:
                # check if removing medial wall has altered correlation or if
                # x and y remain sufficiently correlated
                acorr = np.corrcoef(xs[np.logical_not(np.isnan(xs))], ys[np.logical_not(np.isnan(ys))])[0, 1]
            else:
                acorr = np.corrcoef(xs, ys)[0, 1]
            
        else:
            xs, ys = make_correlated_xy(corr, size=surface_mesh.GetNumberOfPoints(),
                                        seed=rs.integers(MSEED))
    
    if acorr < 0:
        ys *= -1
        
    if normalize:
        xs, ys = sstats.zscore(xs), sstats.zscore(ys)
        
    return xs, ys

def _mod_medial(data, remove=True, val=0):
    """
    Removes (inserts) medial wall from (into) `data` from fsaverage5 surface

    Parameters
    ----------
    data : (20484,) array_like
        Surface data
    remove : bool, optional
        Whether to remove medial wall instead of inserting it. Assumes input
        has (does not have) medial wall. Default: True
    val : float, optional
        What value to insert if `remove=False`. Default: 0

    Returns
    -------
    out : np.ndarray
        Provided surface `data` with medial wall removed/inserted
    """
    if len(data) <= 10242:
        subj, path = check_fs_subjid('fsaverage5')
        lh = [
            nib.freesurfer.read_label(
                os.path.join(path, subj, 'label', f'{h}.Medial_wall.label')
            )
            for h in (['lh'])
        ]
        lhm = np.ones(10242, dtype=bool)
        lhm[lh] = False
    
        if remove:
            return data[lhm]
        else:
            xd = np.ones(10242) * val
            xd[lhm] = data
            return xd
    else:
        subj, path = check_fs_subjid('fsaverage5')
        lh, rh = [
            nib.freesurfer.read_label(
                os.path.join(path, subj, 'label', f'{h}.Medial_wall.label')
            )
            for h in ('lh', 'rh')
        ]
        lhm, rhm = np.ones(10242, dtype=bool), np.ones(10242, dtype=bool)
        lhm[lh], rhm[rh] = False, False
    
        if remove:
            x, y = np.split(data, 2)
            return np.hstack((x[lhm], y[rhm]))
        else:
            x, y = np.split(data, [np.sum(lhm)])
            xd, yd = np.ones(10242) * val, np.ones(10242) * val
            xd[lhm], yd[rhm] = x, y
            return np.hstack((xd, yd))


def matching_multinorm_grfs(corr, tol=0.005, *, alpha=3.0, normalize=True,
                            seed=None, debug=False):
    """
    Generates two surface GRFs (fsaverage5) that correlate at r = `corr`

    Starts by generating two random variables from a multivariate normal
    distribution with correlation `corr`, adds spatial autocorrelation with
    specified `alpha`, and projects to the surface. Continues this procedure
    until two variables are generated that have correlation `corr` on the
    surface.

    Parameters
    ----------
    corr : float
        Desired correlation of generated GRFs
    tol : float
        Tolerance for correlation between generated GRFs
    alpha : float (positive), optional
        Exponent of the power-law distribution. Only used if `use_gstools` is
        set to False. Default: 3.0
    normalize : bool, optional
        Whether to normalize the returned field to unit variance. Default: True
    seed : None, int, default_rng, optional
        Random state to seed GRF generation. Default: None
    debug : bool, optional
        Whether to print debug info

    Return
    ------
    x, y : (20484,) np.ndarray
        Generated surface GRFs
    """

    rs = np.random.default_rng(seed)

    acorr, n = np.inf, 0
    while np.abs(np.abs(acorr) - corr) > tol:
        if alpha > 0:
            x, y = make_correlated_xy(corr, size=902629,
                                      seed=rs.integers(MSEED))
            # smooth correlated noise vectors + project to surface
            xs = create_surface_grf(noise=x, alpha=alpha, normalize=normalize)
            ys = create_surface_grf(noise=y, alpha=alpha, normalize=normalize)
        else:
            xs, ys = make_correlated_xy(corr, size=20484,
                                        seed=rs.integers(MSEED))

        # remove medial wall to ensure data are still sufficiently correlated.
        # this is important for parcellations that will ignore the medial wall
        xs, ys = _mod_medial(xs, remove=True), _mod_medial(ys, remove=True)
        acorr = np.corrcoef(xs, ys)[0, 1]

        if debug:
            # n:>3 because dear lord i hope it doesn't take more than 999 tries
            print(f'{n:>3}: {acorr:>6.3f}')
            n += 1

    if acorr < 0:
        ys *= -1

    if normalize:
        xs, ys = sstats.zscore(xs), sstats.zscore(ys)

    return _mod_medial(xs, remove=False), _mod_medial(ys, remove=False)

def matching_multinorm_exgrfs(corr, tol=0.005, alpha=3.0, params=[0., 1., 1.], normalize=True, seed=None, debug=False):
    """
    Generates two surface ex-GRFs (fsaverage5) that correlate at r = `corr`

    Starts by generating two random variables from a multivariate normal
    distribution with correlation `corr`, adds spatial autocorrelation with
    specified `alpha`, and projects to the surface. Continues this procedure
    until two variables are generated that have correlation `corr` on the
    surface.

    Parameters
    ----------
    corr : float
        Desired correlation of generated GRFs
    tol : float
        Tolerance for correlation between generated GRFs
    alpha : float (positive), optional
        Exponent of the power-law distribution. Only used if `use_gstools` is
        set to False. Default: 3.0
    normalize : bool, optional
        Whether to normalize the returned field to unit variance. Default: True
    seed : None, int, default_rng, optional
        Random state to seed GRF generation. Default: None
    debug : bool, optional
        Whether to print debug info

    Return
    ------
    x, y : (20484,) np.ndarray
        Generated surface GRFs
    """
    
    mu, sigma, tau = params
    rs = np.random.default_rng(seed)

    acorr, n = np.inf, 0
    while np.abs(np.abs(acorr) - corr) > tol:
        if alpha > 0:
            x, y = make_correlated_exgaussian(corr, size=902629,
                                      seed=rs.integers(MSEED), 
                                      mu=mu, sigma=sigma, tau=tau)
            # smooth correlated noise vectors + project to surface
            xs = create_surface_exgaussian(noise=x, alpha=alpha, normalize=normalize)
            ys = create_surface_exgaussian(noise=y, alpha=alpha, normalize=normalize)
        else:
            xs, ys = make_correlated_exgaussian(corr, size=20484,
                                        seed=rs.integers(MSEED),
                                        mu=mu, sigma=sigma, tau=tau)

        # remove medial wall to ensure data are still sufficiently correlated.
        # this is important for parcellations that will ignore the medial wall
        xs, ys = _mod_medial(xs, remove=True), _mod_medial(ys, remove=True)
        acorr = np.corrcoef(xs, ys)[0, 1]

        if debug:
            # n:>3 because dear lord i hope it doesn't take more than 999 tries
            print(f'{n:>3}: {acorr:>6.3f}')
            n += 1

    if acorr < 0:
        ys *= -1

    if normalize:
        xs, ys = sstats.zscore(xs), sstats.zscore(ys)

    return _mod_medial(xs, remove=False), _mod_medial(ys, remove=False)