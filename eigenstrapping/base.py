#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eigenmode resampling on the cortex
"""

from .utils import (
    _get_eigengroups,
    eigen_decomposition,
    transform_to_spheroid,
    transform_to_ellipsoid,
    normalize_data,
    calc_parcellate,
    )

import os
import subprocess

from .geometry import calc_surface_eigenmodes, get_tkrvox2ras

import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from nilearn import plotting, masking
from .dataio import is_string_like, dataio, load_surface
import tempfile
from joblib import Parallel, delayed
from sklearn.utils.validation import check_random_state
from scipy.stats import normaltest
import nibabel as nib
from lapy import TriaIO, TetIO, Solver
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from .rotations import rotate_matrix

cmap = plt.get_cmap('viridis')

norm_types = ['volume', 'constant', 'area', 'number']

global methods
methods = [
    'matrix',
    'regression',
    ]

def eigengroup_wavelength(mode, radius=60.0, group=False):
    """
    Returns the approximate wavelength of a mode given a radius of the surface
    (radii are approximated for non-spherical surfaces, default 60mm). Radius is 
    specified in mm. Based on the degenerate solution of the Laplace-Beltrami
    operator on a sphere in [1]:
        
                        2 * pi * radius
        wavelength ~= -------------------
                      [ l ( l + 1 ) ] ^ 2

    Parameters
    ----------
    mode : int
        Mode at which to return the (approximate) wavelength.
    radius : float, default 60.0
        The (approximate) radius of the surface, measured in millimeters.
    group : bool, default False
        Return the group membership of `mode`.

    Returns
    -------
    float
        Approximate wavelength of `mode` in mm.
    if group is True: int
        Group membership of `mode`

    """
    if mode < 0 or type(mode) != int:
        raise ValueError('Input mode must be an integer zero or greater')
    if mode == 0:
        l = 0
        print('Wavelength of zeroth mode is infinity.')
        if group:
            return l
    
    total = 0
    l = 0
    while total < mode:
        l += 1
        total += 2 * l + 1
    
    wavelength = 2 * np.pi * radius / (l * (l + 1) ** 2)  
    if group:
        return l, wavelength
    
    return wavelength

def compute_psd(data, emodes):
    """Compute normalized power spectral density using emodes."""
    mask = np.isnan(data)
    coeffs = eigen_decomposition(data[~mask], emodes[~mask])
    power = np.abs(coeffs)**2
    normalized_power = power/np.sum(power)
    return normalized_power

def gram_schmidt_randomized(A, rs=None):
    """
    Orthogonalize a set of vectors stored as the columns of matrix A.
    Randomize the order of orthonormalization to explore different parameter
    spaces.
    """
    # get the number of vectors
    n = A.shape[1]
    
    # Randomize the column order
    if rs is None:
        rs = check_random_state(rs)
    random_order = rs.permutation(n)
    
    B = copy.deepcopy(A)
    B = B[:, random_order]
    
    for j in range(n):
        # To orthogonalize the vector in column j with respect to the
        # previous vectors, subtract from it its projection onto
        # each of the previous vectors
        for k in range(j):
            B[:, j] -= np.dot(B[:, k], B[:, j]) *B[:, k]
        
        B[:, j] = B[:, j] / np.linalg.norm(B[:, j])
        
    # reverse the randomization to original order
    reverse_order = np.argsort(random_order)
    A = B[:, reverse_order]
        
    return A

def is_gaussian_distribution(data, significance_level=0.05):
    """
    Check if the given data matches a Gaussian distribution using a normality test.

    Parameters:
        data (array-like): The data to be tested.
        significance_level (float, optional): The significance level for the test. Default is 0.05.

    Returns:
        bool: True if the data is consistent with a Gaussian distribution, False otherwise.
    """
    
    data = normalize_data(data)
    _, p_value = normaltest(data)
    if p_value >= significance_level:
        return True

    return False

class SurfaceEigenstrapping:
    """
    Compute the eigenmodes and eigenvalues of the surface in `surface` and 
    resample the hypersphere bounded by the eigengroups contained within `emodes`,
    to reconstruct the data using coefficients conditioned on the original data
    Based on the degenerate solutions of solving the Laplace-Beltrami operator 
    on the cortex. The power spectrum is perfectly retained (the square of the 
    eigenvalues). If evals and emodes are given (i.e., precomputed) then 
    eigenmodes are not computed. Performs amplitude adjustment by default (see
    `resample`)
    
    @author: Nikitas C. Koussis, School of Psychological Sciences,
             University of Newcastle & Systems Neuroscience Group Newcastle
             
             Michael Breakspear, School of Psychological Sciences,
             University of Newcastle & Systems Neuroscience Group Newcastle
    
    Process
    -------
        - the orthonormal eigenvectors n within eigengroup l give the surface 
            of the hyperellipse of dimensions l-1
            NOTE: for eigengroup 0 with the zeroth mode, we ignore it,
            and for eigengroup 1, this is resampling the surface of the 
            2-sphere
        - the axes of the hyperellipse are given by the sqrt of the eigenvalues 
            corresponding to eigenvectors n
        - linear transform the eigenvectors N to the hypersphere by dividing 
            by the ellipsoid axes
        - finds the set of points `p` on the hypersphere given by the basis 
            modes (the eigenmodes) by normalizing them to unit length
        - rotate the set of points `p` by cos(angle) for 
            even dimensions and sin(angle) for odd dims (resampling step)
        - find the unit vectors of the points `p` by dividing by the 
            Euclidean norm (equivalent to the eigenmodes)
        - make the new unit vectors orthonormal using Gram-Schmidt process
        - return the new eigenmodes of that eigengroup until all eigengroups
            are computed
        - reconstruct the null data by multiplying the original coefficients
            by the new eigenmodes
        - resamples the null data by performing rank-ordering to replicate
            the reconstructed data term, then adds the noise term back into the
            null data to produce a surrogate that replicates the original
            variance of the empirical data.
            * Only performed if `resample` = True.
    
    References
    ----------
        1. Robinson, P. A. et al., (2016), Eigenmodes of brain activity: Neural field 
        theory predictions and comparison with experiment. NeuroImage 142, 79-98.
        <https://dx.doi.org/10.1016/j.neuroimage.2016.04.050>
        
        2. Jorgensen, M., (2014), Volumes of n-dimensional spheres and ellipsoids. 
        <https://www.whitman.edu/documents/Academics/Mathematics/2014/jorgenmd.pdf>
        
        3. <https://math.stackexchange.com/questions/3316373/ellipsoid-in-n-dimension>
        
        4. Blumenson, L. E. (1960). A derivation of n-dimensional spherical 
        coordinates. The American Mathematical Monthly, 67(1), 63-66.
        <https://www.jstor.org/stable/2308932>
        
        5. Trefethen, Lloyd N., Bau, David III, (1997). Numerical linear algebra. 
        Philadelphia, PA: Society for Industrial and Applied Mathematics. 
        ISBN 978-0-898713-61-9.
        
        6. https://en.wikipedia.org/wiki/QR_decomposition
        
        7. Chen, Y. C. et al., (2022). The individuality of shape asymmetries 
        of the human cerebral cortex. Elife, 11, e75056. 
        <https://doi.org/10.7554/eLife.75056>
        

    Parameters
    ----------
    surface : str of path to file or array_like
        Filename of surface to resample of surface.darrays[0].data of shape (n_vertices, 3)
        and surface.darrays[1].data of shape (n_faces, 3). Must be a single
        hemisphere or bounded surface.
        
    data : str of path to file or np.ndarray of (n_vertices,)
        Filename of empirical data to resample
        
    evals : np.ndarray of shape (n,), optional
        Eigenvalues corresponding to the number of eigenmodes n in `emodes`
        
    emodes : np.ndarray of shape (n_vertices, n), optional
        Eigenmodes that are the solution to the generalized eigenvalue problem
        or Helmholtz equation in the Laplace-Beltrami operator of the cortex
        
    num_modes : int, optional
        If `evals` and `emodes` are not given, then they are computed on the
        surface given in `surface`. This variable controls how many modes
        to compute. Cannot exceed the number of vertices in `surface`.
        Default is 200.
    
    medial : np.logical_array or str of path to file, default None
        Medial wall mask for the input surface `surface`. Suitable only if
        surface is cortical hemisphere. Will use the labels for the medial 
        wall to mask out of the surrogates. If None, uses the naive 
        implementation of finding the medial wall by finding 0.0 values
        in `data` - prone to errors if `data` has zero values outside of the
        medial wall. Can also pass `False` to not attempt masking of medial wall
        at all. 
        
        WARNING: If passing `False` to medial and `True` to resample,
        resulting surrogates may have strange distributions since the 
        rank-ordering step may assign medial-wall values outside of the 
        medial wall. USE AT YOUR OWN RISK.
        
    save_surface : bool, optional
        As above, if `evals` and `emodes` are computed and medial wall mask
        is given, then cortical surface with cuts is saved. Default False.
    
    seed : None or int or np.random.RandomState instance
        Specify the seed for random angle generation (or random state instance)
        default None.
        
    decomp_method : str, optional
        method of calculation of coefficients: 'matrix', 'matrix_separate', 
        'regression'.
        
        The default is 'matrix'.
        
    resample : bool, optional
        Set whether to resample surrogate map from original map to preserve
        values, default True
        
    randomize : bool, optional
        Set whether to shuffle coefficients calculated from minimizing
        least-squares error to reconstruct surrogates. Results in better
        randomization of surrogates at the cost of poorer replication of
        empirical spatial autocorrelation.
    
    find_optimal : bool, optional
        Overrides `evals` and `emodes` if "True". Computes a large amount
        of modes (20% of all possible modes of surface) and finds the optimal
        number of modes to make residual error between least-squares error
        estimation of coefficients and empirical data "white" (Gaussian),
        then generates new residuals with the same variance structure.
        Can increase computation time significantly but results in best balance
        of empirical variogram and randomization of values.
    
    n_jobs : int, optional
        Number of workers to use for parallelization. Default 1.
        
    use_cholmod : bool, optional
        Specify whether or not to use sksparse subroutines - requires 
        ``scikit-sparse`` and ``libsuitesparse-dev`` to be installed. See 
        https://github.com/scikit-sparse/scikit-sparse for more information.
        
    Raises
    ------
    ValueError : Inappropriate inputs

    """
    def __init__(self, data, surface=None, evals=None, emodes=None, num_modes=200,
                 save_surface=False, seed=None, decomp_method='matrix',
                 medial=None, randomize=False, resample=False, n_jobs=1, 
                 use_cholmod=False, permute=False, add_res=False, adjust=False, 
                 gen_rotations=False, parcellation=None):
        
        # initialization of variables
        if surface is None:
            print("No surface given, expecting precomputed eigenvalues and eigenmodes")
            if evals is None or emodes is None:
                raise ValueError("Class must have a `surface` input, or eigenvalues `evals` and eigenmodes `emodes`")        
        
        if surface is not None:
            if not is_string_like(surface):
                raise ValueError("Input surface must be a filename")
            self.surface_file = surface
            self.surface = load_surface(self.surface_file)
            self.n_vertices = self.surface[0].shape[0]
        
        self.data = copy.deepcopy(data)
        
        if emodes is not None and evals is not None:
            self.evals = evals[:num_modes]
            self.emodes = emodes[:, :num_modes]
        else:
            self.evals = self.emodes = None
        
        self.num_modes = num_modes
        self.save_surface = save_surface
        self._rs = check_random_state(seed)
        self.decomp_method = decomp_method
        self.medial = medial
        self.resample = resample
        self.randomize = randomize
        self.n_jobs = n_jobs
        self.cholmod = use_cholmod
        self.permute = permute
        self.adjust = adjust
        self.gen_rotations = gen_rotations
        self.parcellation = parcellation
        
        self._lm = LinearRegression(fit_intercept=True)
        self.add_res = add_res
        if self.permute is True and self.add_res is True:
            self.add_res = False
            print('Permuting of residuals passed, `add_res` not passed')
        
        if self.decomp_method is not None:
            if self.decomp_method in methods:
                self.method = self.decomp_method
            else:
                raise ValueError("Eigenmode decomposition method must be 'matrix' or 'regression'")
        
        # mask out medial wall
        if self.medial is None:
            # get index of medial wall hopefully
            self.medial_wall = ~np.isnan(self.data)
            
        elif self.medial is False:
            if self.resample is True:
                raise RuntimeWarning("Resampling without masking out the medial wall "
                                     "may result in erroneous surrogate distributions. "
                                     "The authors of this code do not take responsibility for "
                                     "improper usage.\n"
                                     "USE AT YOUR OWN RISK.")
                
                self.medial_wall = np.zeros_like(self.data)
            
        else: # if given medial array
            if is_string_like(self.medial) == True:
                try:
                    self.medial = dataio(self.medial)
                except:
                    raise RuntimeError("Could not load medial wall file, please check")
            
            if isinstance(self.medial, np.ndarray) == True:
                if self.medial.ndim != 1:
                    raise ValueError("Medial wall array must be a vector")
                if self.medial.shape[0] != self.n_vertices:
                    # try transpose
                    if self.medial.shape[1] != self.n_vertices:
                        raise ValueError("Medial wall array must have the same number of vertices as the brain map")
                    else:
                        self.medial_wall = self.medial.T
                if not np.array_equal(self.medial, self.medial.astype(int)):
                    raise RuntimeError("Medial wall array must be 1 for the ROI (medial wall) and 0 elsewhere")
                else:    
                    self.medial_wall = self.medial
            else:
                raise ValueError("Could not use provided medial wall array or "
                                 "file, please check")
        
        self.medial_wall = self.medial_wall.astype(np.bool_)
        
        # checks
        if self.evals is None and self.emodes is None:
            print(f'Computing eigenmodes on surface using N={num_modes} modes')
            if self.num_modes >= self.n_vertices:
                raise ValueError('Number of modes must not exceed the number of vertices in input surface')
                
            self.evals, self.emodes = calc_surface_eigenmodes(self.surface_file, self.medial_wall, save_cut=True, num_modes=self.num_modes, use_cholmod=self.cholmod)
        
        else:
            # perform checks
            if self.evals.ndim != 1:
                raise ValueError("Eigenvalue array must be 1-dimensional")
            if self.emodes.shape[1] != self.evals.shape[0]:
                # try transpose
                self.emodes = self.emodes.T
                if self.emodes.shape[1] != self.evals.shape[0]:
                    raise ValueError("There must be as many eigenmodes as there are eigenvalues")
                    
        self.n_vertices = self.emodes.shape[0]
        
        self.medial_mask = np.logical_not(self.medial_wall)
        self.data_no_mwall = self.data # deepcopy original data so it's not modified
        self.data_no_mwall = self.data[self.medial_wall]
        self._emodes = copy.deepcopy(self.emodes)
        self._emodes = self._emodes[self.medial_wall]
        
        # try initial modes given
        self.coeffs = self.eigen_decomposition(method=self.method)
        self.reconstructed_data = self.coeffs @ self._emodes.T
        self.reconstructed_data = self.reconstructed_data.squeeze()
        self.residuals = self.data_no_mwall - self.reconstructed_data
        
        # compute original modal PSD
        #self.psd = compute_psd(self.data_no_mwall, self._emodes)
        
        # find eigengroups
        self.groups = _get_eigengroups(self._emodes)
        
    def __call__(self, n=1):
        """
        Generate new surrogate map(s).

        Parameters
        ----------
        n : int, default 1
            Number of surrogate maps to generate

        Returns
        -------
        (n,N) np.ndarray
            Generated map(s) resampled from eigenspace of surface map
            
        Notes
        -----
        Surrogates are returned with nans in place of the mask given by
        ``self.medial_wall``.

        """
        rs = self._rs.randint(np.iinfo(np.int32).max, size=n)
        surrs = np.row_stack(
            Parallel(self.n_jobs)(
                    delayed(self._call_method)(rs=i) for i in rs
            )
        )
        return np.asarray(surrs.squeeze())
    
    def _call_method(self, rs=None):
        """ Subfunction """
        
        # reset randomstate
        self._rs = check_random_state(rs)
        
        surrogate = self.generate()
        
        return surrogate.squeeze()
    
    def generate(self, output_modes=False):
        """
        Generate eigensphere resampled surrogate.

        Returns
        -------
        surrogate_data : np.ndarray
            Eigensphere resampled surrogate

        """
        # initialize data
        data = copy.deepcopy(self.data_no_mwall)
        emodes = copy.deepcopy(self._emodes)
        evals = self.evals
        groups = self.groups
        mask = self.medial_wall
        coeffs = copy.deepcopy(self.coeffs)
        reconstructed_data = copy.deepcopy(self.reconstructed_data)
        residuals = copy.deepcopy(self.residuals)
        
        # initialize the new modes
        new_modes = np.zeros_like(emodes)
        
        # resample the hypersphere (except for groups 1 and 2)
        for idx in range(len(groups)):
            if idx > 100:
                gen = True
            group_modes = emodes[:, groups[idx]]
            group_evals = evals[groups[idx]]
            
            # else, transform to spheroid and index the angles properly
            p = transform_to_spheroid(group_evals, group_modes)
            
            p_rot = self.rotate_modes(p, gen=True)
            
            # transform back to ellipsoid
            group_ellipsoid_modes = transform_to_ellipsoid(group_evals, p_rot)
            
            new_modes[:, groups[idx]] = group_ellipsoid_modes
        
        if output_modes:
            return new_modes
        
        # matrix multiply the estimated coefficients by the new modes
        surrogate = np.zeros_like(self.data)*np.nan # original data
        
        if self.randomize:
            for i in range(len(groups)):
                coeffs[groups[i]] = self._rs.permutation(coeffs[groups[i]])
        
        # adjust weights based on matching the power spectral density of
        # the original data
        surrogate[mask] = coeffs @ new_modes.T
            
        # Mask the data and surrogate_data excluding the medial wall
        surr_no_mwall = copy.deepcopy(surrogate)
        surr_no_mwall = surr_no_mwall[mask]
        
        # now add the residuals of the original data
        if self.permute:
            surr_no_mwall += self._rs.permutation(residuals)
            
        if self.add_res:
            surr_no_mwall += residuals
            
        # if self.resample:
        #     # Get the rank ordered indices
        #     data_ranks = data.argsort()[::-1]
        #     surr_ranks = surr_no_mwall.argsort()[::-1]
            
        #     # Resample surr_no_mwall according to the rank ordering of data_no_mwall
        #     surr_no_mwall[surr_ranks] = data[data_ranks]
        
        if self.resample:  # resample values from empirical map
            sorted_map = np.sort(data)
            ii = np.argsort(surr_no_mwall)
            np.put(surr_no_mwall, ii, sorted_map)
        else: # demean
            surr_no_mwall = surr_no_mwall - np.nanmean(surr_no_mwall)
            
        # else: # force match the minima
        #     indices = np.nonzero(surr_no_mwall)[0]  # Indices where s is non-zero

        #     # Compute the normalization
        #     data_selected = data[indices]
        #     surr_selected = surr_no_mwall[indices]
            
        #     surr_no_mwall[indices] = (surr_selected - surr_selected.min()) * (data_selected.max() - data_selected.min()) / (surr_selected.max() - surr_selected.min()) + data_selected.min()

        output_surr = np.zeros_like(surrogate)*np.nan
        output_surr[mask] = surr_no_mwall
        
        return output_surr.squeeze()
    
    def eigen_decomposition(self, method='matrix'):
        """
        Decompose data using eigenmodes and calculate the coefficient of 
        contribution of each vector.
        
        Parameters:
        -----------
        method : string
            method of calculation of coefficients: 'matrix', 'matrix_separate', 
            'regression'
        
        Returns:
        -------
        coeffs : numpy array of shape (N, 3)
         coefficient values
        
        """
        if self.parcellation:
            return eigen_decomposition(self.data_no_mwall, self._emodes, method='regression')
        
        return eigen_decomposition(self.data_no_mwall, self._emodes, method=method).squeeze()
    
    def shuffle_modes(self, emodes):
        return self._rs.permutation(emodes)
    
    def rotate_modes(self, emodes, gen=True):
        """
        Rotate modes using random rotations.

        Parameters
        ----------
        vectors : np.ndarray of (n_vertices, mu)
            Orthogonal vectors of n_vertices and mu modes
        
        Returns
        -------
        rotated_vectors : np.ndarray of (n_vertices, mu)
            Array containing rotated vectors for each pair.

        """
        if gen is True:
            return rotate_matrix(emodes)
        
        mu = emodes.shape[1]
        if mu % 2 == 0 or mu == 1:
            return rotate_matrix(emodes)
        
        l = (mu - 1) // 2
        return np.dot(emodes, self._loader.get_random_rotation(l))
        
    def resample_eigenspace(self, emodes, distribution='normal'):
        """
        Resample across the eigenspace through linear weighted sums of real
        modes iteratively. Each iteration returns new orthogonal modes that 
        are reweighted combinations of previous modes. NOTE: should only be
        applied to modes that have the same eigenvalue, i.e., on the 
        eigensphere.
        
        Parameters
        ----------
        modes : np.ndarray of (n_vertices, mu)
            Orthogonal vectors of n_vertices and mu modes
        
        Returns
        -------
        new_modes : np.ndarray of (n_vertices, mu)
            Array containing resampled vectors for each pair.
        """
        
        mu = emodes.shape[1]
        new_modes = np.zeros(emodes.shape)
        
        # Derive coefficients
        for i in range(mu):
            if distribution == 'uniform':
                coefficients = self._rs.uniform(low=-1., high=1., size=mu)
            elif distribution == 'normal':
                coefficients = self._rs.normal(size=mu)
            elif distribution == 'none':
                coefficients = np.ones((mu,))
            else:
                raise ValueError("Unsupported distribution {}".format(str(distribution)))
        
            # now combine with new coefficients
            new_mode = np.dot(emodes, coefficients)
            new_modes[:, i] = new_mode
                
        return new_modes

    def plot_data(self, surface, data, hemi='left', view='lateral', cmap='gray', show=True):
        """
        Plots a data map using nilearn.plotting, returns fig and ax handles
        from matplotlib.pyplot for further use. Can also plot values on the
        surface by input to `data`.
    
        Parameters
        ----------
        surface : nib.GiftiImage class or np.ndarray of shape (n_vertices, 3)
            A single surface to plot.
        data : np.ndarray of shape (n_vertices,)
            Data to plot on the surface
        hemi : str, optional
            Which hemisphere to plot. The default is 'left'.
        view : str, optional
            Which view to look at the surface. 
            The default is 'lateral'. Accepted strings are detailed in
            the docs for nilearn.plotting
        cmap : str or matplotlib.cm class
            Which colormap to plot the surface with, default is 'viridis'
        show : bool, optional
            Flag whether to show the plot, default is True
    
        Returns
        -------
        fig : figure handle    
        ax : axes handle
    
        """
        # make figure
        fig = plt.figure(figsize=(15,9), constrained_layout=False)
        mesh = (surface.darrays[0].data, surface.darrays[1].data)
        
        # get colormap
        cmap = plt.get_cmap(cmap)
        vmin = np.min(data)
        vmax = np.max(data)
            
        # plot surface
        ax = fig.add_subplot(projection='3d')
        plotting.plot_surf(mesh, surf_map=data, hemi=hemi, view=view, 
                           vmin=vmin, vmax=vmax, colorbar=False, 
                           cmap=cmap, axes=ax)
        ax.dist = 7
        
        # show figure check
        if show is True:
            plt.show()
        
        return fig, ax
        
    
    def eigen_spectrum(self, data, show=True):
        """
        Plot the modal decomposition power spectrum reconstruction using original
        modes. See ``eigen_decomposition()``. Returns fig and ax handles from 
        matplotlib.pyplot for further use.

        Parameters
        ----------
        data : np.ndarray of shape (n_vertices,)
            Data to decompose and reconstruct for eigen spectrum.
        show : bool, optional
            Flag whether to show plot, default True

        Returns
        -------
        fig : matplotlib.pyplot class
            Figure handle
        ax : matplotlib.pyplot class
            Axes handle

        """
        
        # compute power spectrum = eval^2
        coeffs = eigen_decomposition(data, self._emodes)
        power = np.abs(coeffs)**2
        normalized_power = power/np.sum(power)
        
        # now do figure
        fig = plt.figure(figsize=(12, 12), constrained_layout=False)
        
        ax = fig.add_subplot()
        
        ax.semilogy(np.arange(1,len(coeffs)+1), normalized_power)
        #ax.set_ylim((0.00000, 0.1))
        ax.set_xlim((0, len(coeffs)+1))
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        
        plt.xlabel(r'Mode', fontdict={'fontsize':30})
        plt.ylabel(r'Normalized power (log scale)', fontdict={'fontsize':30})
        
        # show figure check
        if show is True:
            plt.show()
        
        return fig, ax
    
class VolumetricEigenstrapping:
    """
    Compute the eigenmodes and eigenvalues of the volume in `volume` and 
    resample the hypersphere bounded by the eigengroups contained within `emodes`,
    to reconstruct the data using coefficients conditioned on the original data
    Based on the degenerate solutions of solving the Laplace-Beltrami operator 
    on the cortex. The power spectrum is perfectly retained (the square of the 
    eigenvalues). If evals and emodes are given (i.e., precomputed) then 
    eigenmodes are not computed. Performs amplitude adjustment by default (see
    `resample`)
    
    @author: Nikitas C. Koussis, School of Psychological Sciences,
             University of Newcastle & Systems Neuroscience Group Newcastle
             
             Michael Breakspear, School of Psychological Sciences,
             University of Newcastle & Systems Neuroscience Group Newcastle
             
    IMPORTANT
    ---------
        Several software packages must be installed apart from python dependencies
        to use this class:
            - FreeSurfer >5
            - Gmsh
            - fsl >5 (if ``aseg`` not passed)
    
    Process
    -------
        - the orthonormal eigenvectors n within eigengroup l give the surface 
            of the hyperellipse of dimensions l-1
            NOTE: for eigengroup 0 with the zeroth mode, we ignore it,
            and for eigengroup 1, this is resampling the surface of the 
            2-sphere
        - the axes of the hyperellipse are given by the sqrt of the eigenvalues 
            corresponding to eigenvectors n
        - linear transform the eigenvectors N to the hypersphere by dividing 
            by the ellipsoid axes
        - finds the set of points `p` on the hypersphere given by the basis 
            modes (the eigenmodes) by normalizing them to unit length
        - rotate the set of points `p` by cos(angle) for 
            even dimensions and sin(angle) for odd dims (resampling step)
        - find the unit vectors of the points `p` by dividing by the 
            Euclidean norm (equivalent to the eigenmodes)
        - make the new unit vectors orthonormal using Gram-Schmidt process
        - return the new eigenmodes of that eigengroup until all eigengroups
            are computed
        - reconstruct the null data by multiplying the original coefficients
            by the new eigenmodes
        - resamples the null data by performing rank-ordering to replicate
            the reconstructed data term, then adds the noise term back into the
            null data to produce a surrogate that replicates the original
            variance of the empirical data.
            * Only performed if `resample` = True.
    
    References
    ----------
        1. Robinson, P. A. et al., (2016), Eigenmodes of brain activity: Neural field 
        theory predictions and comparison with experiment. NeuroImage 142, 79-98.
        <https://dx.doi.org/10.1016/j.neuroimage.2016.04.050>
        
        2. Jorgensen, M., (2014), Volumes of n-dimensional spheres and ellipsoids. 
        <https://www.whitman.edu/documents/Academics/Mathematics/2014/jorgenmd.pdf>
        
        3. <https://math.stackexchange.com/questions/3316373/ellipsoid-in-n-dimension>
        
        4. Blumenson, L. E. (1960). A derivation of n-dimensional spherical 
        coordinates. The American Mathematical Monthly, 67(1), 63-66.
        <https://www.jstor.org/stable/2308932>
        
        5. Trefethen, Lloyd N., Bau, David III, (1997). Numerical linear algebra. 
        Philadelphia, PA: Society for Industrial and Applied Mathematics. 
        ISBN 978-0-898713-61-9.
        
        6. https://en.wikipedia.org/wiki/QR_decomposition
        
        7. Chen, Y. C. et al., (2022). The individuality of shape asymmetries 
        of the human cerebral cortex. Elife, 11, e75056. 
        <https://doi.org/10.7554/eLife.75056>
        

    Parameters
    ----------
    volume : str to image
        Filename of volume ROI with integer intensity values. Must be in the same
        space (i.e. registered) to the volume file in `data`. Extensions
        are those commonly accepted by nibabel (see nibabel file types). Volume
        is extracted using FreeSurfer or fslmaths if ``aseg`` is not passed.
        
    data : str to image or str to text file
        Filename of empirical data to resample
        
    label : int or list of ints or array of ints, optional
        Label of value in ``volume`` for ROI extraction. If default None, volume
        is considered to be a mask - i.e., all values equal to 1 are extracted
        and transformed to tetrahedral space, and those equal to 0 or otherwise 
        are left out. Default is None
        
    aseg : bool, optional
        Specify whether file in ``volume`` is FreeSurfer preprocessed "aseg.mgz"
        file. Extra steps need to be performed for computation, though labels
        can be combined into structures (e.g., ``label=[11, 12, 26]`` with 
        ``aseg=True`` would combine the left caudate, left putamen, and left NA
        to make the left striatum. See the FreeSurferColorLUT.txt for more
        details about labeling). Default is False
        
    evals : np.ndarray of shape (n,), optional
        Eigenvalues corresponding to the number of eigenmodes n in `emodes`.
        Default is None
        
    emodes : np.ndarray of shape (n_vertices, n), optional
        Eigenmodes that are the solution to the generalized eigenvalue problem
        or Helmholtz equation in the Laplace-Beltrami operator of the volume
        (must be in tetrahedral space). Default is None
        
    num_modes : int, optional
        If `evals` and `emodes` are not given, then they are computed on the
        surface given in `surface`. This variable controls how many modes
        to compute. Cannot exceed the number of vertices after tetrahedral
        resampling of ``volume``.
        Default is 200.
    
    seed : None or int or np.random.RandomState instance
        Specify the seed for random angle generation (or random state instance)
        default None.
        
    decomp_method : str, optional
        method of calculation of coefficients: 'matrix', 'matrix_separate', 
        'regression'.
        
        The default is 'matrix'.
        
    resample : bool, optional
        Set whether to resample surrogate map from original map to preserve
        values, default True
        
    randomize : bool, optional
        Set whether to shuffle coefficients calculated from minimizing
        least-squares error to reconstruct surrogates. Results in better
        randomization of surrogates at the cost of poorer replication of
        empirical spatial autocorrelation.
    
    n_jobs : int, optional
        Number of workers to use for parallelization. Default 1.
        
    use_cholmod : bool, optional
        Specify whether or not to use sksparse subroutines - requires 
        ``scikit-sparse`` and ``libsuitesparse-dev`` to be installed. See 
        https://github.com/scikit-sparse/scikit-sparse for more information.
        
    Raises
    ------
    ValueError : Inappropriate inputs

    """
    
    def __init__(self, volume, data, label=None, aseg=False, norm_file=None,
                 normalization=None, normalization_factor=None, evals=None,
                 emodes=None, num_modes=200, seed=None, decomp_method='matrix', 
                 randomize=False, resample=False, n_jobs=1, use_cholmod=False, 
                 permute=False, adjust=False, distribution='normal'):
        
        # checks
        if not is_string_like(volume):
            raise ValueError('Input volume must be filename')
        self.voldir, self.nifti_input_tail = os.path.split(volume)
        self.nifti_input_main, self.nifti_input_ext = os.path.splitext(self.nifti_input_tail)
        #self.data_input_head, self.data_input_tail = os.path.split(data)
        #self.data_input_main, self.data_input_ext = os.path.splitext(self.data_input_tail)
        
        self.distribution = distribution
        self.adjust = adjust
        self.volume = volume
        self.ROI = nib.load(volume)
        self.roi = self.ROI.get_fdata()
        self.affine = self.ROI.affine
        self.mask = np.logical_not(np.logical_or(np.isclose(self.roi, 0),
                                            np.isnan(self.roi)))
        self.xyz = nib.affines.apply_affine(self.affine, np.column_stack(np.where(self.mask)))
        
        if is_string_like(data):
            self.data_input_main, self.data_input_ext = os.path.splitext(self.data_input_tail)
            if self.data_input_ext == '.txt':
                self.data_array = np.loadtxt(data)
                self.data_array = masking.unmask(self.data_array, self.ROI).get_fdata()
                self.data = nib.Nifti1Image(self.data_array, self.affine, header=self.ROI.header)
            else:
                try:
                    self.data = nib.load(data)
                    if not np.array_equal(nib.load(volume).affine, nib.load(data).affine):
                        raise ValueError('Input volume and input data must have the same affine transformation (i.e., be in the same space')
                except Exception as e:
                    print(f'Error: {e}')
            
                if self.data.ndim > 3:
                    new_data = nib.Nifti1Image(self.data.get_fdata()[:,:,:,0].squeeze(), self.data.affine, header=self.data.header)
                    self.data = new_data
                self.data_array = self.data.get_fdata()
        else:
            self.data_array = masking.unmask(data, self.ROI).get_fdata()
            
        self.norm = norm_file
        if label is None:
            label = 1
            
        self.label = label
        self.roi_number, self.roi_vol = self.calc_volume(self.volume)
        self.method = decomp_method
        self._rs = check_random_state(seed)
        self.randomize = randomize
        self.resample = resample
        self.permute = permute
        self.use_cholmod = use_cholmod
        self.n_jobs = n_jobs
        
        # prepare data and masking
        self.inds_all = np.where(self.roi==self.label)
        self.xx, self.yy, self.zz = self.inds_all
        
        if normalization is not None:
            if normalization not in norm_types:
                raise ValueError('Normalization type must be "area", "constant", "volume", or "number"')
            if normalization == 'constant' and normalization_factor is None:
                raise ValueError('Normalization type of "constant" but no factor given')
        
        self.norm_type = normalization
        self.norm_factor = normalization_factor
        
        self.masked_data = self.data_array[self.mask]
        
        
        # prepare tetra surface
        self.tetra_file = self.make_tetra(self.volume, label=self.label)
        
        self.points = np.zeros([self.xx.shape[0], 4])
        self.points[:,0] = self.xx
        self.points[:,1] = self.yy
        self.points[:,2] = self.zz
        self.points[:,3] = 1
        
        # get transform matrix
        self.T = get_tkrvox2ras(self.ROI.shape, self.ROI.header.get_zooms())
        
        # apply transform
        self.points_trans = np.matmul(self.T, np.transpose(self.points))
        
        # load surface
        self.tetra = TetIO.import_vtk(self.tetra_file)
        
        # calculate number and volume
        self.roi_number, self.roi_volume = self.calc_volume()
        
        # normalize surface
        self.tetra_norm = self.normalize_vtk(self.tetra, self.norm_type, self.norm_factor)
        
        self.num_modes = num_modes
        
        if emodes is None:
            self.evals, self.emodes = self.calc_volume_eigenmodes(self.tetra_norm, 
                                        self.num_modes, self.use_cholmod)
            # remove zeroth mode
            self.evals = self.evals[1:]
            self.emodes = self.emodes[:, 1:]
    
        # reconstruct data in tetrahedral space
        #self.tetra_data = self.project_to_tetra(self.tetra, self.masked_data)
        self._emodes = self.resample_data(self.emodes, method='nearest', vector=True)
        self.coeffs = eigen_decomposition(self.masked_data, self._emodes, method=self.method)
        self.reconstructed_data = self.coeffs @ self._emodes.T
        self.reconstructed_data = self.reconstructed_data.squeeze()
        self.residuals = self.masked_data - self.reconstructed_data
        
        self.psd = compute_psd(self.masked_data, self._emodes)
        # find eigengroups
        self.groups = _get_eigengroups(self.emodes)
        self._lm = LinearRegression(fit_intercept=True)
        
    def __call__(self, n=1):
        """
        Generate new surrogate map(s) in tetra space. Resample using 
        ``.resample_data(surrs)``

        Parameters
        ----------
        n : int, optional
            Number of surrogates to generate. The default is 1.

        Returns
        -------
        (n,N) np.ndarray
            Generated map(s) resampled from eigenspace of tetrahedral surface
            
        Notes
        -----
        Surrogates are in tetrahedral surface space. They will need to be
        resampled to volumetric space with ``.resample_data(surrs)``

        """
        
        rs = self._rs.randint(np.iinfo(np.int32).max, size=n)
        surrs = np.row_stack(
            Parallel(self.n_jobs)(
                delayed(self._call_method)(rs=i) for i in rs
            )
        )        
        return np.asarray(surrs.squeeze())
    
    def _call_method(self, rs=None):
        # reset randomstate
        self._rs = check_random_state(rs)
        
        surrogate = self.generate()
        
        return surrogate.squeeze()
    
    def rotate_modes(self, emodes):
        """
        Rotate modes using random rotations.

        Parameters
        ----------
        vectors : np.ndarray of (n_vertices, mu)
            Orthogonal vectors of n_vertices and mu modes
        
        Returns
        -------
        rotated_vectors : np.ndarray of (n_vertices, mu)
            Array containing rotated vectors for each pair.

        """
        return rotate_matrix(emodes)
    
    def generate(self, output_modes=False):
        """
        Generate eigensphere resampled surrogate.

        Parameters
        ----------
        output_modes : bool, optional
            Output resampled modes for debugging. The default is False

        Returns
        -------
        (n,) : np.ndarray
            Surrogate data in tetrahedral space

        """
        
        # initialize data
        emodes = copy.deepcopy(self._emodes)
        evals = self.evals
        groups = self.groups
        coeffs = copy.deepcopy(self.coeffs)
        reconstructed_data = copy.deepcopy(self.reconstructed_data)
        residuals = copy.deepcopy(self.residuals)
        
        # initialize the new modes
        new_modes = np.zeros_like(emodes)
        
        # resample the hypersphere
        for idx in range(len(groups)):
            group_modes = emodes[:, groups[idx]]
            group_evals = evals[groups[idx]]
            
            # else, transform to spheroid and index the angles properly
            group_new_modes = transform_to_spheroid(group_evals, group_modes)
            p = group_new_modes / np.linalg.norm(group_modes, axis=0)
            
            p = self.rotate_modes(p)
            
            # transform back to ellipsoid
            group_ellipsoid_modes = transform_to_ellipsoid(group_evals, p)
            
            new_modes[:, groups[idx]] = group_ellipsoid_modes / np.linalg.norm(group_ellipsoid_modes, axis=0) 
        
        if output_modes:
            return new_modes
        
        if self.randomize:
            for i in range(len(groups)):
                coeffs[groups[i]] = self._rs.permutation(coeffs[groups[i]])
                
        surrogate = coeffs @ new_modes.T
        
        if self.resample:
            data_ranks = reconstructed_data.argsort()[::-1]
            surr_ranks = surrogate.argsort()[::-1]
            
            surrogate[surr_ranks] = reconstructed_data[data_ranks]
            
        if self.permute:
            surrogate = surrogate + self._rs.permutation(residuals)
            
        return surrogate.squeeze()
    
    def resample_data(self, data, method='linear', vector=False):
        """
        Resample data on tetrahedral surface to volumetric space.

        Parameters
        ----------
        data : np.ndarray of shape (n_tetra_points, N)
            Data to resample
        method : str, optional
            Interpolation method for ``scipy.interpolate.griddata``. 
            The default is 'linear'.
        vector : bool, optional
            Resample data to masked vector (values inside ROI only). Default False

        Returns
        -------
        resampled_data : np.ndarray of shape (A,B,C,N)
            Data in volumetric space.

        """
        
        points_surface = self.tetra.v
        new_shape = np.array(self.roi.shape)
        if self.roi.ndim > 3:
            new_shape[3] = data.shape[-1]
        else:
            new_shape = np.append(new_shape, data.shape[-1])
        
        new_data = np.zeros(new_shape)
        data_vec = np.zeros((self.xx.shape[0], data.shape[-1]))
        
        # perform interpolation of data from tetrahedral space to volume space
        for idx in range(data.shape[-1]):
            interpolated_data = griddata(points_surface, data[:, idx], np.transpose(self.points_trans[:3, :]), method=method)
            data_vec[:, idx] = interpolated_data
            for ind in range(len(interpolated_data)):
                new_data[self.xx[ind], self.yy[ind], self.zz[ind]] = interpolated_data[ind]
        
        if vector is True:
            return data_vec
        
        return new_data
        
    def project_to_tetra(self, tetra, data):
        """
        Project data in volumetric space onto tetrahedral surface.

        Parameters
        ----------
        tetra : lapy compatible object
            Loaded object corresponding to tetrahedral surface.
        data : np.ndarray of shape (xx, yy, zz)
            Data to project to tetrahedral surface with values inside ROI.
        interpolation : str, optional
            Interpolation kind. Default is 'linear'

        Returns
        -------
        tetra_data : np.ndarray of shape (number of tetra points,)
            Resampled data

        """
        v = tetra.v
        xyz = self.xyz
        
        tetra_data = griddata(xyz, data, v, 
                              method='nearest', rescale=True)
        
        # points_orig = T @ points.T
        # points_orig = points_orig.T[:,:3]
        # mesh = (points_orig, t)
        # tetra_data = s.vol_to_surf(
        #     data, mesh, interpolation='nearest', radius=3, kind='ball')
        
        return tetra_data
        
    
    def make_tetra(self, volume, label=None, aseg=False):
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
        output = subprocess.check_output(cmd, shell="True")
        output = output.splitlines()
        
        # pass norm for pretess
        if aseg is True and self.norm is not None:
            cmd = 'mri_pretess ' + tmpf + ' 1 ' + self.norm + ' ' + tmpf
            output = subprocess.check_output(cmd, shell="True")
            output = output.splitlines()
        
        # run marching cubes
        cmd = 'mri_mc ' + tmpf + ' 1 ' + self.voldir + '/tmp_surface.vtk'
        output = subprocess.check_output(cmd, shell="True")
        output = output.splitlines()
        
        geo_file = volume + '.geo'
        tria_file = volume + '.vtk'
        tetra_file = volume + '.tetra.vtk'
        
        cmd = 'mv -f ' + self.voldir + '/tmp_surface.vtk ' + tria_file
        output = subprocess.check_output(cmd, shell='True')
        output = output.splitlines()
        self.tria = TriaIO.import_vtk(tria_file)
        
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
        output = subprocess.check_output(cmd, shell="True")
        output = output.splitlines()
        
        cmd = "sed 's/double/float/g;s/UNSTRUCTURED_GRID/POLYDATA/g;s/CELLS/POLYGONS/g;/CELL_TYPES/,$d' " + tetra_file + " > " + tetra_file + "'_fixed'"            
        output = subprocess.check_output(cmd, shell="True")
        output = output.splitlines()
        
        cmd = 'mv -f ' + tetra_file + '_fixed ' + tetra_file
        output = subprocess.check_output(cmd, shell="True")
        output = output.splitlines()
        
        # remove auxiliary files
        os.remove(geo_file)
        os.remove(tria_file)
        
        return tetra_file
    
    
    def calc_volume_eigenmodes(self, tetra, num_modes=200, use_cholmod=False):
        """
        Calculate the eigenvalues and eigenmodes of the ROI volume.

        Parameters
        ----------
        tetra : lapy compatible object
            Loaded lapy object corresponding to tetrahedral surface of ROI in vtk format
        num_modes : int, optional
            Number of eigenmodes to compute. The default is 200
        use_cholmod : bool, optional
            Specify whether to use ``scikit-sparse`` libraries to compute eigenmodes.
            Much faster but requires libraries to be installed.

        Returns
        -------
        evals : np.ndarray of shape (num_modes,)
            Eigenvalues
        emodes : np.ndarray of shape (number of tetra points, num_modes)
            Eigenmodes

        """
        
        # calc eigenmodes and eigenvalues
        fem = Solver(tetra)
        evals, emodes = fem.eigs(k=num_modes)
        
        return evals, emodes
    
    def calc_volume(self, nifti_volume=None, label=None):
        """
        Calculate the physical volume of the ROI in the nifti file

        Parameters
        ----------
        nifti_volume : str, optional
            Filename of input volume in nifti format
        label : int, optional
            Label value for input ROI in ``nifti_volume``

        Returns
        -------
        ROI_number : int
            Total number of non-zero voxels
        ROI_volume : float
            Total volume of non-zero voxels in physical dimensions

        """
        if nifti_volume is not None:
            roi = nib.load(nifti_volume)
            roi_data = roi.get_fdata()
        else:
            roi = self.ROI
            roi_data = self.roi
        
        if label is None:
            label = 1
        
        roi_data = roi_data[np.where(roi_data==label)]
        
        # get voxel dimensions in mm
        voxel_dims = (roi.header["pixdim"])[1:4]
        voxel_vol = np.prod(voxel_dims)
        
        # compute volume
        ROI_number = np.count_nonzero(roi_data)
        ROI_volume = ROI_number * voxel_vol        
        
        return ROI_number, ROI_volume
    
    def normalize_vtk(self, tetra, normalization_type=None, normalization_factor=None):
        """
        Normalize tetrahedral surface

        Parameters
        ----------
        tetra : lapy compatible object
            Loaded vtk object corresponding to a surface tetrahedral mesh
        normalization_type : str or None, optional
            Type of normalization. The default is None.
        normalization_factor : float or None, optional
            Factor to be used in "constant" normalization. The default is None.

        Returns
        -------
        tetra_norm : lapy compatible object
            Loaded vtk object corresponding to normalized tetrahedral surface

        """
        tetra_norm = tetra
        if normalization_type == 'number':
            tetra_norm.v = tetra.v/(self.roi_number**(1/3))
        elif normalization_type == 'volume':
            tetra_norm.v = tetra.v/(self.roi_volume**(1/3))
        elif normalization_type == 'area':
            tetra_norm.v = tetra.v/(tetra.vertex_areas()**(1/3))
        else:
            pass
        
        if normalization_type in norm_types:
            surface_output_fname = os.path.join(self.voldir, self.nifti_input_main + '_norm=' + normalization_type + '.tetra.vtk')
    
            f = open(surface_output_fname, 'w')
            f.write('# vtk DataFile Version 2.0\n')
            f.write(self.nifti_input_tail + '\n')
            f.write('ASCII\n')
            f.write('DATASET POLYDATA\n')
            f.write('POINTS ' + str(np.shape(tetra.v)[0]) + ' float\n')
            for i in range(np.shape(tetra.v)[0]):
                f.write(' '.join(map(str, tetra_norm.v[i, :])))
                f.write('\n')
            f.write('\n')
            f.write('POLYGONS ' + str(np.shape(tetra.t)[0]) + ' ' + str(5 * np.shape(tetra.t)[0]) + '\n')
            for i in range(np.shape(tetra.t)[0]):
                f.write(' '.join(map(str, np.append(4, tetra.t[i, :]))))
                f.write('\n')
            f.close()
        
        return tetra_norm
    
    def eigen_spectrum(self, show=True):
        """
        Plot the modal decomposition power spectrum. See ``geometry.eigen_decomposition()``
        Returns fig and ax handles from matplotlib.pyplot for further use.

        Parameters
        ----------
        show : bool, optional
            Flag whether to show plot, default True

        Returns
        -------
        fig : matplotlib.pyplot class
            Figure handle
        ax : matplotlib.pyplot class
            Axes handle

        """
        
        # compute power spectrum = eval^2
        coeffs = eigen_decomposition(self.tetra_data, self.emodes)
        power = np.abs(coeffs)**2
        normalized_power = power/np.sum(power)
        
        # now do figure
        fig = plt.figure(figsize=(12, 12), constrained_layout=False)
        
        ax = fig.add_subplot()
        
        ax.semilogy(np.arange(1,len(coeffs)+1), normalized_power)
        #ax.set_ylim((0.00000, 0.1))
        ax.set_xlim((0, len(coeffs)+1))
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        
        plt.xlabel(r'Mode', fontdict={'fontsize':30})
        plt.ylabel(r'Normalized power (log scale)', fontdict={'fontsize':30})
        
        # show figure check
        if show is True:
            plt.show()
        
        return fig, ax
    
    def calculate_distance_matrix(self, return_data=False):
        """
        Calculates Euclidean distance matrix and index memmap files of ROI,
        also returns data masked by ROI.

        Parameters
        ----------
        outdir : str, optional
            Output directory of distance and index np.memmap files.

        Returns
        -------
        masked_data : (N,)
            
        dist : np.memmap
            Distance memmap matrix.
        index : str
            Distance index memmap matrix

        """
        affine = self.affine
        mask = self.mask
        
        xyz = self.xyz
        #xyz = self.inds_all
        distout = tempfile.NamedTemporaryFile(suffix='.mmap')
        dist = np.memmap(distout, mode='w+', dtype='float32',
                         shape=(np.shape(xyz)[0], np.shape(xyz)[0]))
        indout = tempfile.NamedTemporaryFile(suffix='.mmap')
        index = np.memmap(indout, mode='w+', dtype='int32',
                         shape=(np.shape(xyz)[0], np.shape(xyz)[0]))
        
        for n, row in enumerate(xyz):
            xyz_dist = cdist(row[None], xyz).astype('float32')
            index[n] = np.argsort(xyz_dist, axis=-1).astype('int32')
            dist[n] = xyz_dist[:, index[n]]
            dist.flush()
            index.flush()       
        
        masked_data = self.data_array[mask]
        
        if return_data:
            return masked_data, dist, index
            
        return dist, index
    
    def surfplot(self, data, view='dorsal', cmap='viridis', vrange=[-4, 4]):
        """
        Plotting helper function for volumetric data to triangular
        surfaces.

        Parameters
        ----------
        data : np.ndarray of shape (n_vertices,)
            Volumetric data projected to tetrahedral mesh. See self.project_to_tetra().
        view : str or pair of float, optional
            Specify which direction to place camera and view from. See 
            ``nilearn.plotting.plot_surf()`` for accepted strings. If pair of
            float, must be a list or tuple of (elevation, azimuthal) of angles
            in degrees. The default is 'dorsal'.
        cmap : str, optional
            Colormap for values in ``data``. Must be in matplotlib colormaps.
            The default is 'viridis'
        vrange : tuple of two floats, optional
            Range to truncate minimum and maximum values. The default is [-4, 4].
        annotations : bool, optional
            Specify whether to add annotation labels for direction etc

        Returns
        -------
        fig : matplotlib.pyplot.Figure handle
        ax : matplotlib.pyplot.Axes handle

        """
        # define mesh
        tria = self.tria
        mesh = (tria.v, tria.t)
        xyz = self.xyz
        tria_data = griddata(xyz, data, tria.v, method='nearest')
        
        # get colormap
        cmap = plt.get_cmap(cmap)
        vmin, vmax = vrange
        
        # define figure
        fig = plt.figure(figsize=(8, 6), constrained_layout=False)
        ax = fig.add_axes([0.0, 0.2, 0.97, 0.7], projection='3d')
        plotting.plot_surf(mesh, tria_data, view=view, vmin=vmin, vmax=vmax,
                           alpha=1, colorbar=False, cmap=cmap, axes=ax)
        
        cax = plt.axes([0.8, 0.3, 0.03, 0.4])
        cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
        cbar.set_ticks([])
        cbar.ax.set_title(f'{vmax:.2f}', fontdict={'fontsize':20}, pad=20)
        cbar.ax.set_xlabel(f'{vmin:.2f}', fontdict={'fontsize':20}, labelpad=20)
            
        plt.show()
                
        return fig, ax
        
