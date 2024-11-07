"""
Eigenmode resampling on the cortex. Use fit.surface_fit() to determine how many
modes to use to capture the empirical variogram.
"""

from eigenstrapping import (
    SurfaceEigenstrapping, VolumetricEigenstrapping, compute_psd, utils,
    )
import matplotlib.pyplot as plt
import numpy as np
from netneurotools.stats import efficient_pearsonr
from brainspace.null_models.variogram import SampledSurrogateMaps

eigen_args = ['surface', 'evals', 'emodes',
               'num_modes', 'save_surface',
               'seed', 'decomp_method', 'medial',
               'randomize', 'find_optimal', 'resample',
               'step', 'n_jobs', 'use_cholmod', 'permute',
               'distribution', 'shuffle', 'gen_rotations',
               'add_res']

var_args = ['ns', 'pv', 'nh', 'knn', 'pv', 'nh', 'knn']

def surface_fit(x, D=None, index=None, nsurrs=10, num_modes=100, return_data=False, 
                extra_diags=False, surrs=None, **params):
    """
    Evaluate variogram fits for :class: `eigenstrapping.SurfaceEigenstrapping` 
    to determine how many modes to decompose surface map with. Returns two plots:
        left - variogram fit
        middle - surrogate pairwise correlation with original map `x`
        right - histogram of surrogate residuals with original map `x`

    Parameters
    ----------
    x : (N,) np.ndarray
        Target brain map
    D : (N,N) np.ndarray or np.memmap
        Pairwise distance matrix between elements of `x`
    index : (N,N) np.ndarray or np.memmap
        See :method:`variogram.variogram`
    nsurr : int, default 20
        Number of simulated surrogate maps from which to compute variograms
    return_data : bool, default False
        if True, return the surrogates
    extra_diags : bool, default False
        if True, return extra diagnostics
            1. modal power spectra of original and surrogates
            2. Local Moran's I of original and surrogates
    surrs : (m,N) np.ndarray or np.memmap or path to file
        If not `None`, skips null generation and calculates the variogram for the
        given set of surrogates in `surrs`. Expects these to be of shape (m,N) in
        array-like or `str` of path to file (in .txt or .npy format) containing 
        this array.
    params :
        Keyword arguments for :class: `eigenstrapping.SurfaceEigenstrapping` or
        :method: `variogram.variogram`. If eigenmodes and eigenvalues have been
        precomputed for the surface that `x` exists on, then pass them as `evals`
        and `emodes`. If they're not given, then a `surface` must be given.
        See :class: `eigenstrapping.Eigenstrapping` for more details.

    Returns
    -------
    if and only if return_data is True:
    emp_var : (M,) np.ndarray
        empirical smoothed variogram values
    u0 : (M,) np.ndarray
        distances at which variogram values were computed
    surr_var : (nsurr, M) np.ndarray
        surrogate maps' smoothed variogram values

    Notes
    -----
    If `return_data` is False, this function generates and shows a matplotlib
    plot instance illustrating the fit of the surrogates' variograms to the
    target map's variogram. If `return_data` is True, this function returns the
    data needed to generate such a plot (i.e., the variogram values and the
    corresponding distances).

    """
    # check kwargs
    eigen_params = dict()
    var_params = dict()
    for arg in params:
        if arg in eigen_args:
            eigen_params[arg] = params[arg]
        if arg in var_args:
            var_params[arg] = params[arg]

    if surrs is None:
        # initialize
        eigen = SurfaceEigenstrapping(x, num_modes=num_modes, **eigen_params)
    
        # surrogates
        surrs = eigen(n=nsurrs)
    else:
        surrs = surrs
        nsurrs = len(surrs)
    
    # plot variogram
    # Instantiate surrogate map generator
    print('Surrogates computed, computing stats...')
    # if eigen_params['parcellation']:
    #     parcellation = eigen_params['parcellation']
    #     x = calc_parcellate(parcellation, data_input=x)
    #     surrs = calc_parcellate(parcellation, data_input=surrs).T
    #     generator = Base(x=x, D=D)
        
    #     emp_var, u0 = generator.compute_smooth_variogram(x, return_h=True)

    #     # Compute surrogate map variograms
    #     surr_var = np.empty((nsurrs, generator.nh))
    #     for i in range(nsurrs):
    #         surr_var[i] = generator.compute_smooth_variogram(surrs[i])
        
    # else:
    # if D or index is None:
    #     print('No surface distance matrix given, calculating - may take a while...')
    #     distmat = geodesic_distmat(eigen_params['surface'], n_jobs=eigen_params['n_jobs'])
    #     d = os.path.dirname(__file__)
    #     file = os.path.join(d, 'distmat.txt')
    #     np.savetxt(file, distmat)
    #     dict_dist = txt2memmap(file, output_dir=d)
    #     D = np.load(dict_dist['distmat'])
    #     index = np.load(dict_dist['index'])
    generator = SampledSurrogateMaps(**var_params)
    generator.fit(D, index)

    nsurrs_var = nsurrs
    if nsurrs > 50:
        nsurrs_var = 50
    # Compute target & surrogate map variograms
    surr_var = np.empty((nsurrs_var, generator.nh))
    emp_var_samples = np.empty((nsurrs_var, generator.nh))
    u0_samples = np.empty((nsurrs_var, generator.nh))
    for i in range(nsurrs_var):
        xi = generator._check_map(x)
        idx = generator.sample(xi.size)  # Randomly sample a subset of brain areas
        v = generator.compute_variogram(xi, idx)
        u = generator._dist[idx, :]
        umax = np.percentile(u, generator.pv)
        uidx = np.where(u < umax)
        emp_var_i, u0i = generator.smooth_variogram(
            u=u[uidx], v=v[uidx], return_h=True)
        emp_var_samples[i], u0_samples[i] = emp_var_i, u0i
        # Surrogate
        surri = generator._check_map(surrs[i])
        v_null = generator.compute_variogram(surri, idx)
        surr_var[i] = generator.smooth_variogram(
            u=u[uidx], v=v_null[uidx], return_h=False)

    u0 = u0_samples.mean(axis=0)
    emp_var = emp_var_samples.mean(axis=0)
    
    #u0, emp_var, surr_var = variogram(x, surrs, D, index, **var_params)

    # Start with a 1x3 grid for plots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        
    # Variogram plot
    ax = axes[0]
    ax.scatter(u0, emp_var, s=20, facecolor='none', edgecolor='k',
               marker='o', lw=1, label='Empirical')
    mu = surr_var.mean(axis=0)
    sigma = surr_var.std(axis=0)
    ax.fill_between(u0, mu-sigma, mu+sigma, facecolor='#377eb8',
                    edgecolor='none', alpha=0.3)
    ax.plot(u0, mu, color='#377eb8', label='Surrogates', lw=1)
    leg = ax.legend(loc=0)
    leg.get_frame().set_linewidth(0.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Spatial separation\ndistance")
    ax.set_ylabel("Variance")
    ax.set_title("Variogram")
        
    # Pairwise correlation plot
    ax = axes[1]
    correlations = efficient_pearsonr(x, surrs.T, nan_policy='omit')[0]
    ax.hist(correlations, bins=np.linspace(-1, 1, num=nsurrs//10), color='#377eb8', alpha=0.7)
    ax.axvline(x=np.mean(correlations), color='r', linestyle='dashed', linewidth=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Distribution of correlations")
    ax.set_xlabel("Correlation coefficient")
    ax.set_ylabel("Frequency")
        
    # Histogram of residuals plot
    ax = axes[2]
    residuals = [x - surr for surr in surrs]
    for res in residuals:
        ax.hist(res, bins=50, alpha=0.5, histtype='stepfilled')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Histogram of Residuals")
    ax.set_xlabel("Residual value")
    ax.set_ylabel("Frequency")
    
    # Ensure spacing between plots is nice
    plt.tight_layout()
    
    plt.show()
        
    if return_data:
        return surrs
    
def volumetric_fit(x, volume, nsurrs=10, num_modes=100, 
                   return_data=False, extra_diags=False, **params):
    """
    Evaluate variogram fits for :class: `eigenstrapping.VolumetricEigenstrapping` 
    to determine how many modes to decompose volumetric map with.

    Parameters
    ----------
    mask : str to mask file
        Target ROI volume with 1 inside region-of-interest, and 0 elsewhere
    x : str to data inside mask
        Target brain map
    D : (N,N) np.ndarray or np.memmap
        Pairwise distance matrix between elements of `x`
    index : (N,N) np.ndarray or np.memmap
        See :method:`variogram.variogram`
    nsurr : int, default 20
        Number of simulated surrogate maps from which to compute variograms
    return_data : bool, default False
        if True, return: 1, the smoothed variogram values for the target
        brain map; 2, the distances at which the smoothed variograms values
        were computed; and 3, the surrogate maps' smoothed variogram values
    params :
        Keyword arguments for :class: `eigenstrapping.VolumetricEigenstrapping` or
        :method: `variogram.variogram`. If eigenmodes and eigenvalues have been
        precomputed for the surface that `x` exists on, then pass them as `evals`
        and `emodes`. If they're not given, then a `surface` must be given.
        See :class: `eigenstrapping.VolumetricEigenstrapping` for more details.

    Returns
    -------
    if and only if return_data is True:
    surrs : (N,) np.ndarray
        Eigenstrapped surrogates in volume space

    """
    
    #check kwargs
    eigen_params = dict()
    var_params = dict()
    for arg in params:
        if arg in eigen_args:
            eigen_params[arg] = params[arg]
        if arg in var_args:
            var_params[arg] = params[arg]
            
    # initialize
    eigen = VolumetricEigenstrapping(data=x, volume=volume, num_modes=num_modes, **eigen_params)
    
    # surrogates
    surrs = eigen(n=nsurrs)
    
    x, D, index = eigen.calculate_distance_matrix(return_data=True)
    
    # Instantiate surrogate map generator
    generator = SampledSurrogateMaps(**var_params)
    generator.fit(D, index)

    nsurrs_var = nsurrs
    if nsurrs > 50:
        nsurrs_var = 50
    # Compute target & surrogate map variograms
    surr_var = np.empty((nsurrs_var, generator.nh))
    emp_var_samples = np.empty((nsurrs_var, generator.nh))
    u0_samples = np.empty((nsurrs_var, generator.nh))
    for i in range(nsurrs_var):
        xi = generator._check_map(x)
        idx = generator.sample(xi.size)  # Randomly sample a subset of brain areas
        v = generator.compute_variogram(xi, idx)
        u = generator._dist[idx, :]
        umax = np.percentile(u, generator.pv)
        uidx = np.where(u < umax)
        emp_var_i, u0i = generator.smooth_variogram(
            u=u[uidx], v=v[uidx], return_h=True)
        emp_var_samples[i], u0_samples[i] = emp_var_i, u0i
        # Surrogate
        surri = generator._check_map(surrs[i])
        v_null = generator.compute_variogram(surri, idx)
        surr_var[i] = generator.smooth_variogram(
            u=u[uidx], v=v_null[uidx], return_h=False)

    u0 = u0_samples.mean(axis=0)
    emp_var = emp_var_samples.mean(axis=0)

    # Start with a 1x3 grid for plots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        
    # Variogram plot
    ax = axes[0]
    ax.scatter(u0, emp_var, s=20, facecolor='none', edgecolor='k',
               marker='o', lw=1, label='Empirical')
    mu = surr_var.mean(axis=0)
    sigma = surr_var.std(axis=0)
    ax.fill_between(u0, mu-sigma, mu+sigma, facecolor='#377eb8',
                    edgecolor='none', alpha=0.3)
    ax.plot(u0, mu, color='#377eb8', label='Surrogates', lw=1)
    ax.legend(loc=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Spatial separation\ndistance")
    ax.set_ylabel("Variance")
    ax.set_title("Variogram")
        
    # Pairwise correlation plot
    ax = axes[1]
    correlations, _ = efficient_pearsonr(x, surrs.T, nan_policy='omit')
    ax.hist(correlations, bins=nsurrs_var//2, color='#377eb8', alpha=0.7)
    ax.axvline(x=np.mean(correlations), color='r', linestyle='dashed', linewidth=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Distribution of correlations")
    ax.set_xlabel("Correlation coefficient")
    ax.set_ylabel("Frequency")
        
    # Histogram of residuals plot
    ax = axes[2]
    residuals = [x - surr for surr in surrs]
    for res in residuals:
        ax.hist(res, bins=50, alpha=0.5, histtype='stepfilled')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Histogram of Residuals")
    ax.set_xlabel("Residual value")
    ax.set_ylabel("Frequency")
    
    # Ensure spacing between plots is nice
    plt.tight_layout()
    
    plt.show()
    
    if return_data:
        return surrs
