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
from brainsmash.mapgen import Base, Sampled
from brainsmash.mapgen.memmap import txt2memmap
from .geometry import geodesic_distmat
from .utils import parcellate_distmat, calc_parcellate
import os

eigen_args = ['surface', 'evals', 'emodes',
               'num_modes', 'save_surface',
               'seed', 'decomp_method', 'medial',
               'randomize', 'find_optimal', 'resample',
               'step', 'n_jobs', 'use_cholmod', 'permute',
               'distribution', 'shuffle', 'gen_rotations',
               'add_res', 'parcellation']

var_args = ['ns', 'pv', 'nh', 'knn', 'pv', 'nh', 'knn', 'n_jobs', 'seed']

def surface_fit(x, D=None, index=None, nsurrs=10, num_modes=100, return_data=False, 
                extra_diags=False, **params):
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
            
    # initialize
    eigen = SurfaceEigenstrapping(x, num_modes=num_modes, **eigen_params)
    
    # surrogates
    surrs = eigen(n=nsurrs)
    
    # plot variogram
    # Instantiate surrogate map generator
    print('Surrogates computed, computing stats...')
    if eigen_params['parcellation']:
        parcellation = eigen_params['parcellation']
        x = calc_parcellate(parcellation, data_input=x)
        surrs = calc_parcellate(parcellation, data_input=surrs).T
        generator = Base(x=x, D=D)
        
        emp_var, u0 = generator.compute_smooth_variogram(x, return_h=True)

        # Compute surrogate map variograms
        surr_var = np.empty((nsurrs, generator.nh))
        for i in range(nsurrs):
            surr_var[i] = generator.compute_smooth_variogram(surrs[i])
        
    else:
        if D or index is None:
            print('No surface distance matrix given, calculating - may take a while...')
            distmat = geodesic_distmat(eigen_params['surface'], n_jobs=eigen_params['n_jobs'])
            d = os.path.dirname(__file__)
            file = os.path.join(d, 'distmat.txt')
            np.savetxt(file, distmat)
            dict_dist = txt2memmap(file, output_dir=d)
            D = np.load(dict_dist['distmat'])
            index = np.load(dict_dist['index'])
        generator = Sampled(x=x, D=D, index=index)
    
        nsurrs_var = nsurrs
        if nsurrs > 50:
            nsurrs_var = 50
        # Compute target & surrogate map variograms
        surr_var = np.empty((nsurrs_var, generator.nh))
        emp_var_samples = np.empty((nsurrs_var, generator.nh))
        u0_samples = np.empty((nsurrs_var, generator.nh))
        for i in range(nsurrs_var):
            idx = generator.sample()  # Randomly sample a subset of brain areas
            v = generator.compute_variogram(generator.x, idx)
            u = generator.D[idx, :]
            umax = np.percentile(u, generator.pv)
            uidx = np.where(u < umax)
            emp_var_i, u0i = generator.smooth_variogram(
                u=u[uidx], v=v[uidx], return_h=True)
            emp_var_samples[i], u0_samples[i] = emp_var_i, u0i
            # Surrogate
            v_null = generator.compute_variogram(surrs[i], idx)
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
    ax.hist(correlations, bins=np.linspace(-1, 1, num=nsurrs//5), color='#377eb8', alpha=0.7)
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
    
    if extra_diags:
        # compute modal power spectra
        original_power = eigen.psd
        surr_psds = [compute_psd(surr[eigen.medial_wall], eigen._emodes) for surr in surrs]
        # calculate moran
        original_moran = utils.calc_moran(D, x.reshape(1, -1), medmask=eigen_params['medial'] if 'medial'
                                                            in eigen_params else None)
        surr_moran = utils.calc_moran(D, surrs[:nsurrs//2], medmask=eigen_params['medial'] if 'medial'
                                                            in eigen_params else None)
        
        # now plot
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        # Power Spectrum
        ax = axes[0]
        ax.semilogy(original_power, label='Original', color='k', linewidth=2, zorder=1000)
        for i, power in enumerate(surr_psds[:len(surr_psds)//2]):
            ax.semilogy(power, alpha=0.3, linestyle='dashed')
        ax.set_title('Power Spectrum')
        ax.set_xlabel('Mode')
        ax.set_ylabel('Normalized power')
        leg = ax.legend(loc=0)
        leg.get_frame().set_linewidth(0.0)
        
        # Moran's I
        ax = axes[1]
        labels = ['Original'] + [f'Surrogate {i+1}' for i in range(nsurrs//2)]
        
        # Only displaying every second label for clarity
        visible_labels = [label if idx % 2 == 0 else '' for idx, label in enumerate(labels)]
        
        ax.bar(range(len(labels)), np.hstack((original_moran, surr_moran)), color='c', alpha=0.7)
        ax.set_title("Moran's I")
        ax.set_ylabel("Moran's I Value")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(visible_labels, rotation=30, ha='right')
        
        plt.tight_layout()
        plt.show()
        
    if return_data:
        return surrs
    
def volumetric_fit(volume, x, nsurrs=10, num_modes=100, 
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
    eigen = VolumetricEigenstrapping(volume, x, num_modes=num_modes, **eigen_params)
    
    # surrogates
    surrs = eigen(n=nsurrs)
    
    x, D, index = eigen.calculate_distance_matrix(return_data=True)
    
    # Instantiate surrogate map generator
    generator = Sampled(x=x, D=D, index=index, knn=900)

    if nsurrs > 50:
        nsurrs = 50
    # Compute target & surrogate map variograms
    surr_var = np.empty((nsurrs, generator.nh))
    emp_var_samples = np.empty((nsurrs, generator.nh))
    u0_samples = np.empty((nsurrs, generator.nh))
    for i in range(nsurrs):
        idx = generator.sample()  # Randomly sample a subset of brain areas
        v = generator.compute_variogram(generator.x, idx)
        u = generator.D[idx, :]
        umax = np.percentile(u, generator.pv)
        uidx = np.where(u < umax)
        emp_var_i, u0i = generator.smooth_variogram(
            u=u[uidx], v=v[uidx], return_h=True)
        emp_var_samples[i], u0_samples[i] = emp_var_i, u0i
        # Surrogate
        v_null = generator.compute_variogram(surrs[i], idx)
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
    ax.hist(correlations, bins=nsurrs//2, color='#377eb8', alpha=0.7)
    ax.axvline(x=np.mean(correlations), color='r', linestyle='dashed', linewidth=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Pairwise Correlations")
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
    
    if extra_diags:
        # compute modal power spectra
        original_power = eigen.psd
        surr_psds = [compute_psd(surr, eigen._emodes) for surr in surrs]
        # calculate moran
        original_moran = utils.calc_moran(D, x.reshape(1, -1))
        surr_moran = utils.calc_moran(D, surrs[:nsurrs//2])
        
        # now plot
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        # Power Spectrum
        ax = axes[0]
        ax.semilogy(original_power, label='Original', color='k', linewidth=2, zorder=1000)
        for i, power in enumerate(surr_psds[:len(surr_psds)//2]):
            ax.semilogy(power, alpha=0.3, linestyle='dashed')
        ax.set_title('Power Spectrum')
        ax.set_xlabel('Mode')
        ax.set_ylabel('Normalized power')
        leg = ax.legend(loc=0)
        leg.get_frame().set_linewidth(0.0)
        
        # Moran's I
        ax = axes[1]
        labels = ['Original'] + [f'Surrogate {i+1}' for i in range(nsurrs//2)]
        
        # Only displaying every second label for clarity
        visible_labels = [label if idx % 2 == 0 else '' for idx, label in enumerate(labels)]
        
        ax.bar(range(len(labels)), np.hstack((original_moran, surr_moran)), color='c', alpha=0.7)
        ax.set_title("Moran's I")
        ax.set_ylabel("Moran's I Value")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(visible_labels, rotation=30, ha='right')
        
        plt.tight_layout()
        plt.show()
    
    if return_data:
        return surrs