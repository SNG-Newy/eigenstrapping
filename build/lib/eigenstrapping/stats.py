from scipy import special, stats as sstats
from sklearn.neighbors import KernelDensity
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import correlation
from netneurotools import stats
from neuromaps import compare_images

def compare_maps(src, trg, metric='pearsonr', ignore_zero=True, nulls=None,
                   nan_policy='omit', return_nulls=False):
    """
    Compares images `src` and `trg`. Wrapper for `neuromaps.compare_images()`

    If `src` and `trg` represent data from multiple hemispheres the data are
    concatenated across hemispheres prior to comparison

    Parameters
    ----------
    src, trg : tuple or str or os.PathLike or img_like or array-like
        Images (nib.Nifti1Image or nib.GiftiImage) or parcellated data
        to be compared.
    metric : {'pearsonr', 'spearmanr', callable}, optional
        Type of similarity metric to use to compare `src` and `trg` images. If
        a callable function is provided it must accept two inputs and return a
        single value (the similarity metric). Default: 'pearsonr'
    ignore_zero : bool, optional
        Whether to perform comparisons ignoring all zero values in `src` and
        `trg` data. Default: True
    nulls : array_like, optional
        Null data for `src` to use in generating a non-parametric p-value.
        If not specified a parameteric p-value is generated. Default: None
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan. 'propagate' propagates
        the nan values to the callable metric (will return nan if the metric
        is `spearmanr` `or pearsonr`), 'raise' throws an error, 'omit' performs
        the calculations ignoring nan values. Default: 'omit'
    return_nulls : bool, optional
        Whether to return the null distribution of comparisons. Can only be set
        to `True` if `nulls` is not None. Default: False

    Returns
    -------
    similarity : float
         Comparison metric between `src` and `trg`
    pvalue : float
        The p-value of `similarity`, if `nulls` is not None
    nulls : (n_perm, ) array_like
        Null distribution of similarity metrics. Only returned if
        `return_nulls` is True.
    """
    
    return compare_images(src, trg, metric=metric, ignore_zero=ignore_zero, 
                    nulls=nulls.T, nan_policy=nan_policy, return_nulls=return_nulls)

def gpd_inference(perms, stat=0.15, per=0.10):
    """
    Report the tail-estimated p-value for faster inference using a
    Generalized Pareto distribution.

    Parameters
    ----------
    perms : np.ndarray of shape (N,)
        Statistical values with target brain map (e.g., Pearson correlation)
    stat : float, optional
        Empirical stat (e.g. Pearson correlation) of original brain map
        and target brain map. Default is 0.15.
    per : float, optional
        Percentage of highest correlation values to keep. Default is 0.10.

    Returns
    -------
    float
        Estimated p-value

    """
    if per <= 0.0 or per > 1:
        raise ValueError("Percentage of correlation values to keep from null distribution must be between 0 and 1")
    
    # Extract the tail values above the threshold
    tail_values = [value for value in perms if value > per]
    
    # Fit the Generalized Pareto Distribution to the tail values
    params = sstats.genpareto.fit(tail_values)
    
    # Calculate the p-value using the CDF of the fitted GPD
    return 1 - sstats.genpareto.cdf(stat, *params)
    

def msle(y_true, y_pred):
    """Compute the Mean Squared Logarithmic Error."""
    return np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)

def joint_differential_entropy(x, y, b=0.5):
    """ Compute the joint differential entropy of x and y """
    data = np.vstack([x, y]).T
    
    # estimate using KDE
    kde = KernelDensity(kernel="gaussian", bandwidth=b).fit(data)
    
    # evaluate density at data
    log_density = kde.score_samples(data)
    
    # compute the joint differential entropy
    return -np.mean(log_density)

def conditional_entropy(x, array, flatten=True, n_jobs=1, **kwargs):
    """ Calculates conditional entropy of an empirical array given a set of surrogates """
    ent_x = sstats.differential_entropy(x)
    conds = np.row_stack(
        Parallel(n_jobs=n_jobs)(
            delayed(_cent)(x, ent_x, array[i]) for i in range(array.shape[0])
        )
    )
    if flatten:
        return conds.squeeze().flatten()
    
    return conds.squeeze()
    
def _cent(x, ent_x, y):
    joint_ent = joint_differential_entropy(x, y)
    cond = joint_ent - ent_x
    
    return cond
        
def kl(x, array, b=0.5, kernel='gaussian', flatten=True, n_jobs=1):
    """ Computes KL divergence of surrs to empirical array """
    # estimate using KDE
    pdf_x = _kde(x.reshape(-1, 1), kernel, b)
    
    pdfs = np.row_stack(
        Parallel(n_jobs=n_jobs)(
            delayed(_kde)(array[j].reshape(-1, 1), kernel=kernel, b=b) for j in range(array.shape[0])
        )
    )
    ks = []
    for j in range(pdfs.shape[0]):
        pdf_y = pdfs[j]
        k = sstats.entropy(pdf_x, pdf_y)
        ks.append(k)
    
    if flatten:
        np.asarray(ks).flatten()
    
    return np.asarray(ks)

def _kde(x, kernel, b):
    # Avoid division by zero and log(0)
    epsilon = 1e-10
    
    # estimate using KDE
    kde_x = KernelDensity(kernel=kernel, bandwidth=b).fit(x)
    pdf_x = np.exp(kde_x.score_samples(x))
    pdf_x += epsilon
    return pdf_x

def normalize(x):
    """
    Normalize a vector to [0, 1] range.
    """
    min_val = np.min(x)
    max_val = np.max(x)
    return (x - min_val) / (max_val - min_val)

def distance_correlation(x, nulls, bins=100, flatten=True):
    """ 
    Distance correlation between empirical and nulls.
    """
    # main loop over nulls
    Is = []
    for j in range(nulls.shape[0]):
        y = nulls[j]
        I = correlation(x, y)
        # rescale to be between -1 and 1
        I -= 1
        I *= -1
        Is.append(I)
    
    if flatten:
        return np.asarray(Is).flatten()
    
    return np.asarray(Is)
        
def ks(x, nulls, flatten=True):
    """
    Measures difference in underlying distribution (or non-Gaussianity) of a 
    surrogate dataset using the Kolmogorov-Smirnov test
    """
    ks_nulls = []
    for j in range(nulls.shape[0]):
        y = nulls[j]
        ks = -np.log10(sstats.kstest(x, y)[1]+1e-3)
        ks_nulls.append(ks)
    
    if flatten:
        return np.asarray(ks_nulls).flatten()
    
    return np.asarray(ks_nulls)
    
""" Functions for performing statistical inference using surrogate maps. """

import numpy as np
from scipy.stats import rankdata

__all__ = ['spearmanr', 'pearsonr', 'pairwise_r', 'nonparp']


def spearmanr(X, Y):
    """
    Multi-dimensional Spearman rank correlation between rows of `X` and `Y`.

    Parameters
    ----------
    X : (N,P) np.ndarray
    Y : (M,P) np.ndarray

    Returns
    -------
    (N,M) np.ndarray

    Raises
    ------
    TypeError : `X` or `Y` is not array_like
    ValueError : `X` and `Y` are not same size along second axis

    """
    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        raise TypeError('X and Y must be numpy arrays')

    if X.ndim == 1:
        X = X.reshape(1, -1)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)

    n = X.shape[1]
    if n != Y.shape[1]:
        raise ValueError('X and Y must be same size along axis=1')

    return pearsonr(rankdata(X, axis=1), rankdata(Y, axis=1))


def pearsonr(X, Y, ddof=1, nan_policy='omit'):
    """
    Multi-dimensional Pearson correlation between rows of `X` and `Y`.

    Parameters
    ----------
    X : (N,P) np.ndarray
    Y : (M,P) np.ndarray

    Returns
    -------
    (N,M) np.ndarray

    Raises
    ------
    TypeError : `X` or `Y` is not array_like
    ValueError : `X` and `Y` are not same size along second axis

    """
    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
        raise TypeError('X and Y must be numpy arrays')

    if X.ndim == 1:
        X = X.reshape(1, -1)
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)

    n = X.shape[1]
    if n != Y.shape[1]:
        raise ValueError('X and Y must be same size along axis=1')

    if nan_policy not in ('propagate', 'raise', 'omit'):
        raise ValueError(f'Value for nan_policy "{nan_policy}" not allowed')

    X, Y = X.reshape(len(X), -1), Y.reshape(len(Y), -1)
    if (X.shape[1] != Y.shape[1]):
        X, Y = np.broadcast_arrays(X, Y)

    mask = np.logical_or(np.isnan(X), np.isnan(Y))
    if nan_policy == 'raise' and np.any(mask):
        raise ValueError('Input cannot contain NaN when nan_policy is "omit"')
    elif nan_policy == 'omit':
        # avoid making copies of the data, if possible
        X = np.ma.masked_array(X, mask, copy=False, fill_value=np.nan)
        Y = np.ma.masked_array(Y, mask, copy=False, fill_value=np.nan)

    mu_x = X.mean(axis=1)
    mu_y = Y.mean(axis=1)

    s_x = X.std(axis=1, ddof=n - 1)
    s_y = Y.std(axis=1, ddof=n - 1)
    cov = np.dot(X, Y.T) - n * np.dot(
        mu_x[:, np.newaxis], mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])

def pairwise_r(X, flatten=False, nan_policy='omit'):
    """
    Compute pairwise Pearson correlations between rows of `X`.

    Parameters
    ----------
    X : (N,M) np.ndarray
    flatten : bool, default False
        If True, return flattened upper triangular elements of corr. matrix

    Returns
    -------
    (N*(N-1)/2,) or (N,N) np.ndarray
        Pearson correlation coefficients

    """
    if X.ndim == 1:
        n = len(X)
        correlation_matrix = np.zeros((n, n))
    
        for i in range(n):
            for j in range(n):
                shift = j - i
                # Shift to the right
                shifted_x = np.roll(X, shift)
                original_x = X
    
                if len(original_x) > 1 and len(shifted_x) > 1:
                    correlation = pearsonr(original_x, shifted_x)
                    correlation_matrix[i, j] = correlation[0]
    
        return correlation_matrix
    
    rp = pearsonr(X, X, nan_policy='omit')
    if not flatten:
        return rp
    triu_inds = np.triu_indices_from(rp, k=1)
    return rp[triu_inds].flatten()


def nonparp(stat, dist):
    """
    Compute two-sided non-parametric p-value.

    Compute the fraction of elements in `dist` which are more extreme than
    `stat`.

    Parameters
    ----------
    stat : float
        Test statistic
    dist : (N,) np.ndarray
        Null distribution for test statistic

    Returns
    -------
    float
        Fraction of elements in `dist` which are more extreme than `stat`

    """
    n = float(len(dist))
    return np.sum(np.abs(dist) > abs(stat)) / n
