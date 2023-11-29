#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting subroutines
"""

from nilearn import plotting
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm
import numpy as np
import os
from neuromaps.datasets import fetch_atlas

surfaces = ['fsaverage', 'fsLR']

def vrange(scalars, pv=5.):
    # check if nans
    mask = np.isnan(scalars)
    
    # get visual range
    return np.percentile(scalars[~mask], pv), np.percentile(scalars[~mask], 100-pv)

def csplot(scalars, surface='fsLR', hemisphere='left', orientation='landscape', 
           vrange=(-4,4), cmap='viridis'):
    """
    Automated plotting of cortical surfaces. Generates a figure of
    a scalar brainmap `y` and returns `fig` and `ax` handles for
    further use (if needed). Suitable for use with data registered to
    CIFTI 32k space only. Plots data on fs_LR_32k freesurfer-type 
    "midthickness" surface.

    Parameters
    ----------
    scalars : np.ndarray of shape=(32492,)
        Brainmap to visualize registered to standard fs_LR_32k CIFTI space.
    hemisphere : str, optional
        Hemisphere to plot. The default is 'left'.
    orientation : str, optional
        Orientation of plot, 'landscape' or 'portrait'. Default is 'landscape'.
    vrange : tuple, optional
        Maximum range of values to plot. Values higher or lower are truncated.
        Default is (-4, 4).
    cmap : str, optional
        matplotlib.cm color types. The default is 'viridis'.

    Returns
    -------
    fig : matplotlib.figure handle
    ax : matplotlib.ax handle
    
    Notes
    -----
    If there are NaNs present in the brainmap `y`, these are masked out
    of the plotting. See ``nilearn.plotting``

    """
    if surface not in surfaces:
        raise ValueError('chosen surface must be "fsLR" or "fsaverage5"')
        
    if surface == 'fsaverage':
        surf = 'inflated'
        den = '10k'
    else:
        surf = 'midthickness'
        den = '32k'
        
    # make figure
    if orientation == 'landscape':
        fig = plt.figure(figsize=(9, 6), constrained_layout=False)
        grid = gridspec.GridSpec(1, 2)
        rot = 'vertical'
        axes = [1.01, 0.28, 0.03, 0.4]
    elif orientation == 'portrait':
        fig = plt.figure(figsize=(6, 9), constrained_layout=False)
        grid = gridspec.GridSpec(2, 1)
        rot = 'horizontal'
        axes = [1.01, 0.28, 0.03, 0.4]
        
    if hemisphere == 'left':
        mesh = fetch_atlas(atlas=f'{surface}', density=den)[surf][0]
        bg_map = fetch_atlas(atlas=f'{surface}', density=den)['vaavg'][0]
    elif hemisphere == 'right':
        mesh = fetch_atlas(atlas=f'{surface}', density=den)[surf][1]
        bg_map = fetch_atlas(atlas=f'{surface}', density=den)['vaavg'][1]
    else:
        raise ValueError('Hemisphere can only be "left" or "right"')
    
    # get colormap
    cmap = plt.get_cmap(cmap)
    vmin, vmax = vrange
    view = ['lateral', 'medial']
        
    # plot surface
    for idx in range(2):
        ax = fig.add_subplot(grid[idx], projection='3d')
        plotting.plot_surf(mesh, surf_map=scalars, hemi=hemisphere, 
                           view=view[idx], vmin=vmin, vmax=vmax, 
                           colorbar=False, cmap=cmap, axes=ax,
                           bg_map=bg_map)
        ax.dist = 9
    
    # colorbar
    cax = plt.axes(axes)
    cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax, 
                        orientation=rot)
    cbar.set_ticks([])
    cbar.ax.set_title(f'{vmax:.2f}', fontdict={'fontsize':20}, pad=20)
    cbar.ax.set_xlabel(f'{vmin:.2f}', fontdict={'fontsize':20}, labelpad=20)
    
    return fig, ax
    
def flatplot(scalars, hemisphere='left', vrange=(-4,4), cmap='viridis', colorbar=False):
    """
    Automated plotting of cortical surfaces. Generates a figure of
    a scalar brainmap `y` and returns `fig` and `ax` handles for
    further use (if needed). Plots on the fs_LR_32k freesurfer-type "flat" surface.
    Suitable for use with data registered to CIFTI 32k space only.

    Parameters
    ----------
    scalars : np.ndarray of shape=(32492,)
        Brainmap to visualize registered to standard fs_LR_32k CIFTI space.
    hemisphere : str, optional
        Hemisphere to plot. The default is 'left'.
    vrange : tuple, optional
        Maximum range of values to plot. Values higher or lower are truncated.
        Default is (-4, 4).
    cmap : str, optional
        matplotlib.cm color types. The default is 'viridis'.
    colorbar : bool, optional
        If True, plots the colorbar.

    Returns
    -------
    fig : matplotlib.figure handle
    ax : matplotlib.ax handle
    
    Notes
    -----
    If there are NaNs present in the brainmap `y`, these are masked out
    of the plotting. See ``nilearn.plotting``

    """
    # make figure
    fig = plt.figure(figsize=(6, 6), constrained_layout=False)
        
    if hemisphere == 'left':
        mesh = os.path.join(os.path.split(__file__)[0], 'datasets', 'surfaces', 'standard', 'fs_LR.32k.L.flat.surf.gii')
    elif hemisphere == 'right':
        mesh = os.path.join(os.path.split(__file__)[0], 'datasets', 'surfaces', 'standard', 'fs_LR.32k.R.flat.surf.gii')
    else:
        raise ValueError('Hemisphere can only be "left" or "right"')
    
    # get colormap
    cmap = plt.get_cmap(cmap)
    vmin, vmax = vrange
        
    # plot surface
    ax = fig.add_subplot(projection='3d')
    plotting.plot_surf(mesh, surf_map=scalars, hemi=hemisphere, 
                       view='lateral', vmin=vmin, vmax=vmax, 
                       colorbar=False, cmap=cmap, axes=ax)
    ax.dist = 8
    
    # colorbar
    if colorbar:
        cax = plt.axes([1.04, 0.2, 0.03, 0.6])
        cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
        cbar.set_ticks([])
        cbar.ax.set_title(f'{vmax:.2f}', fontdict={'fontsize':15}, pad=20)
        cbar.ax.set_xlabel(f'{vmin:.2f}', fontdict={'fontsize':15}, labelpad=20)
    
    return fig, ax
