#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting subroutines
"""

from nilearn import plotting
from nilearn.plotting.displays import BaseAxes
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm
import numpy as np
import os
from neuromaps.datasets import fetch_atlas
from brainspace import mesh as me
from .utils import is_string_like
from .geometry import remesh

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

def meshplot(mesh, scalars=None, vrange=(-4, 4), cmap='viridis', colorbar=False, cam=[0, 0]):
    """
    Automated plotting of surface in `mesh`. `mesh` must be a `*.vtk`-like file
    or a list-like of arrays (`vertices`, `faces`). If given, `scalars` must be a 1D vector
    and have the same number of entries as the longest axis of `vertices`. If
    not given as default, function simply plots gray surface (`cmap`, `vrange` do
    nothing)

    Parameters
    ----------
    mesh : file or list-like
        Surface mesh to plot `scalars` on.
    scalars : 1D ndarray, optional
        1D vector on mesh. Default None (not given)
    vrange : float or tuple, optional
        Min and max range to visualize `scalars`. Anything higher or lower is truncated.
        If one number is given (positive or negative) then assumed to be maximum
        and sign of `vrange` is flipped to give minimum. Default is (-4, 4).
    cmap : str, optional
        Colormap of `scalars`. Accepts recognized ``matplotlib.cm`` types. 
        Default is 'viridis'.
    colorbar : bool, optional
        Add colorbar to plot. Default is False.
    cam : list-like, optional
        Elevation and azimuthal angle for camera. Default [30, 60] i.e., elev=30, azim=60.

    Returns
    -------
    fig : matplotlib.figure handle
    ax : matplotlib.ax handle

    """
    # check mesh
    if is_string_like(mesh):
        m = me.mesh_io.read_surface(mesh)
        # plot triangular surface
        if m.GetCells2D().shape[-1] == 4:
            mesh = remesh(mesh)
            return meshplot(mesh, scalars=scalars, vrange=vrange, cmap=cmap, colorbar=colorbar, cam=cam)
        else:
            mesh = m.GetPoints(), m.GetCells2D()
        
    
    fig = plt.figure(figsize=(5, 6), constrained_layout=False)
    
    # get colormap
    cmap = plt.get_cmap(cmap)
    if isinstance(vrange, float):
        vmin, vmax = -vrange, vrange
    else:
        vmin, vmax = vrange
        
    ax = fig.add_subplot(111, projection='3d')
    # Set the camera view
    ax.view_init(elev=cam[0], azim=cam[1])

    # Arrows for cardinal directions
    # Lateral (X-axis)
    max_scale = m.GetPoints().max()/4
    origin = -5*max_scale
    ax.quiver(origin, origin, origin, 1*max_scale, 0, 0, color='blue', arrow_length_ratio=0.1)
    ax.text(1.1*max_scale+origin, origin, origin, "L", color='blue')

    # Anterior (Y-axis)
    ax.quiver(-5*max_scale, origin, origin, 0, 1*max_scale, 0, color='red', arrow_length_ratio=0.1)
    ax.text(origin, 1.1*max_scale+origin, origin, "A", color='red')

    # Superior (Z-axis)
    ax.quiver(origin, origin, origin, 0, 0, 1*max_scale, color='green', arrow_length_ratio=0.1)
    ax.text(origin, origin, 1.1*max_scale+origin, "S", color='green')
    
    # plot surface
    plotting.plot_surf(mesh, surf_map=scalars, view=cam,
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       axes=ax, colorbar=False)
    
    if colorbar:
        cax = plt.axes([0.9, 0.4, 0.03, 0.3])
        cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
        cbar.set_ticks([])
        cbar.ax.set_title(f'{vmax:.2f}', fontdict={'fontsize':15}, pad=20)
        cbar.ax.set_xlabel(f'{vmin:.2f}', fontdict={'fontsize':15}, labelpad=20)
        
    return fig, ax










