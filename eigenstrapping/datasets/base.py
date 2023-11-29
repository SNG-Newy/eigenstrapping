#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data fetchers
"""

import os.path as op
from pathlib import Path
import shutil

import numpy as np
from neuromaps import datasets, images
import nibabel as nib
from .utils import (get_data_dir, _groupby_match,
                    get_dataset_info, _match_files)
from eigenstrapping import dataio

from nilearn.datasets.utils import _fetch_file

def load_surface_examples(data_dir=None, with_surface=False):
    """
    Downloads all surface files for fsaverage5 space

    Parameters
    ----------
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'EIGEN_DATA'; if that is not set, will use
        `~/eigenstrapping-data` instead. Default: None
    with_surface : str, optional
        Flag whether to return the surfaces or not. Default False.

    Returns
    -------
    tuple of ndarrays
        Maps, emodes, and evals for left and right hemispheres.

    """

    space = 'fsaverage'
    version = 'fsaverage5'
    den = '10k'
        
    local_path = op.dirname(__file__)
    
    data_fname = '{}_{}.gii'
    mask_fname = '{}_{}.txt'
    
    data_pth = op.join(local_path, 'brainmaps', data_fname)
    mask_pth = op.join(local_path, 'masks', mask_fname)
    
    surfs = [None] * 2
    data = [None] * 2
    emodes = [None] * 2
    evals = [None] * 2
    masks = [None] * 2
    for i, side in enumerate(['lh', 'rh']):
        surfs[i] = fetch_data(name='surfaces', space=space, den=den,
                              hemi=side)
        data[i] = nib.load(data_pth.format(version, side)).agg_data()
        emodes_file = fetch_data(name='eigenmodes', space=space, den=den, hemi=side,
                   format='emodes')
        emodes[i] = np.loadtxt(emodes_file)
        evals_file = fetch_data(name='eigenmodes', space=space, den=den, hemi=side,
                   format='evals')
        evals[i] = np.loadtxt(evals_file)
        masks[i] = np.loadtxt(mask_pth.format(version, side)).astype(np.bool_)
        data[i][~masks[i]] = np.nan
    
    if with_surface:
        return surfs[0], surfs[1], data[0], data[1], emodes[0], emodes[1], evals[0], evals[1]
    
    return data[0], data[1], emodes[0], emodes[1], evals[0], evals[1]

def load_genepc(join=False):
    """
    Download and return the Allen Human Brain Atlas gene PC1 in
    fsaverage5 space.

    Parameters
    ----------
    join : bool, optional
        If True, return both hemispheres in one array. If False, return
        two hemispheres separately. Default is False.

    Returns
    -------
    tuple of ndarrays of shape (10242,) (10242,)
        if join is True then returns ndarray of shape (20484,)
        Gene PC1 data in fsaverage5 space
    """
    genepc = datasets.fetch_annotation(desc='genepc1', return_single=True)
    data = images.load_data(genepc)
    
    if join:
        return data
    
    data_lh, data_rh = data.reshape((2, 10242))
    
    return data_lh, data_rh

def load_distmat(space='fsaverage', den='10k', data_dir=None, hemi='lh', parcellated=False, sort=True):
    """
    Downloads geodesic distance matrices. If the dense distance matrix is retrieved, 
    there is an option to return the sorted distance matrix and index memmapped, ala
    variogram calculation in ``brainsmash`` in `sort=True`. If `sort=False`,
    the unsorted distance matrix (as per normal) is returned.

    Parameters
    ----------
    space : str, optional
        Which space of the files to get. Default is 'fsaverage5'
    data_dir : str, optional
        Path to use as data directory. If not specified, will check for
        environmental variable 'E_DATA'; if that is not set, will use
        `~/e-data` instead. Default: None
    hemi : str, optional
        Which hemisphere to load data for. Default 'lh'
    parcellated : bool, optional
        If True, return parcellated format of distance matrix. If False, return
        sorted dense distance matrix.
    sort : bool, optional
        If True, return sorted dense distance matrix and index memory-mapped using
        `.utils.txt2memmap()` for variogram calculation. Does nothing if
        `parcellation` is passed.
        
    Returns
    -------
    if parcellated:
        ndarray
            Parcellated distance matrix
    else:
        if sort:
            tuple of ndarrays
                Sorted distance matrix and index for hemisphere in 'lh'
        else:
            ndarray
            Dense geodesic distance matrix

    """
    if parcellated:
        distmat = fetch_data(name='distmat', space=space, den=den,
                             hemi=hemi, format='parc')
        return distmat
    
    distmat = fetch_data(name='distmat', space=space, den=den,
                         hemi=hemi, format='dense')
    
    outfiles = txt2memmap(distmat, get_data_dir(data_dir))
    D = load_memmap(outfiles['distmat'])
    index = load_memmap(outfiles['index'])
        
    return D, index

def load_subcort(structure='thalamus', data_dir=None):
    """
    Loads all the files for the subcortical structures used in the paper -
    cortical-subcortical gradients and the subcortical ROIs from the 25% 
    probability Harvard-Oxford atlas.

    Parameters
    ----------
    structure : str, optional
        Which subcortical structure to return. Default is 'thalamus'.

    Returns
    -------
    data : nib.Nifti1Image
        Subcortical data of cortico-subcortical gradients in MNI152 space.
    mask : nib.Nifti1Image
        Mask image of subcortical structure in MNI152 space (i.e., 1 inside 
        structure and 0 outside structure)

    """
    root_pth = op.dirname(__file__)
    data = nib.load(op.join(root_pth, f'brainmaps/{structure}.txt'))
    mask = nib.load(op.join(root_pth, f'masks/{structure}.txt'))
    
    return data, mask

def load_native(data_dir=None):
    """
    Download HCP native surface and data for demonstration purposes.

    Parameters
    ----------
    data_dir : str, optional
        Path to use . The default is Npne.

    Returns
    -------
    native : dict
        Dictionary of filenames 

    """
    native_surface = fetch_data(name='surfaces', space='native', data_dir=data_dir)
    native_data = fetch_data(name='brainmaps', space='native', data_dir=data_dir)
    
    return {'surface' : native_surface[0], 'data' : native_data[0]}

def fetch_data(*, name=None, space=None, den=None, res=None, hemi=None,
                   tags=None, format=None, data_dir=None, verbose=1):
    """
    Downloads files for brain surfaces and eigenmodes matching requested variables

    Parameters
    ----------
    name, space, den, res, hemi, tags, format : str or list-of-str
        Values on which to match surfaces. If not specified surfaces with
        any value for the relevant key will be matched. Default: None
    data_dir : str, optional
        Path to use as data directory. If not specified, will
        use `~/e-data` instead. Default: None
    verbose : int, optional
        Modifies verbosity of download, where higher numbers mean more updates.
        Default: 1

    Returns
    -------
    data : dict
        Dictionary of downloaded annotations where dictionary keys are tuples
        (space, den/res) and values are lists of corresponding
        filenames
    """

    # check input parameters to ensure we're fetching _something_
    supplied = False
    for val in (space, den, res, hemi, tags, format):
        if val is not None:
            supplied = True
            break
    if not supplied:
        raise ValueError('Must provide at least one parameters on which to '
                         'match files. If you want to fetch all '
                         'annotations set any of the parameters to "all".')

    # get info on datasets we need to fetch
    data_dir = get_data_dir(data_dir=data_dir)
    info = _match_files(get_dataset_info(name),
                        space=space, den=den, res=res,
                        hemi=hemi, tags=tags, format=format)
    if verbose > 1:
        print(f'Identified {len(info)} datsets matching specified parameters')

    # TODO: current work-around to handle that _fetch_files() does not support
    # session instances. hopefully a future version will and we can just use
    # that function to handle this instead of calling _fetch_file() directly
    data = []
    for dset in info:
        fn = Path(data_dir) / dset['rel_path'] / dset['fname']
        if not fn.exists():
            dl_file = _fetch_file(dset['url'], str(fn.parent), verbose=verbose,
                                  md5sum=dset['checksum'])
            shutil.move(dl_file, fn)
        data.append(str(fn))
    
    if len(data) == 1:
        return data[0]
    
    return _groupby_match(data)

def txt2memmap(dist_file, output_dir, maskfile=None, delimiter=' '):
    """
    Export distance matrix to memory-mapped array.

    Parameters
    ----------
    dist_file : filename
        Path to `delimiter`-separated distance matrix file
    output_dir : filename
        Path to directory in which output files will be written
    maskfile : filename or np.ndarray or None, default None
        Path to a neuroimaging/txt file containing a mask, or a mask
        represented as a numpy array. Mask scalars are cast to boolean, and
        all elements not equal to zero will be masked.
    delimiter : str
        Delimiting character in `dist_file`

    Returns
    -------
    dict
        Keys are 'D' and 'index'; values are absolute paths to the
        corresponding binary files on disk.

    Notes
    -----
    Each row of the distance matrix is sorted before writing to file. Thus, a
    second mem-mapped array is necessary, the i-th row of which contains
    argsort(d[i]).
    If `maskfile` is not None, a binary mask.txt file will also be written to
    the output directory.

    Raises
    ------
    IOError : `output_dir` doesn't exist
    ValueError : Mask image and distance matrix have inconsistent sizes

    """

    nlines = count_lines(dist_file)
    if not op.exists(output_dir):
        raise IOError("Output directory does not exist: {}".format(output_dir))

    # Load mask if one was provided
    if maskfile is not None:
        mask = dataio(maskfile).astype(bool)
        if mask.size != nlines:
            e = "Incompatible input sizes\n"
            e += "{} rows in {}\n".format(nlines, dist_file)
            e += "{} elements in {}".format(mask.size, maskfile)
            raise ValueError(e)
        mask_fileout = op.join(output_dir, "mask.txt")
        np.savetxt(  # Write to text file
            fname=mask_fileout, X=mask.astype(int), fmt="%i", delimiter=',')
        nv = int((~mask).sum())  # number of non-masked elements
        idx = np.arange(nlines)[~mask]  # indices of non-masked elements
    else:
        nv = nlines
        idx = np.arange(nlines)

    # Build memory-mapped arrays
    with open(dist_file, 'r') as fp:

        npydfile = op.join(output_dir, "distmat.npy")
        npyifile = op.join(output_dir, "index.npy")
        fpd = np.lib.format.open_memmap(
            npydfile, mode='w+', dtype=np.float32, shape=(nv, nv))
        fpi = np.lib.format.open_memmap(
            npyifile, mode='w+', dtype=np.int32, shape=(nv, nv))

        ifp = 0  # Build memory-mapped arrays one row of distances at a time
        for il, l in enumerate(fp):  # Loop over lines of file
            if il not in idx:  # Keep only CIFTI vertices
                continue
            else:
                line = l.rstrip()
                if line:
                    data = np.array(line.split(delimiter), dtype=np.float32)
                    if data.size != nlines:
                        raise RuntimeError(
                            "Distance matrix is not square: {}".format(
                                dist_file))
                    d = data[idx]
                    sort_idx = np.argsort(d)
                    fpd[ifp, :] = d[sort_idx]  # sorted row of distances
                    fpi[ifp, :] = sort_idx  # sort indexes
                    ifp += 1
        del fpd  # Flush memory changes to disk
        del fpi

    return {'distmat': npydfile, 'index': npyifile}  # Return filenames


def load_memmap(filename):
    """
    Load a memory-mapped array.

    Parameters
    ----------
    filename : str
        path to memory-mapped array saved as npy file

    Returns
    -------
    np.memmap

    """
    return np.load(filename, mmap_mode='r')

def count_lines(filename):
    """
    Count number of lines in a file.

    Parameters
    ----------
    filename : filename

    Returns
    -------
    int
        number of lines in file

    """
    with open(filename, 'rb') as f:
        lines = 0
        buf_size = 1024 * 1024
        read_f = f.raw.read
        buf = read_f(buf_size)
        while buf:
            lines += buf.count(b'\n')
            buf = read_f(buf_size)
        return lines