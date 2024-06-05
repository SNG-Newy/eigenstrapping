from .utils import is_string_like
import nibabel as nib
from pathlib import Path
import numpy as np

"""
Flexible data I/O - based on murraylab/brainsmash
"""


def dataio(x):
    """
    Data I/O for core classes.

    To facilitate flexible user inputs, this function loads data from:
        - neuroimaging files
        - txt files
        - npy files (memory-mapped arrays)
        - array_like data

    Parameters
    ----------
    x : filename or np.ndarray or np.memmap

    Returns
    -------
    np.ndarray or np.memmap

    Raises
    ------
    FileExistsError : file does not exist
    RuntimeError : file is empty
    ValueError : file type cannot be determined or is not implemented
    TypeError : input is not a filename or array_like object

    """
    if is_string_like(x):
        if not Path(x).exists():
            raise FileExistsError("file does not exist: {}".format(x))
        if Path(x).stat().st_size == 0:
            raise RuntimeError("file is empty: {}".format(x))
        if Path(x).suffix == ".npy":  # memmap
            return np.load(x, mmap_mode='r')
        if Path(x).suffix == ".txt":  # text file
            return np.loadtxt(x).squeeze()
        try:
            return load(x)
        except TypeError:
            raise ValueError(
                "expected npy or txt or nii or gii file, got {}".format(
                    Path(x).suffix))
    else:
        if not isinstance(x, np.ndarray):
            raise TypeError(
                "expected filename or array_like obj, got {}".format(type(x)))
        return x


def load(filename):
    """
    Load data contained in a CIFTI2-/GIFTI-format neuroimaging file.

    Parameters
    ----------
    filename : filename
        Path to neuroimaging file

    Returns
    -------
    (N,) np.ndarray
        Brain map data stored in `filename`

    Raises
    ------
    TypeError : `filename` has unknown filetype

    """
    try:
        return _load_gifti(filename)
    except AttributeError:
        try:
            return _load_cifti2(filename)
        except AttributeError:
            raise TypeError("This file cannot be loaded: {}".format(filename))


def _load_gifti(filename):
    """
    Load data stored in a GIFTI (.gii) neuroimaging file.

    Parameters
    ----------
    filename : filename
        Path to GIFTI-format (.gii) neuroimaging file

    Returns
    -------
    np.ndarray
        Neuroimaging data in `filename`

    """
    return nib.load(filename).darrays[0].data

def load_surface(filename):
    """
    Load data stored in a GIFTI (.gii) neuroimaging file.

    Parameters
    ----------
    filename : filename
        Path to GIFTI-format (.gii) neuroimaging file

    Returns
    -------
    np.ndarray
        Neuroimaging data in `filename`

    """
    return nib.load(filename).darrays[0].data, nib.load(filename).darrays[1].data


def _load_cifti2(filename):
    """
    Load data stored in a CIFTI-2 format neuroimaging file (e.g., .dscalar.nii
    and .dlabel.nii files).

    Parameters
    ----------
    filename : filename
        Path to CIFTI-2 format (.nii) file

    Returns
    -------
    np.ndarray
        Neuroimaging data in `filename`

    Notes
    -----
    CIFTI-2 files follow the NIFTI-2 file format. CIFTI-2 files may contain
    surface-based and/or volumetric data.

    """
    return np.asanyarray(nib.load(filename).dataobj).squeeze()
