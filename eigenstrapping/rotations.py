"""
Generate random rotation matrices of arbitrary size and rotate matrices
"""
import numpy as np
from numpy.linalg import qr
from sklearn.utils.validation import check_random_state
from scipy.stats import special_ortho_group
from pathlib import Path
import warnings
from scipy.sparse import csc_matrix, block_diag, load_npz, save_npz
from .utils import _get_eigengroups_from_num
import uuid
import pickle

def load_rotation_blocks(file, n_surrs=None):
    """
    Load rotation block matrices from `file`.

    Parameters
    ----------
    file : os.Pathlike
        Filepath to file containing rotations to load
    n_surrs : int, optional
        Number of rotations to retain (i.e., subset data). Default is None (use all)

    Returns
    -------
    rotations : (N, N, P) csc_matrix
        Loaded rotation block matrices

    """
    try:
        fn = Path(file).with_suffix('')
        if fn.exists():
            with open(fn, 'rb') as fp:
                rotations = pickle.load(fp)
        else:
            rotations = np.loadtxt(file, delimiter=' ', dtype='float32')
    except TypeError:
        rotations = np.asarray(file, dtype='float32')
    
    if n_surrs is not None:
        rotations = rotations[..., :n_surrs]
        
    return rotations

def direct_method(n, seed=None):
    rs = check_random_state(seed)
    # 1. Draw n independent random normal N(0, 1) variates
    v = rs.normal(size=n)
    
    # 2. Normalize
    x = v / np.linalg.norm(v)
    
    # 3. Treat the vector as a single column matrix
    X = x[:, np.newaxis]
    
    # 4. Apply QR decomposition
    Q, _ = qr(X)
    
    return Q

def indirect_method(n, seed=None):
    
    rs = check_random_state(seed)
    
    # Compute the QR decomposition

    if n < 2:
        return rs.normal(size=(n, n))
    rotate = special_ortho_group.rvs(dim=n, random_state=rs)
    
    return rotate

def rotate_matrix(M, method='indirect', seed=None):
    """
    Rotate an (n/m)-by-n matrix of arbitrary length n by the two methods 
    as outlined in [1].

    Parameters
    ----------
    M : 2D np.ndarray
        Input matrix
    method : str, optional
        Which method to use. Refers to the nomenclature in [1], where
        'indirect' refers to the Householder QR decomposition method [2],
        while 'direct' refers to the method of selecting random points
        on the unit n-sphere directly as in [3]. The default is 'indirect'.

    Returns
    -------
    X_rotated : TYPE
        DESCRIPTION.
        
    References
    ----------
    [1] Blaser R, Fryzlewicz P. Random Rotation Ensembles. Journal of 
        Machine Learning Research. 2016;17(4):1-26.
    [2] Householder A. S. Unitary triangularization of a nonsymmetric 
        matrix. Journal of the ACM, 5:339–342, 1958.
    [3] Knuth D. E. Art of computer programming, volume 2: 
        seminumerical algorithms. Addison-Wesley Professional, Reading, 
        Mass, 3 edition edition, November 1997.

    """
    n = M.shape[1]
    if method == 'indirect':
        rot = indirect_method(n, seed=seed)
        M_rotated = np.dot(M, rot)
    
    elif method == 'direct':
        rot = direct_method(n)
        M_rotated = np.dot(M, rot)
    
    else:
        raise ValueError("Method must be one of 'indirect' or 'direct'")
        
    return M_rotated

def gen_rotations(num_groups, n_surrs=1000, seed=None, verbose=False, save_rotations=False,
                  savedir='/tmp'):
    """
    Generate an array of block diagonal rotation matrices obtained from the
    SO(n) group. Generates random rotation matrices as a list and arranges them
    in a block diagonal matrix for easier saving and loading.
    
    The format of the block diagonals (size and number) are based on the multiplicity
    factor of spherical harmonic groups with size n = 2l + 1, where l is the group
    and n is the number of harmonics/modes in that group (see eigenstrapping
    paper in `ref: <references>`for more information).

    Parameters
    ----------
    num_groups : int
        Number of eigenmode/SH groups
    n_surrs : int, optional
        Number of rotations to generate. Default is 1000.
    seed : int or np.random.RandomState instance or None, optional
        Seed for random number generation, used for consistency across runs. 
        Default is None.
    verbose : bool, optional
        Print status messages. Default is False.
    save_rotations : bool, optional
        If True, saves rotations to `savedir` under a universal unique identifier.
        Default is False.
    savedir : str, optional
        Directory to save rotations to. Default '/tmp'

    Returns
    -------
    If save_rotations:
        fn : str, path_like
            Filepath to pickle dumped list of rotation block diagonals.
            
    rotations : (N, N, `n_surrs`)
        List of rotation block diagonals

    """
    # get eigengroups for number of groups
    len_groups = _get_eigengroups_from_num(num_groups)
    
    # empty arrays to store block diags
    rotations = []
    
    if save_rotations:
        uid = str(uuid.uuid4())
        fn = Path(savedir, uid)
        ms = 'Initializing rotation blocks and saving to {}'.format(str(fn))
        verbose = True
    
    # generate rotations and convert to block diagonal matrices
    ms = ''
    for n in range(n_surrs):
        
        if verbose:
            ms = 'Generating rotation matrix {:>5} of {:>5}'.format(n, n_surrs)
            print(ms, end='\r', flush=True)
            
        rotated = [indirect_method(m) for m in len_groups]
        rotated = block_diag(rotated, format='csc')
            
        rotations.append(rotated)
    
    if verbose:
        print(' ' * len(ms) + '\b' * len(ms), end='', flush=True)
    
    if save_rotations:
        with open(fn, 'wb') as fp:
            pickle.dump(rotations, fp)
        
        return fn
    
    return rotations
    
# class RotationMatrix:
#     """
#     Support class for saving, loading and generating random rotation matrices
#     """
#     def __init__(self, filename):
#         self.filename = filename
#         self.data = None
        
#     def load_single_matrix(self, l, index):
#         """
#         Load a single matrix for a given group and index
#         """
#         n = 2 * l + 1
#         num_elements = n * n
        
#         # compute starting position
#         offset = sum(10 * (2 * i + 1) ** 2 for i in range(1, l)) + index * num_elements
        
#         # if data is None, memory-map the entire file
#         if self.data is None:
#             self.data = np.memmap(self.filename, dtype='float64', mode='r')
            
#         matrix_data = self.data[offset:offset+num_elements]
#         return np.array(matrix_data).reshape((n, n))
    
    
#     def generate_rotation_matrices(self, n, num_matrices=10):
#         return np.array([indirect_method(n) for _ in range(num_matrices)])
    
#     def generate_list(self, max_l, num_matrices=10):
#         return [self.generate_rotation_matrices(2 * l + 1, num_matrices) for l in range(1, max_l + 1)]

#     def save_structured_matrices(self, all_rotations_list, filename):
#         flat = [matrices.flatten() for matrices in all_rotations_list]
#         con = np.concatenate(flat)
        
#         np.save(filename, con)
    
#     def get_random_rotation(self, l):
#         """
#         Get random rotation matrix from large array.

#         Parameters
#         ----------
#         l : int
#             Group parameter l > 0

#         Returns
#         -------
#         np.ndarray of shape (n, n)
#             Random rotation matrix where n = 2 * l + 1

#         """
#         if l <= 0:
#             raise ValueError('Group parameter must be greater than zero, got {}'.format(l))
            
#         random_index = np.random.randint(0, 1000)
#         return self.load_single_matrix(l, random_index)

    
    
    
    
    
    
    
    
    
    
