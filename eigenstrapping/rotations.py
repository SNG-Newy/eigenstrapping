"""
Generate random rotation matrices of arbitrary size and rotate matrices
"""
import numpy as np
from numpy.linalg import qr
from sklearn.utils.validation import check_random_state

modes = np

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
    # 1. Generate matrix A
    A = rs.normal(size=(n, n))
    
    # 2. Compute the QR decomposition
    Q, R = qr(A)
    
    # 3. Check the determinant of Q
    if np.linalg.det(Q) <= 0:
        A[:, 0] = -A[:, 0]  # Flip the sign of the first column
        Q, R = qr(A)
    
    return Q

def rotate_matrix(M, method='indirect'):
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
        rot = indirect_method(n)
        M_rotated = np.dot(M, rot)
    
    elif method == 'direct':
        rot = direct_method(n)
        M_rotated = np.dot(M, rot)
    
    else:
        raise ValueError("Method must be one of 'indirect' or 'direct'")
        
    return M_rotated
    
class RotationMatrix:
    """
    Support class for saving, loading and generating random rotation matrices
    """
    def __init__(self, filename):
        self.filename = filename
        self.data = None
        
    def load_single_matrix(self, l, index):
        """
        Load a single matrix for a given group and index
        """
        n = 2 * l + 1
        num_elements = n * n
        
        # compute starting position
        offset = sum(10 * (2 * i + 1) ** 2 for i in range(1, l)) + index * num_elements
        
        # if data is None, memory-map the entire file
        if self.data is None:
            self.data = np.memmap(self.filename, dtype='float64', mode='r')
            
        matrix_data = self.data[offset:offset+num_elements]
        return np.array(matrix_data).reshape((n, n))
    
    
    def generate_rotation_matrices(self, n, num_matrices=10):
        return np.array([indirect_method(n) for _ in range(num_matrices)])
    
    def generate_list(self, max_l, num_matrices=10):
        return [self.generate_rotation_matrices(2 * l + 1, num_matrices) for l in range(1, max_l + 1)]

    def save_structured_matrices(self, all_rotations_list, filename):
        flat = [matrices.flatten() for matrices in all_rotations_list]
        con = np.concatenate(flat)
        
        np.save(filename, con)
    
    def get_random_rotation(self, l):
        """
        Get random rotation matrix from large array.

        Parameters
        ----------
        l : int
            Group parameter l > 0

        Returns
        -------
        np.ndarray of shape (n, n)
            Random rotation matrix where n = 2 * l + 1

        """
        if l <= 0:
            raise ValueError('Group parameter must be greater than zero, got {}'.format(l))
            
        random_index = np.random.randint(0, 1000)
        return self.load_single_matrix(l, random_index)

    
    
    
    
    
    
    
    
    
    