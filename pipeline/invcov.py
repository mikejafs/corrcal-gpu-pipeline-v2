"""Add working routines from mat inverse v8 and create class around them"""

#add the full working routines here and create a class, then,
#create another file that imports the class and uses them to perform the validation tests and timing tests


#For benchmarking, should probably create a class that uses the benchmark code or maybe just a function for now
#that is in its own folder called tests or something. The file name can be something like benchmark.py... We can expand
#upon this later...

"""
Module containing the routines required to perform the inverse covariance calculation.
"""

import numpy as np
import cupy as cp
import seaborn as sns
import matplotlib.pyplot as plt
import corrcal
from cupyx.profiler import benchmark

def zeropad(array, edges, xp):
    """
    Pads an input noise, diffuse, or source matrix with zeros according to
    the largest redundant block in the diffuse matrix. Note that although the input arrays
    will have a 1D or 2D shape for noise or diff/sourc matrices respectively, the routine converts 
    to either 2D or 3D shapes so we can easily perform block mutliplication in the inverse covariance
    function.

    Parameters
    ----------
    array: Input noise, diffuse, or source matrix. Should be of shape (n_bl,), (n_bl, n_eig), or (n_bl, n_src) 
        respectively
    edges: Array containing indices corresponding to the edges of redundant blocks in the diffuse matrix
        Note that the "edges" index the beginning row (or "edge") of each redundant block
    
    Returns
    -------
    out: The output zero-padded noise, diffuse, or source matrix where each matrix has also been reshaped 
        to be easily used in the inverse covariance function that performs mutliplication over blocks. The 
        output matrices have shapes of either (n_blocks, largest_red_block), (n_blocks, largest_red_block, n_eig), 
        or (n_blocks, largest_red_block, n_src) respectively.
    """

    largest_block = xp.diff(edges).max()
    n_blocks = edges.size - 1

    if array.ndim == 1:   #should only be the case for the noise matrix
        out = xp.zeros((n_blocks, int(largest_block)))
    else:
        out  = xp.zeros((n_blocks, int(largest_block), int(array.shape[1])))

    for block, (start, stop) in enumerate(zip(edges, edges[1:])):
        start, stop = int(start), int(stop)
        out[block, :stop - start] = array[start:stop]
    
    return out


def undo_zeropad(array, edges, xp):
    """
    Undoes (essentially does the exact opposite of) the work of the zeropad function. Also 'undoes'
    the re-shaping to a vectorized array with n_blocks along the first axis, etc...
    
    Parameters
    ----------
    array: Input noise, diffuse, or source matrix. Should be of shape (n_blocks, largest_block, 1),
        (n_blocks, largest_block, n_eig), or (n_blocks, largest_block, n_src) respectively
    edges: Array containing indices corresponding to the edges of redundant blocks in the diffuse matrix.
        Note that the "edges" index the beginning row (or "edge") of each redundant block

    Returns
    -------
    out: A dense, unzero-padded array. If an array is given to the zeropad function, this routine will return that original array
        provided the edges array is the same as the one used to pad the original array with zeros.
    """

    if array.ndim == 2:   #once again only the case for the noise matrix
        out = xp.zeros((int(edges[-1])))
    else:
        out = xp.zeros((int(edges[-1]), int(array.shape[2])))

    for block, (start, stop) in enumerate(zip(edges, edges[1:])):
        start, stop = int(start), int(stop)
        out[start:stop] = array[block, :stop - start]

    return out


def inverse_covariance(N, Del, Sig, edges, xp, ret_det = False, N_is_inv = True):
    """
    Given the components of the 2-level sparse covariance object, computes 
    the components of the inverse covariance object. Currectly does not 
    support the option to return the determinant of the covariance.

    TODO: Add option to return the determinant

    Parameters
    ----------
    N: Noise 
    Del: \Delta (diffuse) sky component matrix with shape n_bl x n_eig
    Sig: \Sigma Source component matrix with shape n_bl x n_src
    edges: Array controlling the start and stop of the redundant blocks in the sparse diffuse matrix
    xp: Choice of running on the gpu (xp = cp) or cpu (xp = np)
    ret_det: Option to return the log(det(C)) along with the inverse covariance. Defaults to False

    Returns
    -------
    N^-1: Inverse noise matrix
    Del': The primed version of the diffuse sky matrix
    Sig': The primed version of the source component matrix
    """

    if N_is_inv:
        N_inv = N
    else:
        N_inv = 1/N

    temp = N_inv[..., None] * Del  
    temp2 = xp.transpose(Del, [0, 2, 1]) @ temp
    L_del = xp.linalg.cholesky(xp.eye(Del.shape[2])[None, ...] + temp2)   
    Del_prime = temp @ xp.transpose(xp.linalg.inv(L_del).conj(), [0, 2, 1]) 
          
    A = N_inv[..., None] * Sig
    B = xp.transpose(Sig.conj(), [0, 2, 1]) @ Del_prime
    W = A - Del_prime @ xp.transpose(B.conj(), [0, 2, 1])
    L_sig = xp.linalg.cholesky(
        xp.eye(Sig.shape[2]) + xp.sum(
            xp.transpose(A.conj(), [0, 2, 1]) @ Sig, axis = 0
        ) - xp.sum(
            B @ xp.transpose(B.conj(), [0, 2, 1]), axis = 0
        )
    )
    Sig_prime = W @ xp.linalg.inv(L_sig).T.conj()[None, ...]

    #TODO: TRYING TO GET DET PART OF CODE TO WORK... Current problems
    # - problems referencing logdet before assignment (prob just need to restart
    # VScode or something)
    # - L_del and L_sig (apparently) aren't 1- or 2-D.. Need to look into this
    #UPDATE: The problem is that they are 3-d (ie. still in 'block' form)
    #need to figure out the best way to sum all the diags given this is the case
    # print(L_del.shape)
    # print(L_sig.shape)

    #NOTE: Removed the 2 that I was multiplying the det expression by, since I also forgot about the squareroot,
    #meaning taking the log should means that we can factor out the 1/2 when saying the likelihood is proportional 
    #to... (stuff)

    if ret_det:
        # logdet = 2*(xp.sum(xp.diag(L_del)) + xp.sum(xp.diag(L_sig)))
        #the line should actually be -> Need to check why this works and how differ from xp.diag
        logdet = (xp.sum(xp.diagonal(xp.log(L_del), axis2 = 1, axis1 = 2)) + xp.sum(xp.diagonal(xp.log(L_sig))))
        # cp.cuda.Stream.null.synchronize()
        return logdet, N_inv, Del_prime, Sig_prime 
    else:
        pass
    # cp.cuda.Stream.null.synchronize()
    return N_inv, Del_prime, Sig_prime


def sparden_convert(Array, largest_block, n_blocks, n_bl, n_eig, edges, xp, zeroPad=True):
    """
    Converts either the dense diffuse matrix to sparse, or the sparse diffuse matrix to dense.
    The array (either dense or sparse) should be simply handed to the function and the desired operation
    (sparse-to-dense or dense-to-sparse) will be performed automatically

    Parameters
    ----------
    Array: Either dense or sparse diffuse matrix
    n_bls: Number of baselines used in the calculation of the sparse diffuse matrix
    n_eig: Number of eigenmodes being used to construct the sparse diffuse matrix
    edges: An array controlling the edges of the redundant group blocks in the sparse diffuse matrix
    xp: The choice to either run the computation on the gpu (xp = cp) or cpu (xp = np)

    Returns
    -------
    out: If the dense form was provided, the sparse form with shape (n_bls x n_eig) will be returned.
        If the sparse form was provided, the dense form with shape (n_bls x n_eig*n_grps) with
        n_grps = # redundant groups will be returned.
    """
    
    if Array.shape[1] == n_eig:
        n_grp = edges.size - 1

        if zeroPad:
            out = xp.zeros((n_blocks*largest_block, n_eig*n_grp))
            for i in range(n_blocks):
                out[i*largest_block:(i+1)*largest_block, i*n_eig:(i+1)*n_eig
                    ] = Array[i*largest_block:(i+1)*largest_block]
        else:
            out = xp.zeros((n_bl, n_eig*n_grp))
            for i, (start, stop) in enumerate(zip(edges, edges[1:])):
                out[start:stop, i*n_eig : (i+1)*n_eig] = Array[start:stop]
    else:
        if zeroPad:
            raise NotImplementedError("Dense to sparse has not been implimented with zeropadded arrays yet")
            # out = xp.zeros((n_blocks * largest_block))
        else:
            out = xp.zeros((n_bl, n_eig))
            for i, (start, stop) in enumerate(zip(edges, edges[1:])):
                out[start:stop] = Array[start:stop, i*n_eig : (i+1)*n_eig]

    return out


# print("hello world")


