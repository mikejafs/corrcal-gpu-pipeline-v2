from zp_puregpu_funcs_py import *

def sparse_cov_times_vec(N, Del, Sig, N_inv, Del_prime, Sig_prime, vec, isinv, xp):
    """
    Multiplies a sparse covariance object by a vector from the right
    """
    if vec.ndim == 2:
        vec = vec.reshape(vec.shape[0], vec.shape[1], 1)
        N_inv = N_inv.reshape(vec.shape[0], vec.shape[1], 1)
    else:
        pass
    if isinv:
        del_tmp = xp.transpose(Del_prime, [0, 2, 1]) @ vec
        sig_tmp = xp.sum(xp.transpose(Sig_prime, [0, 2, 1]) @ vec, axis=0)
        out = N_inv * vec - Del_prime @ del_tmp - Sig_prime @ sig_tmp
    else:
        del_tmp = xp.transpose(Del, [0, 2, 1]) @ vec
        sig_tmp = xp.sum(xp.transpose(Sig, [0, 2, 1]) @ vec, axis=0)
        out = N * vec + Del_prime @ del_tmp + Sig_prime @ sig_tmp  
    return out


def apply_gains_to_mat(gains: cp.ndarray, 
                       mat, 
                       edges, 
                       ant_1_array, 
                       ant_2_array, 
                       xp, 
                       is_zeropadded=True
                       ):
    """
    Apply a pair of complex gains to a matrix. Utilizes the Re/Im split.
    Only accounts for "one half" of the gain application, meaning the 
    function is really performing eg. (g_1g_2*\Delta_{1,2}), where it is 
    understood that antenna's 1 and 2 belong to the baseline sitting at
    the same row as that baseline row in the \Delta (\Sigma) matrix. Note that 
    although the matrix provided may be zeropadded, the gain matrix is zeropadded 
    here and as such, should always be provided in an un-zeropadded (original) format.

    NOTE: Could be smart in the future to wrap the zeropadding of the gain 
        matrix in a separate function.

    Params
    ------
    gains
        1D array of Re/Im alternating gains to be applied to the source or
        diffuse matrices. Contains one set of Re/Im gains for all antennas 
        in the array. 
    mat
        Gains are applied to this. Can be 2d as in original C-corrcal.
        If 3d, is_zeropadded must be set to True
    edges
        Indices of edges of redundant blocks in the diffuse matrix.
    ant_1_array
        Indices of the first antenna in each baseline
    ant_2_array
        Indices of the second antenna in each baseline
    xp
        np for cpu (Numpy), cp for gpu (CuPy)
    is_zeropadded
        Boolean. Indicate whether the provided matrix has been zeropadded 
        previously

    Returns
    -------
    out
        Matrix with applied gains (explain this a bit better)
    """

    complex_gains = gains[::2] + 1j*gains[1::2]
    out = xp.zeros_like(mat)

    if is_zeropadded:

        #construct gain mat in the original way
        tmp_gain_mat = (
            complex_gains[ant_1_array, None] * complex_gains[ant_2_array, None].conj()
        )

        #initialize gain mat to be zeropadded
        gain_mat = xp.zeros((len(complex_gains),1))   

        #Re/Im split the gain mat and zeropad using edges array
        gain_mat[::2] = tmp_gain_mat.real
        gain_mat[1::2] = tmp_gain_mat.imag
        zp_gain_mat, largest_block, n_blocks = zeroPad(gain_mat, edges, cp)

        #re-assemble and re-shape the (now zeropadded) complex gain mat
        re_zp_gain_mat = zp_gain_mat[::2]
        im_zp_gain_mat = zp_gain_mat[1::2]
        cplex_gain_mat = re_zp_gain_mat + 1j*im_zp_gain_mat
        cplex_gain_mat = cplex_gain_mat.reshape(n_blocks, largest_block//2, 1)

        #apply the gains
        out[:, ::2] = (
            cplex_gain_mat.real * mat[:, ::2] - cplex_gain_mat.imag * mat[:, 1::2]
        )
        out[:, 1::2] = (
            cplex_gain_mat.imag * mat[:, ::2] + cplex_gain_mat.real * mat[:, 1::2]
        )
    else:
        #apply the gains using the original corrcal routine
        gain_mat = (
            complex_gains[ant_1_array,None] * complex_gains[ant_2_array,None].conj()
        )
        out[::2] = gain_mat.real * mat[::2] - gain_mat.imag * mat[1::2]
        out[1::2] = gain_mat.imag * mat[::2] + gain_mat.real * mat[1::2]
    return out

