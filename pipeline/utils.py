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


def apply_gains_to_mat(gains, mat, edges, ant_1_array, ant_2_array, xp, is_zeropadded=True):
    """
    Apply a pair of complex gains to a matrix. Utilizes the Re/Im split.
    Only accounts for "one half" of the gain application, meaning the 
    function is really performing (g_1g_2*\Delta_{1,2}), where it is 
    understood that antenna's 1 and 2 below to the baseline sitting at
    the same row as that baseline row in the \Delta matrix.

    NOTE: Could be smart in the future to wrap the zeropadding of the gain 
        matrix in a separate function.

    Params
    ------
    mat: Gains are applied to this. Can be 2d as in original C-corrcal.
        If 3d, is_zeropadded must be set to True

    Returns
    -------
    out: Matrix with applied gains (explain this a bit better)
    """
    if is_zeropadded:
        complex_gains = gains[::2] + 1j*gains[1::2]
        tmp_gain_mat = complex_gains[ant_1_array, None] * complex_gains[ant_2_array, None].conj()
        
        gain_mat = xp.zeros((len(complex_gains),1))   
        gain_mat[::2] = tmp_gain_mat.real
        gain_mat[1::2] = tmp_gain_mat.imag
        
        zp_gain_mat, largest_block, n_blocks = zeroPad(gain_mat, edges, cp)
        re_zp_gain_mat = zp_gain_mat[::2]
        im_zp_gain_mat = zp_gain_mat[1::2]
        
        cplex_gain_mat = re_zp_gain_mat + 1j*im_zp_gain_mat
        cplex_gain_mat = cplex_gain_mat.reshape(n_blocks, largest_block//2, 1)
        
        out = xp.zeros_like(mat)
        out[:, ::2] = cplex_gain_mat.real * mat[:, ::2] - cplex_gain_mat.imag * mat[:, 1::2]
        out[:, 1::2] = cplex_gain_mat.imag * mat[:, ::2] + cplex_gain_mat.real * mat[:, 1::2]
    else:
        raise NotImplementedError("Under construction")
    
    return out

