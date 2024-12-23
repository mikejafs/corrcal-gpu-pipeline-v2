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


def apply_gains_to_mat(gains, mat, ant_1_array, ant_2_array, xp):
    """
    Apply a pair of complex gains to a matrix. Utilizes the Re/Im split.
    Only accounts for "one half" of the gain application, meaning the 
    function is really performing (g_1g_2*\Delta_{1,2}), where it is 
    understood that antenna's 1 and 2 below to the baseline sitting at
    the same row as that baseline row in the \Delta matrix.
    """
    complex_gains = gains[::2] + 1j*gains[1::2]
    gain_mat = complex_gains[ant_1_array, None] * complex_gains[ant_2_array, None].conj()
    out = xp.zeros_like(mat)
    out[::2] = gain_mat.real * mat[::2] - gain_mat.imag * mat[1::2]
    out[1::2] = gain_mat.imag * mat[::2] + gain_mat.real * mat[1::2]
    return out


