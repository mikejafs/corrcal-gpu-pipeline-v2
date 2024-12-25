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

    Params
    ------
    mat: Gains are applied to this. Can be 2d as in original C-corrcal.
        If 3d, is_zeropadded must be set to True

    Returns
    -------
    out: Matrix with applied gains (explain this a bit better)
    """
    complex_gains = gains[::2] + 1j*gains[1::2]
    gain_mat = complex_gains[ant_1_array, None] * complex_gains[ant_2_array, None].conj()
    # print(gain_mat)

    gain_mat = xp.array(gain_mat, dtype=xp.complex64)
    out = xp.zeros_like(mat)
    print(out.shape)
    print()

    if is_zeropadded:
        re_gains = gains[::2]
        print(re_gains)
        re_gs_a1 = re_gains[ant_1_array]
        # print(re_gs_a1.shape)
        re_gs_a2 = re_gains[ant_2_array]
        re_gains = xp.concatenate((re_gs_a1, re_gs_a2))
        print(re_gains)
        zp_re_gs, _, _ = zeroPad(re_gains, edges, cp)
        print(zp_re_gs.shape)
        print(zp_re_gs)
        zp_re_gs_a1 = zp_re_gs[:int(len(zp_re_gs)/2)]
        zp_re_gs_a2 = zp_re_gs[int(len(zp_re_gs)/2):]
        print(re_gs_a1.shape)
        print(re_gs_a2.shape)
        
        im_gains = gains[1::2]
        im_gs_a1 = im_gains[ant_1_array]
        im_gs_a2 = im_gains[ant_2_array]
        im_gains = xp.concatenate((im_gs_a1, im_gs_a2))
        zp_im_gs, lb, nb = zeroPad(re_gains, edges, cp)
        zp_im_gs_a1 = zp_im_gs[:int(len(zp_im_gs)/2)]
        zp_im_gs_a2 = zp_im_gs[int(len(zp_im_gs)/2):]
        print(im_gs_a1.shape)
        print(zp_re_gs_a1.shape)

        # zp_re_gs_a1, lb, nb = zeroPad(re_gs_a1, edges, cp)
        # zp_im_gs_a1, _, _ = zeroPad(im_gs_a1, edges, cp)
        # zp_re_gs_a2, _, _ = zeroPad(re_gs_a2, edges, cp)
        # zp_im_gs_a2, _, _ = zeroPad(im_gs_a2, edges, cp)
        # print(zp_re_gs_a1.shape)

        zp_cgs_a1 = zp_re_gs_a1 + 1j*zp_im_gs_a1
        zp_cgs_a2 = zp_re_gs_a2 + 1j*zp_im_gs_a2

        print(zp_cgs_a1.shape)
        print(zp_cgs_a2.shape)

        zp_gain_mat = zp_cgs_a1[:, None] * zp_cgs_a2[:, None].conj()
        print(zp_gain_mat)
        
        zp_gain_mat_resh = zp_gain_mat.reshape(mat.shape[0], lb, 1)

        print(f"zp gain mat shape {zp_gain_mat_resh.shape}")
        print(zp_gain_mat_resh.real)
        print(zp_gain_mat_resh.imag)
        
        print(mat[:, ::2].shape)
        print(mat[:, 1::2].shape)
        print(zp_gain_mat_resh.shape)

        out[:, ::2] = zp_gain_mat_resh.real * mat[:, ::2] - zp_gain_mat_resh.imag * mat[:, 1::2]
        out[:, 1::2] = zp_gain_mat_resh.imag * mat[:, ::2] + zp_gain_mat_resh.real * mat[:, 1::2]

    else:
        out[::2] = gain_mat.real * mat[::2] - gain_mat.imag * mat[1::2]
        out[1::2] = gain_mat.imag * mat[::2] + gain_mat.real * mat[1::2]

    return out


