


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