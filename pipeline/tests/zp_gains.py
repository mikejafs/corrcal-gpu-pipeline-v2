import numpy as np
import cupy as cp
from zp_puregpu_funcs_py import *

# def apply_gains_to_mat(gains, mat, edges, ant_1_array, ant_2_array, xp, is_zeropadded=True):
#     """
#     Apply a pair of complex gains to a matrix. Utilizes the Re/Im split.
#     Only accounts for "one half" of the gain application, meaning the 
#     function is really performing (g_1g_2*\Delta_{1,2}), where it is 
#     understood that antenna's 1 and 2 below to the baseline sitting at
#     the same row as that baseline row in the \Delta matrix.

#     Params
#     ------
#     mat: Gains are applied to this. Can be 2d as in original C-corrcal.
#         If 3d, is_zeropadded must be set to True

#     Returns
#     -------
#     out: Matrix with applied gains (explain this a bit better)
#     """
#     complex_gains = gains[::2] + 1j*gains[1::2]
#     gain_mat = complex_gains[ant_1_array, None] * complex_gains[ant_2_array, None].conj()
#     # print(gain_mat)

#     gain_mat = xp.array(gain_mat, dtype=xp.complex64)
#     out = xp.zeros_like(mat)
#     print(out.shape)
#     print()

#     if is_zeropadded:
#         re_gains = gains[::2]
#         print(re_gains.shape)

#         re_gs_a1 = re_gains[ant_1_array]
#         # print(re_gs_a1.shape)
#         re_gs_a2 = re_gains[ant_2_array]
#         re_gains = xp.concatenate((re_gs_a1, re_gs_a2))
#         # print(re_gains)

#         zp_re_gs, lb, nb = zeroPad(re_gains, edges, cp)
#         zp_re_gs = zp_re_gs.reshape(nb, lb, 1)
#         print("....")
#         print(re_gains.shape)
#         print(zp_re_gs.shape)
#         print(zp_re_gs)

#         zp_re_gs_a1 = zp_re_gs[:(len(zp_re_gs)//2)]
#         print(zp_re_gs_a1)
#         zp_re_gs_a2 = zp_re_gs[(len(zp_re_gs)//2):]
#         print()
#         print(zp_re_gs_a2)
#         print(re_gs_a1.shape)
#         print(re_gs_a2.shape)
        
#         print(f"im section")
#         im_gains = gains[1::2]
#         im_gs_a1 = im_gains[ant_1_array]
#         im_gs_a2 = im_gains[ant_2_array]
#         im_gains = xp.concatenate((im_gs_a1, im_gs_a2))
#         print(im_gains)
#         zp_im_gs, lb, nb = zeroPad(re_gains, edges, cp)
#         print()
#         print(zp_im_gs)
#         zp_im_gs_a1 = zp_im_gs[:(len(zp_im_gs)//2)]
#         zp_im_gs_a2 = zp_im_gs[(len(zp_im_gs)//2):]
#         # print(im_gs_a1)
#         print(zp_im_gs_a1)
#         print()
#         print(zp_im_gs_a2)

#         # zp_re_gs_a1, lb, nb = zeroPad(re_gs_a1, edges, cp)
#         # zp_im_gs_a1, _, _ = zeroPad(im_gs_a1, edges, cp)
#         # zp_re_gs_a2, _, _ = zeroPad(re_gs_a2, edges, cp)
#         # zp_im_gs_a2, _, _ = zeroPad(im_gs_a2, edges, cp)
#         # print(zp_re_gs_a1.shape)

#         print(f"printing shapes of zp_re_gs_ak")
#         print(zp_re_gs_a1.shape)
#         print(zp_im_gs_a1.shape)        
#         print(zp_re_gs_a2.shape)
#         print(zp_im_gs_a2.shape)

#         zp_cgs_a1 = zp_re_gs_a1 + 1j*zp_im_gs_a1
#         zp_cgs_a2 = zp_re_gs_a2 + 1j*zp_im_gs_a2

#         print(zp_cgs_a1.shape)
#         print(zp_cgs_a2.shape)

#         zp_gain_mat = zp_cgs_a1[:, None] * zp_cgs_a2[:, None].conj()
#         print(zp_gain_mat)
        
#         zp_gain_mat_resh = zp_gain_mat.reshape(mat.shape[0], lb, 1)

#         print(f"zp gain mat shape {zp_gain_mat_resh.shape}")
#         print(zp_gain_mat_resh.real)
#         print(zp_gain_mat_resh.imag)
        
#         print(mat[:, ::2].shape)
#         print(mat[:, 1::2].shape)
#         print(zp_gain_mat_resh.shape)

#         out[:, ::2] = zp_gain_mat_resh.real * mat[:, ::2] - zp_gain_mat_resh.imag * mat[:, 1::2]
#         out[:, 1::2] = zp_gain_mat_resh.imag * mat[:, ::2] + zp_gain_mat_resh.real * mat[:, 1::2]

#     else:
#         out[::2] = gain_mat.real * mat[::2] - gain_mat.imag * mat[1::2]
#         out[1::2] = gain_mat.imag * mat[::2] + gain_mat.real * mat[1::2]

#     return out


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
    # reg = gains[::2]
    # reg_ant1_sorted = reg[ant_1_array]
    # reg_ant2_sorted = reg[ant_2_array]
    # reg_sorted = xp.concatenate((reg_ant1_sorted, reg_ant2_sorted))

    # img = gains[1::2]
    # img_ant1_sorted = img[ant_1_array]
    # img_ant2_sorted = img[ant_2_array]
    # img_sorted = xp.concatenate((img_ant1_sorted, img_ant2_sorted))

    # zp_reg, lg, ng = zeroPad(reg, edges, cp)
    # zp_img, lg, ng = zeroPad(reg, edges, cp)
    # zp_reg, lg, ng = zeroPad(reg, edges, cp)
    # zp_img, lg, ng = zeroPad(reg, edges, cp)

    # print(zp_reg)
    # zp_complex_gains = zp_reg + 1j*zp_img
    # print(zp_complex_gains)
    complex_gains = gains[::2] + 1j*gains[1::2]
    print(f"cplex gains shape {complex_gains.shape}")

    # print(zp_complex_gains.shape)
    gain_mat = complex_gains[ant_1_array, None] * complex_gains[ant_2_array, None].conj()
    print(gain_mat)
    print(gain_mat.imag.shape)

    new_gain_mat = xp.zeros((len(complex_gains),1))
    print(gain_mat.real.shape)
    print(new_gain_mat.shape)
    # print(new_gain_mat)
    
    
    new_gain_mat[::2] = gain_mat.real
    new_gain_mat[1::2] = gain_mat.imag

    zp_new_gain_mat, lb, nb = zeroPad(new_gain_mat, edges, cp)
    print(lb, nb)
    print(f"new zp length {zp_new_gain_mat.shape}")

    re_zp_new_gain_mat = zp_new_gain_mat[::2]
    im_zp_new_gain_mat = zp_new_gain_mat[1::2]

    new_cplex_gain_mat = re_zp_new_gain_mat + 1j*im_zp_new_gain_mat
    print(new_cplex_gain_mat)
    print(new_cplex_gain_mat.shape)

    new_cplex_gain_mat = new_cplex_gain_mat.reshape(nb, lb//2, 1)
    # print(new_gain_mat)
    # re_gain_mat = gain_mat.real
    # im_gain_mat = gain_mat.imag
    
    # print("printing zp re and im mat")
    # zp_re_gain_mat, lb, nb = zeroPad(re_gain_mat, edges, cp)
    # zp_im_gain_mat, _, _ = zeroPad(im_gain_mat, edges, cp)

    # print(zp_re_gain_mat)
    # print(zp_im_gain_mat)
    
    # zp_gain_mat = zp_re_gain_mat  + 1j*zp_im_gain_mat
    # print(zp_gain_mat)

    # zp_gain_mat_resh = zp_gain_mat.reshape(nb, lb, 1)
    # print(zp_gain_mat_resh)
    # print(zp_gain_mat_resh.shape)

    out = xp.zeros_like(mat)
    print(out.shape)
    

    out[:, ::2] = new_cplex_gain_mat.real * mat[:, ::2] - new_cplex_gain_mat.imag * mat[:, 1::2]
    out[:, 1::2] = new_cplex_gain_mat.imag * mat[:, ::2] + new_cplex_gain_mat.real * mat[:, 1::2]

    # out[:, ::2] = gain_mat.real * mat[:, ::2] - gain_mat.imag * mat[:, 1::2]
    # out[:, 1::2] = gain_mat.imag * mat[:, ::2] + gain_mat.real * mat[:, 1::2]

    return out

if __name__ == "__main__":
    
    #simulation params
    n_ant = 10
    n_bl = 2*n_ant
    n_gains = 4*n_ant
    n_eig = 3
    xp = cp  #run things on the gpu using cupy

    #random array of edges for the diffuse matrix
    # edges = xp.unique(xp.random.randint(1, n_bl-1, size = 3))
    # edges = xp.concatenate((xp.array([0]), edges, xp.array([n_bl])))

    #NOTE: This all only currently works if there are no odd digits in the
    #   edges array. This makes sense due to the Re/Im splits, though
    #   have to double check that things will actually be set up this way
    edges = xp.array([0, 4, 6, 10, 20])
    print(f"The edges of the redundant blocks have indices{edges}")

    #some random noise, diffuse, and source covariance matrices
    sim_diff_mat = xp.random.rand(n_bl, n_eig, dtype = 'float64')
    sim_gains = cp.random.rand(n_gains, dtype = 'float64') #Re/Im split + ant1 & ant 2 = 4*n_ant
    ant_1_array = cp.arange(n_ant)
    ant_2_array = cp.arange(n_ant, 2*n_ant)
    # print(ant_1_array, ant_2_array)
    # print(ant_1_array.shape, ant_2_array.shape)

    #zeropad the noise, diff, source mats
    zp_sim_diff_mat, largest_block, n_blocks = zeroPad(sim_diff_mat, edges, return_inv=False)

    #Need to reshape to give an extra dimension of n_blocks to be compatible with inv cov routine
    sim_diff_mat_3d = zp_sim_diff_mat.reshape(n_blocks, largest_block, n_eig)
    # print(sim_diff_mat_3d)

    applied_gains = apply_gains_to_mat(sim_gains, sim_diff_mat_3d, edges, ant_1_array, ant_2_array, cp, True)
    print(applied_gains)
        