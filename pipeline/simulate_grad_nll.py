"""
Module designed to test the accuracy of the currently (as of 2025_04_24)
most up-to-date version of the grad_nll function (grad_nll3 folder). Note 
that the way things are simulated here represent the most accurate way
of simulating each parameter that I've reached yet. ant arrays correspond
to the number of antennas being used in calibration (not necessarily the
whole array) and n_bls are adjusted accordingly. -> these latter steps
are the most recent update from the work in the grad2 folder.
"""

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from corrcal import SparseCov
from corrcal.optimize import *
from optimize import *
from fancy_plotting import *
from cupyx.profiler import benchmark


def simulate(n_ant, rand_seed, return_benchmark=True ):
    #TODO: construct a class for simulating these parameters
    #with detailed descriptions of the sizes with motivations

    # Main simulation params
    #----------------------------
    #non-repeating ant indices
    ant_1_array, ant_2_array = cp.tril_indices(n_ant, k=-1)

    #only keep around number of baselines used in calibration (length set by ant indices)
    n_bl = 2*len(ant_1_array)
    n_gains = 2*n_ant #re im split
    n_eig = 3         #number of eigenmodes
    n_src = 5         #number of sources
    xp = cp           # run things on the gpu using cupy
    cp.random.seed(rand_seed)   

    #define edges array
    edges = (xp.unique(xp.random.randint(1, int(n_bl / 2) - 1, size=(n_ant,))* 2))
    edges = xp.concatenate((xp.array([0]), edges, xp.array([n_bl])))
    # print(f"The edges of the redundant blocks have indices{edges}")


    #Constructing matrices for simulation using simulation params
    #& running grad_nll for cpu and gpu
    #------------------------------------------------------------
    """ RUNNING WITH GPU """
    #some random noise, diffuse, source covariance matrices, and gain mat
    xp = cp
    sim_noise_mat = xp.random.rand(n_bl, dtype='float64')
    sim_diff_mat = xp.random.rand(n_bl, n_eig, dtype='float64')
    sim_src_mat = xp.random.rand(n_bl, n_src, dtype='float64')
    sim_gains = xp.random.rand(n_gains, dtype='float64')  # Re/Im split + ant1 & ant 2 = 4*n_ant
    sim_data = xp.random.rand(n_bl, dtype='float64')

    #run gpu version of grad_nll
    # gpu_grad = gpu_grad_nll(sim_gains, sim_noise_mat, sim_diff_mat, sim_src_mat, edges, sim_data, n_ant, ant_1_array, ant_2_array, 1, np.inf)
    


    #----------------------------------------------------
    #TESTING GPU GRAD HERE -> comment out full grad return above
    # gpu_s, gpu_t, gpu_P = gpu_grad_nll(sim_gains, sim_noise_mat, sim_diff_mat, sim_src_mat, edges, sim_data, n_ant, ant_1_array, ant_2_array, 1, np.inf)    
    gpu_gradient = gpu_grad_nll(sim_gains, sim_noise_mat, sim_diff_mat, sim_src_mat, edges, sim_data, n_ant, ant_1_array, ant_2_array, 1, np.inf)    
    #----------------------------------------------------



    # gradr, gradi = gpu_grad_nll(n_ant, sim_gains, sim_data, 1, np.inf, sim_noise_mat, sim_diff_mat, sim_src_mat, edges, ant_1_array, ant_2_array)
    # gradr, gradi = cp.asnumpy(gradr), cp.asnumpy(gradi)

    if return_benchmark:
        gpu_times = str(benchmark(gpu_grad_nll, (sim_gains, sim_noise_mat, sim_diff_mat, sim_src_mat, edges, sim_data, n_ant, ant_1_array, ant_2_array, 1, np.inf), n_repeat=10))
        gpu_times = gpu_times.split()
        gpu_cpu_t = float(gpu_times[3])/1e6
        gpu_gpu_t = float(gpu_times[14])/1e6


    """ RUNNING WITH CORRCAL """
    #Convert everything to Numpy arrays first
    noise_mat = cp.asnumpy(sim_noise_mat)
    src_mat = cp.asnumpy(sim_src_mat)
    diff_mat = cp.asnumpy(sim_diff_mat)
    edges_mat = cp.asnumpy(edges)
    gains_mat = cp.asnumpy(sim_gains)
    data_vec = cp.asnumpy(sim_data)
    ant_1_data = cp.asnumpy(ant_1_array)
    ant_2_data = cp.asnumpy(ant_2_array)

    # #use simulated params to create sparse cov object and feed to cpu grad_nll
    cov = SparseCov(noise_mat, src_mat, diff_mat, edges_mat, n_eig, isinv=False)
    # cpu_grad = grad_nll(gains_mat, cov, data_vec, ant_1_data, ant_2_data, scale=1, phs_norm_fac=np.inf)



    #----------------------------------------------------
    #TESTING CPU GRAD HERE -> comment out full grad return above
    # cpu_s, cpu_t, cpu_P = grad_nll(gains_mat, cov, data_vec, ant_1_data, ant_2_data, scale=1, phs_norm_fac=np.inf)
    cpu_gradient = grad_nll(gains_mat, cov, data_vec, ant_1_data, ant_2_data, scale=1, phs_norm_fac=np.inf)
    #----------------------------------------------------
    


    if return_benchmark:
        cpu_times = str(benchmark(grad_nll, (gains_mat, cov, data_vec, ant_1_data, ant_2_data, 1, np.inf), n_repeat=10))
        cpu_times = cpu_times.split()
        cpu_cpu_t = float(cpu_times[3])/1e6
        cpu_gpu_t = float(cpu_times[14])/1e6

    """ COMPARING OUTPUTS BTWN CPU AND GPU """
    #send gpu stuff to cpu for comparison
    # gpu_grad_np = cp.asnumpy(gpu_grad)

    #variable storing whether cpu grad_nll matches gpu grad_nll
    # truth_check =  np.allclose(gpu_grad_np, cpu_grad)

    #----------------------------------
    #TESTING TRUTH BTWN CPU AND GPU HERE -> comment out full truth check above
    # gpu_s = cp.asnumpy(gpu_s)
    # gpu_t = cp.asnumpy(gpu_t)
    # gpu_P = cp.asnumpy(gpu_P)

    gpu_gradient = cp.asnumpy(gpu_gradient)

    atol, rtol = 1e-7, 1e-5
    truth_check = np.allclose(cpu_gradient, gpu_gradient, atol = atol, rtol = rtol)
    return truth_check, atol, rtol
    #----------------------------------


    if return_benchmark:
        return cpu_cpu_t, cpu_gpu_t, gpu_cpu_t, gpu_gpu_t

    return gpu_grad_np - cpu_grad, cpu_grad, truth_check

    # return gradr, gradi

fancy_plotting(use_tex=True)
def present_grad_nll_tests(
        n_ant,
        n_trials,
        rand_seed,
        print_single_check=True,
        plot_truth_check=True,
        plot_comparison=True,
        save_fig=True,    
        benchmark=True,
        debug_grad=True 
        ):
    
    """
    Run a series of GPU-vs-CPU gradient simulations and visualize the results.

    This helper will:
      - Optionally run and print a single “truth” check.
      - Optionally benchmark and print CPU vs GPU timing.
      - Plot how well the GPU gradient matches the CPU gradient across trials.
      - Plot the raw difference between GPU and CPU gradients across trials.
      - Optionally save all figures to disk.

    Parameters
    ----------
    n_ant : int
        Number of antennas (real+imag split) to simulate.
    n_trials : int, optional
        Number of independent simulation trials for plotting (default: 1).
    print_single_check : bool, optional
        If True, print one simulation’s “truth” output for a quick sanity check (default: True).
    plot_truth_check : bool, optional
        If True, plot GPU vs CPU agreement across trials (default: True).
    plot_comparison : bool, optional
        If True, plot the pointwise difference between GPU and CPU gradients (default: True).
    save_fig : bool, optional
        If True, save the generated plots under `test_plots/` (default: True).
    benchmark : bool, optional
        If True, measure and print CPU & GPU execution times (default: True).

    Returns
    -------
    None
    """
    

    if debug_grad:
        results = []
        for i in range(n_trials):
            print(f"on trial {i}")
            gradr, gradi = simulate(n_ant=n_ant, return_benchmark=benchmark, rand_seed=rand_seed)
            results.append([gradr, gradi])

            # gradr = gradr.reshape(n_ant*n_ant)
            # gradi = gradi.reshape(n_ant*n_ant)
            # plt.plot(gradr)
            # plt.plot(gradi)
        # plt.plot(result)

        # print(results[0])
        # print()
        # print(results[1])
        # print(np.allclose(results[0], results[1]))

        results2 = []
        for i in range(len(results)-1):
            # print(results)
            if np.allclose(results[i], results[i+1]):
                # print("true")
                continue
            else:
                results2.append(i)
        print(results2)
        # plt.show()

    if print_single_check:
        result, cpu_grad, truth = simulate(n_ant = n_ant, return_benchmark=benchmark)
        print(truth)

    if benchmark:
        c_cpu_times, c_gpu_times, g_cpu_times, g_gpu_times = simulate(n_ant = n_ant, rand_seed=None, return_benchmark=benchmark)
        # print(f"\n cpu times: \n \n {cpu_times} \n \n \n" 
        #       f"gpu times: \n \n {gpu_times} \n"
        #       )
        return c_cpu_times, c_gpu_times, g_cpu_times, g_gpu_times 


    if plot_truth_check:
        results_n_ant = []
        for i in range(n_trials):   
            print(f"on trial {i}")
            #'full' return 
            # result, cpu_grad, truth = simulate(n_ant = n_ant, rand_seed=rand_seed, return_benchmark=benchmark)

            #just using this for debugging (catching where the missmatch occurs)
            truth, atol, rtol = simulate(n_ant = n_ant, rand_seed=rand_seed, return_benchmark=benchmark)

            # truth = cp.asnumpy(truth)
            results_n_ant.append(truth)

        # return results_n_ant        
        plt.figure(figsize=(18,10))
        plt.plot(results_n_ant, 'o', markersize=20)
        plt.title(f"Number of antennas = {n_ant}, r_seed = {rand_seed}, atol={atol}, rtol={rtol}", fontsize=26)
        plt.xlabel('N Trials', fontsize=25)
        plt.ylabel(r'Agreement with CPU $\nabla (-\text{log}\mathcal{L})$', fontsize=25)
        if save_fig:
            plt.savefig('tests/debugging_tests/grad_nll_truth nant={}, r_seed={}, atol={}, rtol={}.png'.format(n_ant, rand_seed, atol, rtol), dpi=300, format='png', bbox_inches='tight')
        # plt.show()
        plt.show(block=False)   # display the figure without blocking (doesn't block the rest of script if False)
        plt.pause(3)            # wait 5 seconds (updates the GUI event loop)
        plt.close()  


    if plot_comparison:
        plt.figure(figsize=(18,16))
        for i in range(n_trials):
            print(f"on trial {i}")
            #return the full set of parameters needed for this section;
            #cpu_grad is used only if want relative comparison
            result, cpu_grad, truth = simulate(n_ant = n_ant, return_benchmark=benchmark)
            plt.plot(result, marker='.', lw=0, ms=1)
            plt.title(f"Number of realizations = {n_trials}", fontsize=18)
            plt.xlabel("Number of Antennas (Re Im Split)", fontsize=17)
            plt.ylabel(r"$\nabla log\mathcal{L}_{gpu} - \nabla log\mathcal{L}_{cpu}$", fontsize=17)
        if save_fig:
            plt.savefig('comparison_plots/grad_nll_difference_nant={}.png'.format(n_ant), dpi=300, format='png', bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    """
    **!!** -------- **!!**
    random seed 24 causes a missmatch above n_ant=770:
    -> at n_ant = 800, needs rtol = 0 and atol = 1e-4 for allclose to return True

    for random seed 50:
    -> at n_ant = 800, needs rtol = 0 and atol = 1e-3 for allclose to return True
    """
    random_seed = 24
    # random_seed = 50

    full_bool_list = []
    for i in range(1):
        print(f"on seed {random_seed}")
        bool_list = present_grad_nll_tests(
            n_ant = 800,
            n_trials=1000,
            rand_seed=random_seed,
            print_single_check=False,
            plot_truth_check=True,
            plot_comparison=False,
            save_fig=True,
            benchmark=False,
            debug_grad=False
        )

        # full_bool_list.append(bool_list)
    # print(full_bool_list)

    # for i in range(len(full_bool_list)):
    #     if full_bool_list[i] != True:
    #         print(full_bool_list[i])

    # for i, list in enumerate(full_bool_list):
    #     flag = True
    #     for bool in list:
    #         if not bool:
    #             if flag:
    #                 print(f"false on random seed {i}")
    #             flag = False   


    #-------------------------------------
    #Useful for plotting speed comparisons
    #-------------------------------------

    # ant_list = np.array([2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10])
    # # ant_list = np.array([2**4])

    # ccts = []
    # cgts = []
    # gcts = []
    # ggts = []
    # for n_ant in (ant_list):
    #     print(f"{n_ant}")
    #     cct, cgt, gct, ggt = present_grad_nll_tests(
    #                                 n_ant,
    #                                 n_trials=1,
    #                                 rand_seed=None,
    #                                 print_single_check=False,
    #                                 plot_truth_check=False,
    #                                 plot_comparison=False,
    #                                 save_fig=False,
    #                                 benchmark=True,
    #                                 debug_grad=False
    #                             )
    #     ccts.append(cct)
    #     cgts.append(cgt)
    #     gcts.append(gct)
    #     ggts.append(ggt)
        
    # plt.semilogx(ant_list, ccts, marker='o', linestyle='-', label = 'CPU implementation')
    # # plt.plot(ant_list, cgts, marker='o', linestyle='-', label = 'gpu with corrcal')
    # # plt.plot(ant_list, gcts, marker='o', linestyle='-', label = 'GPU Implementation')
    # plt.semilogx(ant_list, ggts, marker='o', linestyle='-', label = 'GPU Implementation')
    # plt.xlabel('Number of Antennas')
    # plt.ylabel('Average Compute Time (s)')
    # plt.legend()
    # plt.savefig('grad_NLL_times.png', dpi = 300, format='png', bbox_inches='tight')
    # plt.show()


