"""
Module designed for simulating the NLL and comparing output
to corrcal's.
"""

import cupy as cp
import numpy as np
from corrcal import SparseCov
from corrcal.optimize import *
from optimize import *
from simulate_params import *
from cupyx.profiler import benchmark
from fancy_plotting import *

def simulate_nll(n_ant, n_eig, n_src, return_ans, return_benchmark):

    spms = SimCorrcalParams(n_ant, n_eig, n_src, xp=cp)
    ant_1_array, ant_2_array = spms.ant_arrays()
    # n_bl = spms.n_bl()
    edges = spms.edges()

    sim_data = spms.sim_data()
    noise = sim_data[0]
    diff = sim_data[1]
    src = sim_data[2]
    gains = sim_data[3]
    data = sim_data[4]

    """ SIMULATE GPU VERSION """
    if return_ans:
        nll_gpu = gpu_nll(gains,
                    noise,
                    diff,
                    src,
                    data,
                    edges,
                    ant_1_array,
                    ant_2_array,
                    scale=1,
                    phs_norm_fac=cp.inf
                    )

    if return_benchmark:
        gpu_times = str(benchmark(gpu_nll, (gains, noise, diff, src, data, edges, ant_1_array, ant_2_array, 1, cp.inf), n_repeat = 100))
        gpu_times = gpu_times.split()
        gpu_cpu_t = float(gpu_times[3])/1e6
        gpu_gpu_t = float(gpu_times[14])/1e6

    # print(f"gpu nll dtype {(nll_gpu.shape)}")
    # print(nll_gpu)

    """ SIMULATE CPU VERSION """
    noise = cp.asnumpy(noise)
    diff = cp.asnumpy(diff)
    src = cp.asnumpy(src)
    gains = cp.asnumpy(gains)
    data = cp.asnumpy(data)
    edges = cp.asnumpy(edges)
    ant_1_array = cp.asnumpy(ant_1_array)
    ant_2_array = cp.asnumpy(ant_2_array)


    cov = SparseCov(noise, src, diff, edges, spms.n_eig, isinv=False)
    # cinv, logd = cov.inv(return_det=True)
    # print(f"cpu log det is {logd}")
    if return_ans:
        nll_cpu = nll(gains, cov, data, ant_1_array, ant_2_array, scale=1, phs_norm_fac=np.inf)

        """return comparison results"""
        #send gpu soln to cpu to compare
        gpu_nll_result = cp.asnumpy(nll_gpu)

        #store comparison check as a variable
        truth_check = np.allclose(nll_cpu, gpu_nll_result)

    if return_benchmark:
        cpu_times = str(benchmark(nll, (gains, cov, data, ant_1_array, ant_2_array, 1, np.inf), n_repeat=100))
        cpu_times = cpu_times.split()
        cpu_cpu_t = float(cpu_times[3])/1e6
        cpu_gpu_t = float(cpu_times[14])/1e6
        # print(f"Time on cpu: {cpu_t:.6f}s")
        # print(f"Time on gpu: {gpu_t:.6f}s")


    if return_benchmark:
        return  cpu_cpu_t, cpu_gpu_t, gpu_cpu_t, gpu_gpu_t
    
    if return_ans:
        return gpu_nll_result - nll_cpu, nll_cpu, truth_check


def present_nll_tests(
        n_ant,
        n_eig,
        n_src,
        n_trials=1,
        return_ans=True,
        print_single_check=True,
        plot_truth_check=True,
        plot_comparison=True,
        save_fig=True,
        benchmark=True,
        ):
    """
    
    """
    if return_ans:    
        if print_single_check:
            comp_result, nll_cpu, truth = simulate_nll(n_ant, n_eig, n_src, return_ans=return_ans, return_benchmark=benchmark)
            print(truth)

        if plot_truth_check:
            results_n_ant = []
            for i in range(n_trials):
                print(f"on trial {i}")
                result, cpu_nll, truth = simulate_nll(n_ant, n_eig, n_src, return_ans=return_ans, return_benchmark=benchmark)
                results_n_ant.append(truth)

            plt.figure(figsize=(18,16))
            plt.plot(results_n_ant, 'o')
            plt.title(f"NLL GPU/CPU Comparison (n_ant{n_ant})", fontsize=18)
            plt.xlabel('N Trials', fontsize=17)
            plt.ylabel(r'Agreement with CPU $(-\text{log}\mathcal{L})$', fontsize=17)
            if save_fig:
                plt.savefig('comparison_plots/nll_truth_comparison_nant={}.png'.format(n_ant), dpi=300, format='png', bbox_inches='tight')
            plt.show()

        
        if plot_comparison:
            plt.figure(figsize=(18,16))
            for i in range(n_trials):
                print(f"on trial {i}")
                result, cpu_nll, truth = simulate_nll(n_ant, n_eig, n_src, return_ans=return_ans, return_benchmark=benchmark)
                plt.plot(result, marker='.', lw=0, ms=1)
                plt.title(f"Number of realizations = {n_trials}", fontsize=18)
                plt.xlabel("Number of Antennas (Re Im Split)", fontsize=17)
                plt.ylabel(r"$ log\mathcal{L}_{gpu} - log\mathcal{L}_{cpu}$", fontsize=17)
            if save_fig:
                plt.savefig('comparison_plots/nll_difference_for_nant={}.png'.format(n_ant), dpi=300, format='png', bbox_inches='tight')
            plt.show()

    if benchmark:
        c_cpu_times, c_gpu_times, g_cpu_times, g_gpu_times = simulate_nll(n_ant, n_eig, n_src, return_ans, return_benchmark=benchmark)
        return c_cpu_times, c_gpu_times, g_cpu_times, g_gpu_times 
        # print(f"\n cpu times: \n \n {cpu_times} \n \n \n" 
        #       f"gpu times: \n \n {gpu_times} \n"
        #       )



if __name__ == "__main__":
    # n_ant = 10
    n_eig = 3
    n_src = 5

    fancy_plotting(use_tex=True)
    # present_nll_tests(
    #     n_ant,
    #     n_eig,
    #     n_src,
    #     n_trials=10,
    #     return_ans=True,
    #     print_single_check=True,
    #     plot_truth_check=False,
    #     plot_comparison=False,
    #     save_fig=False,
    #     benchmark=False,
    # )

    ant_list = np.array([2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10])
    # ant_list = np.array([2**4])

    ccts = []
    cgts = []
    gcts = []
    ggts = []
    for n_ant in (ant_list):
        print(f"{n_ant}")
        cct, cgt, gct, ggt = present_nll_tests(
                                    n_ant,
                                    n_eig,
                                    n_src,
                                    n_trials=1,
                                    return_ans=False,
                                    print_single_check=False,
                                    plot_truth_check=False,
                                    plot_comparison=False,
                                    save_fig=False,
                                    benchmark=True,
                                )
        ccts.append(cct)
        cgts.append(cgt)
        gcts.append(gct)
        ggts.append(ggt)
        
    plt.semilogx(ant_list, ccts, marker='o', linestyle='-', label = 'CPU implementation')
    # plt.plot(ant_list, cgts, marker='o', linestyle='-', label = 'gpu with corrcal')
    # plt.plot(ant_list, gcts, marker='o', linestyle='-', label = 'GPU Implementation')
    plt.semilogx(ant_list, ggts, marker='o', linestyle='-', label = 'GPU Implementation')
    plt.xlabel('Number of Antennas')
    plt.ylabel('Average Compute Time (s)')
    plt.legend()
    plt.savefig('NLL_times.png', dpi = 300, format='png', bbox_inches='tight')
    plt.show()
