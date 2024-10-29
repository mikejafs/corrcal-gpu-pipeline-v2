import cupy as cp
from cupyx.profiler import benchmark
from zp_puregpu_funcs_py import *

def run(benchmark_zp, return_zp, return_plot, nD):
    n_bl = 120000
    n_eig = 10  #really just the number of cols in the source or diffuse mat's
    n_ant = 500

    #can use this array along with the seaborn heatmap
    #to even more easily check things are working
    # array2d = np.full((n_bl, n_eig), 1, dtype=np.double)

    #random 1d array
    array1d = cp.random.rand(n_bl)

    #Random 2d array and simulated edges array
    array2d = cp.random.rand(n_bl, n_eig)
    edges = cp.unique(cp.random.randint(1, n_bl - 1, size = n_ant))
    edges = cp.concatenate((cp.array([0]), edges, cp.array([n_bl], dtype = cp.int64)))

    if return_zp:
        if nD == 1:
            zp_array, largest_block, n_blocks = zeroPad(array1d, edges)
            zp_array = zp_array.reshape(n_blocks, largest_block)
        elif nD == 2:
            zp_array, largest_block, n_blocks = zeroPad(array2d, edges)
            zp_array = zp_array.reshape(n_blocks*largest_block, n_eig)
        print(zp_array)
        print(edges)

    if return_plot:
        try:
            zp_array
        except NameError:
            raise NameError("You need to return the zp_array if you want to make a plot of it")
        else:
            cp.cuda.Stream.null.synchronize()
            zp_array = cp.asnumpy(zp_array)
            fig, ax = plt.subplots(figsize = (14, 14))        
            sns.heatmap(zp_array, ax = ax)
            # plt.pause(15)
            # plt.close(fig)
            plt.show()
    
    if benchmark_zp:
        test_results = str(benchmark(zeroPad, (array2d, edges), n_repeat=100))
        test_results = test_results.split()
        cpu_t = float(test_results[3])/1e6
        gpu_t = float(test_results[14])/1e6
        print(f"Time on cpu: {cpu_t:.6f}s")
        print(f"Time on gpu: {gpu_t:.6f}s")

if __name__ == "__main__":
    run(benchmark_zp=True, return_zp=False, return_plot=False, nD = 2)



