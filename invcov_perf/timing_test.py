import cupy as cp
import numpy as np
import time



def matmul_timing(x, y):
    return cp.matmul(x, y)


start = cp.cuda.Event()
end = cp.cuda.Event()

size = 1000
x1 = cp.random.rand(size, size, dtype=cp.float32)
y1 = cp.random.rand(size, size, dtype=cp.float32)

start.record()
for _ in range(400):
    matmul_timing(x1, y1)

end.record()
end.synchronize()

t_ms = cp.cuda.get_elapsed_time(start, end)/400
print(f"times in ms: {t_ms:.2f}")


def matmul_timing(x, y):
    return np.matmul(x, y)


# start = cp.cuda.Event()
# end = cp.cuda.Event()

size = 1000
x1 = np.random.rand(size, size).astype('float32')
y1 = np.random.rand(size, size).astype('float32')

start = time.time()
for _ in range(400):
    matmul_timing(x1, y1)
stop = time.time()


t_s_cpu = (stop - start)*(1e3)/400

print(f"times in ms: {t_s_cpu:.2f}")