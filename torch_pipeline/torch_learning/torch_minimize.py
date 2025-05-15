import torch
import numpy as np
import time
from torchmin import minimize

# 1) Device + dtype
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float32
print(device)

# 2) N‑dim Rosenbrock
def rosen_nd(x: torch.Tensor) -> torch.Tensor:
    return torch.sum((1 - x[:-1])**2 + 100*(x[1:] - x[:-1]**2)**2)

# 3) Big problem size
N = 100_000
# x0 = torch.ones(N, device=device, dtype=dtype, requires_grad=True)
# x0 = x0 + torch.randn(N, device=device, dtype=dtype, requires_grad=True)
# print(x0[:10])

x0 = torch.randn(N, device=device, dtype=dtype, requires_grad=True) + 10
print(x0[:10])

# 4) Warm‑up
_ = rosen_nd(x0)         # forward
_ = torch.autograd.grad(rosen_nd(x0), x0)[0]  # backward now works!

# 5) Timed forward+backward
torch.cuda.synchronize()
t0 = time.time()
f = rosen_nd(x0)
g, = torch.autograd.grad(f, x0)
torch.cuda.synchronize()
t1 = time.time()
print(f"Forward+backward on N={N}: {t1-t0:.3f} s")

# 6) (Optional) quick CG
res = minimize(
    fun=rosen_nd,
    x0=x0.detach().clone().requires_grad_(True),  # torchmin will re‑attach grad
    method="cg",
    max_iter=10000,
    tol=1e-4,
    disp=1,
)
print(f"CG took {res.nit} its, f*={res.fun:.3e}")

x_final = res.x.cpu().numpy()
print(x_final)
print("Distance to true minimum:", np.linalg.norm(x_final - 1.0))
