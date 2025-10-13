import os, numpy as np, dace
print("DaCe:", getattr(dace, "__version__", "(dev)"))
print("CPU compiler:", os.getenv("DACE_compiler_cpu_executable"))
@dace.program
def gemm(A: dace.float64[64,64], B: dace.float64[64,64]):
    return A @ B
A = np.random.rand(64,64); B = np.random.rand(64,64)
C = gemm(A,B)
print("C shape:", C.shape, "sum:", float(C.sum()))
