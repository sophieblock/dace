# DaCe Installation by Platform (PC and macOS ARM64)

This guide mirrors the official Installation page, adding concrete steps for common platforms and toolchains. It focuses on installing the required and optional tools for GPUs, FPGAs, BLAS/LAPACK/ScaLAPACK, MPI, PAPI, Verilator, and MLIR.

For general information, see the original reStructuredText page in `doc/setup/installation.rst` and configuration details in `doc/setup/config.rst`.

---

## 1) Core prerequisites

DaCe is tested with Python 3.9–3.13. You need a C++14 compiler and CMake 3.15+ in PATH.

- PC (Linux x86_64)
  - Install compiler and CMake:
    - Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y build-essential cmake`
    - Fedora/RHEL: `sudo dnf install -y gcc-c++ cmake`
    - Arch: `sudo pacman -S --needed base-devel cmake`

- macOS ARM64 (Apple Silicon)
  - Install Homebrew if not installed: <https://brew.sh>
  - Install compiler and CMake:
    - `brew install cmake`
    - macOS ships Apple Clang, but DaCe needs OpenMP for many CPU-parallel builds. Either:
      - Use Apple Clang + brew OpenMP runtime, or
      - Install GCC with OpenMP (recommended): `brew install gcc`
  - If you use GCC, set it in `~/.dace.conf` via `compiler.cpu.executable` (see Config section below).

Once system tools are available, install DaCe from PyPI:

```sh
pip install dace
```

Or for development:

```sh
git clone --recursive https://github.com/spcl/dace.git
cd dace
pip install -e .
```

---

## 2) GPU setup (CUDA or HIP)

DaCe supports NVIDIA (CUDA) and AMD (HIP/ROCm). Some tests and workflows also use CuPy for array interop.

- PC (Linux x86_64)
  - NVIDIA/CUDA:
    - Install the CUDA Toolkit (matching your driver). See NVIDIA docs.
    - Optional: CuPy for CUDA: `pip install cupy-cuda12x` (choose wheel matching your CUDA version)
  - AMD/HIP (ROCm):
    - Install ROCm per AMD documentation for your distro and GPU.
    - Optional: CuPy ROCm wheels may be available for specific ROCm versions.

- macOS ARM64
  - There is no native CUDA or ROCm support on Apple Silicon. GPU backends are generally not available on macOS ARM64.
  - You can still run CPU and most library functionality.

Configure DaCe if needed in `~/.dace.conf` (see `dace/config_schema.yml` for details):

```yaml
compiler:
  cuda:
    backend: auto   # or 'cuda' or 'hip'
    path: ""        # Set to CUDA/ROCm root if not in PATH
    cuda_arch: "60" # Optional extra archs (CUDA)
    hip_arch:  "gfx906" # Optional extra archs (HIP)
```

Tips:
- If auto-detection fails, set `compiler.cuda.backend` explicitly to `cuda` or `hip`.
- For HIP, you may also tune `default_block_size: 64,1,1` (wavefront 64).

---

## 3) FPGA toolchains (Xilinx and Intel)

- PC (Linux x86_64)
  - Xilinx:
    - Install Xilinx Vitis (or legacy SDx/SDAccel) + XRT per Xilinx docs.
    - Ensure a valid platform (e.g., `xilinx_u250_xdma_201830_2`) is installed.
  - Intel:
    - Install Intel FPGA OpenCL SDK (AOCL) and a supported BSP (e.g., Arria/Stratix). Set environment per Intel docs.

- macOS ARM64
  - Vendor FPGA toolchains are not supported on Apple Silicon. Use a Linux machine (native or VM) for FPGA builds.

Minimal config examples (`~/.dace.conf`):

```yaml
compiler:
  fpga:
    vendor: xilinx   # or 'intel_fpga'

  xilinx:
    mode: simulation  # simulation|software_emulation|hardware_emulation|hardware
    platform: xilinx_u250_xdma_201830_2
    path: ""          # Vitis root if not on PATH

  intel_fpga:
    mode: emulator    # emulator|simulator|hardware
    board: a10gx
    path: ""          # AOCL root if not on PATH
```

---

## 4) BLAS/LAPACK/ScaLAPACK and friends

DaCe library nodes can use optimized backends. Defaults are OpenBLAS for LAPACK/linalg on CPU; MKL is also supported.

- PC (Linux x86_64)
  - OpenBLAS (Debian/Ubuntu): `sudo apt-get install -y libopenblas-dev liblapacke-dev`
  - MKL (oneAPI): Install Intel oneAPI Base Toolkit, ensure `MKLROOT` is set.
  - ScaLAPACK: Use MKL’s ScaLAPACK or distro packages (requires MPI, see below).

- macOS ARM64
  - OpenBLAS: `brew install openblas`
    - You may need to export include/lib paths for CMake to find headers and libs, e.g.:
      - `export CPATH="/opt/homebrew/opt/openblas/include:$CPATH"`
      - `export LIBRARY_PATH="/opt/homebrew/opt/openblas/lib:$LIBRARY_PATH"`
      - `export DYLD_LIBRARY_PATH="/opt/homebrew/opt/openblas/lib:$DYLD_LIBRARY_PATH"`
  - MKL: Intel MKL is not natively supported on Apple Silicon; use OpenBLAS.

Configure defaults if desired:

```yaml
library:
  lapack:
    default_implementation: OpenBLAS  # or MKL on Linux with MKL installed
  linalg:
    default_implementation: OpenBLAS  # or MKL
  pblas:
    default_implementation: MKLMPICH  # or MKLOpenMPI, when using ScaLAPACK
```

Note: If CMake cannot find your BLAS/LAPACK libs, add include paths to `CPATH` and libraries to `LIBRARY_PATH` and (Linux) `LD_LIBRARY_PATH` or (macOS) `DYLD_LIBRARY_PATH`.

---

## 5) MPI and ScaLAPACK

- PC (Linux x86_64)
  - Install MPI (choose one):
    - MPICH: `sudo apt-get install -y mpich libmpich-dev`
    - OpenMPI: `sudo apt-get install -y libopenmpi-dev openmpi-bin`
  - mpi4py (optional for tests): `pip install mpi4py`

- macOS ARM64
  - Install OpenMPI: `brew install open-mpi`
  - mpi4py: `pip install mpi4py`

Configure the MPI compiler wrapper if needed:

```yaml
compiler:
  mpi:
    executable: mpicxx  # e.g., mpicxx, mpiicpc, or path to wrapper
```

For ScaLAPACK with MKL on Linux, ensure MKL ScaLAPACK libs are on the link path and use a matching MPI (MPICH/OpenMPI) in `library.pblas.default_implementation`.

---

## 6) PAPI (optional instrumentation)

- PC (Linux x86_64)
  - `sudo apt-get install -y libpapi-dev`

- macOS ARM64
  - PAPI via Homebrew may exist but support varies; if unavailable, skip PAPI instrumentation.

Configure counters (optional):

```yaml
instrumentation:
  papi:
    default_counters: "['PAPI_TOT_INS','PAPI_TOT_CYC','PAPI_L2_TCM','PAPI_L3_TCM']"
```

---

## 7) Verilator (RTL simulation)

- PC (Linux x86_64)
  - Ubuntu: `sudo apt-get install -y verilator`
  - Or build latest from source (recommended for SystemVerilog support).

- macOS ARM64
  - `brew install verilator`

Optional DaCe settings:

```yaml
compiler:
  rtl:
    verilator_flags: ""
    verilator_lint_warnings: true
    verilator_enable_debug: false
```

Note: For full RTL-to-hardware flows, you also need the Xilinx FPGA toolchain on Linux.

---

## 8) MLIR support (optional)

- Python side: `pip install pymlir`
- System side: LLVM/MLIR tools (mlir-opt, mlir-translate) if you plan to compile or run MLIR-based tasklets.

- PC (Linux x86_64): Install LLVM with MLIR from your distro or build from source.
- macOS ARM64: Install via Homebrew (note: MLIR packaging varies): `brew install llvm` (then add LLVM bin to PATH).

---

## 9) Troubleshooting highlights

- On macOS ARM64:
  - Prefer GCC for OpenMP builds: `brew install gcc` and set `compiler.cpu.executable` to `g++-<version>`.
  - Use OpenBLAS (MKL is not native on Apple Silicon).
  - GPU backends (CUDA/HIP) are not available.
- If CMake can’t find a library, export CPATH and library paths as noted above.
- Clear caches by deleting `.dacecache` folders when changing compilers or toolchains.

---

## 10) Minimal `.dace.conf` templates

CPU-only (portable):

```yaml
compiler:
  use_cache: true
  cpu:
    executable: ""  # set to full path of g++/clang++ if desired

library:
  lapack:
    default_implementation: OpenBLAS
  linalg:
    default_implementation: OpenBLAS
```

CUDA (Linux):

```yaml
compiler:
  cuda:
    backend: cuda
    path: ""  # e.g., /usr/local/cuda
    cuda_arch: "80"
```

HIP/ROCm (Linux):

```yaml
compiler:
  cuda:
    backend: hip
    hip_arch: "gfx90a"
    hip_args: -std=c++17 -fPIC -O3 -ffast-math -Wno-unused-parameter
```

FPGA (Xilinx, Linux):

```yaml
compiler:
  fpga:
    vendor: xilinx
  xilinx:
    mode: simulation
    platform: xilinx_u250_xdma_201830_2
    host_flags: -Wno-unknown-pragmas -Wno-unused-label
    synthesis_flags: -std=c++14
```

---

## 11) Quick test

After installing DaCe and any chosen toolchains, try a CPU example:

```python
import dace, numpy as np

@dace.program
def add1(a: dace.float32[128]):
    for i in range(128):
        a[i] += 1

A = np.zeros(128, dtype=np.float32)
add1(A)
print(A[:5])
```

For GPU (Linux only, with CUDA/HIP installed), adapt your `.dace.conf` and run a sample from `samples/` or `tests/` marked with `@pytest.mark.gpu`.

---

If you hit platform-specific issues, see the Troubleshooting section in `doc/setup/installation.rst` and the configuration reference generated from `dace/config_schema.yml`.
