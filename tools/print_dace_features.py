#!/usr/bin/env python3
"""
Print a concise summary of DaCe feature availability in the current environment.

It checks DaCe config keys and probes for toolchains/libraries:
- CUDA/HIP backend selection and toolkit presence
- FPGA vendor (Xilinx/Intel) and basic tools presence
- MPI compiler and mpi4py
- PAPI instrumentation
- RTL/Verilator
- MLIR/LLVM tools and pymlir
- BLAS/LAPACK provider heuristics

Usage:
  python tools/print_dace_features.py
"""
from __future__ import annotations

import os
import platform
import shutil
import sys
from typing import Optional

try:
    import dace  # type: ignore
    from dace.config import Config  # type: ignore
except Exception as e:  # pragma: no cover - script utility
    print("DaCe is not importable in this environment:", e)
    sys.exit(1)


def which_any(candidates: list[str]) -> Optional[str]:
    for c in candidates:
        p = shutil.which(c)
        if p:
            return f"{c} -> {p}"
    return None


def bool_yn(v: Optional[bool]) -> str:
    return "yes" if v else "no"


def main() -> int:
    # Basic environment info
    print("DaCe environment summary\n==========================")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.system()} {platform.machine()}")
    try:
        dv = getattr(dace, "version", None)
        if dv is None:
            # Fallback to package metadata
            dv = getattr(dace, "__version__", "unknown")
    except Exception:
        dv = "unknown"
    print(f"name: {dace.__name__} – path: {dace.__path__}")   
    
    print(f" - version: {dace.__version__}")
    print(f" - version: {dace.__version__}")
    print(dace.__builtins__) 
    # # GPU
    # backend = None
    # try:
    #     backend = Config.get("compiler", "cuda", "backend")
    # except Exception:
    #     pass
    # print("\nGPU backend:")
    # print(f"  compiler.cuda.backend: {backend}")
    # if backend in ("cuda", "CUDA", "Cuda"):
    #     cuda_tool = which_any(["nvcc", "cuda-cc"])
    #     print(f"  CUDA toolkit: {'found ' + cuda_tool if cuda_tool else 'not found'}")
    # elif backend in ("hip", "HIP"):
    #     hip_tool = which_any(["hipcc", "clang++"])
    #     rocm = os.environ.get("ROCM_PATH") or shutil.which("rocminfo")
    #     print(f"  HIP toolchain: {'found ' + hip_tool if hip_tool else 'not found'}")
    #     print(f"  ROCm: {'found ' + str(rocm) if rocm else 'not found'}")
    # else:
    #     print("  Backend not set or unknown.")

    # # FPGA
    # vendor = None
    # try:
    #     vendor = Config.get("compiler", "fpga", "vendor")
    # except Exception:
    #     pass
    # print("\nFPGA:")
    # print(f"  compiler.fpga.vendor: {vendor}")
    # if vendor == "xilinx":
    #     vitis = which_any(["v++", "vitis"])
    #     xrt = which_any(["xbutil", "xrt"])
    #     print(f"  Vitis: {'found ' + vitis if vitis else 'not found'}")
    #     print(f"  XRT: {'found ' + xrt if xrt else 'not found'}")
    # elif vendor == "intel_fpga":
    #     aoc = which_any(["aoc", "icpx"])  # aoc for OpenCL, icpx for oneAPI dpcpp
    #     print(f"  Intel FPGA tools: {'found ' + aoc if aoc else 'not found'}")
    # else:
    #     print("  Vendor not set or unsupported.")

    # # MPI
    # print("\nMPI:")
    # try:
    #     mpicxx = Config.get("compiler", "mpi", "executable")
    # except Exception:
    #     mpicxx = None
    # resolved_mpi = mpicxx or which_any(["mpicxx", "mpicc", "mpiCC"])
    # print(f"  compiler.mpi.executable: {mpicxx}")
    # print(f"  MPI compiler: {'found ' + resolved_mpi if resolved_mpi else 'not found'}")
    # try:
    #     import mpi4py  # type: ignore
    #     print("  mpi4py: yes")
    # except Exception:
    #     print("  mpi4py: no")

    # # PAPI
    # print("\nInstrumentation:")
    # try:
    #     papi_enabled = Config.get_bool("instrumentation", "papi", "enabled")
    # except Exception:
    #     papi_enabled = None
    # print(f"  PAPI enabled: {bool_yn(papi_enabled) if papi_enabled is not None else 'unknown'}")
    # # Also check papi library availability heuristically
    # papi_lib = which_any(["papi_avail"])  # CLI utility if installed
    # print(f"  PAPI tools: {'found ' + papi_lib if papi_lib else 'not found'}")

    # # RTL / Verilator
    # print("\nRTL/Verilator:")
    # verilator = which_any(["verilator"])
    # print(f"  Verilator: {'found ' + verilator if verilator else 'not found'}")

    # # MLIR
    # print("\nMLIR:")
    # try:
    #     import mlir  # type: ignore
    #     mlir_import = True
    # except Exception:
    #     mlir_import = False
    # print(f"  pymlir import: {bool_yn(mlir_import)}")
    # mlir_opt = which_any(["mlir-opt", "llvm-mlir-opt"]) or None
    # mlir_translate = which_any(["mlir-translate", "llvm-mlir-translate"]) or None
    # llc = which_any(["llc"]) or None
    # print(f"  mlir-opt: {'found ' + mlir_opt if mlir_opt else 'not found'}")
    # print(f"  mlir-translate: {'found ' + mlir_translate if mlir_translate else 'not found'}")
    # print(f"  llc: {'found ' + llc if llc else 'not found'}")

    # # BLAS/LAPACK heuristics
    # print("\nBLAS/LAPACK:")
    # blas_impl = None
    # try:
    #     # Not a config flag; we can try to infer via numpy/scipy
    #     import numpy as np  # type: ignore
    #     try:
    #         from numpy.distutils.system_info import get_info  # type: ignore
    #     except Exception:
    #         get_info = None
    #     if get_info:
    #         info = get_info("blas_opt_info") or {}
    #         libs = info.get("libraries") or []
    #         for lib in libs:
    #             lname = lib.lower()
    #             if "mkl" in lname:
    #                 blas_impl = "MKL"
    #                 break
    #             if "openblas" in lname or "openblas64" in lname:
    #                 blas_impl = "OpenBLAS"
    #                 break
    #     if blas_impl is None:
    #         # Try simple heuristics
    #         if os.environ.get("MKLROOT"):
    #             blas_impl = "MKL (env)"
    # except Exception:
    #     pass
    # print(f"  Detected BLAS: {blas_impl or 'unknown'}")

    # # LAPACK presence via scipy (optional)
    # try:
    #     import scipy.linalg  # type: ignore
    #     print("  SciPy LAPACK: yes")
    # except Exception:
    #     print("  SciPy LAPACK: no")

    print("\nConfig file:")
    try:
        cfg_path = Config.cfg_filename()
    except Exception:
        cfg_path = None
    print(f"  Active .dace.conf: {cfg_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
