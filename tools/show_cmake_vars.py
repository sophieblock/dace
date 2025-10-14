#!/usr/bin/env python3
"""
Show key CMake variables from dace/codegen/CMakeLists.txt.

Reports two sources:
- Defaults (as defined in CMakeLists.txt)
- Actual configured values if a CMakeCache.txt is found (by --build-dir, or auto-detected under .dacecache)

Variables shown:
  - DACE_PROGRAM_NAME
  - DACE_LIBS
  - HLSLIB_PART_NAME (and its source DACE_XILINX_PART_NAME default)
  - DACE_CUDA_ARCHITECTURES_DEFAULT
  - LOCAL_CUDA_ARCHITECTURES (set only when CUDA is enabled)
  - CUDAToolkit_NVCC_EXECUTABLE (set only when CUDAToolkit is found)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
CMAKE_LISTS = ROOT / "dace" / "codegen" / "CMakeLists.txt"


def read_cache_vars(cache_path: Path) -> Dict[str, str]:
    vals: Dict[str, str] = {}
    if not cache_path.exists():
        return vals
    for line in cache_path.read_text(errors="ignore").splitlines():
        if not line or line.startswith("#"):
            continue
        if ":" in line and "=" in line:
            try:
                key_type, value = line.split("=", 1)
                key, _typ = key_type.split(":", 1)
                vals[key.strip()] = value.strip()
            except ValueError:
                continue
    return vals


def find_latest_cmakecache(base: Path) -> Optional[Path]:
    """Find a recent CMakeCache.txt under base (e.g., .dacecache)."""
    candidates: list[Tuple[float, Path]] = []
    for p in base.rglob("CMakeCache.txt"):
        try:
            candidates.append((p.stat().st_mtime, p))
        except FileNotFoundError:
            pass
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def print_report(cache_vals: Dict[str, str] | None) -> None:
    # Defaults from CMakeLists.txt (mirrored here for clarity)
    defaults = {
        "DACE_PROGRAM_NAME": "dace_program",  # CACHE default
        "DACE_LIBS": "",  # CACHE default, later appended at configure time
        "DACE_XILINX_PART_NAME": "xcu280-fsvh2892-2L-e",  # CACHE default
        "HLSLIB_PART_NAME": "${DACE_XILINX_PART_NAME}",  # set to current value of DACE_XILINX_PART_NAME
        "DACE_CUDA_ARCHITECTURES_DEFAULT": "",  # CACHE default
        # LOCAL_CUDA_ARCHITECTURES: set only if CUDA enabled
        # CUDAToolkit_NVCC_EXECUTABLE: set only if CUDAToolkit found
    }

    keys = [
        "DACE_PROGRAM_NAME",
        "DACE_LIBS",
        "HLSLIB_PART_NAME",
        "DACE_CUDA_ARCHITECTURES_DEFAULT",
        "LOCAL_CUDA_ARCHITECTURES",
        "CUDAToolkit_NVCC_EXECUTABLE",
    ]

    print("CMake variables (defaults)")
    print("--------------------------")
    for k in keys:
        if k == "HLSLIB_PART_NAME":
            print(f"{k}: {defaults['HLSLIB_PART_NAME']}  (DACE_XILINX_PART_NAME default: {defaults['DACE_XILINX_PART_NAME']})")
        else:
            dv = defaults.get(k, "<unset>")
            print(f"{k}: {dv}")

    if cache_vals is None:
        print("\nNo CMakeCache provided/found. To see configured values, pass --build-dir pointing to a CMake build folder.")
        return

    print("\nCMake variables (from CMakeCache.txt)")
    print("------------------------------------")
    for k in keys:
        val = cache_vals.get(k)
        if val is None and k == "HLSLIB_PART_NAME":
            # Not a CACHE var; may not appear in cache
            # Try to infer from DACE_XILINX_PART_NAME
            dx = cache_vals.get("DACE_XILINX_PART_NAME")
            if dx:
                print(f"{k}: {dx} (inferred from DACE_XILINX_PART_NAME)")
            else:
                print(f"{k}: <not cached>")
        else:
            print(f"{k}: {val if val is not None else '<not set>'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=Path,
        help="Path to a CMake build directory containing CMakeCache.txt. If omitted, auto-detects under .dacecache.",
    )
    args = parser.parse_args()

    cache_path: Optional[Path] = None
    if args.build_dir:
        cp = Path(args.build_dir) / "CMakeCache.txt"
        if cp.exists():
            cache_path = cp
        else:
            print(f"No CMakeCache.txt in {args.build_dir}")
    else:
        # Search under .dacecache for a recent cache
        dc = ROOT / ".dacecache"
        if dc.exists():
            found = find_latest_cmakecache(dc)
            if found:
                cache_path = found
                print(f"Using cache: {found}")

    cache_vals = read_cache_vars(cache_path) if cache_path else None
    print_report(cache_vals)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
