# DaCe macOS Setup Guide (Apple Silicon)

This guide documents a clean, reproducible setup for developing and testing DaCe on macOS (Apple Silicon) using conda-forge toolchains. It also covers packaging a wheel and using it in other projects/environments.

The key principle: Avoid mixing Homebrew and conda toolchains/runtimes (especially OpenMP). Prefer a unified conda-forge stack unless you explicitly opt into Homebrew everywhere.

## Prerequisites

- Homebrew installed at `/opt/homebrew` (optional; used only for brew itself and fallback LLVM)
- Miniforge/Conda at `/opt/homebrew/Caskroom/miniforge/base`
- Xcode CLT installed (for basic developer tools)

## 1) Create and activate a conda environment

Use a conda-forge environment to unify compilers, OpenMP, and BLAS/LAPACK.

- Recommended packages (Apple Silicon):
  - python>=3.10
  - compilers (clang/clang++)
  - cmake, ninja
  - llvm-openmp (OpenMP runtime)
  - openblas (BLAS/LAPACK)

Example commands:

```zsh
# Create once (adjust Python version to your preference)
conda create -n dace-dev -c conda-forge \
  python=3.12 compilers cmake ninja llvm-openmp openblas pkg-config

# Activate for development
conda activate dace-dev
```

Notes:

- If you previously installed LLVM/libomp via Homebrew, do not export global CFLAGS/LDFLAGS that point to Homebrew’s libomp while using a conda environment.

## 2) Clone and install DaCe in editable mode

```zsh
# From your workspace
git clone https://github.com/yourfork/dace.git
cd dace

# Upgrade build tooling (inside the env)
pip install -U pip setuptools wheel

# Project Python dependencies
pip install -r requirements.txt

# Editable install
pip install -e .
```

Verify the install:

```zsh
python - <<'PY'
import dace, sys
print('DaCe path:', dace.__file__)
print('Python:', sys.version)
PY
```

You should see DaCe importing from your local repo path.

## 3) Configure your shell for a consistent compiler

To avoid mixing runtimes, prefer the conda compiler when the conda env is active; otherwise, fall back to Homebrew LLVM. Add this to your `~/.zshrc` (or equivalent):

```zsh
# Prefer the active toolchain: use conda's clang++ inside envs, otherwise fall back to Homebrew LLVM.
if [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/clang++" ]; then
  export DACE_compiler_cpu_executable="$CONDA_PREFIX/bin/clang++"
else
  export DACE_compiler_cpu_executable="$(brew --prefix llvm)/bin/clang++"
fi
```

Reload your shell or `source ~/.zshrc` after editing.

## 4) DaCe configuration (.dace.conf)

DaCe reads configuration from (in priority order):

- Environment variables with `DACE_` prefix (highest priority)
- `.dace.conf` in the current working directory (project-local)
- `$HOME/.dace.conf` (user-global)

Recommendations:

- Keep a project-local `.dace.conf` in each repository that builds/uses DaCe, so settings are versioned alongside code and do not leak across projects.
- Avoid global overrides in `~/.zshrc` or `$HOME/.dace.conf` unless you truly want them to affect every project.

A working macOS/conda configuration that prefers OpenBLAS and avoids OpenMP duplication:

```yaml
# Place this file at the root of your DaCe project (repo-local .dace.conf)
debugprint: verbose
compiler:
  build_type: Debug
  codegen_lineinfo: true
  # Help CMake find conda headers/libs and prefer OpenBLAS
  extra_cmake_args: >-
    -DBLA_VENDOR=OpenBLAS
    -DCMAKE_FIND_FRAMEWORK=LAST
    -DCMAKE_INCLUDE_PATH=${CONDA_PREFIX}/include
    -DCMAKE_LIBRARY_PATH=${CONDA_PREFIX}/lib
  cpu:
    # Let the shell env var DACE_compiler_cpu_executable choose clang++ when the env is active
    executable: ''
    # Add include for OpenBLAS headers, keep optimization flags modest
    args: >-
      -std=c++14 -fPIC -Wall -Wextra -O3 -march=native -ffast-math
      -Wno-unused-parameter -Wno-unused-label
      -I${CONDA_PREFIX}/include
    openmp_sections: false
  linker:
    args: ''
  cuda:
    default_block_size: 64,8,1
library:
  blas:
    default_implementation: OpenBLAS
    override: false
  lapack:
    default_implementation: OpenBLAS
  linalg:
    default_implementation: OpenBLAS
```

Notes:

- `library.blas.override: false` is important so nodes that do not provide an OpenBLAS implementation (e.g., some "pure" nodes) are not forcibly redirected.
- `OpenMP` is available via `llvm-openmp` in conda; avoid linking Homebrew’s `libomp` when using a conda toolchain to prevent duplicate runtime errors.

## 5) Running tests (CPU-friendly subset on macOS)

DaCe’s test suite includes backends not suitable for macOS laptops (GPU/FPGA/MLIR/MPI/ScaLAPACK/HPTT/MKL). You can run a CPU-friendly subset as follows:

```zsh
# Optional: start fresh builds
rm -rf .dacecache

pytest -q \
  -m 'not mkl' \
  -k 'not gpu and not fpga and not mlir and not mpi and not scalapack and not hptt and not long' \
  --disable-warnings --color=yes
```

If you hit a failure that references `mkl.h` or other MKL symbols, confirm the `-m 'not mkl'` filter is present. For BLAS header issues (`cblas.h`), ensure `openblas` is installed in the conda env and that your `.dace.conf` contains the `-DCMAKE_INCLUDE_PATH` and `-I${CONDA_PREFIX}/include` hints shown above.

OpenMP reduction correctness:

- On some Apple Silicon setups, aggressively parallel reductions without atomics can be sensitive to flags/tiling choices. We found two mitigations that kept tests correct and performant:
  - Remove `-ffast-math` and add `-fno-strict-aliasing` in `compiler.cpu.args` (already in the template above).
  - If needed, disable partial-parallelism tiling during auto-optimization by setting in `.dace.conf`:
    
    ```yaml
    optimizer:
      autotile_partial_parallelism: false
    ```
    
    This encourages using atomics where necessary. Prefer keeping this project-local.

Targeted runs for quicker iteration:

```zsh
# Example: run OpenBLAS node tests only
rm -rf .dacecache && \
pytest tests/blas/nodes/blas_nodes_test.py -q -k OpenBLAS -s --maxfail=1
```

## 6) Packaging a wheel and using it elsewhere

To share your local DaCe build with another project or environment, build a wheel and install it where needed.

Build the wheel:

```zsh
# From the repo root, with the env active
python -m pip install -U build
python -m build --wheel
# Outputs to the dist/ folder, e.g., dist/dace-<version>-py3-none-any.whl
```

Alternatively, if `python -m build` isn’t available in your toolchain, you can use:

```zsh
python setup.py bdist_wheel
```

Install the wheel into another environment/project:

```zsh
# In the target environment
conda activate <other-env>
pip install /path/to/dace/dist/dace-<version>-py3-none-any.whl
```

Version pinning and provenance:

- Consider adding a local version suffix when building from a fork/branch (e.g., `1.0.0+local.20251006`).
- If your consuming project also needs a specific `.dace.conf`, place it at the consuming project’s root (repo-local). The wheel does not carry your development `.dace.conf`—and that’s intentional.

## 7) Common pitfalls and fixes

- Duplicate OpenMP runtime (Abort with OMP Error #15): Occurs when mixing Homebrew libomp with conda’s llvm-openmp. Fix by standardizing on conda toolchain and remove Homebrew `CPPFLAGS/LDFLAGS` for libomp in your shell, and ensure `DACE_compiler_cpu_executable` points to conda’s clang++ when the env is active.
- MKL failures on macOS ARM: Exclude MKL-marked tests (`-m 'not mkl'`). Prefer OpenBLAS.
- `cblas.h` not found: Ensure `openblas` is installed in the env and that `.dace.conf` includes the conda include/library hints.
- Global overrides leaking across projects: Keep settings project-local (`.dace.conf` in each repo) and keep `~/.zshrc` to only minimal, conditional overrides.

## 8) Next steps and improvements

- Optionally set `-G Ninja` in `compiler.extra_cmake_args` for faster builds if Ninja is installed.
- If you must use Homebrew LLVM everywhere, ensure all parts (BLAS/OpenMP/compiler) come from Homebrew and remove conda-provided counterparts to avoid mixing.
- For CI, mirror these steps in a conda-forge-based job; cache `.dacecache` between test stages when possible.

---

If you run into issues not covered here, capture the failing pytest command output and the following context for debugging:

- `which clang++`, `clang++ --version`
- `echo $DACE_compiler_cpu_executable`
- `python -c "import dace,sys;print(dace.__file__);print(sys.version)"`
- `.dace.conf` in the project root (if present)
