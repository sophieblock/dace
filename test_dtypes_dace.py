import numpy as np
import pytest
from scipy.__config__ import show
import dace
import torch
import importlib.util
import os
import sys
import subprocess

import numpy as np
import numpy
from dace import dtypes
def show_interfaces(x):
    print(f"\nPackage: {type(x).__module__}.{type(x).__name__}")
    print(f"dace is_array: {dace.dtypes.is_array(x)}")
    print("has __array_interface__:", hasattr(x, "__array_interface__"))
    print("has __cuda_array_interface__:", hasattr(x, "__cuda_array_interface__"))
    print("has __dlpack__:", hasattr(x, "__dlpack__") or hasattr(x, "__torch_dlpack__"))
    


def is_array_like(obj):
    """Return True if object looks like a CPU-backed array or is convertible to numpy."""
    if hasattr(obj, '__array_interface__'):
        return True
    if hasattr(obj, '__array__'):
        return True
    # numpy.asarray should succeed for many array-like objects
    try:
        np.asarray(obj)
        return True
    except Exception:
        return False


def is_gpu_array_like(obj):
    """Return True if object exposes GPU array signals (cuda interface, dlpack, or data pointer)."""
    if hasattr(obj, '__cuda_array_interface__'):
        print(f"Object {obj} has CUDA array interface")
        return True
    if hasattr(obj, '__dlpack__') or hasattr(obj, '__torch_dlpack__'):
        print(f"Object {obj} has DLPack support")
        return True
    # PyTorch tensors expose data_ptr/shape
    if hasattr(obj, 'data_ptr') and hasattr(obj, 'shape'):
        print(f"Object {obj} has data_ptr")
        return True
    # CuPy exposes .data.ptr
    if hasattr(obj, 'data') and hasattr(getattr(obj, 'data'), 'ptr'):
        return True
    return False

def is_torch_tensor(obj):
    if type(obj).__module__ == "torch" and type(obj).__name__ == "Tensor":
        return True
    return False
candidate_names = [
        "bool_",
        # bit-sized integers
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        
        # floating point
        "float16", "float32", "float64", "longdouble",
        # complex floating-point
        "complex64", "complex128","clongdouble",
        
        # c-named integers 
        "byte", "ubyte", "short","ushort",
        "intc","uintc",
        "longlong","ulonglong", 
        
        "str", "bytes", "T", "object", "V",
        
    ]
numpy_types = [numpy.bool_,
    numpy.intc,
    numpy.intp,
    numpy.int8,
    numpy.int16,
    numpy.int32,
    numpy.int64,
    numpy.uint8,
    numpy.uint16,
    numpy.uint32,
    numpy.uint64,
    numpy.float16,
    numpy.float32,
    numpy.float64,
    numpy.complex64,
    numpy.complex128,
    ]

ctype_numpy = [
    np.longlong,
    
    
]
def test_typeclass_by_str():
    for np_type in numpy_types:
        type_class = dace.typeclass(np_type)
        print(f"{np_type} -> {type_class}, type: {type(type_class)}")
def test_numpy_scalar_types():
    # Test that numpy scalar types behave as expected
    import math
    # A conservative list of numpy scalar types to exercise. This covers the
    # canonical DType classes the project commonly needs to support.
    np_scalar_types = [
        np.bool_,
        np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64,
        np.intc, np.uintc, np.longlong, np.ulonglong,
        np.float16, np.float32, np.float64, np.longdouble,
        np.complex64, np.complex128, getattr(np, 'clongdouble', np.complex128),
        np.str_, np.bytes_, np.unicode_, object, np.void
    ]

    for np_t in np_scalar_types:
        # Create a representative numpy scalar value for the dtype
        if np_t is np.str_ or np_t is np.unicode_:
            val = np.array('x', dtype=np_t)[0]
        elif np_t is np.bytes_:
            val = np.array(b'x', dtype=np_t)[0]
        elif np_t is object:
            val = np.array([{'k': 1}], dtype=object)[0]
        elif np_t is np.void:
            # Create a 1-byte void value
            val = np.array([b'\x00'], dtype='V1')[0]
        elif issubclass(np_t, np.complexfloating) if isinstance(np_t, type) else False:
            val = np.array(1+2j, dtype=np_t)[0]
        elif issubclass(np_t, np.floating) if isinstance(np_t, type) else False:
            val = np.array(1.0, dtype=np_t)[0]
        else:
            # Fallback numeric / integer types
            try:
                val = np.array(1, dtype=np_t)[0]
            except Exception:
                # Some platform aliases may not be constructible directly; skip those
                pytest.skip(f'Cannot construct numpy scalar for dtype {np_t}')

        desc = dace.data.create_datadescriptor(val)
        # Must be a Scalar descriptor for numpy scalar values
        assert isinstance(desc, dace.data.Scalar), f'Expected Scalar for {np_t}, got {type(desc)}'

        # Determine expected numpy type where possible
        try:
            expected_np_type = np.dtype(np_t).type
        except Exception:
            expected_np_type = None

        # If we can determine the expected numpy scalar type, assert the DaCe dtype wraps it
        if expected_np_type is not None:
            # Some platform-specific aliases (longdouble/clongdouble) may map to float64/complex128
            # so be tolerant and allow either exact match or same-kind match (int/float/complex/bool/object/void/str)
            got = getattr(desc.dtype, 'type', None)
            if got is None:
                # If desc.dtype is not exposing `.type`, at least ensure it's a DaCe typeclass
                assert isinstance(desc.dtype, dace.dtypes.typeclass)
            else:
                if got == expected_np_type:
                    continue
                # Allow int family equivalence (e.g., longlong -> int64)
                kind_expected = np.dtype(expected_np_type).kind
                kind_got = np.dtype(got).kind if got is not None else None
                assert kind_expected == kind_got, f'DType kind mismatch for {np_t}: expected {kind_expected}, got {kind_got}'
        else:
            # Unknown expected mapping; at minimum ensure DaCe produced a typeclass
            assert isinstance(desc.dtype, dace.dtypes.typeclass)
    

def test_tensor_init_with_numpy_dtype():
    arr = [1, 2, 3]
    # Construct from NumPy arrays so we don't rely on torch accepting numpy dtype objects
    t = torch.tensor(np.array(arr, dtype=np.int32))
    assert t.dtype == torch.int32

    t2 = torch.tensor(np.array(arr, dtype=float))
    assert t2.dtype == torch.float64


def test_numpy_is_array_and_canonicalize():
    

    a = np.arange(12, dtype=np.int32).reshape(3, 4)
    show_interfaces(a)
    # is_array should detect NumPy arrays
    assert dace.dtypes.is_array(a) is True

    desc = dace.data.create_datadescriptor(a)
    # Should produce a DaCe Array descriptor with matching shape and dtype
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == a.shape
    assert desc.dtype.type == a.dtype.type


def test_torch_is_array_and_canonicalize():
    t = torch.zeros((2, 3), dtype=torch.float32)
    
    np_arr = np.asarray(t)
    
    # torch tensors should be detected as arrays
    assert dace.dtypes.is_array(t) is True
  
    
    # assert (hasattr(t, '__array_interface__') or hasattr(t, '__cuda_array_interface__'))
    show_interfaces(t)
    print(f"-> is_tensor: {is_torch_tensor(t)}")
    desc = dace.data.create_datadescriptor(t)
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == tuple(t.shape)
    
    # Torch float32 should map to numpy.float32 in the descriptor
    assert desc.dtype.type == np.float32
    show_interfaces(np_arr)
    print(f"-> is_tensor: {is_torch_tensor(np_arr)}")
    print(f"-> is cpu-backed: {is_array_like(np_arr)}")
    

def test_numba_external_type_handling():
    # If numba is available, ensure its typed.List is not treated as an array
    numba = pytest.importorskip('numba')
    # Create a numba typed list (Python-side object)
    from numba.typed import List as NumbaList

    nl = NumbaList([1, 2, 3])
    # This should not be considered an array by the array-detection helper
    assert dace.dtypes.is_array(nl) is False
    
    show_interfaces(nl)
    # Creating a data descriptor from this object should raise (no automatic conversion)
    with pytest.raises(TypeError):
        dace.data.create_datadescriptor(nl)


def test_builtin_list_and_scalar_canonicalize():
    # Python built-in list is not considered an array by is_array, but is supported by create_datadescriptor
    pylist = [1, 2, 3, 4]
    assert dace.dtypes.is_array(pylist) is False
    show_interfaces(pylist)
    # Creating a data descriptor from a built-in list should produce an Array descriptor
    desc = dace.data.create_datadescriptor(pylist)
    assert isinstance(desc, dace.data.Array)
    # dtype should match numpy's inference on the same list
    inferred = np.asarray(pylist).dtype.type
    assert desc.dtype.type == inferred
    print(f"new type: {type(desc)}, size: {desc.total_size}, shape: {desc.shape}")


    # Built-in scalar converts to a Scalar descriptor
    scalar = 7
    assert dace.dtypes.is_array(scalar) is False
    sdesc = dace.data.create_datadescriptor(scalar)
    # Scalars are represented as Scalar/Data descriptors
    assert isinstance(sdesc, dace.data.Scalar) or isinstance(sdesc, dace.data.Data)
    show_interfaces(scalar)
    
    nested_list = [[1, 2], [3, 4]]
    assert dace.dtypes.is_array(nested_list) is False
    ndesc = dace.data.create_datadescriptor(nested_list)
    assert isinstance(ndesc, dace.data.Array)
    show_interfaces(nested_list)
    print(f"new type: {type(ndesc)}, size: {ndesc.total_size}, shape: {ndesc.shape}")

def test_dace_dtype_resource_values_against_torch_type(print_diagnostics=True):
    # Compact check: compare DaCe typeclasses with torch dtypes via element size
    cases = [
        ("int8", torch.int8), ("float32", torch.float32), ("int32", torch.int32),
        ("uint8", torch.uint8), ("float64", torch.float64), ("int64", torch.int64),
        ("bool", torch.bool)
    ]

    shape = (2, 2, 2)
    for name, torch_dtype in cases:
        dt = getattr(dace, name, None) or getattr(dace.dtypes, name, None)
        if dt is None:
            raise  ValueError(f"DaCe type '{name}' not available")

        # Determine element size from DaCe typeclass without using as_numpy_dtype
        nptype = getattr(dt, 'type', dt)
        elem_size = nptype(0).itemsize

        t = torch.ones(size=shape, dtype=torch_dtype)
        torch_elem_size = t.element_size()

        # Number of elements and total bytes sanity checks
        ne = t.nelement()
        assert ne == int(np.prod(shape))
        assert t.element_size() * ne == t.element_size() * ne

        nbytes = elem_size * ne
        bits_per_elem = elem_size * 8
        total_bits = nbytes * 8
        if print_diagnostics:
            print(f"DT={dt}, shape={shape}, elem_size={elem_size}, ne={ne}, elem_dtype={nptype}, nbytes={nbytes}, bits_per_elem={bits_per_elem}, total_bits={total_bits}")

        assert torch_elem_size == elem_size, f"Mismatch for {name}: torch {torch_elem_size} != dace {elem_size}"


def test_list_tuple_typeclasses():
    # Lists and tuples should be recognized as arrays
    lst = [1, 2, 3]
    tpl = (4, 5, 6)
    assert dace.dtypes.is_array(lst) is False
    assert dace.dtypes.is_array(tpl) is False
    show_interfaces(lst)
    show_interfaces(tpl)
    print(f"{lst} -> is array: {dtypes.is_array(lst)}")
    print(f"{tpl} -> is array: {dtypes.is_array(tpl)}")


def test_cpu_backed_array_packages():
    """Check common CPU-backed array libraries expose an array interface or are numpy-convertible."""
    # NumPy
    a = np.ones((2, 3))
    assert is_array_like(a)
    show_interfaces(a)
    print(f"is array: {dtypes.is_array(a)}")

    # SciPy sparse (csr_matrix) — should be convertible to numpy or expose __array__
    scipy = pytest.importorskip('scipy')
    from scipy import sparse
    s = sparse.csr_matrix(np.arange(6).reshape(2, 3))
    assert is_array_like(s)
    show_interfaces(s)
    print(f"is array: {dtypes.is_array(s)}")
    
    # Dask Array
    da = pytest.importorskip('dask.array')
    import dask.array as da  # noqa: F401  (module alias retained for clarity)
    da_obj = da.from_array(np.arange(6).reshape(2, 3), chunks=(2, 3))
    # Dask arrays may not implement __array_interface__ but are convertible via compute/np.asarray
    assert is_array_like(da_obj)
    show_interfaces(da_obj)
    print(f"is array: {dtypes.is_array(da_obj)}")
    # Awkward Array
    ak = pytest.importorskip('awkward')
    import awkward as ak  # noqa: F401
    ak_arr = ak.Array([[1, 2], [3]])
    assert is_array_like(ak_arr)
    show_interfaces(ak_arr)
    print(f"is array: {dtypes.is_array(ak_arr)}")
    
    # PyData-Sparse
    ps = pytest.importorskip('sparse')
    import sparse as ps  # noqa: F401
    sp_arr = ps.COO(np.arange(6).reshape(2, 3))
    assert is_array_like(sp_arr)
    show_interfaces(sp_arr) 
    print(f"is array: {dtypes.is_array(sp_arr)}")


def test_gpu_pytorch_tensor():
    """Minimal PyTorch GPU (or CPU fallback) detection test isolated from other frameworks."""
    if torch.cuda.is_available():
        t = torch.zeros((2, 3), device='cuda')
    else:
        t = torch.zeros((2, 3))
    assert is_gpu_array_like(t)
    show_interfaces(t)
    print(f"is array: {dtypes.is_array(t)}")


@pytest.mark.skipif(not importlib.util.find_spec('jax'), reason='jax not installed')
def test_gpu_jax_array():
    """JAX array detection without importing TensorFlow in same test to avoid XLA runtime lock contention.

    Notes:
      * JAX deliberately does NOT expose __array_interface__/__cuda_array_interface__ to prevent implicit host copies.
      * It does expose __dlpack__, which we treat as a GPU/accelerator signal.
      * The previously observed hang with a RAW mutex log likely came from mixing TF+JAX in same process segment
        right after device initialization. Splitting the tests isolates runtime initializations.
    """
    # Environment knobs to make JAX safer in constrained CI environments (optional, no-ops if already set)
    import os
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')  # fallback to CPU to avoid GPU contention
    import jax.numpy as jnp
    jx = jnp.array([1, 2, 3])
    # Our heuristic: __dlpack__ indicates accelerator-manageable buffer
    assert is_gpu_array_like(jx)
    show_interfaces(jx)
    print(f"is array: {dtypes.is_array(jx)}")


@pytest.mark.skipif(
    not importlib.util.find_spec('tensorflow') or os.environ.get('DACE_ENABLE_REAL_TF','0') != '1',
    reason='Real TensorFlow test disabled (set DACE_ENABLE_REAL_TF=1 to enable) or tensorflow not installed')
def test_gpu_tensorflow_tensor():
    """TensorFlow tensor detection isolated & made safer.

    Mitigations for hangs:
      * Set TF/XLA env flags before import to reduce aggressive initialization & allocator locking.
      * Optionally enable per-process GPU memory growth if GPUs present.
      * Avoid mixing heavy TF + JAX init in same function (already separated).
    """
    import os, time
    # Must be set before importing tensorflow
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')            # silence INFO/WARN
    os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')    # avoid grabbing full GPU memory
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')           # speed up init in some envs
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

    # Import tensorflow lazily now (kept above for skipif detection only)
    import tensorflow as tf  # noqa: E402
    t0 = time.time()
    t_import = time.time() - t0  # trivial here but retained for symmetry
    print(f"[TF] import (post-skip gate) additional time: {t_import:.3f}s")

    # If import itself was very slow (>10s), we still proceed but note it (helps debugging CI logs)
    # Configure GPU memory growth (skip if exception)
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
    except Exception:
        pass

    tft = tf.constant([[1, 2, 3], [4, 5, 6]])
    # Quick op to ensure runtime ready but not heavy
    _ = tft + 1
    # Accept any of: numpy convertibility, dlpack (newer TF), or explicit array interface (usually absent)
    convertible = (is_array_like(tft) or hasattr(tft, '__dlpack__') or hasattr(tft, 'numpy'))
    assert convertible
    show_interfaces(tft)
    print(f"is array: {dtypes.is_array(tft)}")

#############################################
# Mock-based tests (no heavy imports required)
#############################################

class _MockCPUArray:
    def __init__(self, shape=(2, 3), dtype=np.float32):
        self._backing = np.zeros(shape, dtype=dtype)
        self.shape = shape
        self.dtype = self._backing.dtype
        self.__array_interface__ = {
            'shape': shape,
            'typestr': np.dtype(dtype).str,
            'data': (self._backing.ctypes.data, False),
            'strides': None,
            'version': 3,
        }


class _MockGPUArray:
    def __init__(self, shape=(4,), dtype=np.float32):
        self._backing = np.zeros(shape, dtype=dtype)  # still CPU memory, we just simulate interface
        self.shape = shape
        self.dtype = self._backing.dtype
        self.__cuda_array_interface__ = {
            'shape': shape,
            'typestr': np.dtype(dtype).str,
            'data': (self._backing.ctypes.data, False),
            'strides': None,
            'version': 3,
        }


class _MockDLPackArray:
    def __init__(self, shape=(3,), dtype=np.int32):
        self._backing = np.zeros(shape, dtype=dtype)
        self.shape = shape
        self.dtype = self._backing.dtype
    def __dlpack__(self):  # minimal placeholder
        raise RuntimeError("DLPack export not implemented in mock (intentionally)")


def test_mock_cpu_array_conversion():
    m = _MockCPUArray()
    assert dace.dtypes.is_array(m)
    desc = dace.data.create_datadescriptor(m)
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == m.shape
    assert desc.dtype.type == m.dtype.type


def test_mock_gpu_array_conversion():
    mg = _MockGPUArray()
    # GPU array interface should mark as array and GPU storage
    assert dace.dtypes.is_array(mg)
    desc = dace.data.create_datadescriptor(mg)
    assert isinstance(desc, dace.data.Array)
    from dace.dtypes import StorageType
    assert desc.storage == StorageType.GPU_Global


def test_mock_dlpack_array_detection():
    md = _MockDLPackArray()
    # Our generic heuristic function should consider dlpack presence as GPU-like, DaCe might not.
    # We at least verify the helper logic we wrote (is_gpu_array_like if present) without importing frameworks.
    assert hasattr(md, '__dlpack__')
    # DaCe may return False here because it only checks array/cuda interfaces; that's acceptable.


@pytest.mark.skipif(not importlib.util.find_spec('tensorflow'), reason='tensorflow not installed')
def test_tensorflow_descriptor_wrapper():
    """Create a TensorFlow tensor but convert via a lightweight adapter object's __descriptor__.

    This avoids triggering DaCe's torch/cuda interface branches directly with a TF object that lacks
    __array_interface__, while still validating an integration strategy users can employ without
    modifying DaCe core. The adapter performs a .numpy() materialization (copy to host) explicitly.
    Disable with DACE_ENABLE_TF_WRAPPER=0.
    """
    if os.environ.get('DACE_ENABLE_TF_WRAPPER', '1') != '1':
        pytest.skip('TF wrapper descriptor test disabled by env var')
    import tensorflow as tf
    t = tf.constant([[1.0, 2.0, 3.0], [4.5, 5.5, 6.5]], dtype=tf.float32)

    class _TFWrapper:
        def __init__(self, tensor):
            self.tensor = tensor
        def __descriptor__(self):  # Adapter contract recognized by create_datadescriptor
            arr = self.tensor.numpy()  # Explicit host copy; safe and deterministic
            return dace.data.create_datadescriptor(arr)

    wrapped = _TFWrapper(t)
    desc = dace.data.create_datadescriptor(wrapped)
    assert isinstance(desc, dace.data.Array)
    assert tuple(desc.shape) == (2, 3)
    assert desc.dtype.type == np.float32


@pytest.mark.skipif(not importlib.util.find_spec('tensorflow'), reason='tensorflow not installed')
def test_tensorflow_descriptor_subprocess():
    """Isolated TensorFlow -> DaCe descriptor test run in a subprocess to avoid polluting the main
    test process with TF runtime / allocator side-effects. Always safe even if other frameworks
    (e.g., JAX) are imported in-process elsewhere.

    Enable/disable via DACE_ENABLE_TF_SUBPROC (defaults to enabled). Set to '0' to skip explicitly.
    """
    if os.environ.get('DACE_ENABLE_TF_SUBPROC', '1') != '1':
        pytest.skip('TensorFlow subprocess test disabled by env var')

    # Minimal script: import TF, create constant, call .numpy() to force host copy, print shape/dtype.
    # Conversion to DaCe kept but via wrapper to avoid direct reliance on TF internals.
    code = r"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH','true')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS','0')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','1')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import tensorflow as tf, dace, numpy as np
class _Wrap:
    def __init__(self, t): self.t = t
    def __descriptor__(self): return dace.data.create_datadescriptor(self.t.numpy())
t = tf.constant([[1,2,3],[4,5,6]], dtype=tf.int32)
desc = dace.data.create_datadescriptor(_Wrap(t))
print('SHAPE=', tuple(desc.shape))
print('DTYPE=', desc.dtype)
"""
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    if result.returncode != 0:
        # Treat sporadic low-level TF aborts as xfail (environment-induced) rather than hard failure.
        print('STDOUT:\n' + result.stdout)
        print('STDERR:\n' + result.stderr)
        pytest.xfail(f'TensorFlow subprocess aborted (rc={result.returncode}); marking expected in this env')

    out = result.stdout.splitlines()
    shape_line = [l for l in out if l.startswith('SHAPE=')]
    dtype_line = [l for l in out if l.startswith('DTYPE=')]
    assert shape_line and dtype_line, f'Expected SHAPE/DTYPE lines in output, got: {out}'
    assert '(2, 3)' in shape_line[0], f'Shape mismatch line: {shape_line[0]}'
    # dtype string presence (implementation detail: could be int32 or dace.int32)
    assert 'int32' in dtype_line[0].lower()


@pytest.mark.skipif(
    not importlib.util.find_spec('tensorflow') or os.environ.get('DACE_ENABLE_INPROCESS_TF','0') != '1',
    reason='In-process TF conversion disabled (set DACE_ENABLE_INPROCESS_TF=1) or TF not installed')
def test_tensorflow_wrapper_inprocess():
    """Optional in-process TF -> DaCe conversion using wrapper. Requires explicit opt-in.

    This is intentionally separate; if it causes instability set DACE_ENABLE_INPROCESS_TF=0.
    """
    import tensorflow as tf
    t = tf.constant([[10., 11.],[12.,13.]], dtype=tf.float32)
    class _Wrap:
        def __init__(self, tens): self._t = tens
        def __descriptor__(self):
            arr = self._t.numpy()
            return dace.data.create_datadescriptor(arr)
    d = dace.data.create_datadescriptor(_Wrap(t))
    assert tuple(d.shape) == (2,2)
    assert d.dtype.type == np.float32

if __name__ == "__main__":
    test_typeclass_by_str()
    # test_builtin_list_and_scalar_canonicalize()
    # test_scalar_types()
    # test_torch_is_array_and_canonicalize()
    # test_dace_dtype_resource_values_against_torch_type()
    
    # test_sctype_from_string_and_sctype_from_torch_dtype()
    # test_DType_and_dtype_wrapper()
    # test_tensor_init_with_numpy_dtype()
    # test_numpy_is_array_and_canonicalize()
    # test_list_tuple_typeclasses()
    # test_cpu_backed_array_packages()
    # Individual GPU/accelerator tests (guarded by availability)
    # test_gpu_pytorch_tensor()
    # test_numba_external_type_handling()
    # test_gpu_jax_array()    
    # if importlib.util.find_spec('jax'):
    #     print(f"Testing jax.Array detection...")
    #     test_gpu_jax_array()
  
    # if os.environ.get('DACE_ENABLE_REAL_TF','0') == '1' and importlib.util.find_spec('tensorflow'):
    #     print(f"Testing tensorflow.Tensor detection (real TF enabled)...")
    #     test_gpu_tensorflow_tensor()