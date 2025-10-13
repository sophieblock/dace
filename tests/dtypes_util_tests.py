# Tests for result_type_of, pointer, and vector typeclasses in DaCe
# Focus: primitive promotions, symbolic operands, vector precedence, pointer & vector construction,
# JSON roundtrips, and SVE assert_type_compatibility edge cases.
# link to gpt convo: https://chatgpt.com/c/68cf6c77-0a20-8332-9024-5fb065f1145e

import ctypes
import math
import numpy as np
import pytest

import dace
from dace import dtypes, symbolic
from dace.data import create_datadescriptor, Array, Scalar
from dace.dtypes import StorageType
# Optional SVE compatibility import (skip if unavailable)
try:
    from dace.codegen.targets.sve.type_compatibility import assert_type_compatibility, IncompatibleTypeError  # type: ignore
    from collections import OrderedDict
    HAVE_SVE = True
except Exception:  # pragma: no cover
    HAVE_SVE = False

print(f"HAVE_SVE = {HAVE_SVE}")
#############################################
# Helper utilities for introspection asserts
#############################################

def _assert_typeclass(tc, expected_numpy_type=None):
    assert isinstance(tc, dtypes.typeclass) or isinstance(tc, (dtypes.pointer, dtypes.vector, dtypes.struct, dtypes.opaque))
    if expected_numpy_type is not None and not isinstance(tc, dtypes.pointer):  # pointer has no direct numpy scalar mapping
        assert tc.as_numpy_dtype() == np.dtype(expected_numpy_type)
    # as_ctypes should be callable unless opaque
    if not isinstance(tc, dtypes.opaque):
        _ = tc.as_ctypes()


def _np_itemsize(t):
    if isinstance(t, (dtypes.pointer, dtypes.vector)):
        return t.as_numpy_dtype().itemsize
    return t.as_numpy_dtype().itemsize if isinstance(t, dtypes.typeclass) else t(0).itemsize  # fallback
def _assert_gpu_implies_array(x):
    # Current DaCe logic ensures this is always true
    if dtypes.is_gpu_array(x):
        assert dtypes.is_array(x), "is_gpu_array True must imply is_array True"

# ---------------------------
# NumPy / ctypes baselines
# ---------------------------

def test_numpy_array_is_array_not_gpu():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert dtypes.is_array(a)
    assert not dtypes.is_gpu_array(a)
    dd = create_datadescriptor(a)
    assert isinstance(dd, Array)
    assert dd.storage == StorageType.Default

def test_numpy_scalar_not_array():
    s = np.float32(3.14)
    assert not dtypes.is_array(s)  # because shape length == 0
    # create_datadescriptor maps to Scalar
    dd = create_datadescriptor(s)
    assert isinstance(dd, Scalar)

def test_ctypes_array_is_array_not_gpu():
    CA = ctypes.c_float * 12
    a = CA()
    assert dtypes.is_array(a)
    assert not dtypes.is_gpu_array(a)
    dd = create_datadescriptor(a)
    assert isinstance(dd, Array)
    assert dd.storage == StorageType.Default
    
# ---------------------------
# PyTorch
# ---------------------------

@pytest.mark.parametrize("requires_grad", [False, True])
def test_torch_cpu_tensor_is_array_not_gpu(requires_grad):
    torch = pytest.importorskip("torch")
    t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    t.requires_grad_(requires_grad)
    assert dtypes.is_array(t)         # Torch exposes data_ptr/shape
    assert not dtypes.is_gpu_array(t) # CPU tensor => not GPU
    _assert_gpu_implies_array(t)

    dd = create_datadescriptor(t)
    assert isinstance(dd, Array)
    assert dd.storage == StorageType.Default

@pytest.mark.parametrize("requires_grad", [False, True])
def test_torch_cuda_tensor_is_gpu_or_exception_path(requires_grad):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    t = torch.arange(12, dtype=torch.float32, device="cuda").reshape(3, 4)
    t.requires_grad_(requires_grad)

    # Two possible behaviors in PyTorch:
    # 1) __cuda_array_interface__ present => both True
    # 2) Accessing it raises (requires_grad/bool cases) => is_array True, is_gpu_array False
    try:
        _ = hasattr(t, "__cuda_array_interface__")
        # If hasattr returns cleanly and interface exists, DaCe returns True for both
        if hasattr(t, "__cuda_array_interface__"):
            assert dtypes.is_gpu_array(t)
            assert dtypes.is_array(t)
        else:
            # rare: no interface => fallback determines result
            assert dtypes.is_array(t)  # via data_ptr/shape
            assert not dtypes.is_gpu_array(t)
    except (KeyError, RuntimeError):
        # DaCe code path:
        #   is_array -> True (exception means "still array")
        #   is_gpu_array -> False (exception means "not GPU array")
        assert dtypes.is_array(t)
        assert not dtypes.is_gpu_array(t)

    _assert_gpu_implies_array(t)

    # Regardless of the above, create_datadescriptor has a Torch special case
    dd = create_datadescriptor(t)
    assert isinstance(dd, Array)
    assert dd.storage == StorageType.GPU_Global

def test_torch_cuda_bool_tensor_edgecase():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    tb = (torch.arange(10, device="cuda") % 2 == 0)  # boolean CUDA tensor
    # DaCe: is_array True (exception on __cuda_array_interface__), is_gpu_array False
    try:
        _ = hasattr(tb, "__cuda_array_interface__")
        # If attribute access works, ok; otherwise we fall to except:
        pass
    except (KeyError, RuntimeError):
        assert dtypes.is_array(tb)
        assert not dtypes.is_gpu_array(tb)

    dd = create_datadescriptor(tb)
    assert isinstance(dd, Array)
    assert dd.storage == StorageType.GPU_Global

# ---------------------------
# CuPy
# ---------------------------

def test_cupy_gpu_array_is_gpu_and_array():
    cupy = pytest.importorskip("cupy")
    x = cupy.arange(24, dtype=cupy.float32).reshape(4, 6)
    assert dtypes.is_gpu_array(x)
    assert dtypes.is_array(x)
    _assert_gpu_implies_array(x)

    dd = create_datadescriptor(x)
    assert isinstance(dd, Array)
    assert dd.storage == StorageType.GPU_Global

# ---------------------------
# Numba CUDA
# ---------------------------

def test_numba_device_array_is_gpu_and_array():
    numba = pytest.importorskip("numba")
    cuda = pytest.importorskip("numba.cuda")
    a = np.arange(8, dtype=np.float32)
    da = cuda.to_device(a)  # numba.cuda.devicearray.DeviceNDArray
    assert dtypes.is_gpu_array(da)
    assert dtypes.is_array(da)
    _assert_gpu_implies_array(da)
    
# ---------------------------
# Cross-check invariant: if is_gpu_array True, then is_array True
# (Already asserted inline, but here’s a direct property-based check with examples)
# ---------------------------

# @pytest.mark.parametrize("ctor", [])
# def test_gpu_implies_array_smoke():
#     # Placeholder in case you want to plug in synthetic objects later.
#     pass


# ---------------------------
# Cases where is_array True but is_gpu_array False
# ---------------------------

def test_numpy_array_true_false():
    a = np.ones((3, 3), dtype=np.float64)
    assert dtypes.is_array(a)
    assert not dtypes.is_gpu_array(a)


def test_torch_cuda_requires_grad_exception_path_is_array_true_gpu_false():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    t = torch.randn(5, 5, device="cuda", requires_grad=True)
    # DaCe behavior on exception:
    try:
        _ = hasattr(t, "__cuda_array_interface__")
    except (KeyError, RuntimeError):
        assert dtypes.is_array(t)
        assert not dtypes.is_gpu_array(t)

# ---------------------------
# Builtins canonicalization via create_datadescriptor
# ---------------------------

def test_list_tuple_canonicalization_numeric():
    obj = [[1, 2, 3], [4, 5, 6]]
    dd = create_datadescriptor(obj)
    assert isinstance(dd, Array)
    assert dd.storage == StorageType.Default
    assert dd.shape == (2, 3)
    assert dd.dtype.base_type.name in ("int32", "int64")  # depends on DaCe default_data_types config

def test_nested_mixed_types_becomes_object_dtype():
    obj = [[1, 2], ["a", "b"]]  # mixed types
    arr = np.array(obj, dtype=object)  # how create_datadescriptor will view it
    dd = create_datadescriptor(obj)
    assert isinstance(dd, Array)
    # pyobject type inside DaCe for dtype=object
    assert dd.dtype.base_type.name == "pyobject"
    assert dd.shape == arr.shape

def test_jagged_lists_become_object_1d():
    obj = [[1, 2, 3], [4, 5]]  # jagged
    arr = np.array(obj, dtype=object)  # NumPy cannot build a rectangular array → object array
    assert arr.dtype == object
    dd = create_datadescriptor(obj)
    assert isinstance(dd, Array)
    assert dd.dtype.base_type.name == "pyobject"
    assert dd.shape == arr.shape  # usually (2,)

def test_dict_unsupported():
    with pytest.raises(TypeError):
        create_datadescriptor({"a": 1, "b": 2})

def test_set_unsupported():
    with pytest.raises(TypeError):
        create_datadescriptor({1, 2, 3})
        
#############################################
# result_type_of primitive promotions
#############################################

def test_result_type_same_type():
    for tc in [dtypes.int8, dtypes.int16, dtypes.int32, dtypes.uint32, dtypes.float32, dtypes.float64]:
        assert dtypes.result_type_of(tc, tc) is tc


def test_result_type_integer_widening():
    res = dtypes.result_type_of(dtypes.int8, dtypes.int32)
    assert res is dtypes.int32
    res2 = dtypes.result_type_of(dtypes.uint8, dtypes.uint16)
    assert res2 is dtypes.uint16


def test_result_type_signed_unsigned_same_size_rule():
    # Logic: if sizes equal and left is unsigned, return left; else return right
    # int32 vs uint32: left signed -> expect right (uint32)
    r1 = dtypes.result_type_of(dtypes.int32, dtypes.uint32)
    assert r1 is dtypes.uint32
    # uint32 vs int32: left unsigned -> expect left (uint32)
    r2 = dtypes.result_type_of(dtypes.uint32, dtypes.int32)
    assert r2 is dtypes.uint32


def test_result_type_integer_float():
    r = dtypes.result_type_of(dtypes.int32, dtypes.float32)
    assert r is dtypes.float32
    r2 = dtypes.result_type_of(dtypes.uint8, dtypes.float64)
    assert r2 is dtypes.float64


def test_result_type_float_widen():
    r = dtypes.result_type_of(dtypes.float32, dtypes.float64)
    assert r is dtypes.float64
    r2 = dtypes.result_type_of(dtypes.float64, dtypes.float32)
    assert r2 is dtypes.float64  # larger wins irrespective of order


def test_result_type_multi_arg():
    r = dtypes.result_type_of(dtypes.int8, dtypes.int16, dtypes.int32, dtypes.float32)
    assert r is dtypes.float32


def test_result_type_vector_precedence_over_scalar():
    v = dtypes.vector(dtypes.float32, 4)
    r = dtypes.result_type_of(v, dtypes.float64)
    # Vector precedence, even though float64 wider than float32, vector wins
    assert r is v


def test_result_type_vector_vector_same_length_base_promotion():
    v1 = dtypes.vector(dtypes.int16, 2)
    v2 = dtypes.vector(dtypes.int32, 2)
    r = dtypes.result_type_of(v1, v2)
    assert isinstance(r, dtypes.vector)
    assert r.veclen == 2
    assert r.vtype is dtypes.int32  # base type promoted


def test_result_type_vector_vector_different_length():
    v1 = dtypes.vector(dtypes.float32, 2)
    v2 = dtypes.vector(dtypes.float32, 8)
    r = dtypes.result_type_of(v1, v2)
    assert r is v2  # longer vector wins


def test_result_type_with_symbol():
    N = symbolic.symbol('N')  # A symbolic value; result_type_of extracts .dtype if present
    # Make sure symbol has dtype attribute; DaCe symbolic.symbol sets .dtype = int64 typically
    sym_dtype = N.dtype
    r = dtypes.result_type_of(N, dtypes.int32)
    # Expect whichever is larger between symbol dtype and int32
    # Compare sizes via numpy
    lhs_size = sym_dtype.as_numpy_dtype().itemsize
    rhs_size = dtypes.int32.as_numpy_dtype().itemsize
    if lhs_size >= rhs_size:
        assert r is sym_dtype
    else:
        assert r is dtypes.int32


#############################################
# pointer tests
#############################################

def test_pointer_basic_attributes():
    p = dtypes.pointer(dtypes.float32)
    assert isinstance(p, dtypes.pointer)
    assert p.base_type is dtypes.float32
    assert '*' in p.ctype
    # assert p.as_ctypes().__class__.__name__ == 'LP_c_float', f'Got {p.as_ctypes().__class__.__name__}'
    assert p.as_ctypes().__class__.__name__ == 'PyCPointerType', f'Got {p.as_ctypes().__class__.__name__}'
    # numpy dtype from pointer is an object dtype referencing ctypes pointer
    npdt = p.as_numpy_dtype()
    assert isinstance(npdt, np.dtype)
    # JSON roundtrip
    j = p.to_json()
    p2 = dtypes.pointer.from_json({'type': 'pointer', 'dtype': j['dtype']})
    assert isinstance(p2, dtypes.pointer)
    assert p2.base_type.type == p.base_type.type


def test_pointer_struct_nested():
    mystruct = dtypes.struct('Pair', a=dtypes.int32, b=dtypes.float64)
    ps = dtypes.pointer(mystruct)
    assert isinstance(ps.base_type, dtypes.struct)
    ctp = ps.as_ctypes()
    assert issubclass(ctp._type_, ctypes.Structure)


#############################################
# vector tests
#############################################

def test_vector_basic():
    v = dtypes.vector(dtypes.float32, 4)
    assert isinstance(v, dtypes.vector)
    assert v.veclen == 4
    assert v.base_type is dtypes.float32
    assert 'vec' in v.ctype  # C++ representation
    npdt = v.as_numpy_dtype()
    assert npdt.itemsize == dtypes.float32.bytes * 4
    # ctypes representation length
    ctt = v.as_ctypes()
    assert len(ctt()) == 4


def test_vector_json_roundtrip():
    v = dtypes.vector(dtypes.int16, 8)
    j = v.to_json()
    v2 = dtypes.vector.from_json({'type': 'vector', 'dtype': j['dtype'], 'elements': j['elements']})
    assert v2.veclen == v.veclen
    assert v2.vtype.type == v.vtype.type


def test_vector_base_promotion_in_result_type():
    vsmall = dtypes.vector(dtypes.int16, 4)
    vwide = dtypes.vector(dtypes.int32, 4)
    r = dtypes.result_type_of(vsmall, vwide)
    assert r.veclen == 4 and r.vtype is dtypes.int32


#############################################
# SVE compatibility tests (optional)
#############################################
@pytest.mark.skipif(not HAVE_SVE, reason='SVE utilities not available')
def test_sve_compatibility_ok():
    # Provide dummy defined_symbols
    defined = OrderedDict()
    v = dtypes.vector(dtypes.float32, 4)
    # Should not raise
    assert_type_compatibility(defined, (v,))


@pytest.mark.skipif(not HAVE_SVE, reason='SVE utilities not available')
def test_sve_incompatible_mixed_pointer_and_scalar():
    defined = OrderedDict()
    p = dtypes.pointer(dtypes.float32)
    with pytest.raises(IncompatibleTypeError):
        assert_type_compatibility(defined, (p, dtypes.float32))


@pytest.mark.skipif(not HAVE_SVE, reason='SVE utilities not available')
def test_sve_incompatible_mismatched_vectors():
    defined = OrderedDict()
    v1 = dtypes.vector(dtypes.float32, 2)
    v2 = dtypes.vector(dtypes.int32, 2)
    with pytest.raises(IncompatibleTypeError):
        assert_type_compatibility(defined, (v1, v2))


#############################################
# Edge & documentation oriented tests
#############################################

def test_result_type_of_order_dependency_signed_unsigned():
    # Documented behavior: left unsigned of equal size wins, else right
    a = dtypes.result_type_of(dtypes.uint32, dtypes.int32)
    b = dtypes.result_type_of(dtypes.int32, dtypes.uint32)
    assert a is dtypes.uint32 and b is dtypes.uint32


def test_result_type_vector_chain_multiarg():
    v1 = dtypes.vector(dtypes.int8, 2)
    v2 = dtypes.vector(dtypes.int16, 2)
    v3 = dtypes.vector(dtypes.int32, 2)
    r = dtypes.result_type_of(v1, v2, v3)
    assert isinstance(r, dtypes.vector) and r.veclen == 2 and r.vtype is dtypes.int32


def test_pointer_and_vector_different_namespaces_attrs():
    p = dtypes.pointer(dtypes.float64)
    v = dtypes.vector(dtypes.float64, 2)
    # Ensure they expose required attributes
    assert hasattr(p, 'base_type') and hasattr(v, 'base_type')
    assert p.base_type is dtypes.float64
    assert v.base_type is dtypes.float64
    # ctype uniqueness
    assert p.ctype.endswith('*')
    assert 'vec' in v.ctype


if __name__ == '__main__':  # pragma: no cover
    import sys
    import pytest as _pytest
    sys.exit(_pytest.main([__file__]))
