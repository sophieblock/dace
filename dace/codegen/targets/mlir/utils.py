# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
Helps extracting the relevant information from MLIR for CodeGen of an MLIR tasklet
Can handle MLIR in generic form or in the supported dialect of pyMLIR
Requires pyMLIR to run
"""
try:
    import mlir
except (ModuleNotFoundError, NameError, ImportError):
    raise ImportError('To use MLIR tasklets, please install the "pymlir" package.')

import dace
from typing import Union, Optional
import re

# Only these types and the vector version of them are supported
TYPE_DICT = {
    "ui8": dace.uint8,
    "ui16": dace.uint16,
    "ui32": dace.uint32,
    "ui64": dace.uint64,
    "si8": dace.int8,
    "si16": dace.int16,
    "si32": dace.int32,
    "si64": dace.int64,
    "i8": dace.int8,
    "i16": dace.int16,
    "i32": dace.int32,
    "i64": dace.int64,
    "f16": dace.float16,
    "f32": dace.float32,
    "f64": dace.float64
}


class _PseudoModule:
    def __init__(self, code: str):
        self.code = code


class _PseudoFunction:
    def __init__(self, name: str, args: list[tuple[str, str]], result_type: str):
        self._name = name
        self._args = args  # list of (name, type_str)
        self._result_type = result_type  # str

    # For compatibility with get_func_name for dialect AST
    class _Name:
        def __init__(self, value: str):
            self.value = value

    @property
    def name(self):
        return _PseudoFunction._Name(self._name)

    @property
    def args(self):
        # Create a minimal structure with .name.value and .type
        class _ArgName:
            def __init__(self, v: str):
                self.value = v

        class _Arg:
            def __init__(self, n: str, t: str):
                self.name = _ArgName(n)
                self.type = t  # str, handled in get_dace_type

        return [_Arg(n, t) for n, t in self._args]

    @property
    def result_types(self):
        # Return as a single type (not list) for dialect path
        return self._result_type


_FUNC_SIG_RE = re.compile(
    r"func\s+@?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\((?P<args>[^)]*)\)\s*->\s*(?P<ret>[^\s\{]+)",
    re.MULTILINE,
)


def _parse_pseudo_function_from_code(code: str, expected_name: Optional[str]):
    for m in _FUNC_SIG_RE.finditer(code):
        name = m.group('name')
        if expected_name is not None and name != expected_name:
            continue
        args_s = m.group('args').strip()
        ret_s = m.group('ret').strip()

        args = []
        if args_s:
            # split on commas not inside angle brackets
            parts = []
            depth = 0
            buf = ''
            for ch in args_s:
                if ch == '<':
                    depth += 1
                elif ch == '>':
                    depth = max(0, depth - 1)
                if ch == ',' and depth == 0:
                    parts.append(buf.strip())
                    buf = ''
                else:
                    buf += ch
            if buf.strip():
                parts.append(buf.strip())
            for p in parts:
                # format: %a: i32
                if ':' in p:
                    name_part, type_part = p.split(':', 1)
                    aname = name_part.strip().lstrip('%')
                    atype = type_part.strip()
                    args.append((aname, atype))
        return _PseudoFunction(name, args, ret_s)
    return None


def get_ast(code: str):
    # If canonical MLIR function syntax is detected, use pseudo parser to avoid grammar mismatches
    if _FUNC_SIG_RE.search(code) is not None:
        return _PseudoModule(code)
    # Try parsing using pymlir for generic/builtin dialect
    try:
        return mlir.parse_string(code).modules[0]
    except Exception:
        # Fallback to pseudo-module that we can parse signatures from
        return _PseudoModule(code)


def is_generic(ast: Union[mlir.astnodes.Module, mlir.astnodes.GenericModule]):
    return isinstance(ast, mlir.astnodes.GenericModule)


def get_entry_func(ast: Union[mlir.astnodes.Module, mlir.astnodes.GenericModule],
                   is_generic: bool,
                   func_uid: Optional[str] = None):
    # mlir_entry is a reserved keyword for the entry function. In order to allow for multiple MLIR tasklets we append a UID
    entry_func_name = "mlir_entry"
    if func_uid is not None:
        entry_func_name = entry_func_name + func_uid

    # If we have a pseudo module, extract directly from code text
    if isinstance(ast, _PseudoModule):
        # Ensure exactly one entry function exists
        matches = list(_FUNC_SIG_RE.finditer(ast.code))
        matches = [m for m in matches if m.group('name') == entry_func_name]
        if len(matches) == 0:
            raise SyntaxError('No entry function in MLIR tasklet, please make sure a "mlir_entry()" function is present.')
        if len(matches) > 1:
            raise SyntaxError('Multiple entry function in MLIR tasklet.')
        func = _parse_pseudo_function_from_code(ast.code, entry_func_name)
        return func

    # Iterating over every function in the body of every region to check if ast contains exactly one entry and saving the entry function
    entry_func = None

    # pymlir AST changed from singular `region` to `regions`; support both
    regions = getattr(ast, 'regions', None)
    if regions is None:
        region = getattr(ast, 'region', None)
        regions = [region] if region is not None else []

    for region in regions:
        if region is None:
            continue
        for body in getattr(region, 'body', []) or []:
            for op in getattr(body, 'body', []) or []:
                # Some nodes wrap the actual op under `.op`
                func = getattr(op, 'op', op)
                try:
                    func_name = get_func_name(func, is_generic)
                except Exception:
                    continue

                if func_name == entry_func_name:
                    if entry_func is not None:
                        raise SyntaxError("Multiple entry function in MLIR tasklet.")
                    entry_func = func

    if entry_func is None:
        raise SyntaxError('No entry function in MLIR tasklet, please make sure a "mlir_entry()" function is present.')

    return entry_func


def _unwrap_value(v, max_depth: int = 6):
    """Best-effort unwrap of nested value nodes (e.g., .value.value...)."""
    for _ in range(max_depth):
        if hasattr(v, 'value'):
            v = v.value
        else:
            break
    return v


def get_func_name(func: Union[mlir.astnodes.Function, mlir.astnodes.GenericModule, _PseudoFunction], is_generic: bool):
    if isinstance(func, _PseudoFunction):
        return func._name
    if is_generic:
        # In generic AST, locate the 'sym_name' attribute by name
        attrs = getattr(func, 'attributes', None)
        if attrs is None:
            raise SyntaxError('Generic MLIR function missing attributes (sym_name).')

        # Attributes typically hold parallel arrays `names` and `values`
        names = getattr(attrs, 'names', None)
        values = getattr(attrs, 'values', None)
        if names is not None and values is not None:
            # Find index of 'sym_name'
            idx = None
            for i, n in enumerate(names):
                if _unwrap_value(n) == 'sym_name':
                    idx = i
                    break
            if idx is None:
                # Fallback: first value
                name_node = values[0]
            else:
                name_node = values[idx]
            return _unwrap_value(name_node)
        # Fallback to prior behavior if structure is different
        return _unwrap_value(attrs.values[0])
    # In dialect ast the name can be found directly
    return func.name.value


def get_entry_args(entry_func: Union[mlir.astnodes.Function, mlir.astnodes.GenericModule, _PseudoFunction], is_generic: bool):
    ret = []

    if isinstance(entry_func, _PseudoFunction):
        return [(n, t) for n, t in entry_func._args]

    if is_generic:
        # In generic AST the block label contains arg ids and types. Support regions vs region.
        regions = getattr(entry_func, 'regions', None)
        if regions is None:
            region = getattr(entry_func, 'region', None)
            regions = [region] if region is not None else []

        if not regions or regions[0] is None:
            return []

        first_region = regions[0]
        bodies = getattr(first_region, 'body', []) or []
        if not bodies:
            return []
        first_body = bodies[0]
        label = getattr(first_body, 'label', None)
        if label is None:
            return []

        arg_names = getattr(label, 'arg_ids', []) or []
        arg_types = getattr(label, 'arg_types', []) or []

        for idx in range(min(len(arg_names), len(arg_types))):
            arg_name = _unwrap_value(getattr(arg_names[idx], 'value', arg_names[idx]))
            arg_type = arg_types[idx]
            ret.append((arg_name, arg_type))
        return ret

    if getattr(entry_func, 'args', None) is None:
        return []

    for arg in entry_func.args:
        arg_name = arg.name.value
        arg_type = arg.type
        ret.append((arg_name, arg_type))
    return ret


def get_entry_result_type(entry_func: Union[mlir.astnodes.Function, mlir.astnodes.GenericModule, _PseudoFunction], is_generic: bool, code_str: Optional[str] = None):
    if isinstance(entry_func, _PseudoFunction):
        return entry_func._result_type

    if is_generic:
        # First, try the legacy attribute path that works for many pymlir versions
        try:
            vals = getattr(getattr(entry_func, 'attributes', None), 'values', None)
            if isinstance(vals, list) and len(vals) > 1:
                legacy_node = vals[1]
                legacy_ty = getattr(getattr(legacy_node, 'value', None), 'value', None)
                if hasattr(legacy_ty, 'result_types'):
                    legacy_result_list = legacy_ty.result_types
                    if isinstance(legacy_result_list, list):
                        if len(legacy_result_list) != 1:
                            raise SyntaxError('Entry function in MLIR tasklet must return exactly one value.')
                        return legacy_result_list[0]
                    else:
                        return legacy_result_list
        except Exception:
            pass

        # Fallback: robust named-attribute search across AST variants
        attrs = getattr(entry_func, 'attributes', None)
        if attrs is None:
            raise SyntaxError('Generic MLIR function missing attributes (type).')

        names = getattr(attrs, 'names', None)
        values = getattr(attrs, 'values', None)
        func_type_node = None
        if names is not None and values is not None:
            idx = None
            for i, n in enumerate(names):
                if _unwrap_value(n) == 'type':
                    idx = i
                    break
            if idx is not None:
                func_type_node = values[idx]
            else:
                # Fallback to second value by convention if present
                if len(values) > 1:
                    func_type_node = values[1]
                elif len(values) > 0:
                    func_type_node = values[0]

        if func_type_node is None:
            # Some AST variants expose the function type directly as `entry_func.type`
            t = getattr(entry_func, 'type', None)
            if t is not None and hasattr(t, 'result_types'):
                generic_result_list = t.result_types
                # Some generic ASTs wrap single return type in a list; if more than one, fall back to source parsing
                if isinstance(generic_result_list, list):
                    if len(generic_result_list) == 1:
                        return generic_result_list[0]
                else:
                    return generic_result_list
                # If we get here, len != 1; fall through to code_str parsing
            # Fallback: parse from source code if provided
            if code_str is not None:
                try:
                    fn = _parse_pseudo_function_from_code(code_str, get_func_name(entry_func, True))
                    if fn is not None:
                        return fn._result_type
                except Exception:
                    pass
            # raise SyntaxError(f'Could not find function type in MLIR generic attributes. code_str={code_str}, generic result list= {entry_func.attributes.values[1].value.value.result_types}')  

        # Unwrap until we find an object with `result_types`
        t = func_type_node
        for _ in range(8):
            if hasattr(t, 'result_types'):
                generic_result_list = t.result_types
                break
            t = getattr(t, 'value', None)
            if t is None:
                break
        else:
            generic_result_list = None

        if generic_result_list is None:
            raise SyntaxError(f'Could not find function type in MLIR generic attributes. code_str={code_str}, generic result list= {entry_func.attributes.values[1].value.value.result_types}')  

        # Only one return value allowed as we can not match multiple return values
        if len(generic_result_list) != 1:
            raise SyntaxError(f'Entry function in MLIR tasklet must return exactly one value. Got {len(generic_result_list)}.generic_result_list={generic_result_list} vs.  generic result list2= {entry_func.attributes.values[1].value.value.result_types}')

        return generic_result_list[0]

    dialect_result = entry_func.result_types
    # Only one return value allowed as we can not match multiple return values
    if isinstance(dialect_result, list):
        raise SyntaxError('Entry function in MLIR tasklet must return exactly one value.')

    return dialect_result


def get_dace_type(node: Union[mlir.astnodes.IntegerType, mlir.astnodes.FloatType, mlir.astnodes.VectorType, str]):
    # Allow string inputs as a fallback parser path
    if isinstance(node, str):
        s = node.strip()
        # Vector types: vector<4xi32>
        if s.startswith('vector<') and s.endswith('>'):
            inner = s[len('vector<'):-1]
            # format: <len>x<type>
            if 'x' in inner:
                veclen_s, subtype_s = inner.split('x', 1)
                try:
                    veclen = int(veclen_s)
                except ValueError:
                    # Default to 1 if parsing fails
                    veclen = 1
                subtype = get_dace_type(subtype_s)
                return dace.vector(subtype, veclen)
        # Direct scalar type
        if s in TYPE_DICT:
            return TYPE_DICT[s]
        # Some canonical forms use plain i32/f32 without si/ui prefixes
        if s in ('i8', 'i16', 'i32', 'i64', 'f16', 'f32', 'f64'):
            return TYPE_DICT[s]
        raise SyntaxError(f'Unsupported MLIR type: {s}')

    if isinstance(node, mlir.astnodes.IntegerType) or isinstance(node, mlir.astnodes.FloatType):
        return TYPE_DICT[node.dump()]

    if isinstance(node, mlir.astnodes.VectorType):
        result_dim = node.dimensions[0]
        result_subtype = get_dace_type(node.element_type)
        return dace.vector(result_subtype, result_dim)
