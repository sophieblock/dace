#!/usr/bin/env python3
"""
torch_ir.py - PyTorch FX Node Metadata Analysis Tool

This script demonstrates how to analyze the metadata of PyTorch FX Nodes using 
a tabular print method that focuses on runtime validation data and guards.
"""

import os
import sys

# Make sure tabulate is installed
try:
    from tabulate import tabulate
except ImportError:
    import subprocess
    print("Installing tabulate package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
    from tabulate import tabulate

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch location: {torch.__file__}")

from torch.fx.tensor_type import TensorType, Dyn
from torch.fx.annotate import annotate
from torch.fx import symbolic_trace
from torch.export import export
from torch.fx.experimental.proxy_tensor import make_fx
# Import our node utilities or define them if not available

# Define the metadata-focused tabular printer
def print_node_meta_tabular(node, include_full_tensor_data=False):
    """
    Prints a detailed tabular view of a node's metadata, focusing on 
    runtime validation data, guards, and shape information.
    """
    # Basic node identification
    basic_info = [
        ["Property", "Value"],
        ["Node Name", node.name],
        ["Op Type", node.op],
        ["Target", node._pretty_print_target(node.target)],
    ]
    
    # Static type information
    if node.type is not None:
        basic_info.append(["Static Type", str(node.type)])
    
    # Organize metadata by category
    tensor_meta = []
    guard_meta = []
    shape_meta = []
    other_meta = []
    
    # Process all metadata
    for key, value in node.meta.items():
        # Tensor metadata (shapes, dtypes, etc.)
        if key == "tensor_meta":
            tensor_meta.append(["tensor_meta", "See detailed table below"])
            # TensorMetadata is a NamedTuple; use _asdict() to iterate fields
            for tm_key, tm_val in value._asdict().items():
                tensor_meta.append([f"  - {tm_key}", str(tm_val)])
        
        # Actual tensor values (summarized unless full data requested)
        elif key == "val" and isinstance(value, torch.Tensor):
            if include_full_tensor_data:
                other_meta.append([key, str(value)])
            else:
                tensor_info = f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
                if value.device.type != "cpu":
                    tensor_info += f", device={value.device}"
                other_meta.append([key, tensor_info])
        
        # Guard-related information
        elif key.startswith("guard_") or "guard" in key:
            guard_meta.append([key, _format_value(value)])
        
        # Shape-related information
        elif "shape" in key or "dim" in key or "size" in key:
            shape_meta.append([key, _format_value(value)])
        
        # Everything else
        else:
            other_meta.append([key, _format_value(value)])
    
    # Build the complete output
    result = [
        "Node Metadata Table:",
        tabulate(basic_info, headers="firstrow", tablefmt="grid")
    ]
    
    if tensor_meta:
        result.append("\nTensor Metadata:")
        result.append(tabulate([["Property", "Value"]] + tensor_meta, 
                            headers="firstrow", tablefmt="grid"))
    
    if guard_meta:
        result.append("\nGuard Conditions:")
        result.append(tabulate([["Guard", "Condition"]] + guard_meta, 
                            headers="firstrow", tablefmt="grid"))
    
    if shape_meta:
        result.append("\nShape Information:")
        result.append(tabulate([["Property", "Value"]] + shape_meta, 
                            headers="firstrow", tablefmt="grid"))
    
    if other_meta:
        result.append("\nOther Metadata:")
        result.append(tabulate([["Property", "Value"]] + other_meta, 
                            headers="firstrow", tablefmt="grid"))
    
    # Input dependencies
    if hasattr(node, "_input_nodes") and node._input_nodes:
        input_nodes = [["Input Node", "Op Type"]]
        for input_node in node._input_nodes:
            input_nodes.append([input_node.name, input_node.op])
        
        result.append("\nInput Dependencies:")
        result.append(tabulate(input_nodes, headers="firstrow", tablefmt="grid"))
    
    return "\n".join(result)

def _format_value(value):
    """Format a value for tabular display with special handling for common types"""
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
    elif isinstance(value, (list, tuple)) and len(value) > 5:
        return f"{type(value).__name__} of length {len(value)}: {str(value[:3])}..."
    elif isinstance(value, dict) and len(value) > 5:
        keys = list(value.keys())
        return f"dict with {len(value)} keys: {keys[:3] + ['...']}"
    else:
        # Truncate long strings
        str_val = str(value)
        if len(str_val) > 80:
            return str_val[:77] + "..."
        return str_val

# Monkey patch the Node class to add our metadata-focused method
from torch.fx import node
from torch._subclasses.fake_tensor import extract_tensor_metadata

import types
def type_repr(obj: object) -> str:
    """Return the repr() of an object, special-casing types (internal helper).
    If obj is a type, we return a shorter version than the default
    type.__repr__, based on the module and qualified name, which is
    typically enough to uniquely identify a type.  For everything
    else, we fall back on repr(obj).
    """
    # Extension: If we don't ignore GenericAlias then `list[int]` will print
    # simply "list".
    if isinstance(obj, type) and not isinstance(obj, types.GenericAlias):
        if obj.__module__ == "builtins":
            return obj.__qualname__
        return f"{obj.__module__}.{obj.__qualname__}"
    if obj is ...:
        return "..."
    if isinstance(obj, types.FunctionType):
        return obj.__name__
    return repr(obj)
def print_node_spec(node):
    """
    Prints a tabular view of a node's input specifications and output specifications.
    
    Args:
        node: An FX Graph Node
        
    Returns:
        str: A formatted string containing the spec mapping table
    """
    headers = ["Input Name", "Input Spec", "Output Spec"]
    rows = []
    
    # Extract node output specification
    output_spec = ""
    if node.type is not None:
        output_spec = str(node.type)
    elif "tensor_meta" in node.meta:
        shape_str = f"shape={str(node.meta['tensor_meta'].shape)}"
        dtype_str = f"dtype={node.meta['tensor_meta'].dtype}"
        output_spec = f"Tensor({shape_str}, {dtype_str})"
    elif "val" in node.meta and isinstance(node.meta["val"], torch.Tensor):
        tensor = node.meta["val"]
        output_spec = f"Tensor(shape={tuple(tensor.shape)}, dtype={tensor.dtype})"
    
    if node.op == "placeholder":
        # For placeholders, just show one row with the node name
        rows.append([node.name, "N/A", output_spec])
    elif node.op in ["call_function", "call_method", "call_module"]:
        # For each argument, add a row
        for i, arg in enumerate(node.args):
            if isinstance(arg, torch.fx.node.Node):
                # Get input specification
                input_spec = ""
                if arg.type is not None:
                    input_spec = str(arg.type)
                elif "tensor_meta" in arg.meta:
                    shape_str = f"shape={str(arg.meta['tensor_meta'].shape)}"
                    dtype_str = f"dtype={arg.meta['tensor_meta'].dtype}"
                    input_spec = f"Tensor({shape_str}, {dtype_str})"
                elif "val" in arg.meta and isinstance(arg.meta["val"], torch.Tensor):
                    tensor = arg.meta["val"]
                    input_spec = f"Tensor(shape={tuple(tensor.shape)}, dtype={tensor.dtype})"
                
                rows.append([f"args[{i}]: {arg.name}", input_spec, output_spec if i == 0 else ""])
        
        # Handle kwargs
        for name, arg in node.kwargs.items():
            if isinstance(arg, torch.fx.node.Node):
                # Get input specification
                input_spec = ""
                if arg.type is not None:
                    input_spec = str(arg.type)
                elif "tensor_meta" in arg.meta:
                    shape_str = f"shape={str(arg.meta['tensor_meta'].shape)}"
                    dtype_str = f"dtype={arg.meta['tensor_meta'].dtype}"
                    input_spec = f"Tensor({shape_str}, {dtype_str})"
                elif "val" in arg.meta and isinstance(arg.meta["val"], torch.Tensor):
                    tensor = arg.meta["val"]
                    input_spec = f"Tensor(shape={tuple(tensor.shape)}, dtype={tensor.dtype})"
                
                rows.append([f"kwargs[{name}]: {arg.name}", input_spec, ""])
    elif node.op == "output":
        # For output nodes, show the returned value
        if len(node.args) > 0:
            arg = node.args[0]
            if isinstance(arg, torch.fx.node.Node):
                input_spec = ""
                if arg.type is not None:
                    input_spec = str(arg.type)
                elif "tensor_meta" in arg.meta:
                    shape_str = f"shape={str(arg.meta['tensor_meta'].shape)}"
                    dtype_str = f"dtype={arg.meta['tensor_meta'].dtype}"
                    input_spec = f"Tensor({shape_str}, {dtype_str})"
                elif "val" in arg.meta and isinstance(arg.meta["val"], torch.Tensor):
                    tensor = arg.meta["val"]
                    input_spec = f"Tensor(shape={tuple(tensor.shape)}, dtype={tensor.dtype})"
                
                rows.append([f"return: {arg.name}", input_spec, "N/A"])
    
    # Add the method to Node class
    if not rows:
        return "No specification information available"
    
    table = tabulate(rows, headers=headers, tablefmt="grid")
    return f"Node Specification Mapping for {node.name} (op={node.op}):\n{table}"
def print_placeholder_node_spec(node):
    """
    Prints a tabular view of a placeholder node's input specifications.
    
    Args:
        node: An FX Graph Node representing a placeholder
        
    Returns:
        str: A formatted string containing the spec mapping table
    """
    headers = ["Input Name", "Input Spec"]
    rows = []
    

    # For placeholders, just show one row with the node name
    input_spec = ""
    
    assert isinstance(node.target, str)
    arg_str = node.target
    arg_str += arg_str + f": {type_repr(node.type)}" if node.type else ""

    maybe_typename = f"{type_repr(node.type)} " if node.type else ""
    default_val = "(default=" + str(node.args[0]) + ")" if node.args else ""
    formatted_str= f"%{node.name} : {maybe_typename}[num_users={len(node.users)}] = {node.op}[target={node.target}]{default_val}"
    print(f"Formatted Placeholder Node: {formatted_str}")
    print(f" - node.name:               {node.name}")
    print(f" - node.target:             {node.target}")
    print(f" - arg_str (target_...):    {arg_str}")
    
    if node.type is not None:
        input_spec = str(node.type)
    elif "tensor_meta" in node.meta:
        shape_str = f"shape={str(node.meta['tensor_meta'].shape)}"
        dtype_str = f"dtype={node.meta['tensor_meta'].dtype}"
        input_spec = f"Tensor({shape_str}, {dtype_str})"
    elif "val" in node.meta and isinstance(node.meta["val"], torch.Tensor):
        tensor = node.meta["val"]
        input_spec = f"Tensor(shape={tuple(tensor.shape)}, dtype={tensor.dtype})"
    
    rows.append([node.name, input_spec])
    
    table = tabulate(rows, headers=headers, tablefmt="grid")
    return f"Placeholder Node Specification for {node.name}:\n{table}"

def print_call_function_node_spec(node):
    """
    Prints a tabular view of a call_function node's input specifications.
    
    Args:
        node: An FX Graph Node representing a call_function
        
    Returns:
        str: A formatted string containing the spec mapping table
    """
    def stringify_shape(shape) -> str:
        return f"[{', '.join([str(x) for x in shape])}]"
    meta_val = node.meta.get(
                "val",
                node.meta.get("tensor_meta", node.meta.get("example_value", None)),
            )
    print(f"meta_val: {meta_val},\n - .untyped_storage: {meta_val.untyped_storage()},\n - .shape: {meta_val.shape},\n - .dtype: {meta_val.dtype}")
    print(f" - .layout: {meta_val.layout}")
    print(f' -> isinstance(meta_val, torch.Tensor)? {isinstance(meta_val, torch.Tensor)}')

    headers = ["Input Name", "Input Spec"]
    rows = []

    # For call_function nodes, show each argument
    for i, arg in enumerate(node.args):
        if isinstance(arg, torch.fx.node.Node):
            input_spec = ""
            if arg.type is not None:
                input_spec = str(arg.type)
            elif "tensor_meta" in arg.meta:
                shape_str = f"shape={str(arg.meta['tensor_meta'].shape)}"
                dtype_str = f"dtype={arg.meta['tensor_meta'].dtype}"
                input_spec = f"Tensor({shape_str}, {dtype_str})"
            elif "val" in arg.meta and isinstance(arg.meta["val"], torch.Tensor):
                tensor = arg.meta["val"]
                input_spec = f"Tensor(shape={tuple(tensor.shape)}, dtype={tensor.dtype})"
            
            rows.append([f"args[{i}]: {arg.name}", input_spec])
    
    # Handle kwargs
    for name, arg in node.kwargs.items():
        if isinstance(arg, torch.fx.node.Node):
            input_spec = ""
            if arg.type is not None:
                input_spec = str(arg.type)
            elif "tensor_meta" in arg.meta:
                shape_str = f"shape={str(arg.meta['tensor_meta'].shape)}"
                dtype_str = f"dtype={arg.meta['tensor_meta'].dtype}"
                input_spec = f"Tensor({shape_str}, {dtype_str})"
            elif "val" in arg.meta and isinstance(arg.meta["val"], torch.Tensor):
                tensor = arg.meta["val"]
                input_spec = f"Tensor(shape={tuple(tensor.shape)}, dtype={tensor.dtype})"
            
            rows.append([f"kwargs[{name}]: {arg.name}", input_spec])

    
    table = tabulate(rows, headers=headers, tablefmt="grid")
    return f"Call Function Node Specification for {node.name}:\n{table}"
def get_graph_table_v2(G):
    """
    Prints the intermediate representation of the graph in tabular
    format. Note that this API requires the ``tabulate`` module to be
    installed.
    """
    from torch._library.infer_schema import infer_schema
    
    node_specs=[]
    for n in G.nodes:
        if n.op == 'call_function':
            assert callable(n.target)
            # schema = infer_schema(n.target, mutates_args={})
            input_nodes = n.all_input_nodes
        else:
            input_nodes = n.all_input_nodes
        prev_node = n.prev
        next_node = n.next
        node_specs.append([n.op,n.name, n.target, n.args,n.type,prev_node,next_node, input_nodes,n.kwargs])
    # node_specs = [[n.op,n.name, n.target, n.args,n.type, n.kwargs] for n in self.nodes]
    # print(
    #     tabulate(node_specs, headers=["opcode", "name", "target", "args","type", "Prev Node", "Next Node", "Input Nodes", "kwargs"])
    # )
    return tabulate(node_specs, headers=["opcode", "name", "target", "args","type", "Prev Node", "Next Node", "Input Nodes", "kwargs"])
    
# # Add method to Node class
# if not hasattr(node.Node, "print_meta_tabular"):
#     node.Node.print_meta_tabular = _print_meta_tabular
#     print("Successfully added print_meta_tabular method to Node class")

# # Create a module-like interface
# class NodeUtils:
#     @staticmethod
#     def print_node_meta_tabular(node, include_full_tensor_data=False):
#         return print_node_meta_tabular(node, include_full_tensor_data)

# node_utils = NodeUtils()

def demo_exported_node_metadata():
    """Demo the metadata-focused tabular printer on exported module nodes"""
    print("\n" + "="*80)
    print("EXPORTED NODE METADATA DEMO")
    print("="*80)
    
    class MM(torch.nn.Module):
        def forward(self, X: torch.Tensor, Y: torch.Tensor):
            return torch.matmul(X, Y)
   
    # Create test tensors
    x = torch.randn(5, 2)
    y = torch.randn(2, 3)
    
    # Export the module
    matmul_mod = MM()
    exported_program = export(matmul_mod, (x, y))
    graph_signature = exported_program.graph_signature
    input_specs = graph_signature.input_specs
    output_specs = graph_signature.output_specs
    # print(graph_signature)
    
    
    print("\nExported program spec table: ")

    # print(exported_program.graph.print_tabular())
    dag_table = get_graph_table_v2(exported_program.graph)
    print(dag_table)
    print(f"\nGraph Signature:\n input specs")
    for idx,inspec in enumerate(input_specs):
        print(f' {idx}. {inspec}')
    print(f"\n output specs")
    print(f'    {output_specs}')
    
    print("\nExamining nodes in exported program:")
    
    for node in exported_program.graph.nodes:
        print(f"\nNode: {node.name} (op={node.op})")
        print(node.format_node())
        print(print_node_spec(node))
        # print(node.print_meta_tabular(include_full_tensor_data=True))


def demo_exported_node_metadata_2():
    """Demo the metadata-focused tabular printer on exported module nodes"""
    print("\n" + "="*80)
    print("EXPORTED NODE METADATA DEMO")
    print("="*80)
    
    class MM3(torch.nn.Module):
        def __init__(self,a: torch.Tensor):
            super().__init__()
            self.a = a
            
        def forward(self,Y:torch.Tensor):

            X = self.a
            
            return torch.matmul(X, Y)
   
    # Create test tensors
    x = torch.randn(5, 2)
    y = torch.randn(2, 3)
    
    # Export the module
    matmul_mod = MM3(a=x)
    
    exported_program = export(matmul_mod, (y,))
    graph_signature = exported_program.graph_signature
    input_specs = graph_signature.input_specs
    output_specs = graph_signature.output_specs
    # print(graph_signature)
    
    
    print("\nExported program spec table: ")

    # print(exported_program.graph.print_tabular())
    dag_table = get_graph_table_v2(exported_program.graph)
    print(dag_table)
    print(f"\nGraph Signature:\n input specs")
    for idx,inspec in enumerate(input_specs):
        print(f' {idx}. {inspec}')
    print(f"\n output specs")
    print(f'    {output_specs}')
    outspec = output_specs[0]
    print(type(outspec))
    
    print("\nExamining nodes in exported program:")
    
    for node in exported_program.graph.nodes:
        print(f"\nNode: {node.name} (op={node.op})")
        print(node.format_node())
        # print(print_node_spec(node))
        if node.op == 'placeholder':
            print(print_placeholder_node_spec(node))
        elif node.op == 'call_function':
            print(print_call_function_node_spec(node))

def main():
    """Main function to run all demos"""
    # demo_exported_node_metadata()
    demo_exported_node_metadata_2()


if __name__ == "__main__":
    main()