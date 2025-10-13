sep_str = 100*'#'


# from torch.export import export
import torch

from torch.fx.experimental.symbolic_shapes import ShapeEnv,DimDynamic,is_symbolic, StatelessSymbolicContext



# from torch.fx.experimental.rewriter import RewritingTracer
class MM(torch.nn.Module):
        
        def forward(self,X: torch.Tensor, Y: torch.Tensor):

        
            return torch.matmul(X, Y)
class MM2(torch.nn.Module):
    def __init__(self,a: torch.Tensor,b: torch.Tensor):
        super().__init__()
        self.a = a
        self.b = b
    def forward(self):

        X = self.a
        Y = self.b
        return torch.matmul(X, Y)
class MM3(torch.nn.Module):
    def __init__(self,a: torch.Tensor):
        super().__init__()
        self.a = a
        
    def forward(self,Y:torch.Tensor):

        X = self.a
        
        return torch.matmul(X, Y)
def test_compare_MM_signatures():
    print(f"{sep_str}\n running test_compare_MM_signatures()...")
    matmul_mod = MM()
    x = torch.randn(5, 2)
    y = torch.randn(2, 3)
    exported_program = export(matmul_mod, (x, y))
    print("\nExported program spec table: ")
    
    print(exported_program.graph.print_tabular())
    print(f".signature: \n  - in specs: {exported_program.graph_signature.input_specs}\n  - out specs: {exported_program.graph_signature.output_specs}\n")
    m2 = MM2(x,y)
    exported_program = export(m2,())
    print("\nExported program spec table: ")
    
    print(exported_program.graph.print_tabular())
    print(f".signature: \n  - in specs: {exported_program.graph_signature.input_specs}\n  - out specs: {exported_program.graph_signature.output_specs}\n")
    m3 = MM3(a=x)
    exported_program = export(m3,(y,))
    print("\nExported program spec table: ")
    
    print(exported_program.graph.print_tabular())
    print(f"ExportedProgram named_params:")
    for param_name in exported_program.graph_signature.parameters:
        print(param_name, exported_program.state_dict[param_name])
    
    print(f".signature: \n  - in specs: {exported_program.graph_signature.input_specs}\n  - out specs: {exported_program.graph_signature.output_specs}\n")
    for node in exported_program.graph.nodes:
        if node.op == "placeholder":
            print(f"'{node.name}', target={node.target}, {type(node.target)}\n - meta: {node.meta}")
            # print(f"'{node.name}'\n - meta.val: {node.meta['val']}\n - meta.from_node: {node.meta['from_node']}\n - meta.tensor_meta: {node.meta['tensor_meta']}")
    print(exported_program.graph_signature.parameters)
from torch._subclasses import fake_tensor
def test_split_join_dynamo():
    print(f"{sep_str}\n running test_split_join_dynamo()...")
    a_tensor = torch.randn(2,)
    shape_env = ShapeEnv()
    with fake_tensor.FakeTensorMode(shape_env=shape_env) as fake_mode:

        fake_a = fake_mode.from_tensor(
            a_tensor, symbolic_context=StatelessSymbolicContext(
                dynamic_sizes=[DimDynamic.DYNAMIC for _ in range(a_tensor.dim())],
            )
        )
    exported_program, guards = torch._dynamo.export(m)(fake_a,)
    print("\n exported_dynamo_graph_sym (fake input tensors via _dynamo.export()):")
    exported_program.graph.print_tabular()
    print(exported_program.graph)
if __name__ == "__main__":
 
    test_split_join_dynamo()
