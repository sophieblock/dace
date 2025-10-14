import torch
from torch._subclasses import fake_tensor,FakeTensorMode
from torch.fx import symbolic_trace
from torch.fx.experimental.proxy_tensor import make_fx
from torch.export import export

def print_sig_nice(graph_signature):
    # print(graph_signature)
    for inspec in graph_signature.input_specs:
        print(f" - {inspec}")
    for outspec in graph_signature.output_specs:
        print(f" - {outspec}")

class MM(torch.nn.Module):
        
        def forward(self,X: torch.Tensor, Y: torch.Tensor):

            result = torch.mm(X, Y)
            return result
        
        
class MM3(torch.nn.Module):
    def __init__(self,a: torch.Tensor):
        super().__init__()
        self.X = a

    def forward(self, Y:torch.Tensor):

        X = self.X
        
        return torch.mm(X, Y)