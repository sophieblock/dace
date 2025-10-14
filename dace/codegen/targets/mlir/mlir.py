# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from typing import TYPE_CHECKING
from dace import registry, dtypes
from dace.codegen.codeobject import CodeObject
from dace.codegen.targets.target import TargetCodeGenerator
from dace.codegen.targets.cpu import CPUCodeGen
from dace.sdfg import nodes
from dace.sdfg.sdfg import SDFG

if TYPE_CHECKING:
    from dace.codegen.targets.framecode import DaCeCodeGenerator




@registry.autoregister_params(name='mlir')
class MLIRCodeGen(TargetCodeGenerator):
    target_name = 'mlir'
    title = 'MLIR'

    def __init__(self, frame_codegen: 'DaCeCodeGenerator', sdfg: SDFG):
        self._codeobjects = []
        self._cpu_codegen: CPUCodeGen = frame_codegen.dispatcher.get_generic_node_dispatcher()
        frame_codegen.dispatcher.register_node_dispatcher(self, self.node_dispatch_predicate)

    def get_generated_codeobjects(self):
        return self._codeobjects

    def node_dispatch_predicate(self, sdfg, state, node):
        return isinstance(node, nodes.Tasklet) and node.language == dtypes.Language.MLIR

    def generate_node(self, sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream):
        if self.node_dispatch_predicate(sdfg, dfg, node):
            function_uid = str(cfg.cfg_id) + "_" + str(state_id) + "_" + str(dfg.node_id(node))
            code = node.code.code.replace("mlir_entry", "mlir_entry_" + function_uid)

            # Attempt to upgrade canonical MLIR to newer dialect spelling when possible
            # Skip if generic MLIR is used (quotes) or already upgraded
            if '"' not in code:
                # Replace top-level module and func ops
                if 'func.func' not in code:
                    code = code.replace('\n', '\n')  # no-op to keep style
                    code = code.replace('module', 'builtin.module')
                    code = code.replace('func @', 'func.func @')
                # Replace bare return with func.return (avoid touching words containing return)
                code = code.replace('\n        return ', '\n        func.return ')
                code = code.replace('\n    return ', '\n    func.return ')
                code = code.replace('\nreturn ', '\nfunc.return ')
                # Map legacy ops to modern dialects
                code = code.replace(' addi ', ' arith.addi ')
                code = code.replace(' addf ', ' arith.addf ')
            else:
                # Generic MLIR (quoted) - map deprecated std.* ops to current dialects
                # Arithmetic ops
                code = code.replace('"std.addi"', '"arith.addi"')
                code = code.replace('"std.addf"', '"arith.addf"')
                code = code.replace('"std.subi"', '"arith.subi"')
                code = code.replace('"std.cmpi"', '"arith.cmpi"')
                code = code.replace('"std.constant"', '"arith.constant"')
                # Control-flow and function ops
                code = code.replace('"std.return"', '"func.return"')
                code = code.replace('"std.cond_br"', '"cf.cond_br"')
                code = code.replace('"std.call"', '"func.call"')
                code = code.replace('"std.br"', '"cf.br"')
                # Remove deprecated operand_segment_sizes attribute on cond_br
                import re as _re
                code = _re.sub(r'"cf\.cond_br"\(([^)]*)\)\s*\[[^\]]*\]\s*\{[^}]*operand_segment_sizes[^}]*\}', r'"cf.cond_br"(\1) [^bb]', code)

            node.code.code = code
            node.label = node.name + "_" + function_uid
            self._codeobjects.append(CodeObject(node.name, node.code.code, "mlir", MLIRCodeGen, node.name + "_Source"))

        self._cpu_codegen.generate_node(sdfg, cfg, dfg, state_id, node, function_stream, callsite_stream)

    @staticmethod
    def cmake_options():
        options = []
        return options
