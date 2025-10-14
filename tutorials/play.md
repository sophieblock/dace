Tutorial: DaCe IR – Library Nodes and Operator Abstractions

Introduction

Data-Centric (DaCe) is a Python framework that transforms high-level code into a Stateful DataFlow multiGraph (SDFG) IR for performance optimization ￼ ￼. In the SDFG IR, computations are represented as nodes and data movement as edges. Among these nodes, Library Nodes play a crucial role: they represent high-level operator abstractions (such as matrix multiplication, transposition, etc.) as single nodes in the graph ￼. Instead of immediately specifying low-level implementation (loops, kernels, etc.), a library node encapsulates the semantics of an operation, allowing DaCe to delay the choice of implementation. This tutorial will explain how Library Nodes are defined and used in DaCe (including their semantic contracts via @dace.library.node), what compile-time information (“facts”) they carry, and how expansions (concrete implementations) are registered and applied. We will start with the example of matrix multiplication (already provided by DaCe’s BLAS library), and then create a new Library Node for matrix inversion with two expansion variants (Gaussian elimination and an iterative method akin to conjugate gradient).

Prerequisites: We assume you have DaCe installed and can import it in Python. This tutorial is aimed at a beginner level, with detailed explanations alongside code. We focus on CPU implementations for simplicity (no GPUs/FPGA specifics). Let’s begin by discussing what Library Nodes are in DaCe’s IR.

Library Nodes in DaCe IR (Operator Abstractions)

In DaCe, a Library Node is an abstract representation of an operation (e.g., matrix multiplication) in the SDFG. Instead of expanding the operation into lower-level dataflow immediately, the Library Node stands in as a single node with well-defined inputs and outputs ￼. This abstraction serves two main purposes:
• **Deferred Implementation (Expansion)**: The Library Node can later be expanded into a concrete implementation (which could be a native dataflow or a call to an external library) ￼. This allows DaCe to leverage highly-optimized libraries (like BLAS, cuBLAS, MKL, etc.) or fallback to a default implementation if those are unavailable ￼.
• **Semantic Transformations**: Because the Library Node encodes a high-level operation, DaCe can apply transformations using its semantics. For example, a transformation might fuse a matrix-multiplication node with a subsequent transposition by recognizing that a certain implementation can output transposed data directly ￼. Such optimizations are easier at the abstract level than on low-level code.
• **Performance Portability**: Same algorithm can target different hardware by switching implementations

Library Nodes serve as the semantic contract for an operation, defining:
1. Input/output connectors
2. Compile-time properties and parameters 
3. Available implementations

#### How to Define a Library Node

Library Nodes are created by decorating a class with `@dace.library.node` (establishes the semantic contracts) (or `@dace.library.register_node`) on the class. For example, the DaCe library defines an Einstein Summation node as follows:
```python
from dace import library, nodes

@library.node
class Einsum(nodes.LibraryNode):
    # Available implementations (expansions) for this node:
    implementations = { ... }
    # Default implementation name:
    default_implementation = 'specialize'
    # Configurable properties for Einsum would be defined here (omitted for brevity)
    ...
```

As shown above, the class sets an implementations dictionary (mapping string keys to expansion classes or functions) and a default_implementation ￼. The decorator @library.node ensures DaCe registers this class as a library node type. Under the hood, this associates the node with a library (which could be a group like “blas”, “lapack”, “mpi”, etc., depending on where you place the class or how you register it) ￼. For instance, DaCe’s built-in BLAS library nodes (like matrix multiplication) are defined in the dace.libraries.blas module and automatically registered when that module is imported ￼ ￼.

Inputs, Outputs, and Properties: In the library node class, you typically define the expected input and output connectors. This can be done by calling the base class constructor with inputs and outputs sets, or by overriding methods. For example, a matrix multiplication node might have connectors A, B (inputs) and C (output). The library node’s semantic contract might enforce that the shapes of A and B are compatible (inner dimensions match) and that C’s shape is the matrix product shape. These constraints can be checked either in the node’s constructor or via a validate method. At compile-time, once the SDFG is constructed, the library node carries the following known facts about the operation:
•	Data Types and Shapes: The data type of each input/output connector (e.g., float32) and their shapes (possibly symbolic). For instance, a MatMul node might know that if A is of shape (M,K) and B is (K,N), then output C is (M,N). This shape inference can be done during SDFG construction or via memlet propagation in DaCe ￼ ￼.
•	Operation-specific Properties: Any compile-time constants or flags for the operation. For example, a GEMM (general matrix multiply) might have boolean properties like transA/transB to indicate if inputs are transposed, or scalar coefficients alpha and beta (as in C = alpha*A@B + beta*C). These would be defined as class properties (using dace.properties.Property) and set at node creation. Such properties are accessible at compile-time for use in expansions or transformations.
•	Available Implementations: The node class knows which implementations (expansions) are available (from the implementations dict) and which one is the default. Initially, when a Library Node is created in an SDFG, it doesn’t yet choose a concrete implementation – it just records that, say, the default is 'pure' (a native expansion) or 'MKL' if configured otherwise. The actual expansion happens during compilation or when explicitly triggered.

In summary, a Library Node in the IR encapsulates the what of an operation (with known inputs/outputs and semantic parameters), deferring the how to expansions. This design allows DaCe to perform high-level reasoning and choose the best implementation later ￼.

Example: The Matrix Multiplication Library Node (MatMul)

Let’s illustrate these concepts using matrix multiplication. In Python, using the @ operator or numpy.dot in a DaCe program will create a Library Node for matrix multiplication in the SDFG ￼. The DaCe BLAS library provides a MatMul library node for this purpose. We will write a simple DaCe program and inspect its SDFG:
```python
import dace
import numpy as np

# Symbolic dimension (for generality)
N = dace.symbol('N')

# Define a DaCe program for matrix multiplication
@dace.program
def matmul_example(A: dace.float32[N, N], B: dace.float32[N, N]):
    return A @ B  # Using the @ operator triggers a MatMul library node

# Instantiate example data
A = np.random.rand(4, 4).astype(np.float32)
B = np.random.rand(4, 4).astype(np.float32)

# Get the SDFG (Stateful DataFlow Graph) representation
sdfg = matmul_example.to_sdfg()
sdfg.save('matmul_example.sdfg')  # Save to file (optional, for visualization)
print("SDFG states:", sdfg.states)
state = sdfg.states[0]
# Find the library node in the state
for node in state.nodes():
    print(type(node), ":", node.label, "| Implementation:", getattr(node, "implementation", None))
```
When you run this, you should see that the SDFG has one state, and within that state there is a node of type MatMul (a library node) with a label like “MatMul” or similar. The implementation property of the node will likely be None or set to "default" at this point – meaning no specific implementation has been chosen yet, so it will use the class’s default_implementation. By default, DaCe’s BLAS library nodes expand to a "pure" (native) implementation, which is a portable SDFG expansion of the operation ￼ ￼.

**What does the MatMul library node contain at this stage?** It contains connectors for inputs and output (connected to our arrays A, B, and the return), knows the data type (float32), and the symbolic shape N (which is 4 in our test). It is also aware of potential expansions. In fact, DaCe’s BLAS MatMul provides multiple implementations – e.g., 'pure' (a triple nested loop in DaCe), 'MKL' (Intel MKL BLAS call), 'cuBLAS' (NVIDIA cuBLAS call), etc. ￼. These are all listed in its implementations. The default_implementation is 'pure' (as configured in DaCe’s settings, to ensure compatibility) ￼. We will see next how we can select and apply these implementations.

### Expansions: Attaching Concrete Implementations to Library Nodes

**Expansion** is the process of replacing a Library Node with a concrete subgraph or code that performs the operation. In DaCe, each library node type can have one or more expansions available. As noted, this could mean converting the node into an equivalent SDFG subgraph (“native” expansion) or into an external library call. Choosing an expansion allows us to specialize the abstract operation for a target platform or performance preference ￼.

Expansions in DaCe are implemented as subclasses of transformation.ExpandTransformation. Each expansion class typically corresponds to one way of implementing the library node. DaCe uses a registration mechanism for expansions as well:
    - Use the decorator @dace.library.register_expansion(NodeType, 'name') on an ExpandTransformation subclass to tie it to a particular library node class ￼. The 'name' should match a key in the node’s implementations dict.
    - The expansion class defines an expansion method (often as a @staticmethod) that takes the library node and parent SDFG/state, and returns an SDFG (or subgraph) implementing the operation ￼. DaCe will insert this returned SDFG in place of the library node during expansion.

For example, referring again to the Einsum example in DaCe’s docs: the library node Einsum had a default implementation 'specialize'. The expansion was defined and registered like so:
```python
@library.register_expansion(Einsum, 'specialize')
class SpecializeEinsum(xf.ExpandTransformation):
    environments = []  # (Any required environment, e.g., MKL, can be listed here)

    @staticmethod
    def expansion(node: Einsum, parent_state: SDFGState, parent_sdfg: SDFG) -> SDFG:
        # Construct an SDFG that implements the Einstein summation
        sdfg = dace.SDFG('einsum_expansion')
        # ... build the SDFG (states, transients, tasklets) ...
        return sdfg
```

When DaCe needs to expand an `Einsum` node with implementation 'specialize', it will call this expansion method, get an SDFG, and plug it into the parent graph ￼. The inputs and outputs of the library node will be connected to the corresponding connectors of the nested SDFG.

**Selecting Implementations:** By default, if we do nothing, the library node will expand using its default_implementation. We can override this in two ways:
•	Globally: For an entire library. For instance, we can tell BLAS library nodes to use MKL by default if available:

from dace.libraries import blas
print("Default BLAS implementation:", blas.default_implementation)  # likely "pure" initially [oai_citation:23‡spcldace.readthedocs.io](https://spcldace.readthedocs.io/en/v0.16.1/optimization/blas.html#:~:text=from%20dace)
if blas.IntelMKL.is_installed():
    blas.default_implementation = "MKL"

This sets the global default for all BLAS ops (like MatMul) to MKL, if MKL is installed ￼. Similarly, blas.default_implementation = 'cuBLAS' could be used for NVIDIA GPUs, etc. DaCe provides is_installed() checks for each library environment ￼.

•	Per Node: You can also set the implementation property on a specific node in the SDFG before compilation. For example:

# Suppose `matmul_node` is our MatMul library node instance in the SDFG
matmul_node.implementation = "MKL"

This forces that one node to expand with MKL (assuming the environment is available), regardless of the global default.

When does expansion happen? Expansion can be triggered manually by calling sdfg.expand_library_nodes(), or it will happen automatically during SDFG compilation (just before code generation). Until expansion, the SDFG retains the high-level library nodes (which is useful for performing transformations or analyses).

Example: Using Different Expansions for MatMul

Continuing our matrix multiplication example, let’s manually expand the library node in two different ways to see the effect:
```python
# Continue from previous matmul_example
sdfg1 = matmul_example.to_sdfg()
# 1. Use default (pure) expansion
sdfg1.expand_library_nodes()
print("Expanded (pure) SDFG has states:", sdfg1.number_of_states())
print("Nodes in expanded state:", [type(n).__name__ for n in sdfg1.states[0].nodes()])

# 2. Use MKL expansion (if available)
sdfg2 = matmul_example.to_sdfg()
# Set the implementation on the MatMul node
for node in sdfg2.nodes()[0]:
    if isinstance(node, dace.sdfg.nodes.LibraryNode) and node.label.startswith("MatMul"):
        node.implementation = "MKL"
sdfg2.expand_library_nodes()
```
In the first case, expanding with "pure" will turn the MatMul node into a nested SDFG (or a set of maps and tasklets) that perform the matrix multiplication with loops. You would see that the expanded state now contains low-level nodes (maps, accesses, tasklets) instead of the single MatMul node. In the second case, expanding with "MKL" will likely result in a single nested SDFG or tasklet that calls the MKL library (the details are handled by DaCe’s MKL expansion). The structure of the expanded graph is different, but functionally both achieve the same result.

Key point: The ability to switch expansions easily gives performance portability. You can try different implementations (pure Python vs. MKL vs. cuBLAS) by simply toggling a string, without changing your algorithm code. DaCe will also ensure that if an expansion requires an environment (like MKL), it is available; otherwise, you’d get a warning or it would fall back to pure implementation ￼.

Now that we understand library nodes and expansions with a built-in example, let’s create our own operator abstraction: a Matrix Inversion node with multiple expansions. This will demonstrate how to use @dace.library.node and @dace.library.register_expansion in practice.

### Creating a Custom Library Node: Matrix Inversion

Suppose we want to add support for matrix inversion in DaCe’s IR. DaCe does not (as of writing) have a built-in Inv node, but we can create one. We will define a library node MatInv that takes one input matrix (assume square N×N) and produces one output matrix (the inverse). We’ll then implement two expansions:
1.	Gaussian Elimination (GE): A direct method using Gauss-Jordan elimination to compute the inverse. This will serve as a generic CPU implementation.
2.	Iterative Method (Conjugate Gradient): An alternative approach that solves for the inverse by solving linear systems using Conjugate Gradient. This method is applicable to symmetric positive-definite matrices and is more of an illustration of an iterative expansion.

By the end, we’ll see how to register these expansions and use the new MatInv node in an SDFG.

Defining the MatInv Library Node

First, we define the library node class and register it. We use @dace.library.node to register the node (here we won’t tie it to an existing library name, but we could specify a library name if we want to group it – DaCe can auto-group by module). We inherit from dace.sdfg.nodes.LibraryNode. In the constructor, we specify one input connector (let’s call it "A") and one output connector ("Inv" for the inverse). We also set up the implementations dict with placeholders (we will fill these after defining the expansion classes).

```python
import dace
from dace import library, sdfg as sd, dtypes

@library.node
class MatInv(sd.nodes.LibraryNode):
    # Register available implementations (will populate after classes are defined)
    implementations = {}
    default_implementation = None  # we'll set a default later

    def __init__(self, name=None):
        # name: optional label for the node
        super().__init__(name or "MatInv", inputs={"A"}, outputs={"Inv"})
```

A few things to note in this definition:
•	We leave implementations empty for now; we will update this dictionary after we define the expansion classes (DaCe allows adding to it dynamically).
•	We set default_implementation = None initially. We’ll set it to "gaussian" once that expansion is defined.
•	In the constructor, inputs={"A"} and outputs={"Inv"} define the connector names. We assume the matrix to invert will connect to "A", and the result will be given via "Inv". We could have named these differently, but these are descriptive enough.
•	We did not define any extra properties (like matrix size) because the size can be deduced from the input connector’s shape at compile-time. If we wanted to restrict, say, to SPD matrices, we could add a boolean property spd_only or similar, but we’ll keep it general.

With the node class defined, we can proceed to the expansions. Before writing the expansion classes, it’s helpful to implement the algorithms in pure Python (or as DaCe programs) to have a reference. We will do that and then integrate them.

Expansion 1: Gaussian Elimination (Direct Inverse)

Our first expansion will perform Gaussian elimination (Gauss-Jordan method) to compute the inverse. The algorithm for an N×N matrix A is to augment A with the identity matrix and perform row reduction until A becomes identity and the other half becomes A^{-1}. We can implement this with nested loops.

For clarity, let’s first write a DaCe program that does this, then use it in the expansion:
```python
N = dace.symbol('N')  # symbolic size

@dace.program
def invert_gaussian(A: dace.float64[N, N]):  # using double precision for stability
    # Make local copies to avoid modifying input
    A_work = np.copy(A)         # Working copy of A (will be transformed to identity)
    Inv = np.eye(N, dtype=A.dtype)  # Start with Inv as identity matrix
    # Gauss-Jordan elimination
    for i in range(N):
        # Divide i-th row by the pivot A_work[i, i]
        pivot = A_work[i, i]
        for j in range(N):
            A_work[i, j] = A_work[i, j] / pivot
            Inv[i, j]    = Inv[i, j] / pivot
        # Eliminate column i in all other rows
        for k in range(N):
            if k != i:
                factor = A_work[k, i]
                for j in range(N):
                    A_work[k, j] = A_work[k, j] - factor * A_work[i, j]
                    Inv[k, j]    = Inv[k, j] - factor * Inv[i, j]
    return Inv
```
This program uses plain NumPy operations and Python loops, which DaCe will convert into an SDFG (with maps for the loops, etc.). We assume the matrix is invertible (non-zero pivots on diagonal; in a robust implementation we would add partial pivoting to swap rows if needed, but for simplicity we skip that). The result Inv will contain the inverse of the original matrix A.

Now we’ll incorporate this into an expansion. Instead of writing the loop logic by hand as an SDFG, we can leverage this program. DaCe allows us to call invert_gaussian.to_sdfg() to get an SDFG for the expansion. We can then hook that into the parent.

Let’s write the expansion class and register it:
```python
from dace.transformation.transformation import ExpandTransformation

@library.register_expansion(MatInv, 'gaussian')
class ExpandMatInvGaussian(ExpandTransformation):
    environments = []  # no special environment needed (pure Python implementation)

    @staticmethod
    def expansion(node: MatInv, parent_state: sd.SDFGState, parent_sdfg: sd.SDFG):
        # Create an SDFG for the inversion using our DaCe program
        inv_sdfg = invert_gaussian.to_sdfg()  # convert the above program to an SDFG
        inv_sdfg.name = f"expand_matinv_gaussian_{node.label}"  # name it for clarity

        # The returned SDFG will expect an input 'A' and produce a result (likely as __return or an array named 'Inv').
        # We need to connect it to the parent graph. We rely on connector names:
        # The MatInv node has input connector "A" and output connector "Inv".
        # We'll assume invert_gaussian SDFG returns an array named "__return".
        # We can rename that to "Inv" to match the connector.
        if "__return" in inv_sdfg.arrays:
            inv_sdfg.arrays["Inv"] = inv_sdfg.arrays.pop("__return")
            # Also update references in the SDFG (this is a bit low-level; alternatively, ensure program returns an array named Inv)
            for st in inv_sdfg.states():
                for n in st.nodes():
                    if isinstance(n, sd.nodes.AccessNode) and n.data == "__return":
                        n.data = "Inv"
        return inv_sdfg
```
A few details in this expansion:
•	We decorated the class with @library.register_expansion(MatInv, 'gaussian') to tie it to our MatInv node under the name “gaussian”.
•	It subclasses ExpandTransformation (DaCe’s base for expansions).
•	In the expansion method, we obtain an SDFG from our invert_gaussian program. This SDFG has one state that implements the loops. The input to that SDFG is our matrix A and its output is the returned Inv. By default, DaCe names the return value container as __return. We do a small fix: rename __return to "Inv" in the SDFG to match our library node’s output connector. This ensures that when DaCe connects the parent graph to this nested SDFG, it knows to map the Library Node’s "Inv" output to the nested SDFG’s "Inv" array. (We iterate through the SDFG’s states to also rename any AccessNode that was using __return).
•	We then return this inv_sdfg. DaCe will handle replacing the MatInv node with a Nested SDFG node using inv_sdfg and reconnecting edges appropriately.

Now we have our first implementation. We should add it to the MatInv.implementations dict and set the default:
```python
MatInv.implementations['gaussian'] = ExpandMatInvGaussian
MatInv.default_implementation = 'gaussian'
```
This updates our MatInv class to know about the “gaussian” expansion and to use it by default.

### Expansion 2: Iterative Method (Conjugate Gradient)

For the second expansion, we will illustrate using an iterative solver. The idea is to compute the inverse by solving $A \cdot X = I$ for $X$, one column at a time. If $A$ is symmetric positive-definite (SPD), we can use the Conjugate Gradient (CG) algorithm to solve $A x = b$ for each unit vector $b$ (each column of the identity) and assemble the inverse matrix $X$. This is not the most efficient way to invert a matrix in general, but it demonstrates an alternative approach (and could be useful if $A$ is sparse or if we don’t want to do a direct O($N^3$) factorization).

We’ll implement a CG solver inside an expansion. As before, let’s write a DaCe program for clarity:
```python
@dace.program
def invert_cg(A: dace.float64[N, N], maxiter: dace.int32):
    Inv = np.zeros((N, N), dtype=A.dtype)
    # Solve A * x = e_i for each column i
    for i in range(N):
        # Initialize for CG
        b = np.zeros((N,), dtype=A.dtype)
        b[i] = 1  # unit vector (column of identity)
        x = np.zeros((N,), dtype=A.dtype)
        r = np.copy(b)               # initial residual = b (since x=0)
        p = np.copy(r)               # initial direction
        rr = np.dot(r, r)            # r^T r
        for k in range(maxiter):
            Ap = A @ p
            alpha = rr / np.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            new_rr = np.dot(r, r)
            if new_rr < 1e-12:
                break  # converged
            p = r + (new_rr/rr) * p
            rr = new_rr
        # store solution x as column i of Inv
        for j in range(N):
            Inv[j, i] = x[j]
    return Inv
```
A few notes: we introduced a parameter maxiter to limit CG iterations (in worst case CG takes N iterations for an N×N SPD matrix, so we can set maxiter=N). We break early if the residual is very small. This nested loop setup will become a fairly large SDFG (with nested loops inside a loop over columns), but it’s fine for demonstration.

Now, we create an expansion class for this:

```python
@library.register_expansion(MatInv, 'iterative')
class ExpandMatInvIterative(ExpandTransformation):
    environments = []  # no special environment, using CPU

    @staticmethod
    def expansion(node: MatInv, parent_state: sd.SDFGState, parent_sdfg: sd.SDFG):
        # Determine a reasonable max iterations (could use N or a property)
        # If the matrix size N is known as a constant we use that, otherwise use a symbol or a default.
        N_val = None
        for e in parent_state.in_edges(node):
            if e.dst_conn == 'A':
                arr = parent_sdfg.arrays[e.src.data]
                if isinstance(arr.shape[0], int):
                    N_val = arr.shape[0]
        maxiter = N_val if N_val is not None else 1000  # e.g., default 1000 if N not constant
        inv_sdfg = invert_cg.to_sdfg()  # SDFG from the CG program
        inv_sdfg.name = f"expand_matinv_iterative_{node.label}"
        # Set the maxiter constant
        inv_sdfg.add_constant('maxiter', maxiter)
        # As before, handle output naming:
        if "__return" in inv_sdfg.arrays:
            inv_sdfg.arrays["Inv"] = inv_sdfg.arrays.pop("__return")
            for st in inv_sdfg.states():
                for n in st.nodes():
                    if isinstance(n, sd.nodes.AccessNode) and n.data == "__return":
                        n.data = "Inv"
        return inv_sdfg
```
Here we did something slightly different:
•	We attempted to determine N_val, the numeric size of the matrix, by looking at the input descriptor in the parent SDFG. If available as an integer (maybe the SDFG was specialized with a constant size), we use it; otherwise we just set a default maxiter (e.g., 1000) or we could use the symbolic N. In DaCe, we could also pass the symbol N as a parameter, but for simplicity, we use a constant or a high default.
•	We call invert_cg.to_sdfg() to get the SDFG, then use add_constant('maxiter', maxiter) to set the iteration count (binding the maxiter symbol in the SDFG to a value).
•	We rename the output __return to "Inv" similar to before.
•	We return the SDFG.

Finally, we register this expansion in the MatInv class:
```python
MatInv.implementations['iterative'] = ExpandMatInvIterative
# (We already set default to 'gaussian'; we could allow user to switch to iterative as needed)
```
Now our custom Library Node is fully defined with two implementations. We can use it in an SDFG.

Using the MatInv Library Node in an SDFG

To use our new operator, we have a couple of options:
•	Integrate with Python front-end via replacements: We could write a Python function replacement so that calls to numpy.linalg.inv or a custom matinv function produce our MatInv node. This involves DaCe’s frontend (AST parsing) and is beyond our current scope (but for reference, replacements are done via dace.frontend utilities ￼).
•	Manual SDFG construction: We can directly construct an SDFG with a MatInv node, which is what we’ll do here for demonstration.

Let’s manually create an SDFG that inverts a matrix using our Library Node:
```python
import numpy as np

# Create a sample matrix to invert
N_val = 4
A_val = np.random.randn(N_val, N_val).astype(np.float64)
# Ensure it's invertible (e.g., add identity * 0.5 to avoid singular matrix)
A_val += 0.5 * np.eye(N_val)

# Construct SDFG
sdfg = dace.SDFG('invert_matrix')
state = sdfg.add_state()

# Add data descriptors
sdfg.add_array('A', shape=[N_val, N_val], dtype=dace.float64)
sdfg.add_array('Inv_out', shape=[N_val, N_val], dtype=dace.float64)

# Add access nodes for data
A_read = state.add_read('A')
Inv_write = state.add_write('Inv_out')

# Add our MatInv library node
inv_node = MatInv()  # our library node instance
state.add_node(inv_node)

# Connect input and output
state.add_edge(A_read, None, inv_node, 'A', dace.Memlet('A'))          # A -> node.A
state.add_edge(inv_node, 'Inv', Inv_write, None, dace.Memlet('Inv_out'))  # node.Inv -> Inv_out

# By default, this MatInv will use the 'gaussian' expansion.
# We can choose to use 'iterative' by setting:
# inv_node.implementation = 'iterative'

# Compile and run the SDFG
inv_func = sdfg.compile()
Inv_result = inv_func(A=A_val)
print("Input A:\n", A_val)
print("Inverted Inv_out:\n", Inv_result['Inv_out'])
print("Numpy.linalg.inv:\n", np.linalg.inv(A_val))
```
In the above:
•	We created an SDFG with one state.
•	We added an input array A and an output array Inv_out.
•	We added a MatInv node to the state and connected A to its input and Inv_out to its output.
•	We left the implementation as default 'gaussian'. You could uncomment the inv_node.implementation = 'iterative' line to test the iterative variant.
•	We compiled the SDFG and ran it with a random matrix (ensuring it’s invertible). The result is printed alongside NumPy’s inversion for comparison.

Verification: The printed output should show that Inv_out closely matches numpy.linalg.inv(A_val). The Gaussian elimination expansion should produce an exact (within floating-point error) inverse. The iterative expansion (if used) should also be very close, though it might have slight numerical differences depending on the convergence tolerance.

You have now created and used a custom Library Node in DaCe! The MatInv node acts as a high-level abstraction, and we provided two ways to realize it. This pattern can be followed to add all sorts of operations to DaCe’s IR:
•	Define a LibraryNode for the abstract operation (with connectors and properties describing its interface).
•	Write one or more implementations (either by hand-coding the SDFG or, as we did, writing a DaCe program and converting it to an SDFG).
•	Register those implementations with @library.register_expansion and assign them in the node class’s implementations dict.
•	Optionally, handle any integration in the front-end (so that a high-level API call maps to your node), or use it by constructing SDFGs manually.

Conclusion

In this tutorial, we delved into DaCe’s IR level to understand operator abstraction via Library Nodes and the operator expansion system. We saw that Library Nodes serve as semantic placeholders for operations (like matrix multiplication or inversion), carrying compile-time information such as input/output types, shapes, and available implementation variants. This separation of concerns – abstract what vs. concrete how – enables powerful optimization workflows. At compile time (before expansion), the SDFG knows the facts about the operation (e.g., in our MatInv node: the matrix size, dtype, and that two implementation options exist, etc.), but the implementation is chosen later, which could depend on the target hardware or libraries available.

We demonstrated how expansions are registered and used, from the built-in MatMul (with expansions like pure, MKL, cuBLAS) to a custom MatInv with a direct algorithm and an iterative algorithm. This extensibility is one of DaCe’s strengths – users can introduce new high-level operations and ensure they integrate with the transformation and optimization framework of DaCe ￼ ￼.

Further reading and notes:
•	If you plan to integrate your custom nodes with the Python frontend (e.g., hooking numpy.linalg.inv to produce MatInv), look into DaCe’s replacement mechanism, which uses AST transformations to swap function calls with library nodes ￼.
•	Complex operations and domain-specific libraries (e.g., FFTs, quantum operations, etc.) can be integrated similarly by providing library nodes and expansions. For instance, Google’s Qualtran project uses a concept of algorithmic building blocks (“bloqs”) to represent quantum operations, analogous to library nodes in DaCe ￼ – reinforcing how separating an operation’s abstract definition from its implementations is a powerful idea across domains.
•	Always ensure that your expansions are correct and, if necessary, handle corner cases (our Gaussian elimination lacked pivoting for simplicity – in a production scenario you’d want to add that or use an LU decomposition approach).
•	Use DaCe’s visualization (sdfg.view()) to inspect your SDFGs and ensure the expansions yield the expected graphs.

We hope this tutorial serves as a useful reference for implementing and understanding Library Nodes in DaCe. Happy optimizing!

Sources:
•	DaCe Documentation – Library Nodes and Expansions ￼ ￼ ￼
•	DaCe Documentation – BLAS Library Node Usage ￼ ￼ ￼
•	Research Paper (ETH Zurich) – Explanation of Library Nodes in DaCe ￼ ￼
•	GSoC 2023 Report – Using LibraryNodes for MPI in DaCe (Fu-Chiang Chang) ￼ ￼