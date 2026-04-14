# Introduction to Coordinate Graph

A coordinate graph represents *coordinate transforms* and *dataflow*.
The main purpose of this graph is to store these encodings in a structured manner
that are required to perform index calculations and manage storage.

## Coordinate Transform

A *coordinate transform* is essentially a mapping function that describes a logical
transformation from one coordinate space to another coordinate space.
For example, a simple coordinate transform representing a memory address space for
a 2D packed tensor via its dimensions can be defined by a mapping function,
*f : (x,y) -> m*, where *(x,y)* indicates the x and y coordinates of 2D tensor and
*m* indicates the memory address.

In this graph representation, each *coordinate transform* consists of nodes
(or *coordinates* or dimensions) and edges (or *transforms*), where the nodes
indicate coordinate spaces and the edges indicate how to transform the coordinates.

Each dimension has a *size* and a *stride* associated with it. The *size* represents
the number of elements in a dimension while the *stride* represents the number of
elements to skip in order to access the next element in that dimension.

### Flatten

Flatten is a transform that takes one or more input coordinates and outputs a single
"flattened" output coordinate. It always represents a one-to-one mapping function.
A simple coordinate transform representing a memory address space (output) for a 2D
packed tensor via its dimensions (inputs) is an example of a Flatten coordinate transform.

$$\begin{bmatrix}A & B & C \\\ D & E & F \\ \end{bmatrix}$$

For a given 2D packed tensor (stored in row-major order) with input dimensions
$I (size_i = N_i, stride_i = N_j)$ and $J (size_j = N_j, stride_j = 1)$,
the values of output dimension $K (size_k = N_i * N_j, stride_k = 1)$
can be calculated via Flatten as follows:

$$k = Flatten(i, j) = i * stride_i + j$$
, where $i \in I$, $j \in J$ and $k \in K$.

$$Memory = \begin{bmatrix} A & B & C & D & E & F \end{bmatrix}$$

For a 2 x 3 packed tensor ($N_i = 2, N_j = 3$) stored in row-major order,
an element (D) at {1, 0} coordinates (i = 1, j = 0),
the value of k (relative address of the element) will be {1 * 3 + 0} = {3}.

On the other hand, for a given 2D packed tensor (stored in column-major order) with input
dimensions $I (size_i =N_i, stride_i = 1)$ and $J (size_j = N_j, stride_j = N_i)$, the values
of output dimension $K (size_k = N_i \times N_j, stride_k = 1)$ can be calculated via Flatten as follows:

$$k = Flatten(i, j) = i + j * stride_j$$
, where $i \in I$, $j \in J$ and $k \in K$.

$$Memory = \begin{bmatrix}A & D & B & E & C & F \end{bmatrix} $$

For a 2 x 3 packed tensor ($N_i = 2, N_j = 3$) stored in column-major order,
an element (B) at {0,1} coordinates (i = 0, j = 1),
the value of k (relative address of the element) will be {0 + 1 * 2} = {2}.

Please note that the strides of a tensor describe the physical memory layout of its elements.

### Tile

Tile is an inverse transform of Flatten. It takes a single coordinate as an input, and
"tiles" it into one or more output coordinates. It always represents a one-to-one
mapping function.

Given a memory address space, input dimension $K (size_k = N_i \times N_j, stride_k = 1)$ of a
2D packed tensor (stored in row-major order), the values of output dimensions
$I (size_i = N_i, stride_i = N_j)$ and $J (size_j = N_j, stride_j = 1)$ can be calculated via
Tile as follows:

$${i, j} = Tile(k) = {\lfloor\frac{k}{size_j}\rfloor, k \bmod size_j}$$
, where $k \in K$, $i \in I$ and $j \in J$.

For a 2 x 3 packed tensor ($N_i = 2, N_j = 3$), an element at 4th position (k = 3),
the value of {i, j} will be {3 / 3, 3 % 3} = {1, 0}.

### Join

Join is a transform that takes one or more input coordinates and outputs a single
"joined" output coordinate. This transform can represent both one-to-one and
many-to-one mapping functions. Please note that Flatten is a special case of Join.

For a given 2D tensor with input dimensions $I (size_i = N_i, stride_j = S_i)$ and
$J (size_j = N_j, stride_j = S_j)$, the values of output dimension K can be calculated
via Join as follows:

$$k = Join(i, j) = i * stride_i + j * stride_j$$
, where $i \in I$, $j \in J$ and $k \in K$.

### Split

Split is an inverse transform of Join. It takes a single coordinate as an input, and
"splits" it into one or more output coordinates.

In general, this transform can represent one-to-many mappings, so we cannot always
describe it as a function to generate the values of output dimensions I
($size_i = N_i, stride_i = S_i$) and J ($size_j = N_j, stride_j = S_j$),
given a memory address space (input dimension K ($size_k = N_k, stride_k = S_k$))
of a 2D tensor.

Please note that Tile is a special case of Split.

### Sunder

Sunder is a transform takes a single input coordinate, and "sunders" it based on a switch
to generate a single output coordinate at a time.

Given a switch dimension $S (size_s = 2, stride_s = 1)$ and the number of Kloop iterations in
the Stream-K portion and the Data-Parallel portion, represented by output dimensions
$I (size_i = N_i, stride_i = 1)$ and $J (size_j = N_j, stride_j = 1)$,
the values of input dimension (representing the total number of Kloop iterations)
$F (size_f = N_i + N_j, stride_f = 1)$ can be calculated via the inverse of Sunder as follows:

$$f(i,j,s)=\begin{cases}i & \quad \text{s = 0}\\\ j + N_i & \quad \text{s = 1}\end{cases}$$
