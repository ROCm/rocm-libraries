Workgroup Mapping
=================

Workgroup mapping parameters specify how rocRoller maps the GPU
workgroup number (hardware) to tile numbers (software).  Workgroup
mapping is done to increase cache efficiency.

Recall that the GPU workgroup number is available to the kernel
through a pre-populated SGPR (see `AssemblyKernel::workgroupIndex()`);
and that the GPU workgroup number exists in the rocRoller coordinate
transform graph as a `Workgroup` node.  During code-generation,
`Workgroup` nodes resolve to the pre-populated SGPR.

Workgroups (`Workgroup` nodes) are usually attached to leaf (dangling)
`MacroTileNumber` nodes in the coordinate graph.  This is done during
the `ConnectWorkgroups` graph transformation.  Workgroup mapping is
applied during the `ConnectWorkgroups` pass as well.

Simple workgroup mapping
------------------------

To illustrate workgroup-mapping concepts, consider a simple GEMM
kernel, where each workgroup computes all values in a single output
($D$) tile.

Consider a 6 by 8 tile space.  The kernel will be launched with $6
\cdot 8 = 48$ workgroups; where each workgroup will work on one tile.

We can visualize which workgroup will operate on the $(M,N)$ output
tile with a simple table.  The entry in row $M$ and column $N$
corresponds to the hardware workgroup number that will compute the
output values in tile $(M, N)$.

A very simple linear mapping could be::

        |  N
        |  0   1   2   3   4   5   6   7
    ----+-------------------------------
    M 0 |  0   6  12  18  24  30  36  42
      1 |  1   7  13  19  25  31  37  43
      2 |  2   8  14  20  26  32  38  44
      3 |  3   9  15  21  27  33  39  45
      4 |  4  10  16  22  28  34  40  46
      5 |  5  11  17  23  29  35  41  47

In this linear mapping, as the workgroup number increases, we increase
the $M$ tile number until a column is finished before moving onto the
next column.  In this sense, the $M$ dimension is parallel to the
workgroup.

To increase cache efficiency, we may want to do something like::

        |  N
        |  0   1   2   3   4   5   6   7
    ----+-------------------------------
    M 0 |  0   4   8  12  16  20  24  28
      1 |  1   5   9  13  17  21  25  29
      2 |  2   6  10  14  18  22  26  30
      3 |  3   7  11  15  19  23  27  31
      4 | 32  34  36  38  40  42  44  46
      5 | 33  35  37  39  41  43  45  47

In this mapping, as the workgroup number increases, we increase the
$M$ tile number, and move onto the next column every 4 steps.  Once
all columns are done, we return to the first column and continue.

To parameterize the workgroup mapping, we need to specify: the
parallel dimension (either $0$ for $M$ or $1$ for $N$); and the number
of tiles, denoted by $WGM$, that should be traversed in the parallel
dimension before moving on to the next row/column.

The parallel dimension of the workgroup mapping is baked into the
kernel and cannot be changed once the kernel is generated.  The number
of tiles $WGM$ becomes a kernel argument and can be tuned.


XCC remapping
-------------

For architectures that package CUs onto XCDs, another level of
workgroup mapping may be used to help cache efficiency.  This is the
XCC remapping.

For example, consider a system with 8 CUs per XCD.  By default,
workgroups are scheduled onto CUs in a round-robin manner with the XCD
number increasing fastest.  For GEMM workloads we may want flip this,
so that as our new workgroup number increases the CU number increases
fastest, and therefore neighbouring workgroups will land on the
same XCD.
