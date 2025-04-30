.. meta::
  :description: rocFFT documentation and API reference library
  :keywords: rocFFT, ROCm, API, documentation

.. _buffer_assignment:

********************************************************************
Buffer assignment design document for rocFFT
********************************************************************

Summary
=======

Buffer assignment in rocFFT is the work of coordinating the input
and output buffers of each step in a rocFFT plan.

Observations
============

Some observations can be made about the FFT planning and buffer
assignment process:

1. Buffer assignment begins after the plan structure is decided.
   This means all of the node schemes, as well as input lengths and
   input/output strides are known.

   Note that at this time, output lengths are not directly known.

2. The first child of any non-leaf node must read from its parent's
   input, and the last child must write to its parent's output.

3. The input of any other child node must be the same as the output
   of its preceding sibling.

   There is one exception: a CS_KERNEL_CHIRP node is only used to
   populate the Bluestein temporary buffer, and can essentially be
   ignored during buffer processing as it does not actually read
   input.

4. The top-level node in the tree must read from the user-defined
   input buffer, and write to the user-defined output buffer.  These
   buffers will be the same for in-place transforms.

5. When deciding buffer assignments for a node, only the output
   buffer for nodes besides the last requires actual decision-making.
   The input buffer follows either from the top-level input, a
   preceding sibling, or a parent node's input.  The last node's
   output must be the user-defined output.

6. During buffer assignment, some number of buffers are available.
   At minimum, we have the user input and output buffers (which may
   be the same for in-place transforms) whose sizes are defined by
   the user.

   Zero or more temporary buffers may be needed, whose sizes are
   dynamic and can be as big as necessary for the transform to
   succeed.

7. Some choices of output buffers are clearly invalid.  For example:

   * Transpose nodes must always be out-of-place (i.e. input buffer
     cannot equal output buffer).

   * Some internal kernels only support interleaved formats as their
     input or output.  For example, the input of a copy-kernel (like
     COPY_CMPLX_TO_R or COPY_CMPLX_TO_HERM) must be interleaved.

   * Internal temp buffers are allocated contiguously, so they can be
     used on both planar and interleaved formats.  This is not always
     true for user-provided buffers.  An obvious example of this
     planar data: users typically create these using two buffers.

   * A node cannot write data to a buffer if the buffer is too small
     for that data.  This really only applies to user input/output
     buffers, as temp buffers are always made large enough.

Solution
========

We implement a decision function that determines whether a buffer
assignment is valid based on the observations above.

Buffer assignment should do an exhaustive search through the space of
possible buffer assignments for the tree, calling the decision
function for each potential choice.  If we arrive at the end of the
tree and all assignments are valid, then the buffer assignment
operation is complete.

Returning the first valid buffer assignment found is a simple
solution.  However, not all valid buffer assignments are equal in
terms of memory usage and/or performance: some buffer assignments
allow more kernel fusions and/or use more in-place kernels.  This
implies that we should keep all valid assignment candidates in a list
and subsequently return the "best" one.

The first pass of buffer assignment shall attempt to assign buffers
starting with just the user input buffer (and output buffer, if
distinct) and a Bluestein temp buffer (if the plan requires it), but
without any other temp buffers.

If that first pass is not successful, we retry with one temp buffer
added to the list of available buffers.  If that is still not
successful, we retry with a second temp buffer.

Implementation
==============

A structure storing a try
-------------------------

We store our current assignment try in a tree-like structure.  We
don't assign to the tree-node directly since there could be many valid
assignment paths for one plan.  Once we determine the best assignment
path from this tree, we fill the assignment back to the real
tree-node.

.. code-block:: cpp

  struct PlacementTrace
  {
      TreeNode* referedNode;
      Buffer inBuffer, outBuffer;
      ArrayType inArrayType, outArrayType;

      // each branch stands for a possible try on the next node
      vector<PlacementTrace*> branches;

      // a parent pointer to backtracing
      PlacementTrace* parent;

      // propagate these values from the root to leaves
      size_t numFusedNodes;
      size_t numInplace;
      size_t numTypwSwithching;
  }


Exhaustive search
-----------------

All possible assignments on each node are attempted.  There are
several limitations on each node that allow us to reject many illegal
assignments and prevent the assignment tree from growing
exponentially.  For example, SBRC and transpose kernels can only be
done using out-of-place buffers.

The exhaustive search is implemented in pseudocode like:

.. code-block:: cpp

   // ------------------------------------------------------------------------------------
   // Recursive function enumerates all the possible assignments
   // Returns a subtree, starting from execSeq[curSeqID], with input startBuf & startAType
   // ------------------------------------------------------------------------------------
   Function: void Enumerate(PlacementTrace* parent, ExecPlan, curSeqID, startBuf, startAType)
   // for terminal condition:
   - if curSeqID is the last nodes
     - if the end buffer and array-type fit the root-plan setting
       - calculate the number of eligible kernel-fusions.
       - add this candidate to the winnerCandidates list.
       - finish this path, return

   // not terminal condition:
   // add a single assignment on current node and append to parent's branches
   - if current node->isPlacementAllowed(inplace)
     // add a branch which uses inplace (if allowed) on this node and test validity
     - if ValidOutBuffer(execPlan, *curNode, startBuf, startType)
       - append an assignIP = PlacementTrace(curNode, startBuf, startBuf, startType, startType, parent)
       - call Enumerate(IPAssign, execPlan, curSeqID + 1, startBuf, startType);

   - if current node->isPlacementAllowed(out-of-place)
     // add branches which use out-of-place (if allowed) on this node and test validity
     - for each testOutputBuf in the availableBuffers set, (where testOutputBuf != startBuf)
       - if ValidOutBuffer(execPlan, *curNode, testOutputBuf, testOutType)
         - append an assignOP = PlacementTrace(curNode, startBuf, testOutputBuf, startType, testOutType, parent)
         - call Enumerate(OPAssign, execPlan, curSeqID + 1, testOutputBuf, testOutType);

   // --------------------------------------------------------
   // Decision maker: choose the best one from all candidates
   // This function is a sorting function, pick the first one
   // --------------------------------------------------------
   Function: void ValidPathExists(ExecPlan)
   - if winnerCandidates is empty, simply return false
   - using std::sort, sort by:
     // the one can fuse more kernels is better
     - lhs->numFusedNodes > rhs->numFusedNodes ?
     // if tie, compare inplace kernels, more is better
     - lhs->numInplace > rhs->numInplace ?
     // if tie, compare the times of switching-array-type, less is better
     - lhs->numTypeSwitching < rhs->numTypeSwitching ?

   - pick the first one, and do the Backtracking()
     - fill-in the assignment back to the real tree-nodes

   // ---------------------------------------------------------
   // Top-level function that assigns buffers on the root plan
   // ---------------------------------------------------------
   Function: void AssignBuffers(ExecPlan)
   - add rootPlan in/out buffer to availableBuffers set
     - Note: For C2C out-of-place, we can't add USER_IN to the set to prevent it from being modified.
   - add rootPlan in/out array-type to availableArrayTypes set
   - add OB_TEMP_BLUESTEIN to availableBuffers set, if plan uses Bluestein
   - initialize a winnerCandidates list to save all valid results.
   - initialize a dummyRoot of PlacementTrace as tree root, this dummyRoot pretends it's a parent of the first node (in execSeq).
     So dummyRoot.outBuf = rootPlan->obIn, and dummyRoot.oType = rootPlan->inArrayType

   // The 1st round try
   - call Enumerate(&dummyRoot, execPlan, 0, dummyRoot.outBuf, dummyRoot.oType)
     here 0 is curSeqID, which means starting from the first leafNode
   - call ValidPathExists() to pick the best solution
   - if successful, return

   // The 2nd round try
   - add OB_TEMP to availableBuffers
   - call Enumerate(&dummyRoot, execPlan, 0, dummyRoot.outBuf, dummyRoot.oType)
     here 0 is curSeqID, which means starting from the first leafNode
   - call ValidPathExists() to pick the best solution
   - if successful, return

   // The last round try
   - add OB_TEMP_CMPLX_FOR_REAL to availableBuffers
   - call Enumerate(&dummyRoot, execPlan, 0, dummyRoot.outBuf, dummyRoot.oType)
     here 0 is curSeqID, which means starting from the first leafNode
   - call ValidPathExists() to pick the best solution
   - if successful, return

   // Failed
   - if not found, throw exception.

Decision function and output lengths
------------------------------------

Much of the remaining complexity lies in the ValidOutBuffer()
decision function mentioned above.

Output lengths often differ from input lengths on a node.  For
example, R2C/C2R transforms change the data length from the input,
and transpose kernels swap dimension lengths between input and
output.

Tree nodes need to store their output length explicitly so that the
decision function does not need to guess at what lengths any node
will output.  This information is also helpful to log, so humans
reading the plan don't need to guess either.

As the exhaustive search proceeds, it likely needs to call the
decision function multiple times with identical inputs.  This is
because it might need to decide validity of two plans that might only
have tiny buffer assignment differences. The results of the function
are cached to reduce extra work during the search.

Fusions
-------

Kernel-fusion is essential for improving performance.  Unfortunately
fusion depends heavily on buffer assignment.  Two (or more) kernels
can be fused into one kernel only when the resulting buffer assignment
remains valid.

To maximize kernel fusion, we also implement a FuseShim framework. A
FuseShim class is a container/shell indicating that there is a
potentially-fusable kernel-fusion.  Each FuseShim defines its own
requirements to fulfill the fusion, including the expected buffer
assignment.

During the buffer assignment process, we can use the test function to
get the final number of the achievable kernel fusions.  This number
plays the most important role when making the final decision: we
always pick the one which can fuse the most kernels.

Padding
-------

We have cases where reading/writing along certain strides is bad for
performance (e.g. power-of-2).  While we are unable to adjust strides
for user-provided input and output buffers, we can potentially pad
temp buffers to avoid bad strides.

Once a plan candidate is constructed and buffers are assigned
(including any kernel fusion), a padding pass can adjust the output
strides of any node that writes to a temp buffer with bad strides.

The padding pass must also consider the input lengths and strides of
subsequent nodes that continue to use the same temp buffer, and
adjust them accordingly.  The writing and reading nodes might also
decompose the problem differently, so the logic needs to be aware
that a change to one dimension's stride on the write side may affect
multiple dimensions' strides on the reading side, and vice-versa.

Padding example
^^^^^^^^^^^^^^^

For example, consider this excerpt of a large plan:

.. code-block::

    scheme: CS_KERNEL_TRANSPOSE
    length: 4096 262144
    outputLength: 262144 4096
    iStrides: 1 4096
    oStrides: 1 262144
    OB_USER_OUT -> OB_TEMP

    scheme: CS_KERNEL_STOCKHAM_BLOCK_CC
    length: 512 512 4096
    outputLength: 512 512 4096
    iStrides: 512 1 262144
    oStrides: 512 1 262144
    OB_TEMP -> OB_TEMP

    scheme: CS_KERNEL_STOCKHAM_BLOCK_RC
    length: 512 512 4096
    outputLength: 512 512 4096
    iStrides: 1 512 262144
    oStrides: 1 512 262144
    OB_TEMP -> OB_USER_OUT


The first kernel writes 262144 elements on the fastest dimension, and
the higher dimension of 4096 elements is written along large
power-of-2 strides, making it a good candidate for padding.  The
following two kernels decompose the 262144 length to 512x512 along
their fastest dimensions.

Padded output of the first kernel needs to modify the following
strides using the same buffer, until the data leaves that temp
buffer:

.. code-block::

    scheme: CS_KERNEL_TRANSPOSE
    length: 4096 262144
    outputLength: 262144 4096
    iStrides: 1 4096
    oStrides: 1 262208
    OB_USER_OUT -> OB_TEMP

    scheme: CS_KERNEL_STOCKHAM_BLOCK_CC
    length: 512 512 4096
    outputLength: 512 512 4096
    iStrides: 512 1 262208
    oStrides: 512 1 262208
    OB_TEMP -> OB_TEMP

    scheme: CS_KERNEL_STOCKHAM_BLOCK_RC
    length: 512 512 4096
    outputLength: 512 512 4096
    iStrides: 1 512 262208
    oStrides: 1 512 262144
    OB_TEMP -> OB_USER_OUT

The second kernel is in-place, and would need iStrides == oStrides.
The padding pass would need to continue through the execution plan to
keep the third kernel's input strides consistent with the second's
output.  The output of the third kernel is a user buffer, so we
cannot change its padding.

When to pad
^^^^^^^^^^^

The exact criteria for when to add padding to a temp buffer (and how
much) are an implementation detail, but ad-hoc planning we've done in
the past has padded strides if higher dimension data longer than a
threshold is written along sufficiently large powers of two.

The decision logic around padding is centralized in one place in this
design, making it more feasible to have per-architecture decisions
around padding, should they become necessary.

Choosing a winner
-----------------

The exhaustive search is a depth-first-search that produces a list of
valid plans, each of which would produce correct results.  The list
is sorted to decide which option is best, and the best plan is
ultimately given to the user for execution.

The sort criteria are:

1. Number of fused kernels (more is better, to minimize kernel launches and global memory reads/writes)
2. Number of buffers used (fewer is better, to minimize temporary memory usage)
3. Number of padded reads/writes (more is better, to maximize use of padding once we've accepted the memory cost)
4. Number of in-place operations (more is better)
5. Number of type changes (e.g. planar -> interleaved, or vice-versa) in the plan (fewer is better, as a tiebreaker)

Future work
===========

Strides
-------

Currently, rocFFT does not guarantee that strides on user buffers are
respected if temporary data is written to those buffers.

With this implementation, it would be simpler to begin enforcing such
a guarantee.

Enforcing read-only input
-------------------------

rocFFT may currently overwrite user input buffers for out-of-place
real-transforms (not C2C-transform).  Although we have documented this
behavior, and it is common practice in other libraries, it might still
be unintuitive for some users.

If we ever wanted to start guaranteeing that user input is left
unmodified, this buffer assignment implementation would make that work
trivial - only the decision function needs to be made aware of this
policy change, and buffer assignment will work fine.

However, we may need to introduce yet another temp buffer, since we'd
be taking away a potential work space from existing plans.

Flexibility between minimizing memory or maximizing fusions
-----------------------------------------------------------

We can't always expect there is a perfect assignment that maximizes
kernel fusions while also minimizing temporary buffers.  In some
cases, these two goals are contradictory: if we choose an assignment
using minimal buffers, we may lose the opportunity to fuse more
kernels.  On the other hand, if we are allowed to use more memory, we
have more buffers available for out-of-place kernel-fusions.

With this implementation, it is possible to introduce an optimization
strategy option to users.

For example, if the memory usage is the main concern of users, we can
return the assignment with the least buffer usage.  Otherwise, we return
the result which maximizes the kernel fusions regardless of the memory
consumption.

Make C Buffer as Temp2 Buffer
-----------------------------

There is no reason to limit the "C" buffer to real-transforms only.
We can make the C buffer as another generic temporary buffer throughout;
this can also avoid any confusion about the purpose of C and T.

Copyright and disclaimer
------------------------

The information contained herein is for informational purposes only,
and is subject to change without notice. While every precaution has
been taken in the preparation of this document, it may contain
technical inaccuracies, omissions and typographical errors, and AMD is
under no obligation to update or otherwise correct this information.
Advanced Micro Devices, Inc. makes no representations or warranties
with respect to the accuracy or completeness of the contents of this
document, and assumes no liability of any kind, including the implied
warranties of non-infringement, merchantability or fitness for
particular purposes, with respect to the operation or use of AMD
hardware, software or other products described herein.  No license,
including implied or arising by estoppel, to any intellectual property
rights is granted by this document.  Terms and limitations applicable
to the purchase or use of AMD’s products are as set forth in a signed
agreement between the parties or in AMD's Standard Terms and
Conditions of Sale.

AMD is a trademark of Advanced Micro Devices, Inc. Other product names
used in this publication are for identification purposes only and may
be trademarks of their respective companies.

Copyright (C) 2021 - 2024 Advanced Micro Devices, Inc. All rights reserved.

