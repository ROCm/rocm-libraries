.. meta::
  :description: rocPRIM scope 
  :keywords: rocPRIM, ROCm, API, documentation, scope

********************************************************************
rocPRIM operation scope
********************************************************************

The scope of a rocPRIM operation determines the parts of the GPU that will cooperate to compute the result. 

The scope has a direct influence on how the data will be subdivided and processed by the computing units (CUs) and vector arithmetic logic units (VALUs). 

A device operation runs at the grid level. Both the operation and the data are broken down into function calls that are dispatched to the CUs. Synchronization at the grid level is done through wait lists and queue barriers.

Each block is made up of warps which are groups of threads. The function calls in the blocks are distributed over warps. Each warp computes an operation. All the warps on the same VALU run the same operation. 

Operations then run sequentially in the threads within the warps. 

Synchronization at the block, warp, and thread level is done through memory barriers.
