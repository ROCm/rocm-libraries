.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _glossary:

********************************************************************
Glossary of rocPRIM terms
********************************************************************

This glossary is to help users understand the basic concepts or terminologies used in the rocPRIM library.

.. glossary::

    Warp
        A group of threads that runs using the single instruction, multiple thread (SIMT) model. Also referred to as "wavefronts" on AMD GPUs.

    Hardware warp size
        The number of threads in a warp as defined by the hardware. On NVIDIA GPUs, the warp size is thirty-two (32) threads. On AMD GPUs, the warp size is sixty-four (64) threads.

    Logical warp size
        The number of threads in a warp as defined by the user. This can be equal to or less than the size of the hardware warp size.

    Block
        A group of warps that run on the same streaming multiprocessor (SM). Threads in the block can be indexed using one dimension, {X}, two dimensions, {X, Y}, or three dimensions, {X, Y, Z}. 
    
    Stride
        The number of threads per block.

    Tile
        The name for a block in C++ AMP and HIPCC.

    Grid
        A group of blocks that all run the same kernel call.

    Warp ID
        The identifier of the warp within a block. A warp's warp ID is guaranteed to be unique.

    Thread ID
        The identifier of the thread within a block.

    Lane ID
        The identifier of the thread within the warp. 

    Flat ID
        The flattened block or thread idex. The flat ID is a one-dimensional index created from two-dimensional or three-dimensional indices. For example the flat ID of a two-dimensional thread ID {X, Y} in a two-dimensional ``128x4`` block is ``Y*128*X``.
    
