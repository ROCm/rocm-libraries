.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _glossary:

********************************************************************
Glossary of rocPRIM terms
********************************************************************

This glossary is to help users understand the basic concepts or terminologies used in the rocPRIM library.

.. glossary::
    :sorted:

    Wavefront
        A group of threads that runs using the single instruction, multiple thread (SIMT) model. 
    
    Wave
        See :term:`wavefront<Wavefront>`. rocPRIM uses "warp", "wave", and "wavefront" interchangeably.

    Work-item
        A work-item is the smallest unit of parallel execution. A work-item runs a single independent instruction stream on a single data element. 

    Thread
        See :term:`work-item<Work-item>`. rocPrim uses "thread" and "work-item" interchangeably.

    Warp
        Alternate term for a :term:`wavefront<Wavefront>`. rocPRIM uses "warp", "wave", and "wavefront" interchangeably.

    Hardware warp size
        The number of threads in a warp as defined by the hardware. On AMD GPUs, the warp size can be either thirty-two (32) or sixty-four (64) threads.

    Logical warp size
        The number of threads in a warp as defined by the user. This can be equal to or less than the size of the hardware warp size.

    Block
        See :term:`tile<Tile>`. rocPRIM uses "block" and "tile" interchangeably.
    
    Stride
        The number of threads per block.

    Tile
        A group of warps that run on the same streaming multiprocessor (SM). Threads in the block can be indexed using one dimension, {X}, two dimensions, {X, Y}, or three dimensions, {X, Y, Z}. In rocPRIM the tile size is always the same as the block size. 

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
    
    SIMT 
        See :term:`Single-Instruction, Multi-Thread`.

    Single-Instruction, Multi-Thread 
        Single-instruction, multi-thread (SIMT) is a parallel computing model where all the :term:`work-items<work-item>` within a :term:`wavefront` run the same instruction on different data. 