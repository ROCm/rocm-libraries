.. meta::
   :description: rocRAND compatibility with cuRAND
   :keywords: rocRAND, ROCm, API, documentation, cuRAND

.. _curand-compatibility:

====================
cuRAND compatibility
====================

The following table shows which rocRAND generators produce the exact same sequence as the equivalent NVIDIA CUDA cuRAND generator when using legacy ordering, given the same seed, number of dimensions, and offset.
cuRAND is a closed-source library, which makes achieving compatibility difficult. Even when the same underlying algorithm is used, implementation details can differ for example, how the initial state is initialized, the order in which threads generate and write output values, and how the seed is interpreted or applied. However, despite these differences, the statistical properties and randomness characteristics remain equivalent.

.. table:: cuRAND compatibility
    :widths: auto

    =================  =====================
    Generator          Compatible
    =================  =====================
    XORWOW             No
    MRG32K3A           No
    MTGP32             No
    Philox 32x4-10     No
    MT19937            No
    Sobol32            Yes
    Scrambled Sobol32  No
    Sobol64            Yes
    Scrambled Sobol64  No
    =================  =====================