.. meta::
   :description: rocRAND compatibility with cuRAND
   :keywords: rocRAND, ROCm, API, documentation, cuRAND

.. _curand-compatibility:

====================
cuRAND compatibility
====================

The following table shows which rocRAND generators produce the exact same sequence as the equivalent NVIDIA CUDA cuRAND generator when using legacy ordering, given the same seed, number of dimensions, and offset.

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
    Sobol64            Yes, with same offset
    Scrambled Sobol64  No
    =================  =====================