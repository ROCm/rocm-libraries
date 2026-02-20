.. meta::
   :description: Data type support for rocRAND documentation
   :keywords: rocRAND, ROCm, API, documentation, cuRAND, data types

.. _data-type-support:

rocRAND data type support
******************************************

This topic discusses the various data types supported by rocRAND and provides a comparison
with the data type support in NVIDIA CUDA cuRAND.

Host API
========

This section covers the data types supported for the host API.

Generator types
---------------

.. list-table:: Supported generators on the host
    :header-rows: 1
    :name: host-supported-generators

    *
      - Generator
      - rocRAND support
      - cuRAND support
    *
      - XORWOW
      - ✅
      - ✅
    *
      - MRG32K3A
      - ✅
      - ✅
    *
      - MTGP32
      - ✅
      - ✅
    *
      - Philox 4x32-10
      - ✅
      - ✅
    *
      - MT19937
      - ✅
      - ✅
    *
      - MRG31K3P
      - ✅
      - ❌
    *
      - LFSR113
      - ✅
      - ❌
    *
      - ThreeFry 2x32-20
      - ✅
      - ❌
    *
      - ThreeFry 4x32-20
      - ✅
      - ❌
    *
      - ThreeFry 2x64-20
      - ✅
      - ❌
    *
      - ThreeFry 4x64-20
      - ✅
      - ❌
    *
      - Sobol32
      - ✅
      - ✅
    *
      - Scrambled Sobol32
      - ✅
      - ✅
    *
      - Sobol64
      - ✅
      - ✅
    *
      - Scrambled Sobol64
      - ✅
      - ✅

Only Sobol64, Scrambled Sobol64, ThreeFry 2x64-20, and ThreeFry 4x64-20 support the generation of 64-bit :code:`unsigned long long int` integers.
The other generators generate 32-bit :code:`unsigned int` integers.

Seed types
----------

All generators can be seeded with :code:`unsigned long long`. However, LFSR113 can also be seeded using a :code:`uint4`.

Output types
------------

The generators produce pseudo-random numbers chosen from a given distribution.
The following distributions and corresponding output types are supported for the host API:

Uniform distribution
^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Supported types for uniform distributions on the host
    :header-rows: 1
    :name: host-types-uniform-distribution

    *
      - Type
      - Size of type
      - rocRAND support
      - cuRAND support
    *
      - :code:`unsigned char`
      - 8 bit
      - ✅
      - ❌
    *
      - :code:`unsigned short`
      - 16 bit
      - ✅
      - ❌
    *
      - :code:`unsigned int`
      - 32 bit
      - ✅
      - ✅
    *
      - :code:`unsigned long long`
      - 64 bit (see note)
      - ✅
      - ✅
    *
      - :code:`half`
      - 16 bit
      - ✅
      - ❌
    *
      - :code:`float`
      - 32 bit
      - ✅
      - ✅
    *
      - :code:`double`
      - 64 bit
      - ✅
      - ✅

Uniform distributions of integral types return a number between 0 and 2^(size in bits) - 1,
whereas floating-point types return a number between 0.0 and 1.0, excluding 1.0.

.. note::

   The generation of 64-bit :code:`unsigned long long` integers is only supported by 64-bit generators
   (Scrambled Sobol 64, Sobol64, Threefry 2x64-20, and Threefry 4x64-20).

Poisson distribution
^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Supported types for the Poisson distribution on the host
    :header-rows: 1

    *
      - Type
      - Size of type
      - rocRAND support
      - cuRAND support
    *
      - :code:`unsigned int`
      - 32 bit
      - ✅
      - ✅

Normal distribution
^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Supported types for normal distributions on the host
    :header-rows: 1
    :name: host-types-normal-distribution

    *
      - Type
      - Size of type
      - rocRAND support
      - cuRAND support
    *
      - :code:`half`
      - 16 bit
      - ✅
      - ❌
    *
      - :code:`float`
      - 32 bit
      - ✅
      - ✅
    *
      - :code:`double`
      - 64 bit
      - ✅
      - ✅

Log-normal distributions
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Supported types for log-normal distributions on the host
    :header-rows: 1
    :name: host-types-log-normal-distribution

    *
      - Type
      - Size of type
      - rocRAND support
      - cuRAND support
    *
      - :code:`half`
      - 16 bit
      - ✅
      - ❌
    *
      - :code:`float`
      - 32 bit
      - ✅
      - ✅
    *
      - :code:`double`
      - 64 bit
      - ✅
      - ✅

Device API
==========

This section covers the supported data types for the device API.


Generator types
---------------

.. list-table:: Supported generators on the device
    :header-rows: 1
    :name: device-supported-generators

    *
      - Generator
      - rocRAND support
      - cuRAND support
    *
      - XORWOW
      - ✅
      - ✅
    *
      - MRG32K3A
      - ✅
      - ✅
    *
      - MTGP32
      - ✅
      - ✅
    *
      - Philox 4x32-10
      - ✅
      - ✅
    *
      - MT19937
      - ❌
      - ❌
    *
      - MRG31K3P
      - ✅
      - ❌
    *
      - LFSR113
      - ✅
      - ❌
    *
      - ThreeFry 2x32-20
      - ✅
      - ❌
    *
      - ThreeFry 4x32-20
      - ✅
      - ❌
    *
      - ThreeFry 2x64-20
      - ✅
      - ❌
    *
      - ThreeFry 4x64-20
      - ✅
      - ❌
    *
      - Sobol32
      - ✅
      - ✅
    *
      - Scrambled Sobol32
      - ✅
      - ✅
    *
      - Sobol64
      - ✅
      - ✅
    *
      - Scrambled Sobol64
      - ✅
      - ✅

Seed types
----------

All generators can be seeded with :code:`unsigned long long`, however LFSR113 can also be seeded using an :code:`uint4`.

Output types
------------

The generators produce pseudo-random numbers chosen from a given distribution.
The following distributions and corresponding output types are supported for the device API.
However, not all generators support all types.


Uniform distribution
^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Supported types for uniform distributions on the device
    :header-rows: 1
    :name: device-types-uniform-distribution

    *
      - Type
      - rocRAND support
      - Supported rocRAND generators
      - cuRAND support
    *
      - :code:`unsigned int`
      - ✅
      - All native 32-bit generators
      - ✅
    *
      - :code:`unsigned long long int`
      - ✅
      - All native 64-bit generators
      - ✅
    *
      - :code:`float`
      - ✅
      - All generators
      - ✅
    *
      - :code:`float2`
      - ✅
      - Philox 4x32-10
      - ❌
    *
      - :code:`float4`
      - ✅
      - Philox 4x32-10
      - ✅
    *
      - :code:`double`
      - ✅
      - All generators
      - ✅
    *
      - :code:`double2`
      - ✅
      - Philox 4x32-10
      - ✅
    *
      - :code:`double4`
      - ✅
      - Philox 4x32-10
      - ❌


Normal distribution
^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Supported types for normal distributions on the device
    :header-rows: 1
    :name: device-types-normal-distribution

    *
      - Type
      - rocRAND support
      - Supported rocRAND generators
      - cuRAND support
    *
      - :code:`float`
      - ✅
      - All generators
      - ✅
    *
      - :code:`float2`
      - ✅
      - Philox 4x32-10, MRG31K3P, MRG32K3A, XORWOW, LFSR113, all ThreeFry generators
      - ✅
    *
      - :code:`float4`
      - ✅
      - Philox 4x32-10
      - ✅
    *
      - :code:`double`
      - ✅
      - All generators
      - ✅
    *
      - :code:`double2`
      - ✅
      - Philox 4x32-10, MRG31K3P, MRG32K3A, XORWOW, LFSR113, all ThreeFry generators
      - ✅
    *
      - :code:`double4`
      - ✅
      - Philox 4x32-10
      - ❌

Log-normal distributions
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Supported types for log-normal distributions on the device
    :header-rows: 1
    :name: device-types-log-normal-distribution

    *
      - Type
      - rocRAND support
      - Supported rocRAND generators
      - cuRAND support
    *
      - :code:`float`
      - ✅
      - All generators
      - ✅
    *
      - :code:`float2`
      - ✅
      - Philox 4x32-10, MRG31K3P, MRG32K3A, XORWOW, LFSR113, all ThreeFry generators
      - ✅
    *
      - :code:`float4`
      - ✅
      - Philox 4x32-10
      - ✅
    *
      - :code:`double`
      - ✅
      - All generators
      - ✅
    *
      - :code:`double2`
      - ✅
      - Philox 4x32-10, MRG31K3P, MRG32K3A, XORWOW, LFSR113, all ThreeFry generators
      - ✅
    *
      - :code:`double4`
      - ✅
      - Philox 4x32-10
      - ❌

Poisson distributions
^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Supported types for Poisson distributions on the device
    :header-rows: 1
    :name: device-types-poisson-distribution

    *
      - Type
      - rocRAND support
      - Supported rocRAND generators
      - cuRAND support
    *
      - :code:`unsigned int`
      - ✅
      - Philox 4x32-10, MRG31k3p, MRG32K3A, XORWOW, MTGP32, Sobol32, Scrambled Sobol32, LFSR113, all ThreeFry generators
      - ✅
    *
      - :code:`unsigned long long int`
      - ✅
      - Sobol64, Scrambled Sobol64
      - ❌
    *
      - :code:`uint4`
      - ✅
      - Philox 4x32-10
      - ✅

Discrete distributions
^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Supported types for discrete distributions on the device
    :header-rows: 1
    :name: device-types-discrete-distribution

    *
      - Type
      - rocRAND support
      - Supported rocRAND generators
      - cuRAND support
    *
      - :code:`unsigned int`
      - ✅
      - All generators
      - ✅
    *
      - :code:`uint4`
      - ✅
      - Philox 4x32-10
      - ✅ - only Philox - 4x32-10
