.. meta::
  :description: rocRAND documentation and API reference library
  :keywords: rocRAND, ROCm, API, documentation
  
.. _programmers-guide:

==================
Programmer's guide
==================

Generator types
===============

There are two main classes of generator in rocRAND: Pseudo-Random Number Generators (PRNGs), and Quasi-Random Number Generators (QRNGs). The following pseudo-random number generators are available:

* XORWOW.
* MRG32K3A.
* MTGP32.
* Philox 4x32-10.
* MRG31K3P.
* LFSR113.
* MT19937.
* ThreeFry 2x32-20, 4x32-30, 2x64-20 and 4x64-20.

Additionally, the following quasi-random number generators are available:

* Sobol32.
* Sobol64.
* Scrambled Sobol32.
* Scrambled Sobol64.

Ordering
========

rocRAND generators can be configured to change how results are ordered in global memory. These parameters can be used to, for example, tune the performance versus the reproducibility of rocRAND generators. The following ordering types are available:

* `ROCRAND_ORDERING_PSEUDO_BEST`
* `ROCRAND_ORDERING_PSEUDO_DEFAULT`
* `ROCRAND_ORDERING_PSEUDO_SEEDED`
* `ROCRAND_ORDERING_PSEUDO_LEGACY`
* `ROCRAND_ORDERING_PSEUDO_DYNAMIC`
* `ROCRAND_ORDERING_QUASI_DEFAULT`

`ROCRAND_ORDERING_PSEUDO_DEFAULT` and `ROCRAND_ORDERING_QUASI_DEFAULT` are the default ordering for pseudo- and quasi-random number generators respectively. `ROCRAND_ORDERING_PSEUDO_DEFAULT` is currently the same as `ROCRAND_ORDERING_PSEUDO_BEST` and `ROCRAND_ORDERING_PSEUDO_LEGACY`.

`ROCRAND_ORDERING_PSEUDO_DYNAMIC` indicates that rocRAND may change the output ordering such that the best performance is obtained for a particular generator on a particular GPU. Using this ordering, the generated sequences can vary between different GPU models and rocRAND versions. More information about generating such configurations can be found at :doc:`dynamic_ordering_configuration`. `ROCRAND_ORDERING_PSEUDO_DYNAMIC` is not supported for generators created with `rocrand_create_generator_host`.

`ROCRAND_ORDERING_PSEUDO_LEGACY` indicates that rocRAND should generate values in a way that is backward compatible. When it is set, rocRAND generates exactly the same sequences across releases.

All supported orderings for all generators are detailed below:

.. table:: XORWOW ordering support
    :widths: auto

    ==================================  ====================================================================================================================
    Ordering                            
    ==================================  ====================================================================================================================
    `ROCRAND_ORDERING_PSEUDO_BEST`      The same as `ROCRAND_ORDERING_PSEUDO_LEGACY`.
    `ROCRAND_ORDERING_PSEUDO_DEFAULT`   The same as `ROCRAND_ORDERING_PSEUDO_LEGACY`.
    `ROCRAND_ORDERING_PSEUDO_SEEDED`    The same as `ROCRAND_ORDERING_PSEUDO_LEGACY`.
    `ROCRAND_ORDERING_PSEUDO_LEGACY`    There are :math:`131072` generators in total, each of which are separated by :math:`2^{67}` values. The results are generated in an interleaved fashion. The result at offset :math:`n` in memory is generated from offset :math:`(n\;\mathrm{mod}\; 131072) \cdot 2^{67} + \lfloor n / 131072 \rfloor` in the XORWOW sequence for a particular seed.
    `ROCRAND_ORDERING_PSEUDO_DYNAMIC`   The ordering depends on the GPU that is used.
    ==================================  ====================================================================================================================

.. table:: MRG32K3A ordering support
    :widths: auto

    ==================================  ====================================================================================================================
    Ordering                            
    ==================================  ====================================================================================================================
    `ROCRAND_ORDERING_PSEUDO_BEST`      The same as `ROCRAND_ORDERING_PSEUDO_LEGACY`.
    `ROCRAND_ORDERING_PSEUDO_DEFAULT`   The same as `ROCRAND_ORDERING_PSEUDO_LEGACY`.
    `ROCRAND_ORDERING_PSEUDO_LEGACY`    There are :math:`131072` generators in total, each of which are separated by :math:`2^{76}` values. The results are generated in an interleaved fashion. The result at offset :math:`n` in memory is generated from offset :math:`(n\;\mathrm{mod}\; 131072) \cdot 2^{76} + \lfloor n / 131072 \rfloor` in the MRG32K3A sequence for a particular seed.
    `ROCRAND_ORDERING_PSEUDO_DYNAMIC`   The ordering depends on the GPU that is used.
    ==================================  ====================================================================================================================

.. table:: MTGP32 ordering support
    :widths: auto

    ==================================  ====================================================================================================================
    Ordering                            
    ==================================  ====================================================================================================================
    `ROCRAND_ORDERING_PSEUDO_BEST`      The same as `ROCRAND_ORDERING_PSEUDO_LEGACY`.
    `ROCRAND_ORDERING_PSEUDO_DEFAULT`   The same as `ROCRAND_ORDERING_PSEUDO_LEGACY`.
    `ROCRAND_ORDERING_PSEUDO_LEGACY`    There are :math:`512` generators in total, each of which generates :math:`256` values per iteration. Blocks of :math:`256` elements from generators are concatenated to form the output. The result at offset :math:`n` in memory is generated from generator :math:`\lfloor n / 256\rfloor\;\mathrm{mod}\; 512`.
    `ROCRAND_ORDERING_PSEUDO_DYNAMIC`   The ordering depends on the GPU that is used.
    ==================================  ====================================================================================================================

.. table:: Philox 4x32-10 ordering support
    :widths: auto

    ==================================  ====================================================================================================================
    Ordering                            
    ==================================  ====================================================================================================================
    `ROCRAND_ORDERING_PSEUDO_BEST`      The same as `ROCRAND_ORDERING_PSEUDO_LEGACY`.
    `ROCRAND_ORDERING_PSEUDO_DEFAULT`   The same as `ROCRAND_ORDERING_PSEUDO_LEGACY`.
    `ROCRAND_ORDERING_PSEUDO_LEGACY`    There is only one Philox generator, and the result at offset :math:`n` is simply the :math:`n`-th value from this generator.
    `ROCRAND_ORDERING_PSEUDO_DYNAMIC`   The ordering depends on the GPU that is used.
    ==================================  ====================================================================================================================

.. table:: MT19937 ordering support
    :widths: auto

    ==================================  ====================================================================================================================
    Ordering                            
    ==================================  ====================================================================================================================
    `ROCRAND_ORDERING_PSEUDO_BEST`      The same as `ROCRAND_ORDERING_PSEUDO_LEGACY`.
    `ROCRAND_ORDERING_PSEUDO_DEFAULT`   The same as `ROCRAND_ORDERING_PSEUDO_LEGACY`.
    `ROCRAND_ORDERING_PSEUDO_LEGACY`    The Mersenne Twister sequence is generated from :math:`8192` generators in total, and each of these are separated by :math:`2^{1000}` values. Each generator generates :math:`8` elements per iteration. The result at offset :math:`n` is generated from generator :math:`(\lfloor n / 8\rfloor\;\mathrm{mod}\; 8192) \cdot 2^{1000} + \lfloor n / (8 \cdot 8192) \rfloor + \lfloor n / 8 \rfloor`.
    ==================================  ====================================================================================================================

.. table:: MRG31K3P ordering support
    :widths: auto

    ==================================  ====================================================================================================================
    Ordering                            
    ==================================  ====================================================================================================================
    `ROCRAND_ORDERING_PSEUDO_BEST`      The same as `ROCRAND_ORDERING_PSEUDO_LEGACY`.
    `ROCRAND_ORDERING_PSEUDO_DEFAULT`   The same as `ROCRAND_ORDERING_PSEUDO_LEGACY`.
    `ROCRAND_ORDERING_PSEUDO_LEGACY`    There are :math:`131072` generators in total, each of which are separated by :math:`2^{72}` values. The results are generated in an interleaved fashion. The result at offset :math:`n` in memory is generated from offset :math:`(n\;\mathrm{mod}\; 131072) \cdot 2^{72} + \lfloor n / 131072 \rfloor` in the MRG31K3P sequence for a particular seed.
    `ROCRAND_ORDERING_PSEUDO_DYNAMIC`   The ordering depends on the GPU that is used.
    ==================================  ====================================================================================================================

.. table:: LFSR113 ordering support
    :widths: auto

    ==================================  ====================================================================================================================
    Ordering                            
    ==================================  ====================================================================================================================
    `ROCRAND_ORDERING_PSEUDO_BEST`      The same as `ROCRAND_ORDERING_PSEUDO_LEGACY`.
    `ROCRAND_ORDERING_PSEUDO_DEFAULT`   The same as `ROCRAND_ORDERING_PSEUDO_LEGACY`.
    `ROCRAND_ORDERING_PSEUDO_LEGACY`    There are :math:`131072` generators in total, each of which are separated by :math:`2^{55}` values. The results are generated in an interleaved fashion. The result at offset :math:`n` in memory is generated from offset :math:`(n\;\mathrm{mod}\; 131072) \cdot 2^{55} + \lfloor n / 131072 \rfloor` in the LFSR113 sequence for a particular seed.
    `ROCRAND_ORDERING_PSEUDO_DYNAMIC`   The ordering depends on the GPU that is used.
    ==================================  ====================================================================================================================

.. table:: ThreeFry ordering support
    :widths: auto

    ==================================  ====================================================================================================================
    Ordering                            
    ==================================  ====================================================================================================================
    `ROCRAND_ORDERING_PSEUDO_BEST`      The same as `ROCRAND_ORDERING_PSEUDO_LEGACY`.
    `ROCRAND_ORDERING_PSEUDO_DEFAULT`   The same as `ROCRAND_ORDERING_PSEUDO_LEGACY`.
    `ROCRAND_ORDERING_PSEUDO_LEGACY`    There is only one ThreeFry generator, and the results at offset :math:`n` is simply the :math:`n`-th value from this generator.
    `ROCRAND_ORDERING_PSEUDO_DYNAMIC`   The ordering depends on the GPU that is used.
    ==================================  ====================================================================================================================

.. table:: Sobol ordering support
    :widths: auto

    ==================================  ====================================================================================================================
    Ordering                            
    ==================================  ====================================================================================================================
    `ROCRAND_ORDERING_QUASI_DEFAULT`    The (scrambled) 32- and 64-bit sobol quasi-random number generators generated the result from :math:`d` dimensions by flattening them into the output. The result at offset :math:`n` in memory is generated from offset :math:`n\;\mathrm{mod}\; d` in dimension :math:`\lfloor n / d \rfloor`, where :math:`d` is the generator's number of dimensions.
    ==================================  ====================================================================================================================

Using rocRAND in HIP Graphs
===========================

rocRAND supports the capturing of the random number generation with HIP Graphs. However, the construction, initialization and cleanup of the generator objects must take place outside of the recorded section. See the following example (error handling omitted for brevity):

.. literalinclude:: ../../test/hipgraphs_doc_sample.hpp
    :language: c++
