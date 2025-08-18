.. meta::
   :description: hipRAND interface usage notes
   :keywords: hipRAND, ROCm, library, API, tool, interface, usage


*******************************************************************
Using hipRAND interfaces
*******************************************************************

The hipRAND interface is compatible with the rocRAND API.
Porting a CUDA application that calls the cuRAND API to an application that calls the hipRAND API is relatively straightforward.

Host API
===============================

For example, to create a generator, follow this example:

.. code-block:: cpp

   hiprandStatus_t
   hiprandCreateGenerator(
      hiprandGenerator_t* generator,
      hiprandRngType_t rng_type
   )

Device API
===============================

Here is an example that generates a log-normally distributed float from a generator.
These functions are templated for all generators.

.. code-block:: cpp

   __device__ double
   hiprand_log_normal_double(
      hiprandStateSobol64_t* state,
      double mean,
      double stddev
   )