.. meta::
  :description: hipFFT documentation and API reference library
  :keywords: FFT, hipFFT, rocFFT, ROCm, API, documentation

.. _hipfft-api-usage:

********************************************************************
hipFFT API Usage
********************************************************************

This section describes usage of the hipFFT library API.  The hipFFT
API follows the NVIDIA `cuFFT`_ API.

.. _cuFFT: https://docs.nvidia.com/cuda/cufft/

Types
=====

There are a few data structures that are internal to the library.  The
pointer types to these structures are given below.  The user would
need to use these types to create handles and pass them between
different library functions.

.. doxygendefine:: HIPFFT_FORWARD

.. doxygendefine:: HIPFFT_BACKWARD

.. doxygenenum:: hipfftType

.. doxygentypedef:: hipfftHandle

.. doxygenenum:: hipfftResult


Simple plans
============

These planning routines allocate a plan for you.  If execution of the
plan requires a work buffer, it will be created (and destroyed)
automatically.

.. doxygenfunction:: hipfftPlan1d

.. doxygenfunction:: hipfftPlan2d

.. doxygenfunction:: hipfftPlan3d


User managed simple plans
-------------------------

These planning routines assume that you have allocated a plan
(`hipfftHandle`) yourself; and that you will manage a work area as
well.

If you want to manage your own work buffer... XXX

.. doxygenfunction:: hipfftCreate

.. doxygenfunction:: hipfftDestroy

.. doxygenfunction:: hipfftSetAutoAllocation

.. doxygenfunction:: hipfftMakePlan1d

.. doxygenfunction:: hipfftMakePlan2d

.. doxygenfunction:: hipfftMakePlan3d


Advanced plans
===================

.. doxygenfunction:: hipfftMakePlanMany
.. doxygenfunction:: hipfftXtMakePlanMany



Estimating work area sizes
==========================

These call return estimates of the work area required to support a
plan generated with the same parameters (either with the simple or
extensible API).  Callers who choose to manage work area allocation
within their application must use this call after plan generation, and
after any hipfftSet*() calls subsequent to plan generation, if those
calls might alter the required work space size.

.. doxygenfunction:: hipfftEstimate1d

.. doxygenfunction:: hipfftEstimate2d

.. doxygenfunction:: hipfftEstimate3d

.. doxygenfunction:: hipfftEstimateMany


Accurate work area sizes
------------------------

After plan generation is complete, an accurate work area size can be
obtained with these routines.

.. doxygenfunction:: hipfftGetSize1d

.. doxygenfunction:: hipfftGetSize2d

.. doxygenfunction:: hipfftGetSize3d

.. doxygenfunction:: hipfftGetSizeMany

.. doxygenfunction:: hipfftXtGetSizeMany
		     

Executing plans
===============

Once you have created an FFT plan, you can execute it using one of the
`hipfftExec*` functions.

For real-to-complex transforms, the output buffer XXX

For complex-to-real transforms, the output buffer XXX

.. doxygenfunction:: hipfftExecC2C

.. doxygenfunction:: hipfftExecR2C

.. doxygenfunction:: hipfftExecC2R

.. doxygenfunction:: hipfftExecZ2Z

.. doxygenfunction:: hipfftExecD2Z

.. doxygenfunction:: hipfftExecZ2D

.. doxygenfunction:: hipfftXtExec
		     

HIP graph support for hipFFT
============================

hipFFT supports capturing kernels launched during FFT execution into
HIP graph nodes.  This way, users can capture FFT execution, along
with other work, into a HIP graph and launch the work in the graph
multiple times.

The following hipFFT APIs can be used with graph capture:

* :cpp:func:`hipfftExecC2C`

* :cpp:func:`hipfftExecR2C`

* :cpp:func:`hipfftExecC2R`

* :cpp:func:`hipfftExecZ2Z`

* :cpp:func:`hipfftExecD2Z`

* :cpp:func:`hipfftExecZ2D`

Note that each launch of a HIP graph will provide the same arguments
to the kernels in the graph.  In particular, this implies that all of
the parameters to the above APIs remain valid while the HIP graph is
in use:

* The hipFFT plan

* The input and output buffers

hipFFT does not support capturing work performed by other API
functions aside from those listed above.

Callbacks
=========

.. doxygenfunction:: hipfftXtSetCallback
.. doxygenfunction:: hipfftXtClearCallback	     
.. doxygenfunction:: hipfftXtSetCallbackSharedSize

		     
Single-process Multi-GPU Transforms
===================================

hipFFT offers experimental support for distributing a transform
across multiple GPUs in a single process.

The API usage works as follows:

1. Create a hipFFT plan handle with :cpp:func:`hipfftCreate`.

2. Associate a set of GPU devices to the plan by calling :cpp:func:`hipfftXtSetGPUs`.

3. Make the plan by calling one of:

   * :cpp:func:`hipfftMakePlan1d`
   * :cpp:func:`hipfftMakePlan2d`
   * :cpp:func:`hipfftMakePlan3d`
   * :cpp:func:`hipfftMakePlanMany`
   * :cpp:func:`hipfftMakePlanMany64`
   * :cpp:func:`hipfftXtMakePlanMany`

4. Allocate memory for the data on the devices with
   :cpp:func:`hipfftXtMalloc`, which returns the allocated memory as
   a :cpp:struct:`hipLibXtDesc` descriptor.

5. Copy data from the host to the descriptor with :cpp:func:`hipfftXtMemcpy`.

6. Execute the plan by calling one of:

   * :cpp:func:`hipfftXtExecDescriptor`
   * :cpp:func:`hipfftXtExecDescriptorC2C`
   * :cpp:func:`hipfftXtExecDescriptorR2C`
   * :cpp:func:`hipfftXtExecDescriptorC2R`
   * :cpp:func:`hipfftXtExecDescriptorZ2Z`
   * :cpp:func:`hipfftXtExecDescriptorD2Z`
   * :cpp:func:`hipfftXtExecDescriptorZ2D`

   Pass the descriptor as input and output.

7. Copy the output from the descriptor back to the host with :cpp:func:`hipfftXtMemcpy`.

8. Free the descriptor with :cpp:func:`hipfftXtFree`.

9. Clean up the plan by calling :cpp:func:`hipfftDestroy`.

.. doxygenfunction:: hipfftXtSetGPUs

.. doxygenstruct:: hipXtDesc
.. doxygenstruct:: hipLibXtDesc

.. doxygenfunction:: hipfftXtMalloc
.. doxygenfunction:: hipfftXtFree
.. doxygenfunction:: hipfftXtMemcpy
		     
.. doxygengroup:: hipfftXtExecDescriptor

Multi-process transforms
========================

hipFFT has experimental support for transforms distributed across MPI (Message 
Passing Interface) processes.

Support for MPI transforms was introduced in ROCm 6.4 with hipFFT 1.0.18.

.. note::

   hipFFT MPI support is only available when the library is built
   with the `HIPFFT_MPI_ENABLE` CMake option enabled. By default, it
   is off.

   Additionally, hipFFT MPI support requires its backend FFT library
   to also support MPI.  This means that either an MPI-enabled rocFFT
   library or cuFFTMp must be used.

   Finally, hipFFT API calls made on different ranks might return
   different values.  Users must take care to ensure that all ranks
   have successfully created their plans before attempting to execute
   a distributed transform, and it is possible for one rank to fail
   to create/execute a plan while the others succeed.

MPI must be initialized before creating multi-process hipFFT plans.

Built-in decomposition
----------------------

hipFFT can automatically decide the data decomposition for
distributed transforms.  The API usage is similar to the
single-process, multi-GPU case described above.

1. On all ranks in the MPI communicator:

   i. Create a hipFFT plan handle with :cpp:func:`hipfftCreate`.

   ii. Attach the MPI communicator to the plan with :cpp:func:`hipfftMpAttachComm`.

   iii. Make the plan by calling one of:

      * :cpp:func:`hipfftMakePlan1d`
      * :cpp:func:`hipfftMakePlan2d`
      * :cpp:func:`hipfftMakePlan3d`
      * :cpp:func:`hipfftMakePlanMany`
      * :cpp:func:`hipfftMakePlanMany64`
      * :cpp:func:`hipfftXtMakePlanMany`

      .. note::

         Not all backend FFT libraries support distributing all
         transforms.  Check the documentation for the backend FFT library
         for any restrictions on distributed transform types, placement,
         sizes, or data layouts.

2. Copy data from the host to the descriptor with :cpp:func:`hipfftXtMemcpy`.

3. Execute the plan by calling one of:

   * :cpp:func:`hipfftXtExec`
   * :cpp:func:`hipfftXtExecDescriptorC2C`
   * :cpp:func:`hipfftXtExecDescriptorR2C`
   * :cpp:func:`hipfftXtExecDescriptorC2R`
   * :cpp:func:`hipfftXtExecDescriptorZ2Z`
   * :cpp:func:`hipfftXtExecDescriptorD2Z`
   * :cpp:func:`hipfftXtExecDescriptorZ2D`

4. Copy the output from the descriptor back to the host with :cpp:func:`hipfftXtMemcpy`.

5. Free the descriptor with :cpp:func:`hipfftXtFree`.

6. On all ranks in the MPI communicator:

   i. Clean up the plan by calling :cpp:func:`hipfftDestroy`.

Custom decomposition
--------------------

hipFFT also allows an arbitrary decomposition of the FFT into 1D, 2D, or
3D bricks.  Each MPI rank calls :cpp:func:`hipfftXtSetDistribution`
during plan creation to declare which input and output brick resides
on that rank.

The same API calls are made on each rank in the MPI communicator, as follows:

1. Create a hipFFT plan handle with :cpp:func:`hipfftCreate`.

2. Attach the MPI communicator to the plan with :cpp:func:`hipfftMpAttachComm`.

3. Call :cpp:func:`hipfftXtSetDistribution` to specify the input and output brick for the current rank.

   Bricks are specified by their lower and upper coordinates in
   the input/output index space.  The lower coordinate is
   inclusive (contained within the brick) and the upper
   coordinate is exclusive (first index past the end of the
   brick).

   Strides for the input/output data are also provided, to
   describe how the bricks are laid out in physical memory.

   Each coordinate and stride contains the same number of elements as
   the number of dimensions in the FFT.  Note that this also implies
   that batched FFTs are not supported when using MPI, because the
   coordinates and strides do not contain information about the batch
   dimension.

4. Make the plan by calling one of:

   * :cpp:func:`hipfftMakePlan1d`
   * :cpp:func:`hipfftMakePlan2d`
   * :cpp:func:`hipfftMakePlan3d`

   "PlanMany" APIs enable batched FFTs and are not usable with
   MPI.

   .. note::

      Not all backend FFT libraries support distributing all
      transforms.  Check the documentation for the backend FFT library
      for any restrictions on distributed transform types, placement,
      sizes, or data layouts.

5. Call :cpp:func:`hipfftXtMalloc` with
   :cpp:enum:`HIPFFT_XT_FORMAT_DISTRIBUTED_INPUT` to
   allocate the input brick on the current rank.  The allocated
   memory is returned as a :cpp:struct:`hipLibXtDesc` descriptor.

6. Call :cpp:func:`hipfftXtMalloc` with
   :cpp:enum:`HIPFFT_XT_FORMAT_DISTRIBUTED_OUTPUT` to
   allocate the output brick on the current rank.  The allocated
   memory is returned as a :cpp:struct:`hipLibXtDesc` descriptor.

7. Initialize the memory pointed to by the descriptor.

8. Execute the plan by calling one of:

   * :cpp:func:`hipfftXtExecDescriptor`
   * :cpp:func:`hipfftXtExecDescriptorC2C`
   * :cpp:func:`hipfftXtExecDescriptorR2C`
   * :cpp:func:`hipfftXtExecDescriptorC2R`
   * :cpp:func:`hipfftXtExecDescriptorZ2Z`
   * :cpp:func:`hipfftXtExecDescriptorD2Z`
   * :cpp:func:`hipfftXtExecDescriptorZ2D`

   Pass the input descriptor as input and the output descriptor as output.

9. Use the transformed data pointed to by the output descriptor.

10. Free the descriptors with :cpp:func:`hipfftXtFree`.

11. Clean up the plan by calling :cpp:func:`hipfftDestroy`.

.. doxygenfunction:: hipfftMpAttachComm
.. doxygenfunction:: hipfftXtSetDistribution
.. doxygenfunction:: hipfftXtSetSubformatDefault
