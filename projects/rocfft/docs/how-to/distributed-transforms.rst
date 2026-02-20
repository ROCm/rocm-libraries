.. meta::
  :description: Distributed transforms in rocFFT
  :keywords: rocFFT, ROCm, API, documentation, distributed transform


.. _distributed-transforms:

********************************************************************
Distributed transforms
********************************************************************

rocFFT can optionally distribute FFTs across multiple devices in a
single process or across multiple Message Passing Interface (MPI) ranks. To perform distributed
transforms, describe the input and output data layouts
as :ref:`fields<input_output_fields>`.

Multiple devices in a single process
====================================

A transform can be distributed across multiple devices in a single
process by passing distinct device IDs to
:cpp:func:`rocfft_brick_create` to create bricks in the input and output
fields.

Support for single-process multi-device transforms was introduced in
ROCm 6.0 with rocFFT 1.0.25.

Message Passing Interface
===============================

MPI lets you distribute the transform across multiple processes,
organized into MPI ranks.

To turn on rocFFT MPI support, enable the ``ROCFFT_MPI_ENABLE`` CMake option
when building the library. By default, this option
is off. To use Cray MPI, enable the ``ROCFFT_CRAY_MPI_ENABLE`` CMake option.

Additionally, rocFFT MPI support requires a GPU-aware MPI library
that supports transferring data to and from HIP devices.

Support for MPI transforms was introduced in ROCm 6.3 with rocFFT
1.0.29.

.. note::

   rocFFT API calls made on different ranks might return
   different values. Application developers must ensure that all ranks
   have successfully created their plans before attempting to execute
   a distributed transform. One rank can fail
   to create or execute a plan while the others succeed.

To distribute a transform across multiple MPI ranks, the
following additional steps are required:

#. Each rank calls :cpp:func:`rocfft_plan_description_set_comm` to
   add an MPI communicator to an allocated plan description. rocFFT
   distributes the computation across all ranks in the
   communicator.

#. Each rank allocates the same fields and calls
   :cpp:func:`rocfft_plan_description_add_infield` and
   :cpp:func:`rocfft_plan_description_add_outfield` on the plan
   description. However, each rank must only call
   :cpp:func:`rocfft_brick_create` and
   :cpp:func:`rocfft_field_add_brick` for bricks that reside on that
   rank. A brick resides on exactly one rank. Each rank can have zero
   or more bricks associated to it.

#. Each rank in the communicator calls
   :cpp:func:`rocfft_plan_create`. rocFFT then uses this information to distribute
   the supplied brick information between all of the ranks.

#. Each rank in the communicator calls :cpp:func:`rocfft_execute`.
   This function accepts arrays of pointers for input and output.
   The arrays contain pointers to each brick in the input or output
   of the current rank.

   The pointers must be provided in the same order in which the bricks were
   added to the field (using calls to :cpp:func:`rocfft_field_add_brick`) and
   must point to the memory on the device that was specified at that time.

   For in-place transforms, only pass the input pointers and use an
   empty array of output pointers.
