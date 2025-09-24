.. meta::
  :description: Programmers guide for the rocBLAS documentation and API reference library
  :keywords: rocBLAS, ROCm, API, Linear Algebra, documentation, programming,

.. _programmers-guide:

********************************************************************
rocBLAS programming guide
********************************************************************

This topic covers internal details that are required to program with rocBLAS. It includes
a discussion of the source code organization, stream and device management, device
memory allocation, and other technical considerations.

================================
Source code organization
================================

The rocBLAS code can be found in the `rocBLAS folder <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocblas>`_
of the `rocm-libraries GitHub <https://github.com/ROCm/rocm-libraries>`_.
It is split into three major parts:

* The ``library`` directory contains all the source code for the library.
* The ``clients`` directory contains all the test code and code to build clients.
* Infrastructure such as ``docs`` and ``cmake`` to support the library.

The library directory
-----------------------

Here is the structure of the ``library`` directory, which contains the following content for rocBLAS.

library/include
^^^^^^^^^^^^^^^^^^^^^^^^

This directory contains C98 include files for the external API. These files also contain Doxygen
comments that document the API.

library/src/blas[1,2,3]
^^^^^^^^^^^^^^^^^^^^^^^^

This includes source code for Level 1, 2, and 3 BLAS functions in ``.cpp`` and ``.hpp`` files.

*  The ``*.cpp`` files contain the following:

   *  External C functions that call or instantiate templated functions with an ``_impl`` extension.
   *  The ``_impl`` functions, which have argument checking and logging. In turn, they
      call functions with a ``_template`` extension.

*  The ``*_imp.hpp`` files contain:

   *  ``_template`` functions that can be exported to rocSOLVER. They usually call the ``_launcher`` functions.
   *  API implementations that can be instantiated for the original APIs with integer arguments using ``rocblas_int`` and
      again for the ILP64 API with integer arguments as ``int64_t``.

*  The ``*_kernels.cpp`` files contain:

   * ``_launcher`` functions that invoke or launch kernels using ``ROCBLAS_LAUNCH_KERNEL`` or related macros.
   * ``_kernel`` functions that run on the device.

library/src/blas_ex
^^^^^^^^^^^^^^^^^^^^^^^^

This contains the source code for mixed-precision BLAS.

library/src/src64
^^^^^^^^^^^^^^^^^^^^^^^^

This directory contains the ILP64 source code for Level 1, 2, and 3 BLAS and mixed-precision functions in ``blas_ex``.
The files normally end with ``_64`` before the file type extension, for example, ``_64.cpp``.
The API integers are ``int64_t`` instead of ``rocblas_int``.
The function behavior is kept identical at the higher level by instantiable macros and C++ templates.
Only at the kernel dispatch level does the code diverge by
providing a ``_64`` version, in which the invocation is controlled using the ``ROCBLAS_API`` macro.
The directory structure mirrors the level organization used for the parent directory ``library/src``.

Device kernel code
^^^^^^^^^^^^^^^^^^^^^^^^

Most BLAS device functions (kernels) are C++ templated functions based on the data type.
In C++ host code, any duplicate instantiations of templates can be handled by the linker
and the duplicates are ignored. LLVM device code instantiations, however, are not handled this way,
so you must avoid duplicate instantiations in multiple code units.
Kernel templates should therefore only be provided as C++ template prototypes in the include files
unless they must be instantiated. Try to instantiate all forms in a single
unit, for example, in a ``.cpp`` file, and expose a launcher C++ interface to invoke the device calls where possible.
This is especially important for ILP64 implementations where it's best to
reuse the LP64 instantiations without any duplication to avoid bloating the library size.

library/src/blas3/Tensile
^^^^^^^^^^^^^^^^^^^^^^^^^

This includes code for calling Tensile from rocBLAS and YAML files with Tensile tuning configurations.

library/src/include
^^^^^^^^^^^^^^^^^^^^^^^^

This includes the internal include files for:

*  Handle-related code
*  Device memory allocation
*  Logging
*  Numerical checking
*  Utility code

The clients directory
-----------------------

The ``clients`` directory contains all test code and code to build clients.

clients/gtest
^^^^^^^^^^^^^^^^^^^^^^^^

Code for the rocblas-test client. This client is used to test rocBLAS.

clients/benchmarks
^^^^^^^^^^^^^^^^^^^^^^^^

Code for the rocblas-benchmark client. This client is used to benchmark rocBLAS functions.

clients/include
^^^^^^^^^^^^^^^^^^^^^^^^

This contains code for testing and benchmarking individual rocBLAS functions and utility code for testing.
The test harness functions are templated by data type and are defined in separate files for
each function form: non-batched, batched, and strided_batched.
When a function also supports the ILP64 API, then both forms can be tested by the same
template and are controlled by the ``Arguments`` API member variable.
This follows the pattern for Fortran API testing and includes ``FORTRAN_64`` for the ILP64 format.

The code for benchmarking ``gemm_ex`` (in ``testing_gemm_ex.hpp``) using rocblas-bench tries to reuse device memory between consecutive calls.
To get the best memory reuse, leading to better performance, the larger GEMMs should be listed first in the YAML input file.
Device memory is only reused between consecutive calls with the same precision, so GEMM operations should be grouped by their precisions.

clients/common
^^^^^^^^^^^^^^^^^^^^^^^^

This folder includes common code used by both rocblas-benchmark and rocblas-test.

clients/samples
^^^^^^^^^^^^^^^^^^^^^^^^

This directory contains sample code for calling the rocBLAS functions.

Infrastructure
--------------

*  CMake is used to build and package rocBLAS. There are ``CMakeLists.txt`` files throughout the code.
*  Doxygen, Breathe, Sphinx, and ReadTheDocs are used to produce the documentation. The content for the documentation is taken from these sources:

   *  Doxygen comments in the include files in the directory ``library/include``
   *  Files in the ``docs`` folder.

*  Jenkins is used to automate continuous integration (CI) testing.
*  clang-format is used to format the C++ code.

=====================================
Handle, stream, and device management
=====================================

This section covers handles, streams, and devices, including multiple streams and devices.

Handle
-------

You must create a ``rocBLAS_handle``, as shown below, before calling any other rocBLAS functions:

.. code-block:: cpp

    rocblas_handle handle;
    if(rocblas_create_handle(&handle) != rocblas_status_success) return EXIT_FAILURE;

After you have finished calling the rocBLAS functions, destroy the handle:

.. code-block:: cpp

    if(rocblas_destroy_handle(handle) != rocblas_status_success) return EXIT_FAILURE;

The handle you create uses the default stream and device. To use a non-default
stream and non-default device, follow this approach:

.. code-block:: cpp

    int deviceId = non_default_device_id;
    if(hipSetDevice(deviceId) != hipSuccess) return EXIT_FAILURE;

    //optional call to rocblas_initialize
    rocblas_initialize();

    // note the order, call hipSetDevice before hipStreamCreate
    hipStream_t stream;
    if(hipStreamCreate(&stream) != hipSuccess) return EXIT_FAILURE;

    rocblas_handle handle;
    if(rocblas_create_handle(&handle) != rocblas_status_success) return EXIT_FAILURE;

    if(rocblas_set_stream(handle, stream) != rocblas_status_success) return EXIT_FAILURE;


To use the library with a non-default device within a host thread, the device must be set using ``hipSetDevice()`` before creating the handle.

The device in the host thread must not be changed between ``hipStreamCreate`` and ``hipStreamDestroy``.
If the device in the host thread is changed between creating and destroying the stream, then the behavior is undefined.

If you create a non-default stream, it is your responsibility to synchronize the old non-default stream
and update the rocBLAS handle with the default or new non-default stream before destroying the old non-default stream.

.. code-block:: cpp

    // Synchronize the non-default stream before destroying it
    if(hipStreamSynchronize(stream) != hipSuccess) return EXIT_FAILURE;

    // Reset the stream reference in the handle to either default or new non-default
    if(rocblas_set_stream(handle, 0) != rocblas_status_success) return EXIT_FAILURE;

    if(hipStreamDestroy(stream) != hipSuccess) return EXIT_FAILURE;

.. note::

   It is essential to reset the rocBLAS handle stream reference to avoid a ``hipErrorContextIsDestroyed`` error, which is handled internally.
   If this step is skipped, you might encounter this error with ``AMD_LOG_LEVEL`` logging or when using ``hipPeekAtLastError( )``.

When switching from one non-default stream to another, you must complete
all rocBLAS operations previously submitted with this handle on the old stream using
the ``hipStreamSynchronize(old_stream)`` API before setting the new stream.

.. code-block:: cpp

    // Synchronize the old stream (optional)
    if(hipStreamSynchronize(old_stream) != hipSuccess) return EXIT_FAILURE;

    // Create a new stream (this step can be done before the steps above)
    if(hipStreamCreate(&new_stream) != hipSuccess) return EXIT_FAILURE;

    // Set the handle to use the new stream (must come after synchronization & before deletion of old stream)
    if(rocblas_set_stream(handle, new_stream) != rocblas_status_success) return EXIT_FAILURE;

    // Destroy the old stream (this step is optional but must come after synchronization)
    if(hipStreamDestroy(old_stream) != hipSuccess) return EXIT_FAILURE;

The call to ``hipStreamSynchronize`` above is necessary for the ``user_owned`` allocation scheme because the ``rocBLAS_handle`` contains allocated device
memory provided by the user that must not be shared by multiple asynchronous streams at the same time.

If either the old or new stream is the default or ``NULL`` stream, it is not necessary to
synchronize the old stream before destroying it, or before setting the new stream,
because the synchronization is implicit.

.. note::

   You can switch from one non-default stream to another without calling ``hipStreamSynchronize()`` as the default memory allocation scheme (``rocBLAS_managed``) uses stream order allocation.
   For more information, see :ref:`Device Memory Allocation Usage`.

Creating the handle incurs a startup cost. There is an additional startup cost for
GEMM functions to load GEMM kernels for a specific device. You can shift the
GEMM startup cost to occur later after setting the device by calling ``rocblas_initialize()``
after calling ``hipSetDevice()``. This needs to happen once for each device.
If you have two rocBLAS handles which use the same device, then you only need to call ``rocblas_initialize()``
once. If ``rocblas_initialize()`` is not called, then the first GEMM call incurs
the startup cost.

The ``rocBLAS_handle`` stores the following information:

*  Stream
*  Logging mode
*  Pointer mode
*  Atomics mode

Stream and device management
-----------------------------

HIP kernels are launched in a queue, which is also known as a stream. A stream represents a queue of
work on a particular device.

A ``rocBLAS_handle`` always has one stream, while a stream is always associated with one device.
The ``rocBLAS_handle`` is passed as an argument to all rocBLAS functions that launch kernels. These kernels are
launched in the handle's stream to run on that stream's device.

If you do not create a stream, the ``rocBLAS_handle`` uses the default or ``NULL``
stream, which is maintained by the system. You cannot create or destroy the default
stream. However, you can create a new non-default stream and bind it to the ``rocBLAS_handle`` using the
commands ``hipStreamCreate()`` and ``rocblas_set_stream()``.

rocBLAS supports non-blocking streams for functions requiring synchronization to guarantee results on the host.
For functions like ``rocblas_Xnrm2``, the scalar result is copied from device to host when ``rocblas_pointer_mode == rocblas_pointer_mode_host``.
This is accomplished by using ``hipMemcpyAsync()``, followed by ``hipStreamSynchronize()``.
The stream in the ``rocBLAS_handle`` is synchronized.

.. note::

   The rocBLAS functions :any:`rocblas_set_vector`, :any:`rocblas_get_vector`, :any:`rocblas_set_matrix`, and :any:`rocblas_get_matrix`
   block on the default stream and are exceptions to the pattern above.

If you create a stream, you are responsible for destroying it using ``hipStreamDestroy()``. If the handle
has to switch from one non-default stream to another, then the old stream needs to be synchronized.
After that, you need to create and set the new non-default stream using ``hipStreamCreate()`` and ``rocblas_set_stream()``, respectively.
Then you can optionally destroy the old stream.

HIP has two important device management functions:

*  ``hipSetDevice()``: Sets the default device to be used for subsequent HIP API calls from the thread.
*  ``hipGetDevice()``: Returns the default device ID for the calling host thread.

The device which was set using ``hipSetDevice()`` when ``hipStreamCreate()`` was called
is the one that is associated with a stream. If the device was not set using ``hipSetDevice()``, then the default device is used.

You cannot switch the device in a stream between ``hipStreamCreate()`` and ``hipStreamDestroy()``.
To use another device, create another stream.

rocBLAS never sets a device. It only queries the device using ``hipGetDevice()``. If rocBLAS does not see a
valid device, it returns an error message.

Multiple streams and multiple devices
-------------------------------------

If a machine has ``num`` GPU devices, they have the ``deviceID`` numbers 0, 1, 2, and so forth, which are equivalent to ``num - 1``. The
default device has ``deviceID == 0``. Each ``rocBLAS_handle`` can only be used with a single device,
but you can run ``<num>`` handles on ``<num>`` devices concurrently.

.. _Device Memory allocation in detail:

========================
Device memory allocation
========================

This section presents the requirements and design details for rocBLAS device memory allocation, along with a series of examples.

Requirements
-------------

The following list of requirements motivate the design implementation for device memory allocation.

*  Some rocBLAS functions require temporary device memory.
*  Allocating and deallocating device memory is expensive and needs synchronizing.
*  Temporary device memory should be recycled across multiple rocBLAS function calls using the same ``rocblas_handle``.
*  The following schemes need to be supported:

   *  **Default**: Functions allocate required device memory automatically. This has the disadvantage that allocation is a synchronizing event.
   *  **Preallocate**: Query all the functions called using a ``rocblas_handle`` to find out how much device memory is needed.
      Preallocate the required device memory when the ``rocblas_handle`` is created. There are no more synchronizing allocations or deallocations.
   *  **Manual**: Query a function to find out how much device memory is required.
      Allocate and deallocate the device memory before and after the function calls.
      This allows the user to control where the synchronizing allocation and deallocation occur.

In all of the above schemes, temporary device memory needs to be held by the ``rocblas_handle`` and recycled if a subsequent function using the handle needs it.

Design
------

*  rocBLAS uses per-handle device memory allocation with out-of-band management.
*  The state of the device memory is stored in the ``rocblas_handle``.
*  For the rocBLAS user:

   *  Functions are provided to query how much device memory a function needs.
   *  An environment variable is provided to preallocate when the ``rocblas_handle`` is created.
   *  Functions are provided to manually allocate and deallocate after the ``rocblas_handle`` is created.
   *  The following two values are added to the ``rocblas_status`` enum to indicate how a rocBLAS function is changing the state of the temporary device memory in the ``rocblas_handle``:

      *  ``rocblas_status_size_unchanged``
      *  ``rocblas_status_size_increased``

*  For the rocBLAS developer:

   *  Functions are provided to answer device memory size queries.
   *  Functions are provided to allocate temporary device memory.
   *  Opaque RAII objects are used to hold the temporary device memory. Allocated memory is returned to the handle automatically when it is no longer needed.

The functions for the rocBLAS user are described in the :ref:`api-reference-guide`. The functions for the rocBLAS developer are described below.

Answering device memory size queries in functions that need memory
------------------------------------------------------------------

Functions should contain code like the sample below to answer a query on how much temporary device memory is required.
In this case, ``m * n * sizeof(T)`` bytes of memory is required.

Here is an example:

.. code-block:: c++

    rocblas_status rocblas_function(rocblas_handle handle, ...)
    {
        if(!handle) return rocblas_status_invalid_handle;

        if (handle->is_device_memory_size_query())
        {
            size_t size = m * n * sizeof(T);
            return handle->set_optimal_device_memory_size(size);
        }

        //  rest of function
    }

is_device_memory_size_query function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

    bool _rocblas_handle::is_device_memory_size_query() const

Indicates if the current function call is collecting information about the optimal device memory allocation size.

Return value:

*  **true**: Information is being collected
*  **false**: Information is not being collected

set_optimal_device_memory_size function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

    rocblas_status _rocblas_handle::set_optimal_device_memory_size(size...)

Sets the optimal sizes of device memory buffers in bytes for this function. The sizes are rounded up to the next multiple of 64 (or some other chunk size), and the running maximum is updated.

Return value:

*  **rocblas_status_size_unchanged**: The maximum optimal device memory size did not change. This is the case where the function does not use device memory.
*  **rocblas_satus_size_increased**: The maximum optimal device memory size increased.
*  **rocblas_status_internal_error**: This function is not supposed to be collecting size information.

rocblas_sizeof_datatype function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

    size_t rocblas_sizeof_datatype(rocblas_datatype type)

Returns the size of a rocBLAS data type.


Answering device memory size queries in functions that do not need memory
--------------------------------------------------------------------------

Here is an example:

.. code-block:: c++

    rocblas_status rocblas_function(rocblas_handle handle, ...)
    {
        if(!handle) return rocblas_status_invalid_handle;

        RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);

    //  rest of function
    }

RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED macro
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

    RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle)

This is a convenience macro that returns ``rocblas_status_size_unchanged`` if the function call is a memory size query.

rocBLAS kernel device memory allocation
-----------------------------------------

Device memory can be allocated for ``n`` floats using ``device_malloc`` as in this example:

.. code-block:: c++

     auto workspace = handle->device_malloc(n * sizeof(float));
     if (!workspace) return rocblas_status_memory_error;
     float* ptr = static_cast<float*>(workspace);

Example
^^^^^^^

To allocate multiple buffers:

.. code-block:: c++

    size_t size1 = m * n;
    size_t size2 = m * k;

    auto workspace = handle->device_malloc(size1, size2);
    if (!workspace) return rocblas_status_memory_error;

    void * w_buf1, * w_buf2;
    w_buf1 = workspace[0];
    w_buf2 = workspace[1];


device_malloc function
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: c++

    auto workspace = handle->device_malloc(size...)

*  Returns an opaque RAII object lending allocated device memory to a particular rocBLAS function.
*  The object returned is convertible to ``void *`` or other pointer types if only one size is specified.
*  The individual pointers can be accessed using the subscript ``operator[]``.
*  The lifetime of the returned object is the lifetime of the borrowed device memory (RAII).
*  To simplify and optimize the code, only one successful allocation object can be alive at a time.
*  If the handle's device memory is currently being managed by rocBLAS, as in the default scheme, it is expanded in size as necessary.
*  If the user allocated (or pre-allocated) an explicit size of device memory, then that size is used as the limit, and no resizing or synchronization ever occurs.

Parameters:

- **size**: The size in bytes of memory to be allocated.

Return value:

- **On success**: Returns an opaque RAII object that evaluates to ``true`` when converted to ``bool``.
- **On failure**: Returns an opaque RAII object that evaluates to ``false`` when converted to ``bool``.


Performance degradation
-----------------------

The ``rocblas_status`` enum value ``rocblas_status_perf_degraded`` indicates that a slower algorithm was used because of insufficient device memory for the optimal algorithm.

Example
^^^^^^^

.. code-block:: c++

    rocblas_status ret = rocblas_status_success;
    size_t size_for_optimal_algorithm = m + n + k;
    size_t size_for_degraded_algorithm = m;
    auto workspace_optimal = handle->device_malloc(size_for_optimal_algorithm);
    if (workspace_optimal)
    {
        // Algorithm using larger optimal memory
    }
    else
    {
        auto workspace_degraded = handle->device_malloc(size_for_degraded_algorithm);
        if (workspace_degraded)
        {
            // Algorithm using smaller degraded memory
            ret = rocblas_status_perf_degraded;
        }
        else
        {
            // Not enough device memory for either optimal or degraded algorithm
            ret = rocblas_status_memory_error;
        }
    }
    return ret;


===================
Thread-safe logging
===================

rocBLAS has thread-safe logging. This prevents garbled output when multiple threads are writing to the same file.

Thread-safe logging is achieved by using ``rocblas_internal_ostream``, a class that can be used similarly to ``std::ostream``.
It provides standardized methods for formatted output to either strings or files.
The default constructor for ``rocblas_internal_ostream`` writes to strings, which are thread safe because they are owned by the calling thread.
There are also ``rocblas_internal_ostream`` constructors for writing to files.
The ``rocblas_internal_ostream::yaml_on`` and ``rocblas_internal_ostream::yaml_off`` I/O modifiers turn YAML formatting mode on and off.

``rocblas_cout`` and ``rocblas_cerr`` are the thread-safe versions of ``std::cout`` and ``std::cerr``.

Many output identifiers have been marked "poisoned" in rocblas-test and rocblas-bench, to detect the use of non-thread-safe I/O.
These include ``std::cout``, ``std::cerr``, ``printf``, ``fprintf``, ``fputs``, ``puts``, and others.
The poisoning is not turned on in the library itself or in the samples
to avoid imposing restrictions on the use of these symbols on outside users.

``rocblas_handle`` contains three ``rocblas_internal_ostream`` pointers for logging output:

*  ``static rocblas_internal_ostream* log_trace_os``
*  ``static rocblas_internal_ostream* log_bench_os``
*  ``static rocblas_internal_ostream* log_profile_os``

The user can also create ``rocblas_internal_ostream`` pointers and objects outside the handle.

The following usage notes apply to ``rocblas_internal_ostream``:

*  Each ``rocblas_internal_ostream`` associated with a file points to a single ``rocblas_internal_ostream::worker``
   with a ``std::shared_ptr`` for writing to the file. The worker is mapped from the device ID and ``inode`` corresponding to the file.
   More than one ``rocblas_internal_ostream`` can point to the same worker.

*  This means that if more than one ``rocblas_internal_ostream`` is writing to a single output file,
   they will share the same ``rocblas_internal_ostream::worker``.

*  The ``<<`` operator for ``rocblas_internal_ostream`` is overloaded. Output is first accumulated
   in ``rocblas_internal_ostream::os``, a ``std::ostringstream`` buffer. Each ``rocblas_internal_ostream`` has
   its own ``os`` ``std::ostringstream`` buffer, so strings in ``os`` are not garbled.

*  When ``rocblas_internal_ostream.os`` is flushed with either a ``std::endl`` or an explicit flush
   of ``rocblas_internal_ostream``, then ``rocblas_internal_ostream::worker::send`` pushes the string contents
   of ``rocblas_internal_ostream.os`` and a promise, which together are called a task, onto ``rocblas_internal_ostream.worker.queue``.

*  The ``send`` function uses the promise to asynchronously transfer data from ``rocblas_internal_ostream.os`` to
   ``rocblas_internal_ostream.worker.queue`` and wait for the worker to finish writing the string to the file.
   It also locks a mutex to ensure pushing the task onto the queue is atomic.

*  The ``ostream.worker.queue`` contains a number of tasks. When ``rocblas_internal_ostream`` is destroyed,
   all the ``tasks.string`` in ``rocblas_internal_ostream.worker.queue`` are printed to the ``rocblas_internal_ostream`` file and
   the ``std::shared_ptr`` to the ``ostream.worker`` is destroyed. If the reference count to the worker becomes ``0``,
   the worker's thread is sent a zero-length string telling it to exit.

===========================
rocBLAS numerical checking
===========================

rocBLAS provides the environment variable ``ROCBLAS_CHECK_NUMERICS``, which allows users to debug numerical abnormalities.
Setting a value for ``ROCBLAS_CHECK_NUMERICS`` enables checks on the input and the output vectors/matrices
of the rocBLAS functions for NaN (not-a-number), zero, infinity, and denormal/subnormal values.
Numerical checking is available for the input and the output vectors for all level-1 and level-2 functions.
In level 2 functions, only the general (ge) type input and the output matrix can be checked for numerical abnormalities.
In level 3, GEMM is the only function to have numerical checking.

.. note::

   Performance degrades when numerical checking is enabled.

``ROCBLAS_CHECK_NUMERICS`` is a bitwise OR of zero or more bit masks with the following possible values:

*  ``ROCBLAS_CHECK_NUMERICS = 0``: The variable is not set, so there is no numerical checking.

*  ``ROCBLAS_CHECK_NUMERICS = 1``: Prints a fully informative message to the console. Indicates whether the input
   and the output Matrices/Vectors have a NaN, zero, infinity, or denormal value.

*  ``ROCBLAS_CHECK_NUMERICS = 2``: Prints the result of numerical checking only if the input and the output
   Matrices/Vectors have a NaN, infinity, or denormal value.

*  ``ROCBLAS_CHECK_NUMERICS = 4``: Returns ``rocblas_status_check_numeric_fail`` status if there is a NaN, infinity, or denormal value.

*  ``ROCBLAS_CHECK_NUMERICS = 8``: Ignores denormal values if there are no NaN or infinity values present.

Here is an example showing how to use ``ROCBLAS_CHECK_NUMERICS``:

.. code-block:: bash

    ROCBLAS_CHECK_NUMERICS=4 ./rocblas-bench -f gemm -i 1 -j 0

This command returns ``rocblas_status_check_numeric_fail`` if the input and the output matrices
of a BLAS level-3 GEMM function have a NaN, infinity, or denormal value.
If there are no numerical abnormalities, then ``rocblas_status_success`` is returned.

.. note::

   In stream capture mode, all numerical checking is skipped and ``rocblas_status_success`` is returned.

===============================================
rocBLAS order of argument checking and logging
===============================================

Argument checking differs between legacy BLAS and rocBLAS.

Legacy BLAS
-------------

Legacy BLAS has two types of argument checking:

*  Error-return for an incorrect argument. Legacy BLAS implements this with a call to the function ``XERBLA``.
*  Quick-return-success when an argument allows for the subprogram to be a no-operation or a constant result.

Level-2 and Level-3 BLAS subprograms have both error-return and quick-return-success.
Level-1 BLAS subprograms have only quick-return-success.

rocBLAS
--------

For a full list of error return codes, see the ``rocblas_status`` enumeration in :ref:`api-datatypes`.

*  ``rocblas_status_invalid_handle``: If the handle is a NULL pointer.
*  ``rocblas_status_invalid_size``: For an invalid size, increment, or leading dimension argument.
*  ``rocblas_status_invalid_value``: For unsupported enum values.
*  ``rocblas_status_success``: For quick-return-success.
*  ``rocblas_status_invalid_pointer``: For NULL argument pointers.

Differences between rocBLAS and legacy BLAS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

rocBLAS has the following differences from legacy BLAS:

*  It has a C API, returning a ``rocblas_status`` type indicating the success of the call.
*  In rocBLAS, a pointer to a scalar return value is passed as the last argument.
   In legacy BLAS, the following functions return a scalar result: ``dot``, ``nrm2``, ``asum``, ``amax``, and ``amin``.
*  The first argument is a ``rocblas_handle`` argument. This is an opaque pointer to rocBLAS resources,
   corresponding to a single HIP stream.
*  Scalar arguments like alpha and beta are pointers on either the host or device, controlled by the pointer mode of the rocBLAS handle.
   In cases where the other arguments do not dictate an early return, if the alpha and beta pointers are ``NULL``,
   the function returns ``rocblas_status_invalid_pointer``.
*  Vector and matrix arguments are always pointers to device memory.
*  When ``rocblas_pointer_mode == rocblas_pointer_mode_host``, the alpha and beta values are inspected. Based on their
   values, a decision is made regarding which vector and matrix pointers must be dereferenced.
   If any of the dereferenced pointers is a NULL pointer, ``rocblas_status_invalid_pointer`` is returned.
*  If ``rocblas_pointer_mode == rocblas_pointer_mode_device``, rocBLAS does NOT check if the vector or matrix pointers will dereference a NULL pointer.
   This is to avoid slowing down execution to fetch and inspect alpha and beta values.
*  The ``ROCBLAS_LAYER`` environment variable controls the option to log argument values.
*  rocBLAS has added functionality, including the following:

   *  batched
   *  strided_batched
   *  mixed precision in ``gemm_ex``, ``gemm_batched_ex``, and ``gemm_strided_batched_ex``

The following changes were made to accommodate the new features:

*  Changes to the logging functionality. See the Logging section below for more details.
*  For batched and strided_batched L2 and L3 functions, there is a quick-return-success for ``batch_count == 0``
   and an invalid-size error for ``batch_count < 0``.
*  For batched and strided_batched L1 functions, there is a quick-return-success for ``batch_count <= 0``.
*  When ``rocblas_pointer_mode == rocblas_pointer_mode_device``, alpha and beta are not copied
   from the device to host for quick-return-success checks. In this case, the quick-return-success checks are omitted.
   This still provides a correct result, but the operation is slower.
*  For strided_batched functions, there is no argument checking for the stride.
   To access elements in a strided_batched_matrix, for example, the C matrix in GEMM, the zero-based index is calculated
   as ``i1 + i2 * ldc + i3 * stride_c``, where ``i1 = 0, 1, 2, ..., m-1``, ``i2 = 0, 1, 2, ..., n-1``, and ``i3 = 0, 1, 2, ..., batch_count -1``.
   An incorrect stride can result in a core dump due a segmentation fault.
   It can also produce an indeterminate result if there is a memory overlap in the output matrix between different values of ``i3``.

Device memory size queries
--------------------------

The following details apply when performing queries on the memory size:

*  When ``handle->is_device_memory_size_query()`` is ``true``, the call is a device memory size query, not a normal call.
*  No logging should be performed during device memory size queries.
*  If the rocBLAS kernel doesn't require temporary device memory, the macro ``RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle)`` can be called after checking that ``handle != nullptr``.
*  If the rocBLAS kernel requires temporary device memory, then it should be set, and the kernel returned, by calling ``return handle->set_optimal_device_memory_size(size...)``,
   where ``size...`` is a list of one or more sizes for different sub-problems. The sizes are rounded up and added.

Logging
--------

There is logging before a quick-return-success or error-return, except under these circumstances:

*  When ``handle == nullptr``, it returns ``rocblas_status_invalid_handle``.
*  When ``handle->is_device_memory_size_query()``, it returns ``true``.

Vectors and matrices are logged with their addresses and are always on device memory. Scalar values in device memory are logged
as their addresses. Scalar values in host memory are logged as their values, with a ``nullptr`` logged
as ``NaN`` (``std::numeric_limits<T>::quiet_NaN()``).

rocBLAS control flow
--------------------

1. If ``handle == nullptr``, then return ``rocblas_status_invalid_handle``.

2. If the function does not require temporary device memory, then call the macro ``RETURN_ZERO_DEVICE_MEMORY_SIZE_IF_QUERIED(handle);``.

3. If the function requires temporary device memory and ``handle->is_device_memory_size_query()`` is ``true``, then validate
   any pointers and arguments required to determine the optimal size of temporary device memory.
   Return ``rocblas_status_invalid_pointer`` or ``rocblas_status_invalid_size`` if the arguments are invalid.
   Otherwise return ``handle->set_optimal_device_memory_size(size...);``, where ``size...`` is a list of one or more sizes of
   temporary buffers, which are allocated later using ``handle->device_malloc(size...)``.

4. Perform logging if it is enabled, ensuring not to dereference ``nullptr`` arguments.

5. Check for unsupported enum values. Return ``rocblas_status_invalid_value`` if an enum value is invalid.

6. Check for invalid sizes. Return ``rocblas_status_invalid_size`` if the size arguments are invalid.

7. Return ``rocblas_status_invalid_pointer`` if any pointers used to determine quick return conditions are ``NULL``.

8. If the quick return conditions are met:

   *  If there is no return value, return ``rocblas_status_success``.
   *  If there is a return value:

      * If the return value pointer argument is a ``NULL`` pointer, return ``rocblas_status_invalid_pointer``.
      * Otherwise, return ``rocblas_status_success``

9.  If any pointers not checked in step 7 are ``NULL`` but must be dereferenced, return ``rocblas_status_invalid_pointer``.
    Only when ``rocblas_pointer_mode == rocblas_pointer_mode_host`` can it be efficiently determined whether some vector/matrix arguments
    must be dereferenced.

10. (Optional) Allocate device memory, returning ``rocblas_status_memory_error`` if the allocation fails.

11. If all the checks above pass, launch the kernel and return ``rocblas_status_success``.


Legacy L1 BLAS "single vector"
-------------------------------

Below are four code snippets from NETLIB for "single vector" legacy Level-1 BLAS. They have quick-return-success for
``(n <= 0) || (incx <= 0)``:

.. code-block:: bash

      DOUBLE PRECISION FUNCTION DASUM(N,DX,INCX)
      IF (N.LE.0 .OR. INCX.LE.0) RETURN

      DOUBLE PRECISION FUNCTION DNRM2(N,X,INCX)
      IF (N.LT.1 .OR. INCX.LT.1) THEN
          return = ZERO

      SUBROUTINE DSCAL(N,DA,DX,INCX)
      IF (N.LE.0 .OR. INCX.LE.0) RETURN

      INTEGER FUNCTION IDAMAX(N,DX,INCX)
      IDAMAX = 0
      IF (N.LT.1 .OR. INCX.LE.0) RETURN
      IDAMAX = 1
      IF (N.EQ.1) RETURN

Legacy L1 BLAS "two vector"
---------------------------

Below are seven legacy Level-1 BLAS codes from NETLIB. They have quick-return-success for ``(n <= 0)``.
In addition, for ``DAXPY``, there is quick-return-success for ``(alpha == 0)``:

.. code-block::

      SUBROUTINE DAXPY(N,alpha,DX,INCX,DY,INCY)
      IF (N.LE.0) RETURN
      IF (alpha.EQ.0.0d0) RETURN

      SUBROUTINE DCOPY(N,DX,INCX,DY,INCY)
      IF (N.LE.0) RETURN

      DOUBLE PRECISION FUNCTION DDOT(N,DX,INCX,DY,INCY)
      IF (N.LE.0) RETURN

      SUBROUTINE DROT(N,DX,INCX,DY,INCY,C,S)
      IF (N.LE.0) RETURN

      SUBROUTINE DSWAP(N,DX,INCX,DY,INCY)
      IF (N.LE.0) RETURN

      DOUBLE PRECISION FUNCTION DSDOT(N,SX,INCX,SY,INCY)
      IF (N.LE.0) RETURN

      SUBROUTINE DROTM(N,DX,INCX,DY,INCY,DPARAM)
      DFLAG = DPARAM(1)
      IF (N.LE.0 .OR. (DFLAG+TWO.EQ.ZERO)) RETURN

Legacy L2 BLAS
-----------------

Below are the code snippets from NETLIB for legacy Level-2 BLAS. They have both argument checking and quick-return-success:

.. code-block::

      SUBROUTINE DGER(M,N,ALPHA,X,INCX,Y,INCY,A,LDA)
      INFO = 0
      IF (M.LT.0) THEN
          INFO = 1
      ELSE IF (N.LT.0) THEN
          INFO = 2
      ELSE IF (INCX.EQ.0) THEN
          INFO = 5
      ELSE IF (INCY.EQ.0) THEN
          INFO = 7
      ELSE IF (LDA.LT.MAX(1,M)) THEN
          INFO = 9
      END IF
      IF (INFO.NE.0) THEN
          CALL XERBLA('DGER  ',INFO)
          RETURN
      END IF

      IF ((M.EQ.0) .OR. (N.EQ.0) .OR. (ALPHA.EQ.ZERO)) RETURN

.. code-block::

      SUBROUTINE DSYR(UPLO,N,ALPHA,X,INCX,A,LDA)

      INFO = 0
      IF (.NOT.LSAME(UPLO,'U') .AND. .NOT.LSAME(UPLO,'L')) THEN
          INFO = 1
      ELSE IF (N.LT.0) THEN
          INFO = 2
      ELSE IF (INCX.EQ.0) THEN
          INFO = 5
      ELSE IF (LDA.LT.MAX(1,N)) THEN
          INFO = 7
      END IF
      IF (INFO.NE.0) THEN
          CALL XERBLA('DSYR  ',INFO)
          RETURN
      END IF

      IF ((N.EQ.0) .OR. (ALPHA.EQ.ZERO)) RETURN

.. code-block::

      SUBROUTINE DGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)

      INFO = 0
      IF (.NOT.LSAME(TRANS,'N') .AND. .NOT.LSAME(TRANS,'T') .AND. .NOT.LSAME(TRANS,'C')) THEN
          INFO = 1
      ELSE IF (M.LT.0) THEN
          INFO = 2
      ELSE IF (N.LT.0) THEN
          INFO = 3
      ELSE IF (LDA.LT.MAX(1,M)) THEN
          INFO = 6
      ELSE IF (INCX.EQ.0) THEN
          INFO = 8
      ELSE IF (INCY.EQ.0) THEN
          INFO = 11
      END IF
      IF (INFO.NE.0) THEN
          CALL XERBLA('DGEMV ',INFO)
          RETURN
      END IF

      IF ((M.EQ.0) .OR. (N.EQ.0) .OR. ((ALPHA.EQ.ZERO).AND. (BETA.EQ.ONE))) RETURN

.. code-block::

      SUBROUTINE DTRSV(UPLO,TRANS,DIAG,N,A,LDA,X,INCX)

      INFO = 0
      IF (.NOT.LSAME(UPLO,'U') .AND. .NOT.LSAME(UPLO,'L')) THEN
          INFO = 1
      ELSE IF (.NOT.LSAME(TRANS,'N') .AND. .NOT.LSAME(TRANS,'T') .AND. .NOT.LSAME(TRANS,'C')) THEN
          INFO = 2
      ELSE IF (.NOT.LSAME(DIAG,'U') .AND. .NOT.LSAME(DIAG,'N')) THEN
          INFO = 3
      ELSE IF (N.LT.0) THEN
          INFO = 4
      ELSE IF (LDA.LT.MAX(1,N)) THEN
          INFO = 6
      ELSE IF (INCX.EQ.0) THEN
          INFO = 8
      END IF
      IF (INFO.NE.0) THEN
          CALL XERBLA('DTRSV ',INFO)
          RETURN
      END IF

      IF (N.EQ.0) RETURN

Legacy L3 BLAS
----------------

Below is a code snippet from NETLIB for legacy Level-3 BLAS dgemm. It has both argument checking and quick-return-success:

.. code-block::

      SUBROUTINE DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)

      NOTA = LSAME(TRANSA,'N')
      NOTB = LSAME(TRANSB,'N')
      IF (NOTA) THEN
          NROWA = M
          NCOLA = K
      ELSE
          NROWA = K
          NCOLA = M
      END IF
      IF (NOTB) THEN
          NROWB = K
      ELSE
          NROWB = N
      END IF

  //  Test the input parameters.

      INFO = 0
      IF ((.NOT.NOTA) .AND. (.NOT.LSAME(TRANSA,'C')) .AND.
     +    (.NOT.LSAME(TRANSA,'T'))) THEN
          INFO = 1
      ELSE IF ((.NOT.NOTB) .AND. (.NOT.LSAME(TRANSB,'C')) .AND.
     +         (.NOT.LSAME(TRANSB,'T'))) THEN
          INFO = 2
      ELSE IF (M.LT.0) THEN
          INFO = 3
      ELSE IF (N.LT.0) THEN
          INFO = 4
      ELSE IF (K.LT.0) THEN
          INFO = 5
      ELSE IF (LDA.LT.MAX(1,NROWA)) THEN
          INFO = 8
      ELSE IF (LDB.LT.MAX(1,NROWB)) THEN
          INFO = 10
      ELSE IF (LDC.LT.MAX(1,M)) THEN
          INFO = 13
      END IF
      IF (INFO.NE.0) THEN
          CALL XERBLA('DGEMM ',INFO)
          RETURN
      END IF

  //  Quick return if possible.

      IF ((M.EQ.0) .OR. (N.EQ.0) .OR. (((ALPHA.EQ.ZERO).OR. (K.EQ.0)).AND. (BETA.EQ.ONE))) RETURN

.. raw:: latex

    \newpage

=================================
rocBLAS benchmarking and testing
=================================

There are three client executables that can be used with rocBLAS. They are:

*  rocblas-bench
*  rocblas-gemm-tune
*  rocblas-test

To build these clients, follow the instructions in the :doc:`../install/Linux_Install_Guide` or :doc:`../install/Windows_Install_Guide` guides.
After the build, the rocBLAS clients can be found in the ``rocBLAS/build/release/clients/staging`` directory.

.. note::

   The ``rocblas-bench`` and ``rocblas-test`` executables use AMD's ILP64 version of AOCL-BLAS 4.2 as the host
   reference BLAS to verify correctness. However, there is a known issue with multiple threads
   in AOCL-BLAS that can cause these executables to hang.
   If the number of threads matches the total number of CPU threads, thread oversubscription can occur, which causes the process to hang.

   To prevent this issue, the number of threads used the AOCL-BLAS library should be smaller than the number of available CPU cores.
   You can configure this setting using the ``OMP_NUM_THREADS`` environment variable.
   For example, on a server with 32 cores, limit the number of threads to 28 by setting ``export OMP_NUM_THREADS=28``.

The next three sections provide a brief explanation of each rocBLAS client and how to use it.

rocblas-bench
--------------

rocblas-bench is used to measure performance and verify the correctness of rocBLAS functions.

It includes a command line interface. For more information, run this command:

.. code-block:: bash

   rocBLAS/build/release/clients/staging/rocblas-bench --help

The following table lists all the data types in rocBLAS:

.. list-table:: Data types in rocBLAS
   :widths: 25 25
   :header-rows: 1

   * - Data type
     - acronym
   * - Real 16-bit brain floating point
     - bf16_r
   * - Real half
     - f16_r (h)
   * - Real float
     - f32_r (s)
   * - Real double
     - f64_r (d)
   * - Complex float
     - f32_c (c)
   * - Complex double
     - f64_c (z)
   * - Integer 32
     - i32_r
   * - Integer 8
     - i8_r

All options for problem types in rocBLAS for GEMM are shown here:

*  N: Not transposed
*  T: Transposed
*  C: Complex conjugate (for a real data type, C is the same as T)


.. list-table:: Various matrix operations
   :widths: 25 25 25
   :header-rows: 1

   * - Problem types
     - problem_type
     - Data type
   * - NN
     - Cijk_Ailk_Bljk
     - real/complex
   * - NT
     - Cijk_Ailk_Bjlk
     - real/complex
   * - TN
     - Cijk_Alik_Bljk
     - real/complex
   * - TT
     - Cijk_Alik_Bjlk
     - real/complex
   * - NC
     - Cijk_Ailk_BjlkC
     - complex
   * - CN
     - Cijk_AlikC_Bljk
     - complex
   * - CC
     - Cijk_AlikC_BjlkC
     - complex
   * - TC
     - Cijk_Alik_BjlkC
     - complex
   * - CT
     - Cijk_AlikC_Bjlk
     - complex

For example, NT means A * B\ :sup:`T`\.

GEMM functions can be divided into two main categories.

*  HPA functions (HighPrecisionAccumulate) where the compute data type is different from the input data type (A/B):
   All HPA functions must be called using the ``gemm_ex`` API in rocblas-bench (and not ``gemm``).
   The ``gemm_ex`` function name consists of three letters: A/B data type, C/D data type, and compute data type.

*  Non-HPA functions where the input (A/B), output (C/D), and compute data types are all the same:
   Non-HPA cases can be called using ``gemm`` or ``gemm_ex`` but ``gemm`` is recommended.

The following table shows all possible GEMM functions in rocBLAS.

.. list-table:: All GEMM functions in rocBLAS
   :widths: 20 30 10 10 10
   :header-rows: 1

   * - Function
     - Kernel name
     - A/B data type
     - C/D data type
     - Compute data type
   * - hgemm
     - <arch>_<problem_type>_HB
     - f16_r
     - f16_r
     - f16_r
   * - hgemm_batched
     - <arch>_<problem_type>_HB_GB
     - f16_r
     - f16_r
     - f16_r
   * - hgemm_strided_batched
     - <arch>_<problem_type>_HB
     - f16_r
     - f16_r
     - f16_r
   * - sgemm
     - <arch>_<problem_type>_SB
     - f32_r
     - f32_r
     - f32_r
   * - sgemm_batched
     - <arch>_<problem_type>_SB_GB
     - f32_r
     - f32_r
     - f32_r
   * - sgemm_strided_batched
     - <arch>_<problem_type>_SB
     - f32_r
     - f32_r
     - f32_r
   * - dgemm
     - <arch>_<problem_type>_DB
     - f64_r
     - f64_r
     - f64_r
   * - dgemm_batched
     - <arch>_<problem_type>_DB_GB
     - f64_r
     - f64_r
     - f64_r
   * - dgemm_strided_batched
     - <arch>_<problem_type>_DB
     - f64_r
     - f64_r
     - f64_r
   * - cgemm
     - <arch>_<problem_type>_CB
     - f32_c
     - f32_c
     - f32_c
   * - cgemm_batched
     - <arch>_<problem_type>_CB_GB
     - f32_c
     - f32_c
     - f32_c
   * - cgemm_strided_batched
     - <arch>_<problem_type>_CB
     - f32_c
     - f32_c
     - f32_c
   * - zgemm
     - <arch>_<problem_type>_ZB
     - f64_c
     - f64_c
     - f64_c
   * - zgemm_batched
     - <arch>_<problem_type>_ZB_GB
     - f64_c
     - f64_c
     - f64_c
   * - zgemm_strided_batched
     - <arch>_<problem_type>_ZB
     - f64_c
     - f64_c
     - f64_c
   * - HHS
     - <arch>_<problem_type>_HHS_BH
     - f16_r
     - f16_r
     - f32_r
   * - HHS_batched
     - <arch>_<problem_type>_HHS_BH_GB
     - f16_r
     - f16_r
     - f32_r
   * - HHS_strided_batched
     - <arch>_<problem_type>_HHS_BH
     - f16_r
     - f16_r
     - f32_r
   * - HSS
     - <arch>_<problem_type>_HSS_BH
     - f16_r
     - f32_r
     - f32_r
   * - HSS_batched
     - <arch>_<problem_type>_HSS_BH_GB
     - f16_r
     - f32_r
     - f32_r
   * - HSS_strided_batched
     - <arch>_<problem_type>_HSS_BH
     - f16_r
     - f32_r
     - f32_r
   * - BBS
     - <arch>_<problem_type>_BBS_BH
     - bf16_r
     - bf16_r
     - f32_r
   * - BBS_batched
     - <arch>_<problem_type>_BBS_BH_GB
     - bf16_r
     - bf16_r
     - f32_r
   * - BBS_strided_batched
     - <arch>_<problem_type>_BBS_BH
     - bf16_r
     - bf16_r
     - f32_r
   * - BSS
     - <arch>_<problem_type>_BSS_BH
     - bf16_r
     - f32_r
     - f32_r
   * - BSS_batched
     - <arch>_<problem_type>_BSS_BH_GB
     - bf16_r
     - f32_r
     - f32_r
   * - BSS_strided_batched
     - <arch>_<problem_type>_BSS_BH
     - bf16_r
     - f32_r
     - f32_r
   * - I8II
     - <arch>_<problem_type>_I8II_BH
     - I8
     - I
     - I
   * - I8II_batched
     - <arch>_<problem_type>_I8II_BH_GB
     - I8
     - I
     - I
   * - I8II_strided_batched
     - <arch>_<problem_type>_I8II_BH
     - I8
     - I
     - I

Benchmarking the performance of a GEMM function using rocblas-bench
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This method is only recommended to test a few sizes. Otherwise, refer to the next section.
The following listing shows how to configure rocblas-bench to call each of the GEMM functions:

Non-HPA cases (``gemm``)

.. code-block:: bash

   #dgemm
   $ ./rocblas-bench -f gemm --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r d --lda 1024 --ldb 2048 --ldc 1024 --ldd 1024 --alpha 1.1 --beta 1.0
   # dgemm batched
   $ ./rocblas-bench -f gemm_batched --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r d --lda 1024 --ldb 2048 --ldc 1024 --ldd 1024 --alpha 1.1 --beta 1 --batch_count 5
   # dgemm strided batched
   $ ./rocblas-bench -f gemm_strided_batched --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r d --lda 1024 --stride_a 4096 --ldb 2048 --stride_b 4096 --ldc 1024 --stride_c 2097152 --ldd 1024 --stride_d 2097152 --alpha 1.1 --beta 1 --batch_count 5

   # sgemm
   $ ./rocblas-bench -f gemm --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r s --lda 1024 --ldb 2048 --ldc 1024 --ldd 1024 --alpha 1.1 --beta 1
   # sgemm batched
   $ ./rocblas-bench -f gemm_batched --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r s --lda 1024 --ldb 2048 --ldc 1024 --ldd 1024 --alpha 1.1 --beta 1 --batch_count 5
   # sgemm strided batched
   $ ./rocblas-bench -f gemm_strided_batched --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r s --lda 1024 --stride_a 4096 --ldb 2048 --stride_b 4096 --ldc 1024 --stride_c 2097152 --ldd 1024 --stride_d 2097152 --alpha 1.1 --beta 1 --batch_count 5

   # hgemm (this function is not really very fast. Use HHS instead, which is faster and more accurate)
   $ ./rocblas-bench -f gemm --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r h --lda 1024 --ldb 2048 --ldc 1024 --ldd 1024 --alpha 1.1 --beta 1
   # hgemm batched
   $ ./rocblas-bench -f gemm_batched --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r h --lda 1024 --ldb 2048 --ldc 1024 --ldd 1024 --alpha 1.1 --beta 1 --batch_count 5
   # hgemm strided batched
   $ ./rocblas-bench -f gemm_strided_batched --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r h --lda 1024 --stride_a 4096 --ldb 2048 --stride_b 4096 --ldc 1024 --stride_c 2097152 --ldd 1024 --stride_d 2097152 --alpha 1.1 --beta 1 --batch_count 5

   # cgemm
   $ ./rocblas-bench -f gemm --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r c --lda 1024 --ldb 2048 --ldc 1024 --ldd 1024 --alpha 1.1 --beta 1
   # cgemm batched
   $ ./rocblas-bench -f gemm_batched --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r c --lda 1024 --ldb 2048 --ldc 1024 --ldd 1024 --alpha 1.1 --beta 1 --batch_count 5
   # cgemm strided batched
   $ ./rocblas-bench -f gemm_strided_batched --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r c --lda 1024 --stride_a 4096 --ldb 2048 --stride_b 4096 --ldc 1024 --stride_c 2097152 --ldd 1024 --stride_d 2097152 --alpha 1.1 --beta 1 --batch_count 5

   # zgemm
   $ ./rocblas-bench -f gemm --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r z --lda 1024 --ldb 2048 --ldc 1024 --ldd 1024 --alpha 1.1 --beta 1
   # zgemm batched
   $ ./rocblas-bench -f gemm_batched --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r z --lda 1024 --ldb 2048 --ldc 1024 --ldd 1024 --alpha 1.1 --beta 1 --batch_count 5
   # zgemm strided batched
   $ ./rocblas-bench -f gemm_strided_batched --transposeA N --transposeB T -m 1024 -n 2048 -k 512 -r z --lda 1024 --stride_a 4096 --ldb 2048 --stride_b 4096 --ldc 1024 --stride_c 2097152 --ldd 1024 --stride_d 2097152 --alpha 1.1 --beta 1 --batch_count 5

   # cgemm (NC)
   $ ./rocblas-bench -f gemm --transposeA N --transposeB C -m 1024 -n 2048 -k 512 -r c --lda 1024 --ldb 2048 --ldc 1024 --ldd 1024 --alpha 1.1 --beta 1
   # cgemm batched (NC)
   $ ./rocblas-bench -f gemm_batched --transposeA N --transposeB C -m 1024 -n 2048 -k 512 -r c --lda 1024 --ldb 2048 --ldc 1024 --ldd 1024 --alpha 1.1 --beta 1 --batch_count 5
   # cgemm strided batched (NC)
   $ ./rocblas-bench -f gemm_strided_batched --transposeA N --transposeB C -m 1024 -n 2048 -k 512 -r c --lda 1024 --stride_a 4096 --ldb 2048 --stride_b 4096 --ldc 1024 --stride_c 2097152 --ldd 1024 --stride_d 2097152 --alpha 1.1 --beta 1 --batch_count 5


HPA cases (``gemm_ex``)

.. code-block:: bash

   # HHS
   $ ./rocblas-bench -f gemm_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type h --lda 1024 --b_type h --ldb 2048 --c_type h --ldc 1024 --d_type h --ldd 1024 --compute_type s --alpha 1.1 --beta 1
   # HHS batched
   $ ./rocblas-bench -f gemm_batched_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type h --lda 1024 --b_type h --ldb 2048 --c_type h --ldc 1024 --d_type h --ldd 1024 --compute_type s --alpha 1.1 --beta 1 --batch_count 5
   # HHS strided batched
   $ ./rocblas-bench -f gemm_strided_batched_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type h --lda 1024 --stride_a 4096 --b_type h --ldb 2048 --stride_b 4096 --c_type h --ldc 1024 --stride_c 2097152 --d_type h --ldd 1024 --stride_d 2097152 --compute_type s --alpha 1.1 --beta 1 --batch_count 5

   # HSS
   $ ./rocblas-bench -f gemm_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type h --lda 1024 --b_type h --ldb 2048 --c_type s --ldc 1024 --d_type s --ldd 1024 --compute_type s --alpha 1.1 --beta 1
   # HSS batched
   $ ./rocblas-bench -f gemm_batched_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type h --lda 1024 --b_type h --ldb 2048 --c_type s --ldc 1024 --d_type s --ldd 1024 --compute_type s --alpha 1.1 --beta 1 --batch_count 5
   # HSS strided batched
   $ ./rocblas-bench -f gemm_strided_batched_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type h --lda 1024 --stride_a 4096 --b_type h --ldb 2048 --stride_b 4096 --c_type s --ldc 1024 --stride_c 2097152 --d_type s --ldd 1024 --stride_d 2097152 --compute_type s --alpha 1.1 --beta 1 --batch_count 5

   # BBS
   $ ./rocblas-bench -f gemm_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type bf16_r --lda 1024 --b_type bf16_r --ldb 2048 --c_type bf16_r --ldc 1024 --d_type bf16_r --ldd 1024 --compute_type s --alpha 1.1 --beta 1
   # BBS batched
   $ ./rocblas-bench -f gemm_batched_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type bf16_r --lda 1024 --b_type bf16_r --ldb 2048 --c_type bf16_r --ldc 1024 --d_type bf16_r --ldd 1024 --compute_type s --alpha 1.1 --beta 1 --batch_count 5
   # BBS strided batched
   $ ./rocblas-bench -f gemm_strided_batched_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type bf16_r --lda 1024 --stride_a 4096 --b_type bf16_r --ldb 2048 --stride_b 4096 --c_type bf16_r --ldc 1024 --stride_c 2097152 --d_type bf16_r --ldd 1024 --stride_d 2097152 --compute_type s --alpha 1.1 --beta 1 --batch_count 5

   # BSS
   $ ./rocblas-bench -f gemm_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type bf16_r --lda 1024 --b_type bf16_r --ldb 2048 --c_type s --ldc 1024 --d_type s --ldd 1024 --compute_type s --alpha 1.1 --beta 1
   # BSS batched
   $ ./rocblas-bench -f gemm_batched_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type bf16_r --lda 1024 --b_type bf16_r --ldb 2048 --c_type s --ldc 1024 --d_type s --ldd 1024 --compute_type s --alpha 1.1 --beta 1 --batch_count 5
   # BSS strided batched
   $ ./rocblas-bench -f gemm_strided_batched_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type bf16_r --lda 1024 --stride_a 4096 --b_type bf16_r --ldb 2048 --stride_b 4096 --c_type s --ldc 1024 --stride_c 2097152 --d_type s --ldd 1024 --stride_d 2097152 --compute_type s --alpha 1.1 --beta 1 --batch_count 5

   # I8II
   $ ./rocblas-bench -f gemm_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type i8_r --lda 1024 --b_type i8_r --ldb 2048 --c_type i32_r --ldc 1024 --d_type i32_r --ldd 1024 --compute_type i32_r --alpha 1.1 --beta 1
   # I8II batched
   $ ./rocblas-bench -f gemm_batched_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type i8_r --lda 1024 --b_type i8_r --ldb 2048 --c_type i32_r --ldc 1024 --d_type i32_r --ldd 1024 --compute_type i32_r --alpha 1.1 --beta 1 --batch_count 5
   # I8II strided batched
   $ ./rocblas-bench -f gemm_strided_batched_ex --transposeA N --transposeB T -m 1024 -n 2048 -k 512 --a_type i8_r --lda 1024 --stride_a 4096 --b_type i8_r --ldb 2048 --stride_b 4096 --c_type i32_r --ldc 1024 --stride_c 2097152 --d_type i32_r --ldd 1024 --stride_d 2097152 --compute_type i32_r --alpha 1.1 --beta 1 --batch_count 5

Setting rocblas-bench parameters in a YAML file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To benchmark many sizes, it is recommended to use rocblas-bench with the batch call to eliminate the latency
involved with loading the GEMM library that rocBLAS links to.
The batch call takes a YAML file with a list of all problem sizes.
You can have multiple sizes of different types in one YAML file.
The benchmark setting is different from the direct call to rocblas-bench.
A sample setting for each function is listed below. After you have created the YAML file, benchmark the sizes as follows:

.. code-block:: bash

   rocBLAS/build/release/clients/staging/rocblas-bench --yaml problem-sizes.yaml


Here are the configurations for each function:


Non-HPA cases (``gemm``)

.. code-block:: bash

    # dgemm
    - { rocblas_function: "rocblas_dgemm",         transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10  }
    # dgemm batched
    - { rocblas_function: "rocblas_dgemm_batched", transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5  }
    # dgemm strided batched
    - { rocblas_function: "rocblas_dgemm_strided_batched", transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5, stride_a: 4096, stride_b: 4096, stride_c: 2097152, stride_d: 2097152 }

    # sgemm
    - { rocblas_function: "rocblas_sgemm",         transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10  }
    # sgemm batched
    - { rocblas_function: "rocblas_sgemm_batched", transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5  }
    # sgemm strided batched
    - { rocblas_function: "rocblas_sgemm_strided_batched", transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5, stride_a: 4096, stride_b: 4096, stride_c: 2097152, stride_d: 2097152 }

    # hgemm
    - { rocblas_function: "rocblas_hgemm",         transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10  }
    # hgemm batched
    - { rocblas_function: "rocblas_hgemm_batched", transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5  }
    # hgemm strided batched
    - { rocblas_function: "rocblas_hgemm_strided_batched", transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5, stride_a: 4096, stride_b: 4096, stride_c: 2097152, stride_d: 2097152 }

    # cgemm
    - { rocblas_function: "rocblas_cgemm",         transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10  }
    # cgemm batched
    - { rocblas_function: "rocblas_cgemm_batched", transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5  }
    # cgemm strided batched
    - { rocblas_function: "rocblas_cgemm_strided_batched", transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5, stride_a: 4096, stride_b: 4096, stride_c: 2097152, stride_d: 2097152 }

    # zgemm
    - { rocblas_function: "rocblas_zgemm",         transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10  }
    # zgemm batched
    - { rocblas_function: "rocblas_zgemm_batched", transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5  }
    # zgemm strided batched
    - { rocblas_function: "rocblas_zgemm_strided_batched", transA: "N", transB: "T", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5, stride_a: 4096, stride_b: 4096, stride_c: 2097152, stride_d: 2097152 }

    # cgemm
    - { rocblas_function: "rocblas_cgemm",         transA: "N", transB: "C", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10  }
    # cgemm batched
    - { rocblas_function: "rocblas_cgemm_batched", transA: "N", transB: "C", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5  }
    # cgemm strided batched
    - { rocblas_function: "rocblas_cgemm_strided_batched", transA: "N", transB: "C", M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5, stride_a: 4096, stride_b: 4096, stride_c: 2097152, stride_d: 2097152 }


HPA cases (``gemm_ex``)

.. code-block:: bash

    # HHS
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: f16_r, b_type: f16_r, c_type: f16_r, d_type: f16_r, compute_type: f32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10  }
    # HHS batched
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: f16_r, b_type: f16_r, c_type: f16_r, d_type: f16_r, compute_type: f32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5  }
    # HHS strided batched
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: f16_r, b_type: f16_r, c_type: f16_r, d_type: f16_r, compute_type: f32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5, stride_a: 4096, stride_b: 4096, stride_c: 2097152, stride_d: 2097152 }

    # HSS
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: f16_r, b_type: f16_r, c_type: f16_r, d_type: f16_r, compute_type: f32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10  }
    # HSS batched
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: f16_r, b_type: f16_r, c_type: f32_r, d_type: f32_r, compute_type: f32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5  }
    # HSS strided batched
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: f16_r, b_type: f16_r, c_type: f32_r, d_type: f32_r, compute_type: f32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5, stride_a: 4096, stride_b: 4096, stride_c: 2097152, stride_d: 2097152 }

    # BBS
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: bf16_r, b_type: bf16_r, c_type: bf16_r, d_type: bf16_r, compute_type: f32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10  }
    # BBS batched
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: bf16_r, b_type: bf16_r, c_type: bf16_r, d_type: bf16_r, compute_type: f32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5  }
    # BBS strided batched
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: bf16_r, b_type: bf16_r, c_type: bf16_r, d_type: bf16_r, compute_type: f32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5, stride_a: 4096, stride_b: 4096, stride_c: 2097152, stride_d: 2097152 }

    # BSS
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: bf16_r, b_type: bf16_r, c_type: f32_r, d_type: f32_r, compute_type: f32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10  }
    # BSS batched
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: bf16_r, b_type: bf16_r, c_type: f32_r, d_type: f32_r, compute_type: f32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5  }
    # BSS strided batched
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: bf16_r, b_type: bf16_r, c_type: f32_r, d_type: f32_r, compute_type: f32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5, stride_a: 4096, stride_b: 4096, stride_c: 2097152, stride_d: 2097152 }

    # I8II
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: i8_r, b_type: i8_r, c_type: i32_r, d_type: i32_r, compute_type: i32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10  }
    # I8II batched
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: i8_r, b_type: i8_r, c_type: i32_r, d_type: i32_r, compute_type: i32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5  }
    # I8II strided batched
    - { rocblas_function: "rocblas_gemm_ex", transA: "N", transB: "T", a_type: i8_r, b_type: i8_r, c_type: i32_r, d_type: i32_r, compute_type: i32_r, M:    1024, N:    2048, K:    512, lda:   1024, ldb:   2048, ldc:   1024,  ldd:   1024, cold_iters: 2, iters: 10, batch_count: 5, stride_a: 4096, stride_b: 4096, stride_c: 2097152, stride_d: 2097152 }


For example, the performance of SGEMM using rocblas-bench on an AMD vega20 machine returns the following:

.. code-block:: bash

   ./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 4096 -n 4096 -k 4096 --alpha 1 --lda 4096 --ldb 4096 --beta 0 --ldc 4096
   transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,rocblas-Gflops,us
   N,N,4096,4096,4096,1,4096,4096,0,4096,11941.5,11509.4

A useful way of finding the parameters that can be used with ``./rocblas-bench -f gemm`` is to turn on logging
by setting the environment variable ``ROCBLAS_LAYER=2``. For example, if the user runs:

.. code-block:: bash

   ROCBLAS_LAYER=2 ./rocblas-bench -f gemm -i 1 -j 0

The above command logs the following:

.. code-block:: bash

   ./rocblas-bench -f gemm -r f32_r --transposeA N --transposeB N -m 128 -n 128 -k 128 --alpha 1 --lda 128 --ldb 128 --beta 0 --ldc 128

The user can copy and change the above command. For example, to change the datatype to IEEE-64 bit and the size to 2048, use the following:

.. code-block:: bash

   ./rocblas-bench -f gemm -r f64_r --transposeA N --transposeB N -m 2048 -n 2048 -k 2048 --alpha 1 --lda 2048 --ldb 2048 --beta 0 --ldc 2048

To measure performance on the ILP64 API functions, when they exist, add the argument ``--api 1`` rather
than changing the function name set in ``-f``.
Logging affects performance, so only use it to log the command under evaluation,
then run the command without logging to measure performance.


.. note::

   rocblas-bench also has the flag ``-v 1`` for correctness checks.

Benchmarking special case gemv_batched and gemv_strided_batched functions using rocblas-bench
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This covers how to benchmark the performance of special case ``gemv_batched`` and ``gemv_strided_batched`` functions
for mixed precision (HSH, HSS, TST, TSS).
The following command launches rocblas-bench for ``rocblas_hshgemv_batched`` with half-precision input,
single-precision compute, and half-precision output (HSH):

.. code-block:: bash

   ./rocblas-bench -f gemv_batched --a_type f16_r --c_type f16_r --compute_type f32_r --transposeA N -m 128 -n 128 --alpha 1  --lda 128  --incx 1 --beta 1 --incy 1  --batch_count 2

For the above command, instead of using the ``-r`` parameter to specify the precision, pass three additional arguments (``a_type``, ``c_type``, and ``compute_type``) to resolve the ambiguity of using mixed-precision compute.

This mixed-precision support is only available for ``gemv_batched``, ``gemv_strided_batched``, and rocBLAS extension
functions (for example, ``axpy_ex``, ``scal_ex``, and ``gemm_ex``). For more information, see the :ref:`api-reference-guide`.


.. _rocblas_bench_stream_sync:

rocblas-bench timing
^^^^^^^^^^^^^^^^^^^^^

rocblas-bench uses ``hipEvent_t`` recording to time API calls and ignore the overhead of any ``hipStreamSynchronize`` call.
To switch back to the earlier timing approach that uses ``hipStreamSynchronize`` to ensure work completion, set the environment variable ``ROCBLAS_BENCH_STREAM_SYNC=1``.

rocblas-gemm-tune
-----------------

rocblas-gemm-tune is used to find the best performing GEMM kernel for each of a given set of GEMM problems.

It has a command line interface, which mimics the ``--yaml`` input used by rocblas-bench (see the section above for details).

To generate the expected ``--yaml`` input, you can use profile logging by setting the environment variable ``ROCBLAS_LAYER=4``.

For more information on rocBLAS logging, see :ref:`logging`.

Here is an example input file:

.. code-block:: bash

    - {'rocblas_function': 'gemm_ex', 'transA': 'N', 'transB': 'N', 'M': 320, 'N': 588, 'K': 4096, 'alpha': 1, 'a_type': 'f32_r', 'lda': 320, 'b_type': 'f32_r', 'ldb': 6144, 'beta': 0, 'c_type': 'f32_r', 'ldc': 320, 'd_type': 'f32_r', 'ldd': 320, 'compute_type': 'f32_r', 'device': 0}
    - {'rocblas_function': 'gemm_ex', 'transA': 'N', 'transB': 'N', 'M': 512, 'N': 3096, 'K': 512, 'alpha': 1, 'a_type': 'f16_r', 'lda': 512, 'b_type': 'f16_r', 'ldb': 512, 'beta': 0, 'c_type': 'f16_r', 'ldc': 512, 'd_type': 'f16_r', 'ldd': 512, 'compute_type': 'f32_r', 'device': 0}

The expected output looks like this (the selected GEMM idx might differ):

.. code-block:: bash

    transA,transB,M,N,batch_count,K,alpha,beta,lda,ldb,ldc,input_type,output_type,compute_type,solution_index
    N,N,320,588,1,4096,1,0,320,6144,320,f32_r,f32_r,f32_r,3788
    N,N,512,3096,1,512,1,0,512,512,512,f16_r,f16_r,f32_r,4546

Where the far right values (``solution_index``) are the indices of the best performing kernels for those GEMMs in the rocBLAS kernel library.
These indices can be used directly in future GEMM calls but cannot be reused across library
releases or different device architectures.

See `example_user_driven_tuning.cpp <https://github.com/ROCm/rocm-libraries/blob/develop/projects/rocblas/clients/samples/example_user_driven_tuning.cpp>`_ for
sample code showing how to use kernels directly via their indices.

If the output is stored in a file, you can use the results to override the default kernel selection
by setting the environment variable ``ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH=<path>``, where ``<path>`` points to the file.


rocblas-test
-------------

rocblas-test is used to perform rocBLAS unit tests. It uses the GoogleTest framework.

The tests are in five categories:

*  quick
*  pre_checkin
*  nightly
*  stress
*  known_bug

To run the quick tests, use the following command:

.. code-block:: bash

   ./rocblas-test --gtest_filter=*quick*

To run the other tests using the ``rocblas-test`` command, replace ``*quick*`` with ``*pre_checkin*``, ``*nightly*``, or ``*known_bug*``.

The pattern for ``--gtest_filter`` is:

.. code-block:: bash

   --gtest_filter=POSTIVE_PATTERNS[-NEGATIVE_PATTERNS]

``gtest_filter`` can also be used to run tests for a particular function and a particular set of input parameters.
For example, to run all quick tests for the function ``rocblas_saxpy``, use this example:

.. code-block:: bash

   ./rocblas-test --gtest_filter=*quick*axpy*f32_r*

The default verbosity shows test category totals and specific test failure details, matching an implicit environment
variable setting of ``GTEST_LISTENER=NO_PASS_LINE_IN_LOG``.
To generate output listing each individual test that runs, use the following command:

.. code-block:: bash

   GTEST_LISTENER=PASS_LINE_IN_LOG ./rocblas-test --gtest_filter=*quick*

``rocblas-test`` can be driven by tests specified in a YAML file using the ``--yaml`` argument.
As the pre_checkin and nightly test categories can require hours to run, a short smoke test set is provided using a YAML file.
This ``rocblas_smoke.yaml`` test set only takes a few minutes to test a few small problem sizes for every function:

.. code-block:: bash

   ./rocblas-test --yaml rocblas_smoke.yaml

The following test situations require additional consideration:

*  YAML extension for lock-step multiple variable scanning

   Both rocblas-test and rocblas-bench can use an extension to scan over multiple variables in lock step implemented by the ``Arguments`` class.
   For this purpose, set the ``Arguments`` member variable
   ``scan`` to the range to scan over and use ``*c_scan_value`` to retrieve the values. This can be used to avoid
   all combinations of YAML variable values that are normally generated,
   for example, ``- { scan: [32..256..32], M: *c_scan_value, N: *c_scan_value, lda: *c_scan_value }``.

*  Large memory tests (stress category)

   Some tests in the stress category might attempt to allocate more RAM than available.
   While these tests should automatically get skipped, in some cases, such
   as when running in a Docker container, they could result in a process termination.
   To limit the peak RAM allocation in GB, use this environment variable:

   .. code-block:: bash

      ROCBLAS_CLIENT_RAM_GB_LIMIT=32 ./rocblas-test --gtest_filter=*stress*

*  Long-running tests

   The rocblas-test process will be terminated if a single test takes longer than the configured timeout.
   To change the timeout, use the environment variable ``ROCBLAS_TEST_TIMEOUT``,
   which takes a value in seconds (with a default of 600 seconds):

   .. code-block:: bash

      ROCBLAS_TEST_TIMEOUT=900 ./rocblas-test --gtest_filter=*stress*

*  Debugging rocblas-test

   The rocblas-test process internally catches signals which might interfere with debugger use. To disable this feature,
   set the environment variable ``ROCBLAS_TEST_NO_SIGACTION``:

   .. code-block:: bash

      ROCBLAS_TEST_NO_SIGACTION=1 rocgdb ./rocblas-test --gtest_filter=*stress*


Adding a new rocBLAS unit test
-------------------------------

To add new data-driven tests to the rocBLAS GoogleTest Framework, follow these steps:

#. Create a C++ header file with the name ``testing_<function>.hpp`` in the
   ``include`` subdirectory, with templated functions for a specific rocBLAS
   routine. Some examples include:

   *  ``testing_gemm.hpp``
   *  ``testing_gemm_ex.hpp``

   In the new ``testing_*.hpp`` file, create a templated function which returns ``void``
   and accepts a ``const Arguments&`` parameter, for example:

   .. code-block:: cpp

      template<typename Ti, typename To, typename Tc>
      void testing_gemm_ex(const Arguments& arg)
      {
      // ...
      }

   This function is used for a YAML file-driven argument testing. It will be invoked by the dispatch code for each permutation of the YAML-driven parameters.
   Additionally, a template function for tests that handle bad arguments should be created, as follows:

   .. code-block:: cpp

      template <typename T>
      void testing_gemv_bad_arg(const Arguments& arg)
      {
      // ...
      }

   These ``bad_arg`` test function templates can be used to set arguments programmatically when it is simpler than the YAML approach,
   for example, to pass NULL pointers.
   It is expected that the member variable values in the ``Arguments`` parameter will not be utilized with the common
   exception of the ``api`` member variable of ``Arguments`` which can drive selection of C, FORTRAN,
   C_64, or FORTRAN_64 API bad argument tests.

   All functions should be generalized with template parameters as much as possible,
   to avoid copy-and-paste code.

   In this function, use the following macros and functions to check results:

   .. code-block:: cpp

      HIP_CHECK_ERROR             Verifies that a HIP call returns success
      ROCBLAS_CHECK_ERROR         Verifies that a rocBLAS call returns success
      EXPECT_ROCBLAS_STATUS       Verifies that a rocBLAS call returns a certain status
      unit_check_general          Check that two answers agree (see unit.hpp)
      near_check_general          Check that two answers are close (see near.hpp)

   .. code-block:: cpp

      DAPI_CHECK                  Verifies either LP64 or ILP64 function form returns success (based on Arguments member variable API)
      DAPI_EXPECT                 Verifies either LP64 or ILP64 function form returns a certain status
      DAPI_DISPATCH               Invoke either LP64 or ILP64 function form

   In addition, you can use GoogleTest macros such as the ones below, provided they are
   guarded by ``#ifdef GOOGLE_TEST``:

   .. code-block:: cpp

      EXPECT_EQ
      ASSERT_EQ
      EXPECT_TRUE
      ASSERT_TRUE
      ...

   .. note::

      The ``device_vector`` template allocates memory on the device. You must check whether
      converting the ``device_vector`` to ``bool`` returns ``false``\ . If so, report a HIP memory
      error and then exit the current function, following this example:

      .. code-block:: cpp

         // allocate memory on device
         device_vector<T> dx(size_x);
         device_vector<T> dy(size_y);
         if(!dx || !dy)
         {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
         }

   The general outline of the function should be:

   #. Convert any scalar arguments (for example, ``alpha`` and ``beta``\ ) to ``double``.
   #. If the problem-size arguments are invalid, use a ``safe_size`` to allocate arrays,
      call the rocBLAS routine with the original arguments, and verify that it returns
      ``rocblas_status_invalid_size``, then return.
   #. Set up the host and device arrays (see ``rocblas_vector.hpp`` and ``rocblas_init.hpp``\ ).
   #. Call a CBLAS or other reference implementation on the host arrays.
   #. Call rocBLAS using both device pointer mode and host pointer mode, verifying that
      every rocBLAS call is successful by wrapping it in ``ROCBLAS_CHECK_ERROR()``.
   #. If ``arg.unit_check`` is enabled, use ``unit_check_general`` or ``near_check_general`` to validate the results.
   #. If ``arg.norm_check`` is enabled, calculate and print out norms. (This is now deprecated.)
   #. If ``arg.timing`` is enabled, perform benchmarking (currently under refactoring).

#. Create a C++ file with the name ``<function>_gtest.cpp`` in the ``gtest``
   subdirectory, where ``<function>`` is a non-type-specific shorthand for the
   function(s) being tested. Here are some examples:

   *  ``gemm_gtest.cpp``
   *  ``trsm_gtest.cpp``
   *  ``blas1_gtest.cpp``

   In the C++ file, follow these steps:

   A. Include the header files related to the tests, as well as ``type_dispatch.hpp``, for example:

      .. code-block:: c++

         #include "testing_syr.hpp"
         #include "type_dispatch.hpp"

   B. Wrap the body with an anonymous namespace, to minimize namespace collisions:

      .. code-block:: c++

         namespace {

   C. Create a templated class which accepts any number of type parameters followed by one anonymous trailing type
      parameter defaulted to ``void`` (to be used with ``enable_if``\ ).

      Choose the number of type parameters based on how likely it is in the future that
      the function will support a mixture of that many different types, for example, input
      type (\ ``Ti``\ ), output type (\ ``To``\ ), and compute type (\ ``Tc``\ ). If the function will
      never support more than one or two type parameters, then that many can be used.
      If the function might be expanded later to support mixed types, then those
      should be planned for ahead of time and placed in the template parameters.

      Unless the number of type parameters is greater than one and is always
      fixed, later type parameters should default to earlier ones. This is so that
      a subset of type arguments can used and that code which works for
      functions that take one type parameter can be used for functions that
      take one or more type parameters. Here is an example:

      .. code-block:: c++

         template< typename Ti, typename To = Ti, typename Tc = To, typename = void>

      Have the primary definition of this class template derive from the ``rocblas_test_invalid`` class, for example:

      .. code-block:: c++

         template <typename T, typename = void>
         struct syr_testing : rocblas_test_invalid
         {
         };

   D. Create one or more partial specializations of the class template conditionally enabled by the type parameters
      matching legal combinations of types.

      If the first type argument is ``void``\ , then these partial specializations must not apply.
      This is so the default based on ``rocblas_test_invalid`` can behave correctly when ``void`` is passed to indicate failure.

      In the partial specializations, derive from the ``rocblas_test_valid`` class.

      In the partial specializations, create a functional ``operator()`` which takes a ``const Arguments&`` parameter
      and calls the templated test functions (usually in ``include/testing_*.hpp``\ ) with the specialization's template arguments
      when the ``arg.function`` string matches the function name. If ``arg.function`` does not match any function related
      to this test, mark it as a test failure. Follow this example:

      .. code-block:: c++

         template <typename T>
         struct syr_testing<T,
                           std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, double>>
                           > : rocblas_test_valid
         {
            void operator()(const Arguments& arg)
            {
               if(!strcmp(arg.function, "syr"))
                     testing_syr<T>(arg);
               else
                     FAIL() << "Internal error: Test called with unknown function: "
                           << arg.function;
            }
         };

   E. If necessary, create a type dispatch function for this function (or group of functions it belongs to) in ``include/type_dispatch.hpp``.
      If possible, use one of the existing dispatch functions, even if it covers a superset of the allowable types.
      The purpose of ``type_dispatch.hpp`` is to perform runtime type dispatch in a single place, rather than
      copying it across several test files.

      The type dispatch function takes a ``template`` template parameter of ``template<typename...> class`` and a function parameter
      of type ``const Arguments&``. It looks at the runtime type values in ``Arguments``\  and instantiates the template with one
      or more static type arguments, corresponding to the dynamic runtime type arguments.

      It treats the passed template as a functor, passing the ``Arguments`` argument to a particular instantiation of it.

      The combinations of types handled by this "runtime type to template type instantiation mapping" function can be general
      because the type combinations which do not apply to a particular test case will have the template argument set
      to derive from ``rocblas_test_invalid``\ , which will not create any unresolved instantiations.
      If unresolved instantiation compile or link errors occur, then the ``enable_if<>`` condition in step D needs to be
      reworked to indicate ``false`` for type combinations which do not apply.

      The return type of this function needs to be ``auto``\ , picking up the return type of the functor.

      If the runtime type combinations do not apply, then this function should return ``TEST<void>{}(arg)``\ , where ``TEST`` is the template
      parameter. However, this is less important than step D above in excluding invalid type
      combinations with ``enable_if``\ , since this only excludes them at run-time and they need to be excluded at compile-time by step D
      to avoid unresolved references or invalid instantiations, for example:

      .. code-block:: c++

         template <template <typename...> class TEST>
         auto rocblas_simple_dispatch(const Arguments& arg)
         {
            switch(arg.a_type)
            {
               case rocblas_datatype_f16_r: return TEST<rocblas_half>{}(arg);
               case rocblas_datatype_f32_r: return TEST<float>{}(arg);
               case rocblas_datatype_f64_r: return TEST<double>{}(arg);
               case rocblas_datatype_bf16_r: return TEST<rocblas_bfloat16>{}(arg);
               case rocblas_datatype_f16_c: return TEST<rocblas_half_complex>{}(arg);
               case rocblas_datatype_f32_c: return TEST<rocblas_float_complex>{}(arg);
               case rocblas_datatype_f64_c: return TEST<rocblas_double_complex>{}(arg);
               default: return TEST<void>{}(arg);
            }
         }

   F. Create a (possibly-templated) test implementation class which derives from the ``RocBLAS_Test`` template class and
      passes itself to ``RocBLAS_Test`` (the CRTP pattern) as well as the template class defined above, for example:

      .. code-block:: c++

         struct syr : RocBLAS_Test<syr, syr_testing>
         {
            // ...
         };

      In this class, implement three static functions:

      *  ``static bool type_filter(const Arguments& arg)``: returns ``true`` if the types described by ``*_type`` in the ``Arguments`` structure
         match a valid type combination.

         This is usually implemented by calling the dispatch function in step E, passing it the helper ``type_filter_functor`` template class
         defined in ``RocBLAS_Test``. This functor uses the same runtime type checks that are used to instantiate test functions
         with particular type arguments, but it instead returns ``true`` or ``false`` depending on whether a function would have been called.
         It is used to filter out tests whose runtime parameters do not match a valid test.

         ``RocBLAS_Test`` is a dependent base class if this test implementation class is templated, so you might need to use a
         fully-qualified name (\ ``A::B``\ ) to resolve ``type_filter_functor``\ .
         In the last part of this name, the keyword ``template`` needs to precede ``type_filter_functor``. The first half of the fully
         qualified name can be this class itself, or the full instantiation of ``RocBLAS_Test<...>``, for example:

         .. code-block:: c++

            static bool type_filter(const Arguments& arg)
            {
               return rocblas_blas1_dispatch<
                  blas1_test_template::template type_filter_functor>(arg);
            }


      *  ``static bool function_filter(const Arguments& arg)``: returns ``true`` if the function name in ``Arguments`` matches one of
         the functions handled by this test, for example:

         .. code-block:: c++

            // Filter for which functions apply to this suite
            static bool function_filter(const Arguments& arg)
            {
            return !strcmp(arg.function, "ger") || !strcmp(arg.function, "ger_bad_arg");
            }


      *  ``static std::string name_suffix(const Arguments& arg)``: returns a string which is used as the suffix for the GoogleTest name.
         It provides an alphanumeric representation of the test arguments.

         Use the ``RocBLAS_TestName`` helper class template to create the name. It accepts ``ostream`` output (like ``std::cout``\ )
         and can be automatically converted to ``std::string`` after all the text of the name has been streamed to it.

         The ``RocBLAS_TestName`` helper class constructor accepts a string argument which is included in the test name.
         It is generally passed the ``Arguments`` structure's ``name`` member.

         The ``RocBLAS_TestName`` helper class template should be passed the name of this test implementation class,
         including any implicit template arguments, as a template argument, so that every instantiation of this
         test implementation class creates a unique instantiation of ``RocBLAS_TestName``. ``RocBLAS_TestName`` has
         some static data that needs to be kept local to each test.

         ``RocBLAS_TestName`` converts non-alphanumeric characters into suitable replacements and disambiguates test names
         when the same arguments appear more than once.

         The conversion of the stream into a ``std::string`` is a destructive one-time operation, so
         the ``RocBLAS_TestName`` value converted to ``std::string`` must be an rvalue, for example:

         .. code-block:: c++

            static std::string name_suffix(const Arguments& arg)
            {
               // Okay: rvalue RocBLAS_TestName object streamed to and returned
               return RocBLAS_TestName<syr>() << rocblas_datatype2string(arg.a_type)
                  << '_' << (char) std::toupper(arg.uplo) << '_' << arg.N
                  << '_' << arg.alpha << '_' << arg.incx << '_' << arg.lda;
            }

            static std::string name_suffix(const Arguments& arg)
            {
               RocBLAS_TestName<gemm_test_template> name;
               name << rocblas_datatype2string(arg.a_type);
               if(GEMM_TYPE == GEMM_EX || GEMM_TYPE == GEMM_STRIDED_BATCHED_EX)
                  name << rocblas_datatype2string(arg.b_type)
                        << rocblas_datatype2string(arg.c_type)
                        << rocblas_datatype2string(arg.d_type)
                        << rocblas_datatype2string(arg.compute_type);
               name << '_' << (char) std::toupper(arg.transA)
                           << (char) std::toupper(arg.transB) << '_' << arg.M
                           << '_' << arg.N << '_' << arg.K << '_' << arg.alpha << '_'
                           << arg.lda << '_' << arg.ldb << '_' << arg.beta << '_'
                           << arg.ldc;
               // name is an lvalue: Must use std::move to convert it to rvalue.
               // name cannot be used after it's converted to a string, which is
               // why it must be "moved" to a string.
               return std::move(name);
            }

   G. Choose a non-type-specific shorthand name for the test, which is displayed as part of the test name in the GoogleTest output
      and is stringified. Create a type alias for this name, unless the name is already the name of the class defined in step F
      and is not templated. For example, for a templated class defined in step F, create an alias for one of its instantiations:

      .. code-block:: c++

         using gemm = gemm_test_template<gemm_testing, GEMM>;

   H. Pass the name created in step G to the ``TEST_P`` macro, along with a broad test category name for the test to belong to
      so that GoogleTest filtering can be used to select all tests in a category. The broad test category suffix should be ``_tensile``
      if it requires Tensile.

      In the body following this ``TEST_P`` macro, call the dispatch function from step E, passing it the class from step C
      as a template template argument, passing the result of ``GetParam()`` as an ``Arguments`` structure, and wrapping the call
      in the ``CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES()`` macro. Here is an example:

      .. code-block:: c++

         TEST_P(gemm, blas3_tensile) { CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES(rocblas_gemm_dispatch<gemm_testing>(GetParam())); }

      The ``CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES()`` macro detects signals such as ``SIGSEGV`` and uncaught C++ exceptions returned from
      rocBLAS C APIs as failures, without terminating the test program.

   I. Call the ``INSTANTIATE_TEST_CATEGORIES`` macro which instantiates the GoogleTests across all test
      categories (\ ``quick``\ , ``pre_checkin``\ , ``nightly``\ , ``known_bug``\ ), passing it the same test name as in steps G and H, for example:

      .. code-block:: c++

         INSTANTIATE_TEST_CATEGORIES(gemm);

   J. Close the anonymous namespace:

      .. code-block:: c++

         } // namespace

#. Create a ``<function>.yaml`` file with the same name as the C++ file, but with a ``.yaml`` extension.

   In the YAML file, define tests with combinations of parameters.

   The YAML files are organized as files which include each other (an extension to YAML).
   Define anchors for the data types and data structures, list of test parameters or subsets thereof, and ``Tests``, which describe
   a combination of parameters including ``category`` and ``function``.

   The ``category`` must be one of ``quick``\ , ``pre_checkin``\ , ``nightly``\ , or ``known_bug``.
   The category is automatically changed to ``known_bug`` if the test matches a test in ``known_bugs.yaml``.

   The ``function`` must be one of the functions tested for and recognized in steps D-F.

   The syntax and idioms of the YAML files is best understood by looking at the
   existing ``*_gtest.yaml`` files as examples.

#. Add the YAML file to ``rocblas_gtest.yaml``\  to be included, for example:

   .. code-block:: yaml

      include: blas1_gtest.yaml

#. Add the YAML file to the list of dependencies for ``rocblas_gtest.data`` in ``CMakeLists.txt``, for example:

   .. code-block:: cmake

      add_custom_command( OUTPUT "${ROCBLAS_TEST_DATA}"
                        COMMAND ../common/rocblas_gentest.py -I ../include rocblas_gtest.yaml -o "${ROCBLAS_TEST_DATA}"
                        DEPENDS ../common/rocblas_gentest.py rocblas_gtest.yaml ../include/rocblas_common.yaml known_bugs.yaml blas1_gtest.yaml gemm_gtest.yaml gemm_batched_gtest.yaml gemm_strided_batched_gtest.yaml gemv_gtest.yaml symv_gtest.yaml syr_gtest.yaml ger_gtest.yaml trsm_gtest.yaml trtri_gtest.yaml geam_gtest.yaml set_get_vector_gtest.yaml set_get_matrix_gtest.yaml
                        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

#. Add the ``.cpp`` file to the list of sources for ``rocblas-test`` in ``CMakeLists.txt``, for example:

   .. code-block:: c++

      set(rocblas_test_source
         rocblas_gtest_main.cpp
         ${Tensile_TEST_SRC}
         set_get_pointer_mode_gtest.cpp
         logging_mode_gtest.cpp
         set_get_vector_gtest.cpp
         set_get_matrix_gtest.cpp
         blas1_gtest.cpp
         gemv_gtest.cpp
         ger_gtest.cpp
         syr_gtest.cpp
         symv_gtest.cpp
         geam_gtest.cpp
         trtri_gtest.cpp
         )

#. Aim for a function to have tests in each of the categories: quick, pre_checkin, and nightly.
   Aim for tests for each function to have the runtime fall within the parameters in the table below:

   .. csv-table::
      :header: "Level","quick","pre_checkin","nightly"
      :widths: 20, 30, 30, 30

      "Level 1", "2 - 12 sec", "20 - 36 sec", "70 - 200 sec"
      "Level 2", "6 - 36 sec", "35 - 100 sec", "200 - 650 sec"
      "Level 3", "20 sec - 2 min", "2 - 6 min", "12 - 24 min"

   Many examples are available in ``gtest/*_gtest.{cpp,yaml}``.

Testing during development
--------------------------

ILP64 APIs require such large problem sizes that getting code coverage during tests is cost prohibitive.
Therefore, there are some hooks to help with early developer testing using smaller sizes.
You can compile with ``-DROCBLAS_DEV_TEST_ILP64`` to test ILP64 code when it would otherwise not be invoked.
For example, a ``scal`` implementation can call the original 32-bit API code when ``N`` and ``incx`` are less than ``c_ILP64_i32_max``.
``c_ILP64_i32_max`` is usually defined as ``std::numeric_limits<int32_t>::max()``,
but with ``ROCBLAS_DEV_TEST_ILP64`` defined, then ``c_ILP64_i32_max`` is defined as zero.
Therefore, for small sizes it branches and uses ILP64 support code instead of the 32-bit original API.

The specifics vary for each implementation and require a YAML configuration to test ``C_64`` APIs with small sizes.
This is intended as a by-pass for when early detection of small sizes invokes the 32-bit APIs.
This is for developer testing only and should not be used for production code.

Test coverage during development should be much more exhaustive than the final versions of test sets.
Test times are limited, so a trade-off between coverage and test duration must be made.
During development, it is expected that the problem space will be covered in more depth to look for potential anomalies.
Any special cases should be analyzed, reduced in scope, and represented in the final test category.
