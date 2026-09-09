.. meta::
  :description: How to load and store callbacks in rocFFT
  :keywords: rocFFT, ROCm, API, documentation, callbacks

.. _load-store-callbacks:

********************************************************************
Load and store callbacks
********************************************************************

rocFFT includes experimental functionality to call user-defined
device functions when loading input from global memory at the
transform start or when storing output to global memory at the
transform end.  If specified, these functions are Just-In-Time (JIT)
compiled to combine them with rocFFT's own device code.

.. note::
   JIT callbacks cannot currently be used on transforms that
   have fields or bricks also specified on the same plan description.
   This support will be added in a future release of rocFFT.

These optional user-defined callback functions can be supplied
to the library using
:cpp:func:`rocfft_plan_description_set_load_callback` and
:cpp:func:`rocfft_plan_description_set_store_callback`.

Device functions supplied as callbacks must load and store element
data types appropriate for the transform being executed.

+-------------------------+----------------------+------------------------+
|Transform type           | Load element type    | Store element type     |
+=========================+======================+========================+
|Complex-to-complex,      | ``_Float16_2``       | ``_Float16_2``         |
|half-precision           |                      |                        |
+-------------------------+----------------------+------------------------+
|Complex-to-complex,      | ``float2``           | ``float2``             |
|single-precision         |                      |                        |
+-------------------------+----------------------+------------------------+
|Complex-to-complex,      | ``double2``          | ``double2``            |
|double-precision         |                      |                        |
+-------------------------+----------------------+------------------------+
|Real-to-complex,         | ``float``            | ``float2``             |
|single-precision         |                      |                        |
+-------------------------+----------------------+------------------------+
|Real-to-complex,         | ``_Float16``         | ``_Float16_2``         |
|half-precision           |                      |                        |
+-------------------------+----------------------+------------------------+
|Real-to-complex,         | ``double``           | ``double2``            |
|double-precision         |                      |                        |
+-------------------------+----------------------+------------------------+
|Complex-to-real,         | ``_Float16_2``       | ``_Float16``           |
|half-precision           |                      |                        |
+-------------------------+----------------------+------------------------+
|Complex-to-real,         | ``float2``           | ``float``              |
|single-precision         |                      |                        |
+-------------------------+----------------------+------------------------+
|Complex-to-real,         | ``double2``          | ``double``             |
|double-precision         |                      |                        |
+-------------------------+----------------------+------------------------+

The callback function signatures must match the specifications
below.

.. code-block:: c

  Tdata load_callback(Tdata* buffer, size_t offset, void* callback_data, void* shared_memory);
  void store_callback(Tdata* buffer, size_t offset, Tdata element, void* callback_data, void* shared_memory);

The parameters for the functions are as follows:

* ``Tdata``: The data type of each element being loaded or stored from the
  input or output.
* ``buffer``: Pointer to the input (for load callbacks) or
  output (for store callbacks) in device memory that was passed to
  :cpp:func:`rocfft_execute`.
* ``offset``: The offset of the location being read from or written
  to. This counts by elements from the ``buffer`` pointer.
* ``element``: For store callbacks only, the element to be stored.
* ``callback_data``: A pointer value accepted by
  :cpp:func:`rocfft_execution_info_set_load_callback_data` and
  :cpp:func:`rocfft_execution_info_set_store_callback_data` which is passed
  through to the callback function.
* ``shared_memory``: A pointer to an amount of shared memory requested
  when the callback is set. Shared memory is not supported,
  so this parameter is always null.

Callback functions are called exactly once for each element being
loaded or stored in a transform. Multiple kernels can be
launched to decompose a transform, which means that separate kernels
might call the load and store callbacks for a transform if both are
specified.

Callback functions are only supported for transforms that do not use planar format for input or output.

Compiling functions to SPIR-V for JIT callbacks
-----------------------------------------------

:cpp:func:`rocfft_plan_description_set_load_callback` and
:cpp:func:`rocfft_plan_description_set_store_callback` accept
callback functions as a named symbol in compiled SPIR-V bitcode.

Symbol names can only contain digits (0-9), letters (a-z, A-Z), and
underscores, and cannot begin with a digit.

A callback function written as HIP code must first be compiled to
SPIR-V bitcode before it can be added to a plan description.  The
following example demonstrates how to compile such code using the
``amdclang++`` compiler.

An example load callback function for a single-precision real-complex
forward transform might look like:

.. code-block:: c++

  #include <hip/hip_runtime.h>

  // Give the function C linkage so that it is not given a mangled C++ name
  extern "C"
  __device__ float load_callback(float* buffer, size_t offset, void* callback_data, void* shared_memory)
  {
    // Scale the input values by 2
    return buffer[offset] * 2.0f;
  }

The ``amdclang++`` compiler can compile this code to SPIR-V, once this
code is written to a file (named ``load_callback.hip`` in this
example):

.. code-block:: shell

  amdclang++ -I/opt/rocm/include load_callback.hip -c -D__HIP_PLATFORM_AMD__=1 --offload-device-only --offload-arch=amdgcnspirv -o load_callback.spv

The compiler outputs a file (``load_callback.spv``).  The contents of
the file and the file's length are then passed to rocFFT:

.. code-block:: c++

  #include <vector>
  #include <fstream>

  rocfft_plan_description create_plan_desc_with_callback()
  {
      // Read the compiled callback into a vector
      std::vector<char> bitcode;
      std::ifstream     infile("load_callback.spv", std::ios::binary | std::ios::ate);
      auto              size = infile.tellg();
      bitcode.resize(size);
      infile.seekg(0);
      infile.read(bitcode.data(), size);

      // Create a plan description and set the load callback
      rocfft_plan_description desc = nullptr;
      if(rocfft_plan_description_create(&desc) != rocfft_status_success)
          return nullptr;
      if(rocfft_plan_description_set_load_callback(desc, "load_callback",
                                                   bitcode.data(), size, 0) != rocfft_status_success)
        {
          rocfft_plan_description_destroy(desc);
          return nullptr;
        }
      return desc;
  }

Passing data to callback functions
----------------------------------

rocFFT can optionally pass a user-specified pointer value to callback
functions.  This is useful in cases where the callback function
requires extra data on top of the input/output pointer and offset
that are already provided.

Callback data is specified on a rocFFT execution info object using
:cpp:func:`rocfft_execution_info_set_load_callback_data` and
:cpp:func:`rocfft_execution_info_set_store_callback_data` for load
and store callbacks, respectively.

These functions accept an array of callback data pointers, one per
brick in the input fields (for load callbacks) or output fields (for
store callbacks) of the transform.  A transform which does not
specify a field and brick layout for input (or output) is
considered to have a single brick for input (or output).

.. note::
   As JIT callbacks cannot currently be used on transforms that have
   fields or bricks specified on the plan description, the length of the
   array of callback data pointers will always be 1 if callback data is
   specified.


Here is an example showing how to pass filtering data to a load
callback.

.. code-block:: c++

  // Define a structure to hold arbitrary amounts of data to pass to
  // the callback function.  This example has just one data member
  // but it could be extended with additional data members.
  struct load_callback_data
  {
      hipDoubleComplex* filter = nullptr;
  };

  // Initialize the structure on the host
  load_callback_data cbdata_host;

  // Set the filter pointer in the host structure.  Code to allocate and
  // initialize this filter on the device has been omitted but would
  // depend on the details of the filtering operation.
  cbdata_host.filter = device_filter;

  // Copy the structure to the device
  load_callback_data* cbdata_device = nullptr;
  hipMalloc(&cbdata_device, sizeof(load_callback_data));
  hipMemcpy(cbdata_device, &cbdata_host, sizeof(load_callback_data), hipMemcpyHostToDevice);

  // Initialize an array of device pointers on the host.  This example
  // creates an array of length 1 as the input has only one brick.
  void* cbdata_ptrs[1];
  cbdata_ptrs[0] = cbdata_device;

  // Create an execution info object and set the device pointer array on it.
  rocfft_execution_info info = nullptr;
  rocfft_execution_info_create(&info);
  rocfft_execution_info_set_load_callback_data(info, cbdata_ptrs, 1);

  // When the execution info is passed to rocfft_execute, the load
  // callback receives the cbdata_device pointer as its callback_data
  // parameter.  The callback can then cast that pointer from 'void*' to
  // 'load_callback_data*' and access the filter.

Legacy function pointer callbacks (deprecated)
----------------------------------------------

rocFFT also includes deprecated functionality to call user-defined
device functions specified as function pointers to
:cpp:func:`rocfft_execution_info_set_load_callback` and
:cpp:func:`rocfft_execution_info_set_store_callback`.  This
functionality will be removed in a future release.

.. note::

   Function pointer callbacks are not functional on the gfx1250
   architecture and :cpp:func:`rocfft_execute` will return an error
   on this architecture if they are specified.

Legacy callback functions are passed as arrays of function pointers, with
one function per brick in the :ref:`input or output field<input_output_fields>`.  For example, to
specify a load callback on a transform with 4 input bricks, pass an
array of 4 function pointers to
:cpp:func:`rocfft_execution_info_set_load_callback`.  Or, to specify
a store callback on a transform with 6 output bricks, pass an array of
6 function pointers to
:cpp:func:`rocfft_execution_info_set_store_callback`.  The order of
the function pointers must match the order that the bricks were added
to the input or output fields with
:cpp:func:`rocfft_field_add_brick`.  If the input or output field of
a transform is unspecified, the input or output is considered to have
one brick.

All functions in an array must perform the same logical operation.
That is, any function in an array must be substitutable for any other
function in the array if the data being loaded or stored were moved
to another brick.  Behavior of the transform is not defined if
functions in an array do not behave the same.

.. note::

   Legacy function pointer callbacks must be built as relocatable
   device code by passing the ``-fgpu-rdc`` option to the compiler
   and linker.

JIT callbacks are preferred over legacy function pointer callbacks
because they allow for rocFFT to properly optimize the combined
callback and FFT code.  Legacy callback functions are already
compiled by the time they are passed to rocFFT, and no further
optimization can be done.
