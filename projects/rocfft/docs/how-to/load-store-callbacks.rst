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
  :cpp:func:`rocfft_plan_description_set_load_callback` and
  :cpp:func:`rocfft_plan_description_set_store_callback` which is passed
  through to the callback function.
* ``shared_memory``: A pointer to an amount of shared memory requested
  when the callback is set. Shared memory is not supported,
  so this parameter is always null.

Callback functions are called exactly once for each element being
loaded or stored in a transform. Multiple kernels can be
launched to decompose a transform, which means that separate kernels
might call the load and store callbacks for a transform if both are
specified.

Callbacks functions are only supported for transforms that do not use planar format for input or output.

Compiling callback functions to SPIR-V
--------------------------------------

:cpp:func:`rocfft_plan_description_set_load_callback` and
:cpp:func:`rocfft_plan_description_set_store_callback` accept
callback functions as a named symbol in compiled SPIR-V code.

Symbol names can only contain digits (0-9), letters (a-z, A-Z), and
underscores, and cannot begin with a digit.

A callback function written as HIP code must first be compiled to
SPIR-V before it can be added to a plan description.  The following
example demonstrates how to compile such code using the ``clang++``
compiler.

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
      std::vector<char> code;
      std::ifstream     infile("load_callback.spv", std::ios::binary | std::ios::ate);
      auto              size = infile.tellg();
      code.resize(size);
      infile.seekg(0);
      infile.read(code.data(), size);

      // Create a plan description and set the load callback
      rocfft_plan_description desc = nullptr;
      if(rocfft_plan_description_create(&desc) != rocfft_status_success)
          return nullptr;
      if(rocfft_plan_description_set_load_callback(desc, "load_callback",
                                                   code.data(), size, nullptr, 0) != rocfft_status_success)
        {
          rocfft_plan_description_destroy(desc);
          return nullptr;
        }
      return desc;
  }

Legacy function pointer callbacks (deprecated)
----------------------------------------------

rocFFT also includes deprecated functionality to call user-defined
device functions specified as function pointers to
:cpp:func:`rocfft_execution_info_set_load_callback` and
:cpp:func:`rocfft_execution_info_set_store_callback`.  This
functionality will be removed in a future release.

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

SPIR-V callbacks are preferred over legacy function pointer callbacks
because they allow for rocFFT to properly optimize the the combined
callback and FFT code.  Legacy callback functions are already
compiled by the time they are passed to rocFFT, and no further
optimization can be done.
