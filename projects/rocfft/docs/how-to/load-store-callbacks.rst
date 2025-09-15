.. meta::
  :description: How to load and store callbacks in rocFFT
  :keywords: rocFFT, ROCm, API, documentation, callbacks

.. _load-store-callbacks:

********************************************************************
Load and store callbacks
********************************************************************

rocFFT includes experimental functionality to call user-defined device functions
when loading input from global memory at the transform start or
when storing output to global memory at the transform end.

These optional user-defined callback functions can be supplied
to the library using
:cpp:func:`rocfft_execution_info_set_load_callback` and
:cpp:func:`rocfft_execution_info_set_store_callback`.

.. note::

   Callback functions must be built as relocatable device code by
   passing the ``-fgpu-rdc`` option to the compiler and linker.

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
  :cpp:func:`rocfft_execution_info_set_load_callback` and
  :cpp:func:`rocfft_execution_info_set_store_callback` which is passed
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
