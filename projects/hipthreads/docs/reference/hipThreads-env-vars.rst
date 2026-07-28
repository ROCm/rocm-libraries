.. meta::
  :description: hipThreads environment variables
  :keywords: hipThreads, environment variables, ROCm, AMD, HIPTHREADS_VCORES_PER_WGP, scheduler, vcores

********************************************************************
hipThreads environment variables
********************************************************************

The following environment variables affect the hipThreads runtime.

.. list-table::
  :header-rows: 1
  :widths: 28 72

  * - Environment variable
    - Values
  * - ``HIPTHREADS_VCORES_PER_WGP``
    - | Sets the number of scheduler :term:`vcores<vcore>` launched per :term:`WGP` at process start.
      | ``hip::wthread::hardware_concurrency()`` returns ``multiprocessorCount * vcoresPerWgp``, where ``multiprocessorCount`` comes from ``hipDeviceAttributeMultiprocessorCount`` on device 0.
      | Accepted values: a base-10 unsigned integer string with no trailing characters.
      | ``1`` through ``4294967295``: Uses that count per WGP.
      | Unset, empty, ``0``, non-numeric text, or any value with trailing characters, including a trailing space: Falls back to the compiled-in default.
      | Negative values: Parsed as unsigned by ``strtoul()``. For example, ``-1`` becomes ``4294967295`` per WGP rather than falling back to the default.
      | Default when unset or unparseable: ``HIPTHREADS_DEFAULT_VCORES_PER_WGP`` from the hipThreads build, or ``16`` when that macro wasn't set at compile time.
      | Overrides the compiled-in default whenever the parsed value is accepted.
      | Read once on the first host call to ``hip::wthread::hardware_concurrency()``. Later changes to the environment don't affect an already-initialized process.
