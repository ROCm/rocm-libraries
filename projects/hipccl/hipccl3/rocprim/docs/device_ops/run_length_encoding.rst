Run Length Encode
-----------------

Configuring the kernel
~~~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct:: rocprim::run_length_encode_config

run_length_encode
~~~~~~~~~~~~~~~~~

.. doxygenfunction:: rocprim::run_length_encode(void *temporary_storage, size_t &storage_size, InputIterator input, unsigned int size, UniqueOutputIterator unique_output, CountsOutputIterator counts_output, RunsCountOutputIterator runs_count_output, hipStream_t stream=0, bool debug_synchronous=false)

run_length_encode_non_trivial_runs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: rocprim::run_length_encode_non_trivial_runs(void *temporary_storage, size_t &storage_size, InputIterator input, unsigned int size, OffsetsOutputIterator offsets_output, CountsOutputIterator counts_output, RunsCountOutputIterator runs_count_output, hipStream_t stream=0, bool debug_synchronous=false)
