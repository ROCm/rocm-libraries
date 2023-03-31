Select
------

Configuring the kernel
~~~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct:: rocprim::select_config

select
~~~~~~

.. doxygenfunction:: rocprim::select(void *temporary_storage, size_t &storage_size, InputIterator input, FlagIterator flags, OutputIterator output, SelectedCountOutputIterator selected_count_output, const size_t size, const hipStream_t stream=0, const bool debug_synchronous=false)
.. doxygenfunction:: rocprim::select(void *temporary_storage, size_t &storage_size, InputIterator input, OutputIterator output, SelectedCountOutputIterator selected_count_output, const size_t size, UnaryPredicate predicate, const hipStream_t stream=0, const bool debug_synchronous=false)

