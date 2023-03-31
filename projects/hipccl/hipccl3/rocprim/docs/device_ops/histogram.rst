Histogram
---------

Configuring the kernel
~~~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct:: rocprim::histogram_config

histogram_even
~~~~~~~~~~~~~~

.. doxygenfunction:: rocprim::histogram_even(void *temporary_storage, size_t &storage_size, SampleIterator samples, unsigned int size, Counter *histogram, unsigned int levels, Level lower_level, Level upper_level, hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::histogram_even(void *temporary_storage, size_t &storage_size, SampleIterator samples, unsigned int columns, unsigned int rows, size_t row_stride_bytes, Counter *histogram, unsigned int levels, Level lower_level, Level upper_level, hipStream_t stream=0, bool debug_synchronous=false)

multi_histogram_even
~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: rocprim::multi_histogram_even(void *temporary_storage, size_t &storage_size, SampleIterator samples, unsigned int size, Counter *histogram[ActiveChannels], unsigned int levels[ActiveChannels], Level lower_level[ActiveChannels], Level upper_level[ActiveChannels], hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::multi_histogram_even(void *temporary_storage, size_t &storage_size, SampleIterator samples, unsigned int columns, unsigned int rows, size_t row_stride_bytes, Counter *histogram[ActiveChannels], unsigned int levels[ActiveChannels], Level lower_level[ActiveChannels], Level upper_level[ActiveChannels], hipStream_t stream=0, bool debug_synchronous=false)

histogram_range
~~~~~~~~~~~~~~~

.. doxygenfunction:: rocprim::histogram_range(void *temporary_storage, size_t &storage_size, SampleIterator samples, unsigned int size, Counter *histogram, unsigned int levels, Level *level_values, hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::histogram_range(void *temporary_storage, size_t &storage_size, SampleIterator samples, unsigned int columns, unsigned int rows, size_t row_stride_bytes, Counter *histogram, unsigned int levels, Level *level_values, hipStream_t stream=0, bool debug_synchronous=false)

multi_histogram_range
~~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: rocprim::multi_histogram_range(void *temporary_storage, size_t &storage_size, SampleIterator samples, unsigned int size, Counter *histogram[ActiveChannels], unsigned int levels[ActiveChannels], Level *level_values[ActiveChannels], hipStream_t stream=0, bool debug_synchronous=false)
.. doxygenfunction:: rocprim::multi_histogram_range(void *temporary_storage, size_t &storage_size, SampleIterator samples, unsigned int columns, unsigned int rows, size_t row_stride_bytes, Counter *histogram[ActiveChannels], unsigned int levels[ActiveChannels], Level *level_values[ActiveChannels], hipStream_t stream=0, bool debug_synchronous=false)
