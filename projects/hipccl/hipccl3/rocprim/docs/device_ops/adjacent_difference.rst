Adjacent difference
-------------------

Configuring the kernel
~~~~~~~~~~~~~~~~~~~~~~

.. doxygenstruct:: rocprim::adjacent_difference_config

left
~~~~

.. doxygenfunction:: rocprim::adjacent_difference(void *const temporary_storage, std::size_t &storage_size, const InputIt input, const OutputIt output, const std::size_t size, const BinaryFunction op=BinaryFunction {}, const hipStream_t stream=0, const bool debug_synchronous=false)

left, inplace
~~~~~~~~~~~~~

.. doxygenfunction:: rocprim::adjacent_difference_inplace(void *const temporary_storage, std::size_t &storage_size, const InputIt values, const std::size_t size, const BinaryFunction op=BinaryFunction {}, const hipStream_t stream=0, const bool debug_synchronous=false)

right
~~~~~

.. doxygenfunction:: rocprim::adjacent_difference_right(void *const temporary_storage, std::size_t &storage_size, const InputIt input, const OutputIt output, const std::size_t size, const BinaryFunction op=BinaryFunction {}, const hipStream_t stream=0, const bool debug_synchronous=false)

right, inplace
~~~~~~~~~~~~~~

.. doxygenfunction:: rocprim::adjacent_difference_right_inplace(void *const temporary_storage, std::size_t &storage_size, const InputIt values, const std::size_t size, const BinaryFunction op=BinaryFunction {}, const hipStream_t stream=0, const bool debug_synchronous=false)

