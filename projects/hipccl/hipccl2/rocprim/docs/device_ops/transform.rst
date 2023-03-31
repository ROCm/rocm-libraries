Transform
---------

Configuring the kernel
~~~~~~~~~~~~~~~~~~~~~~

.. doxygentypedef:: rocprim::transform_config

transform
~~~~~~~~~

.. doxygenfunction:: rocprim::transform(InputIterator, OutputIterator, const size_t, UnaryFunction, const hipStream_t stream, bool)
.. doxygenfunction:: rocprim::transform(InputIterator1, InputIterator2, OutputIterator, const size_t, BinaryFunction, const hipStream_t, bool)
