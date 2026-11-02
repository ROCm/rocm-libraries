.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _dev-transform:

********************************************************************
 Transform
********************************************************************

Configuring the kernel
======================

.. doxygenstruct:: rocprim::transform_config

transform
==========

.. doxygenfunction:: rocprim::transform(InputIterator, OutputIterator, const size_t, UnaryFunction, const hipStream_t stream, bool)
.. doxygenfunction:: rocprim::transform(InputIterator1, InputIterator2, OutputIterator, const size_t, BinaryFunction, const hipStream_t, bool)
