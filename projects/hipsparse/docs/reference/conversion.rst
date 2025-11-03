.. meta::
  :description: hipSPARSE sparse conversion functions API documentation
  :keywords: hipSPARSE, rocSPARSE, ROCm, API, documentation, conversion functions

.. _hipsparse_conversion_functions:

********************************************************************
Sparse conversion functions
********************************************************************

This module contains all sparse conversion routines.

The sparse conversion routines describe operations performed on a matrix in sparse format
to obtain a matrix in a different sparse format.

hipsparseXnnz()
===============

.. doxygenfunction:: hipsparseSnnz
  :outline:
.. doxygenfunction:: hipsparseDnnz
  :outline:
.. doxygenfunction:: hipsparseCnnz
  :outline:
.. doxygenfunction:: hipsparseZnnz

hipsparseXdense2csr()
=====================

.. doxygenfunction:: hipsparseSdense2csr
  :outline:
.. doxygenfunction:: hipsparseDdense2csr
  :outline:
.. doxygenfunction:: hipsparseCdense2csr
  :outline:
.. doxygenfunction:: hipsparseZdense2csr

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_dense2csr_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]  
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_dense2csr_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXpruneDense2csr_bufferSize()
=====================================

.. doxygenfunction:: hipsparseSpruneDense2csr_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDpruneDense2csr_bufferSize

hipsparseXpruneDense2csr_bufferSizeExt()
========================================

.. doxygenfunction:: hipsparseSpruneDense2csr_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDpruneDense2csr_bufferSizeExt

hipsparseXpruneDense2csrNnz()
=================================

.. doxygenfunction:: hipsparseSpruneDense2csrNnz
  :outline:
.. doxygenfunction:: hipsparseDpruneDense2csrNnz

hipsparseXpruneDense2csr()
=================================

.. doxygenfunction:: hipsparseSpruneDense2csr
  :outline:
.. doxygenfunction:: hipsparseDpruneDense2csr

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_prune_dense2csr_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_prune_dense2csr_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXpruneDense2csrByPercentage_bufferSize()
===================================================

.. doxygenfunction:: hipsparseSpruneDense2csrByPercentage_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDpruneDense2csrByPercentage_bufferSize

hipsparseXpruneDense2csrByPercentage_bufferSizeExt()
============================================================

.. doxygenfunction:: hipsparseSpruneDense2csrByPercentage_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDpruneDense2csrByPercentage_bufferSizeExt

hipsparseXpruneDense2csrNnzByPercentage()
==========================================

.. doxygenfunction:: hipsparseSpruneDense2csrNnzByPercentage
  :outline:
.. doxygenfunction:: hipsparseDpruneDense2csrNnzByPercentage

hipsparseXpruneDense2csrByPercentage()
==========================================

.. doxygenfunction:: hipsparseSpruneDense2csrByPercentage
  :outline:
.. doxygenfunction:: hipsparseDpruneDense2csrByPercentage

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_prune_dense2csr_by_percentage_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_prune_dense2csr_by_percentage_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXdense2csc()
========================

.. doxygenfunction:: hipsparseSdense2csc
  :outline:
.. doxygenfunction:: hipsparseDdense2csc
  :outline:
.. doxygenfunction:: hipsparseCdense2csc
  :outline:
.. doxygenfunction:: hipsparseZdense2csc

hipsparseXcsr2dense()
========================

.. doxygenfunction:: hipsparseScsr2dense
  :outline:
.. doxygenfunction:: hipsparseDcsr2dense
  :outline:
.. doxygenfunction:: hipsparseCcsr2dense
  :outline:
.. doxygenfunction:: hipsparseZcsr2dense

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_csr2dense_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_csr2dense_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXcsc2dense()
========================

.. doxygenfunction:: hipsparseScsc2dense
  :outline:
.. doxygenfunction:: hipsparseDcsc2dense
  :outline:
.. doxygenfunction:: hipsparseCcsc2dense
  :outline:
.. doxygenfunction:: hipsparseZcsc2dense

hipsparseXcsr2bsrNnz()
========================

.. doxygenfunction:: hipsparseXcsr2bsrNnz

hipsparseXcsr2bsr()
========================

.. doxygenfunction:: hipsparseScsr2bsr
  :outline:
.. doxygenfunction:: hipsparseDcsr2bsr
  :outline:
.. doxygenfunction:: hipsparseCcsr2bsr
  :outline:
.. doxygenfunction:: hipsparseZcsr2bsr

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_csr2bsr_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_csr2bsr_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXnnz_compress()
========================

.. doxygenfunction:: hipsparseSnnz_compress
  :outline:
.. doxygenfunction:: hipsparseDnnz_compress
  :outline:
.. doxygenfunction:: hipsparseCnnz_compress
  :outline:
.. doxygenfunction:: hipsparseZnnz_compress

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_nnz_compress_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_nnz_compress_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXcsr2coo()
========================

.. doxygenfunction:: hipsparseXcsr2coo


.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_coo2csr_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_coo2csr_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXcsr2csc()
========================

.. doxygenfunction:: hipsparseScsr2csc
  :outline:
.. doxygenfunction:: hipsparseDcsr2csc
  :outline:
.. doxygenfunction:: hipsparseCcsr2csc
  :outline:
.. doxygenfunction:: hipsparseZcsr2csc

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_csr2csc_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_csr2csc_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXcsr2cscEx2_bufferSize()
=================================

.. doxygenfunction:: hipsparseCsr2cscEx2_bufferSize

hipsparseXcsr2cscEx2()
======================

.. doxygenfunction:: hipsparseCsr2cscEx2

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_csr2csc_ex2_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_csr2csc_ex2_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXcsr2hyb()
========================

.. doxygenfunction:: hipsparseScsr2hyb
  :outline:
.. doxygenfunction:: hipsparseDcsr2hyb
  :outline:
.. doxygenfunction:: hipsparseCcsr2hyb
  :outline:
.. doxygenfunction:: hipsparseZcsr2hyb

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_csr2hyb_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_csr2hyb_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXgebsr2gebsc_bufferSize()
==================================

.. doxygenfunction:: hipsparseSgebsr2gebsc_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDgebsr2gebsc_bufferSize
  :outline:
.. doxygenfunction:: hipsparseCgebsr2gebsc_bufferSize
  :outline:
.. doxygenfunction:: hipsparseZgebsr2gebsc_bufferSize

hipsparseXgebsr2gebsc()
========================

.. doxygenfunction:: hipsparseSgebsr2gebsc
  :outline:
.. doxygenfunction:: hipsparseDgebsr2gebsc
  :outline:
.. doxygenfunction:: hipsparseCgebsr2gebsc
  :outline:
.. doxygenfunction:: hipsparseZgebsr2gebsc

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_gebsr2gebsc_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_gebsr2gebsc_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXcsr2gebsr_bufferSize()
=================================

.. doxygenfunction:: hipsparseScsr2gebsr_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDcsr2gebsr_bufferSize
  :outline:
.. doxygenfunction:: hipsparseCcsr2gebsr_bufferSize
  :outline:
.. doxygenfunction:: hipsparseZcsr2gebsr_bufferSize

hipsparseXcsr2gebsrNnz()
========================

.. doxygenfunction:: hipsparseXcsr2gebsrNnz


.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_csr2gebsr_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_csr2gebsr_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXcsr2gebsr()
========================

.. doxygenfunction:: hipsparseScsr2gebsr
  :outline:
.. doxygenfunction:: hipsparseDcsr2gebsr
  :outline:
.. doxygenfunction:: hipsparseCcsr2gebsr
  :outline:
.. doxygenfunction:: hipsparseZcsr2gebsr

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_csr2gebsr_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_csr2gebsr_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXbsr2csr()
========================

.. doxygenfunction:: hipsparseSbsr2csr
  :outline:
.. doxygenfunction:: hipsparseDbsr2csr
  :outline:
.. doxygenfunction:: hipsparseCbsr2csr
  :outline:
.. doxygenfunction:: hipsparseZbsr2csr

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_bsr2csr_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_bsr2csr_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXgebsr2csr()
========================

.. doxygenfunction:: hipsparseSgebsr2csr
  :outline:
.. doxygenfunction:: hipsparseDgebsr2csr
  :outline:
.. doxygenfunction:: hipsparseCgebsr2csr
  :outline:
.. doxygenfunction:: hipsparseZgebsr2csr

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_gebsr2csr_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_gebsr2csr_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXcsr2csr_compress()
=============================

.. doxygenfunction:: hipsparseScsr2csr_compress
  :outline:
.. doxygenfunction:: hipsparseDcsr2csr_compress
  :outline:
.. doxygenfunction:: hipsparseCcsr2csr_compress
  :outline:
.. doxygenfunction:: hipsparseZcsr2csr_compress

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_csr2csr_compress_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_csr2csr_compress_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXpruneCsr2csr_bufferSize()
==========================================

.. doxygenfunction:: hipsparseSpruneCsr2csr_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDpruneCsr2csr_bufferSize

hipsparseXpruneCsr2csr_bufferSizeExt()
==========================================

.. doxygenfunction:: hipsparseSpruneCsr2csr_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDpruneCsr2csr_bufferSizeExt

hipsparseXpruneCsr2csrNnz()
=================================

.. doxygenfunction:: hipsparseSpruneCsr2csrNnz
  :outline:
.. doxygenfunction:: hipsparseDpruneCsr2csrNnz

hipsparseXpruneCsr2csr()
========================

.. doxygenfunction:: hipsparseSpruneCsr2csr
  :outline:
.. doxygenfunction:: hipsparseDpruneCsr2csr

hipsparseXpruneCsr2csrByPercentage_bufferSize()
===================================================

.. doxygenfunction:: hipsparseSpruneCsr2csrByPercentage_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDpruneCsr2csrByPercentage_bufferSize

hipsparseXpruneCsr2csrByPercentage_bufferSizeExt()
===================================================

.. doxygenfunction:: hipsparseSpruneCsr2csrByPercentage_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDpruneCsr2csrByPercentage_bufferSizeExt

hipsparseXpruneCsr2csrNnzByPercentage()
=======================================

.. doxygenfunction:: hipsparseSpruneCsr2csrNnzByPercentage
  :outline:
.. doxygenfunction:: hipsparseDpruneCsr2csrNnzByPercentage

hipsparseXpruneCsr2csrByPercentage()
==========================================

.. doxygenfunction:: hipsparseSpruneCsr2csrByPercentage
  :outline:
.. doxygenfunction:: hipsparseDpruneCsr2csrByPercentage

hipsparseXhyb2csr()
===================

.. doxygenfunction:: hipsparseShyb2csr
  :outline:
.. doxygenfunction:: hipsparseDhyb2csr
  :outline:
.. doxygenfunction:: hipsparseChyb2csr
  :outline:
.. doxygenfunction:: hipsparseZhyb2csr

hipsparseXcoo2csr()
========================

.. doxygenfunction:: hipsparseXcoo2csr


.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_coo2csr_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_coo2csr_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseCreateIdentityPermutation()
==========================================

.. doxygenfunction:: hipsparseCreateIdentityPermutation

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_create_identity_permutation_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_create_identity_permutation_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXcsrsort_bufferSizeExt()
=================================

.. doxygenfunction:: hipsparseXcsrsort_bufferSizeExt


.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_csrsort_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_csrsort_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXcsrsort()
========================

.. doxygenfunction:: hipsparseXcsrsort

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_csrsort_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_csrsort_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXcscsort_bufferSizeExt()
=================================

.. doxygenfunction:: hipsparseXcscsort_bufferSizeExt


.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_cscsort_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_cscsort_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXcscsort()
========================

.. doxygenfunction:: hipsparseXcscsort

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_cscsort_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_cscsort_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXcoosort_bufferSizeExt()
=================================

.. doxygenfunction:: hipsparseXcoosort_bufferSizeExt


.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_coosort_by_row_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_coosort_by_row_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXcoosortByRow()
========================

.. doxygenfunction:: hipsparseXcoosortByRow

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_coosort_by_row_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_coosort_by_row_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXcoosortByColumn()
=================================

.. doxygenfunction:: hipsparseXcoosortByColumn


.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_coosort_by_column_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_coosort_by_column_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXgebsr2gebsr_bufferSize()
==========================================

.. doxygenfunction:: hipsparseSgebsr2gebsr_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDgebsr2gebsr_bufferSize
  :outline:
.. doxygenfunction:: hipsparseCgebsr2gebsr_bufferSize
  :outline:
.. doxygenfunction:: hipsparseZgebsr2gebsr_bufferSize

hipsparseXgebsr2gebsrNnz()
=================================

.. doxygenfunction:: hipsparseXgebsr2gebsrNnz


.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_gebsr2gebsr_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_gebsr2gebsr_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXgebsr2gebsr()
========================

.. doxygenfunction:: hipsparseSgebsr2gebsr
  :outline:
.. doxygenfunction:: hipsparseDgebsr2gebsr
  :outline:
.. doxygenfunction:: hipsparseCgebsr2gebsr
  :outline:
.. doxygenfunction:: hipsparseZgebsr2gebsr

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_gebsr2gebsr_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/conversion/example_hipsparse_gebsr2gebsr_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXcsru2csr_bufferSizeExt()
==================================

.. doxygenfunction:: hipsparseScsru2csr_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDcsru2csr_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseCcsru2csr_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseZcsru2csr_bufferSizeExt

hipsparseXcsru2csr()
========================

.. doxygenfunction:: hipsparseScsru2csr
  :outline:
.. doxygenfunction:: hipsparseDcsru2csr
  :outline:
.. doxygenfunction:: hipsparseCcsru2csr
  :outline:
.. doxygenfunction:: hipsparseZcsru2csr

hipsparseXcsr2csru()
========================

.. doxygenfunction:: hipsparseScsr2csru
  :outline:
.. doxygenfunction:: hipsparseDcsr2csru
  :outline:
.. doxygenfunction:: hipsparseCcsr2csru
  :outline:
.. doxygenfunction:: hipsparseZcsr2csru
