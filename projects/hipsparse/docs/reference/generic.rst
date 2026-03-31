.. meta::
  :description: hipSPARSE sparse generic functions API documentation
  :keywords: hipSPARSE, rocSPARSE, ROCm, API, documentation, generic functions

.. _hipsparse_generic_functions:

********************************************************************
Sparse generic functions
********************************************************************

This module contains all sparse generic routines.

The sparse generic routines describe operations that manipulate sparse matrices.

hipsparseAxpby()
================

.. doxygenfunction:: hipsparseAxpby

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_axpby_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_axpby_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseGather()
=================

.. doxygenfunction:: hipsparseGather

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_gather_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_gather_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseScatter()
==================

.. doxygenfunction:: hipsparseScatter

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_scatter_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_scatter_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseRot()
==============

.. doxygenfunction:: hipsparseRot

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_rot_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_rot_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseSparseToDense_bufferSize()
===================================

.. doxygenfunction:: hipsparseSparseToDense_bufferSize

hipsparseSparseToDense()
========================

.. doxygenfunction:: hipsparseSparseToDense

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_sparse_to_dense_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_sparse_to_dense_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseDenseToSparse_bufferSize()
===================================

.. doxygenfunction:: hipsparseDenseToSparse_bufferSize

hipsparseDenseToSparse_analysis()
=================================

.. doxygenfunction:: hipsparseDenseToSparse_analysis

hipsparseDenseToSparse_convert()
================================

.. doxygenfunction:: hipsparseDenseToSparse_convert

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_dense_to_sparse_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_dense_to_sparse_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseSpVV_bufferSize()
==========================

.. doxygenfunction:: hipsparseSpVV_bufferSize

hipsparseSpVV()
===============

.. doxygenfunction:: hipsparseSpVV

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_spvv_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_spvv_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseSpMV_bufferSize()
==========================

.. doxygenfunction:: hipsparseSpMV_bufferSize

hipsparseSpMV_preprocess()
==========================

.. doxygenfunction:: hipsparseSpMV_preprocess

hipsparseSpMV()
===============

.. doxygenfunction:: hipsparseSpMV

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_spmv_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_spmv_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseSpMM_bufferSize()
==========================

.. doxygenfunction:: hipsparseSpMM_bufferSize

hipsparseSpMM_preprocess()
==========================

.. doxygenfunction:: hipsparseSpMM_preprocess

hipsparseSpMM()
===============

.. doxygenfunction:: hipsparseSpMM

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_spmm_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_spmm_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseSpGEMM_createDescr()
=============================

.. doxygenfunction:: hipsparseSpGEMM_createDescr

hipsparseSpGEMM_destroyDescr()
==============================

.. doxygenfunction:: hipsparseSpGEMM_destroyDescr

hipsparseSpGEMM_workEstimation()
================================

.. doxygenfunction:: hipsparseSpGEMM_workEstimation

hipsparseSpGEMM_compute()
=========================

.. doxygenfunction:: hipsparseSpGEMM_compute

hipsparseSpGEMM_copy()
======================

.. doxygenfunction:: hipsparseSpGEMM_copy

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_spgemm_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_spgemm_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseSpGEMMreuse_workEstimation()
=====================================

.. doxygenfunction:: hipsparseSpGEMMreuse_workEstimation

hipsparseSpGEMMreuse_nnz()
==========================

.. doxygenfunction:: hipsparseSpGEMMreuse_nnz

hipsparseSpGEMMreuse_copy()
===========================

.. doxygenfunction:: hipsparseSpGEMMreuse_copy

hipsparseSpGEMMreuse_compute()
==============================

.. doxygenfunction:: hipsparseSpGEMMreuse_compute

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_spgemm_reuse_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_spgemm_reuse_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseSDDMM_bufferSize()
===========================

.. doxygenfunction:: hipsparseSDDMM_bufferSize

hipsparseSDDMM_preprocess()
===========================

.. doxygenfunction:: hipsparseSDDMM_preprocess

hipsparseSDDMM()
================

.. doxygenfunction:: hipsparseSDDMM

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_sddmm_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_sddmm_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseSpSV_createDescr()
===========================

.. doxygenfunction:: hipsparseSpSV_createDescr

hipsparseSpSV_destroyDescr()
============================

.. doxygenfunction:: hipsparseSpSV_destroyDescr

hipsparseSpSV_bufferSize()
==========================

.. doxygenfunction:: hipsparseSpSV_bufferSize

hipsparseSpSV_analysis()
========================

.. doxygenfunction:: hipsparseSpSV_analysis

hipsparseSpSV_solve()
=====================

.. doxygenfunction:: hipsparseSpSV_solve

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_spsv_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_spsv_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseSpSM_createDescr()
===========================

.. doxygenfunction:: hipsparseSpSM_createDescr

hipsparseSpSM_destroyDescr()
============================

.. doxygenfunction:: hipsparseSpSM_destroyDescr

hipsparseSpSM_bufferSize()
==========================

.. doxygenfunction:: hipsparseSpSM_bufferSize

hipsparseSpSM_analysis()
========================

.. doxygenfunction:: hipsparseSpSM_analysis

hipsparseSpSM_solve()
=====================

.. doxygenfunction:: hipsparseSpSM_solve

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_spsm_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/generic/example_hipsparse_spsm_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseSpSM_solve_ex()
========================

.. doxygenfunction:: hipsparseSpSM_solve_ex