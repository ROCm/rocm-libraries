.. meta::
  :description: hipSPARSE preconditioner functions API documentation
  :keywords: hipSPARSE, rocSPARSE, ROCm, API, documentation, preconditioner functions

.. _hipsparse_precond_functions:

********************************************************************
Preconditioner functions
********************************************************************

This module contains all sparse preconditioners.

The sparse preconditioners describe manipulations on a matrix in sparse format to obtain a sparse preconditioner matrix.

hipsparseXbsrilu02_zeroPivot()
==============================

.. doxygenfunction:: hipsparseXbsrilu02_zeroPivot

hipsparseXbsrilu02_numericBoost()
=================================

.. doxygenfunction:: hipsparseSbsrilu02_numericBoost
  :outline:
.. doxygenfunction:: hipsparseDbsrilu02_numericBoost
  :outline:
.. doxygenfunction:: hipsparseCbsrilu02_numericBoost
  :outline:
.. doxygenfunction:: hipsparseZbsrilu02_numericBoost

hipsparseXbsrilu02_bufferSize()
===============================

.. doxygenfunction:: hipsparseSbsrilu02_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDbsrilu02_bufferSize
  :outline:
.. doxygenfunction:: hipsparseCbsrilu02_bufferSize
  :outline:
.. doxygenfunction:: hipsparseZbsrilu02_bufferSize

hipsparseXbsrilu02_analysis()
=============================

.. doxygenfunction:: hipsparseSbsrilu02_analysis
  :outline:
.. doxygenfunction:: hipsparseDbsrilu02_analysis
  :outline:
.. doxygenfunction:: hipsparseCbsrilu02_analysis
  :outline:
.. doxygenfunction:: hipsparseZbsrilu02_analysis

hipsparseXbsrilu02()
====================

.. doxygenfunction:: hipsparseSbsrilu02
  :outline:
.. doxygenfunction:: hipsparseDbsrilu02
  :outline:
.. doxygenfunction:: hipsparseCbsrilu02
  :outline:
.. doxygenfunction:: hipsparseZbsrilu02

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_hipsparse_bsrilu02_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_hipsparse_bsrilu02_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

  .. tab:: Fortran

    .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_hipsparse_bsrilu02_fortran.f90
      :language: fortran
      :start-after: ! [doc example start]
      :end-before: ! [doc example end]
      :linenos:

hipsparseXcsrilu02_zeroPivot()
==============================

.. doxygenfunction:: hipsparseXcsrilu02_zeroPivot

hipsparseXcsrilu02_numericBoost()
=================================

.. doxygenfunction:: hipsparseScsrilu02_numericBoost
  :outline:
.. doxygenfunction:: hipsparseDcsrilu02_numericBoost
  :outline:
.. doxygenfunction:: hipsparseCcsrilu02_numericBoost
  :outline:
.. doxygenfunction:: hipsparseZcsrilu02_numericBoost

hipsparseXcsrilu02_bufferSize()
================================

.. doxygenfunction:: hipsparseScsrilu02_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDcsrilu02_bufferSize
  :outline:
.. doxygenfunction:: hipsparseCcsrilu02_bufferSize
  :outline:
.. doxygenfunction:: hipsparseZcsrilu02_bufferSize

hipsparseXcsrilu02_bufferSizeExt()
===================================

.. doxygenfunction:: hipsparseScsrilu02_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDcsrilu02_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseCcsrilu02_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseZcsrilu02_bufferSizeExt

hipsparseXcsrilu02_analysis()
=============================

.. doxygenfunction:: hipsparseScsrilu02_analysis
  :outline:
.. doxygenfunction:: hipsparseDcsrilu02_analysis
  :outline:
.. doxygenfunction:: hipsparseCcsrilu02_analysis
  :outline:
.. doxygenfunction:: hipsparseZcsrilu02_analysis

hipsparseXcsrilu02()
====================

.. doxygenfunction:: hipsparseScsrilu02
  :outline:
.. doxygenfunction:: hipsparseDcsrilu02
  :outline:
.. doxygenfunction:: hipsparseCcsrilu02
  :outline:
.. doxygenfunction:: hipsparseZcsrilu02

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_hipsparse_csrilu02_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_hipsparse_csrilu02_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

  .. tab:: Fortran

    .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_hipsparse_csrilu02_fortran.f90
      :language: fortran
      :start-after: ! [doc example start]
      :end-before: ! [doc example end]
      :linenos:

hipsparseXbsric02_zeroPivot()
=============================

.. doxygenfunction:: hipsparseXbsric02_zeroPivot

hipsparseXbsric02_bufferSize()
==============================

.. doxygenfunction:: hipsparseSbsric02_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDbsric02_bufferSize
  :outline:
.. doxygenfunction:: hipsparseCbsric02_bufferSize
  :outline:
.. doxygenfunction:: hipsparseZbsric02_bufferSize

hipsparseXbsric02_analysis()
=============================

.. doxygenfunction:: hipsparseSbsric02_analysis
  :outline:
.. doxygenfunction:: hipsparseDbsric02_analysis
  :outline:
.. doxygenfunction:: hipsparseCbsric02_analysis
  :outline:
.. doxygenfunction:: hipsparseZbsric02_analysis

hipsparseXbsric02()
====================

.. doxygenfunction:: hipsparseSbsric02
  :outline:
.. doxygenfunction:: hipsparseDbsric02
  :outline:
.. doxygenfunction:: hipsparseCbsric02
  :outline:
.. doxygenfunction:: hipsparseZbsric02

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_hipsparse_bsric02_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_hipsparse_bsric02_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:


  .. tab:: Fortran

    .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_hipsparse_bsric02_fortran.f90
      :language: fortran
      :start-after: ! [doc example start]
      :end-before: ! [doc example end]
      :linenos:

hipsparseXcsric02_zeroPivot()
=============================

.. doxygenfunction:: hipsparseXcsric02_zeroPivot

hipsparseXcsric02_bufferSize()
==============================

.. doxygenfunction:: hipsparseScsric02_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDcsric02_bufferSize
  :outline:
.. doxygenfunction:: hipsparseCcsric02_bufferSize
  :outline:
.. doxygenfunction:: hipsparseZcsric02_bufferSize

hipsparseXcsric02_bufferSizeExt()
=================================

.. doxygenfunction:: hipsparseScsric02_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDcsric02_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseCcsric02_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseZcsric02_bufferSizeExt

hipsparseXcsric02_analysis()
============================

.. doxygenfunction:: hipsparseScsric02_analysis
  :outline:
.. doxygenfunction:: hipsparseDcsric02_analysis
  :outline:
.. doxygenfunction:: hipsparseCcsric02_analysis
  :outline:
.. doxygenfunction:: hipsparseZcsric02_analysis

hipsparseXcsric02()
===================

.. doxygenfunction:: hipsparseScsric02
  :outline:
.. doxygenfunction:: hipsparseDcsric02
  :outline:
.. doxygenfunction:: hipsparseCcsric02
  :outline:
.. doxygenfunction:: hipsparseZcsric02

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_hipsparse_csric02_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_hipsparse_csric02_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

  .. tab:: Fortran

    .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_hipsparse_csric02_fortran.f90
      :language: fortran
      :start-after: ! [doc example start]
      :end-before: ! [doc example end]
      :linenos:

hipsparseXgtsv2_bufferSizeExt()
===============================

.. doxygenfunction:: hipsparseSgtsv2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDgtsv2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseCgtsv2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseZgtsv2_bufferSizeExt

hipsparseXgtsv2()
=================

.. doxygenfunction:: hipsparseSgtsv2
  :outline:
.. doxygenfunction:: hipsparseDgtsv2
  :outline:
.. doxygenfunction:: hipsparseCgtsv2
  :outline:
.. doxygenfunction:: hipsparseZgtsv2

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_hipsparse_gtsv_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_hipsparse_gtsv_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXgtsv2_nopivot_bufferSizeExt()
=======================================

.. doxygenfunction:: hipsparseSgtsv2_nopivot_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDgtsv2_nopivot_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseCgtsv2_nopivot_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseZgtsv2_nopivot_bufferSizeExt

hipsparseXgtsv2_nopivot()
=========================

.. doxygenfunction:: hipsparseSgtsv2_nopivot
  :outline:
.. doxygenfunction:: hipsparseDgtsv2_nopivot
  :outline:
.. doxygenfunction:: hipsparseCgtsv2_nopivot
  :outline:
.. doxygenfunction:: hipsparseZgtsv2_nopivot

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_hipsparse_gtsv_no_pivot_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_hipsparse_gtsv_no_pivot_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXgtsv2StridedBatch_bufferSizeExt()
===========================================

.. doxygenfunction:: hipsparseSgtsv2StridedBatch_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDgtsv2StridedBatch_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseCgtsv2StridedBatch_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseZgtsv2StridedBatch_bufferSizeExt

hipsparseXgtsv2StridedBatch()
=============================

.. doxygenfunction:: hipsparseSgtsv2StridedBatch
  :outline:
.. doxygenfunction:: hipsparseDgtsv2StridedBatch
  :outline:
.. doxygenfunction:: hipsparseCgtsv2StridedBatch
  :outline:
.. doxygenfunction:: hipsparseZgtsv2StridedBatch

hipsparseXgtsvInterleavedBatch_bufferSizeExt()
==============================================

.. doxygenfunction:: hipsparseSgtsvInterleavedBatch_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDgtsvInterleavedBatch_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseCgtsvInterleavedBatch_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseZgtsvInterleavedBatch_bufferSizeExt

hipsparseXgtsvInterleavedBatch()
================================

.. doxygenfunction:: hipsparseSgtsvInterleavedBatch
  :outline:
.. doxygenfunction:: hipsparseDgtsvInterleavedBatch
  :outline:
.. doxygenfunction:: hipsparseCgtsvInterleavedBatch
  :outline:
.. doxygenfunction:: hipsparseZgtsvInterleavedBatch

.. tabs::

  .. tab:: C++

    .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_hipsparse_gtsv_interleaved_batch_cpp.cpp
      :language: cpp
      :start-after: //! [doc example start]
      :end-before: //! [doc example end]
      :linenos:

  .. tab:: C

    .. literalinclude:: ../../clients/samples/documentation_examples/precond/example_hipsparse_gtsv_interleaved_batch_c.c
      :language: c
      :start-after: /*! [doc example start] */
      :end-before: /*! [doc example end] */
      :linenos:

hipsparseXgpsvInterleavedBatch_bufferSizeExt()
==============================================

.. doxygenfunction:: hipsparseSgpsvInterleavedBatch_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDgpsvInterleavedBatch_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseCgpsvInterleavedBatch_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseZgpsvInterleavedBatch_bufferSizeExt

hipsparseXgpsvInterleavedBatch()
================================

.. doxygenfunction:: hipsparseSgpsvInterleavedBatch
  :outline:
.. doxygenfunction:: hipsparseDgpsvInterleavedBatch
  :outline:
.. doxygenfunction:: hipsparseCgpsvInterleavedBatch
  :outline:
.. doxygenfunction:: hipsparseZgpsvInterleavedBatch
