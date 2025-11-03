.. meta::
  :description: hipSPARSE sparse level 3 functions API documentation
  :keywords: hipSPARSE, rocSPARSE, ROCm, API, documentation, level 3 functions

.. _hipsparse_level3_functions:

********************************************************************
Sparse level 3 functions
********************************************************************

This module contains all sparse level 3 routines.

The sparse level 3 routines describe operations between a matrix in sparse format and multiple vectors in dense format
that can also be seen as a dense matrix.

hipsparseXbsrmm()
=================

.. doxygenfunction:: hipsparseSbsrmm
  :outline:
.. doxygenfunction:: hipsparseDbsrmm
  :outline:
.. doxygenfunction:: hipsparseCbsrmm
  :outline:
.. doxygenfunction:: hipsparseZbsrmm

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level3/example_hipsparse_bsrmm_cpp.cpp
         :language: cpp
         :start-after: //! [doc example start]
         :end-before: //! [doc example end]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level3/example_hipsparse_bsrmm_c.c
         :language: c
         :start-after: /*! [doc example start] */
         :end-before: /*! [doc example end] */
         :linenos:

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/level3/example_hipsparse_bsrmm_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

hipsparseXcsrmm()
=================

.. doxygenfunction:: hipsparseScsrmm
  :outline:
.. doxygenfunction:: hipsparseDcsrmm
  :outline:
.. doxygenfunction:: hipsparseCcsrmm
  :outline:
.. doxygenfunction:: hipsparseZcsrmm

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level3/example_hipsparse_csrmm_cpp.cpp
         :language: cpp
         :start-after: //! [doc example start]
         :end-before: //! [doc example end]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level3/example_hipsparse_csrmm_c.c
         :language: c
         :start-after: /*! [doc example start] */
         :end-before: /*! [doc example end] */
         :linenos:

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/level3/example_hipsparse_csrmm_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

hipsparseXcsrmm2()
==================

.. doxygenfunction:: hipsparseScsrmm2
  :outline:
.. doxygenfunction:: hipsparseDcsrmm2
  :outline:
.. doxygenfunction:: hipsparseCcsrmm2
  :outline:
.. doxygenfunction:: hipsparseZcsrmm2

hipsparseXbsrsm2_zeroPivot()
============================

.. doxygenfunction:: hipsparseXbsrsm2_zeroPivot

hipsparseXbsrsm2_bufferSize()
=============================

.. doxygenfunction:: hipsparseSbsrsm2_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDbsrsm2_bufferSize
  :outline:
.. doxygenfunction:: hipsparseCbsrsm2_bufferSize
  :outline:
.. doxygenfunction:: hipsparseZbsrsm2_bufferSize

hipsparseXbsrsm2_analysis()
===========================

.. doxygenfunction:: hipsparseSbsrsm2_analysis
  :outline:
.. doxygenfunction:: hipsparseDbsrsm2_analysis
  :outline:
.. doxygenfunction:: hipsparseCbsrsm2_analysis
  :outline:
.. doxygenfunction:: hipsparseZbsrsm2_analysis

hipsparseXbsrsm2_solve()
========================

.. doxygenfunction:: hipsparseSbsrsm2_solve
  :outline:
.. doxygenfunction:: hipsparseDbsrsm2_solve
  :outline:
.. doxygenfunction:: hipsparseCbsrsm2_solve
  :outline:
.. doxygenfunction:: hipsparseZbsrsm2_solve

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level3/example_hipsparse_bsrsm_cpp.cpp
         :language: cpp
         :start-after: //! [doc example start]
         :end-before: //! [doc example end]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level3/example_hipsparse_bsrsm_c.c
         :language: c
         :start-after: /*! [doc example start] */
         :end-before: /*! [doc example end] */
         :linenos:

hipsparseXcsrsm2_zeroPivot()
=============================

.. doxygenfunction:: hipsparseXcsrsm2_zeroPivot

hipsparseXcsrsm2_bufferSizeExt()
================================

.. doxygenfunction:: hipsparseScsrsm2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDcsrsm2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseCcsrsm2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseZcsrsm2_bufferSizeExt

hipsparseXcsrsm2_analysis()
===========================

.. doxygenfunction:: hipsparseScsrsm2_analysis
  :outline:
.. doxygenfunction:: hipsparseDcsrsm2_analysis
  :outline:
.. doxygenfunction:: hipsparseCcsrsm2_analysis
  :outline:
.. doxygenfunction:: hipsparseZcsrsm2_analysis

hipsparseXcsrsm2_solve()
========================

.. doxygenfunction:: hipsparseScsrsm2_solve
  :outline:
.. doxygenfunction:: hipsparseDcsrsm2_solve
  :outline:
.. doxygenfunction:: hipsparseCcsrsm2_solve
  :outline:
.. doxygenfunction:: hipsparseZcsrsm2_solve

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level3/example_hipsparse_csrsm2_cpp.cpp
         :language: cpp
         :start-after: //! [doc example start]
         :end-before: //! [doc example end]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level3/example_hipsparse_csrsm2_c.c
         :language: c
         :start-after: /*! [doc example start] */
         :end-before: /*! [doc example end] */
         :linenos:

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/level3/example_hipsparse_csrsm2_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

hipsparseXgemmi()
=================

.. doxygenfunction:: hipsparseSgemmi
  :outline:
.. doxygenfunction:: hipsparseDgemmi
  :outline:
.. doxygenfunction:: hipsparseCgemmi
  :outline:
.. doxygenfunction:: hipsparseZgemmi

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level3/example_hipsparse_gemmi_cpp.cpp
         :language: cpp
         :start-after: //! [doc example start]
         :end-before: //! [doc example end]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level3/example_hipsparse_gemmi_c.c
         :language: c
         :start-after: /*! [doc example start] */
         :end-before: /*! [doc example end] */
         :linenos:

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/level3/example_hipsparse_gemmi_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:
