.. meta::
  :description: hipSPARSE sparse level 2 functions API documentation
  :keywords: hipSPARSE, rocSPARSE, ROCm, API, documentation, level 2 functions

.. _hipsparse_level2_functions:

********************************************************************
Sparse level 2 functions
********************************************************************

This module contains all sparse level 2 routines.

The sparse level 2 routines describe operations between a matrix in sparse format and a vector in dense format.

hipsparseXcsrmv()
==================

.. doxygenfunction:: hipsparseScsrmv
  :outline:
.. doxygenfunction:: hipsparseDcsrmv
  :outline:
.. doxygenfunction:: hipsparseCcsrmv
  :outline:
.. doxygenfunction:: hipsparseZcsrmv

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_hipsparse_csrmv_cpp.cpp
         :language: cpp
         :start-after: //! [doc example start]
         :end-before: //! [doc example end]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_hipsparse_csrmv_c.c
         :language: c
         :start-after: /*! [doc example start] */
         :end-before: /*! [doc example end] */
         :linenos:

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_hipsparse_csrmv_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

hipsparseXcsrsv2_zeroPivot()
=============================

.. doxygenfunction:: hipsparseXcsrsv2_zeroPivot

hipsparseXcsrsv2_bufferSize()
=============================

.. doxygenfunction:: hipsparseScsrsv2_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDcsrsv2_bufferSize
  :outline:
.. doxygenfunction:: hipsparseCcsrsv2_bufferSize
  :outline:
.. doxygenfunction:: hipsparseZcsrsv2_bufferSize

hipsparseXcsrsv2_bufferSizeExt()
================================

.. doxygenfunction:: hipsparseScsrsv2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDcsrsv2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseCcsrsv2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseZcsrsv2_bufferSizeExt

hipsparseXcsrsv2_analysis()
===========================

.. doxygenfunction:: hipsparseScsrsv2_analysis
  :outline:
.. doxygenfunction:: hipsparseDcsrsv2_analysis
  :outline:
.. doxygenfunction:: hipsparseCcsrsv2_analysis
  :outline:
.. doxygenfunction:: hipsparseZcsrsv2_analysis

hipsparseXcsrsv2_solve()
========================

.. doxygenfunction:: hipsparseScsrsv2_solve
  :outline:
.. doxygenfunction:: hipsparseDcsrsv2_solve
  :outline:
.. doxygenfunction:: hipsparseCcsrsv2_solve
  :outline:
.. doxygenfunction:: hipsparseZcsrsv2_solve

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_hipsparse_csrsv2_cpp.cpp
         :language: cpp
         :start-after: //! [doc example start]
         :end-before: //! [doc example end]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_hipsparse_csrsv2_c.c
         :language: c
         :start-after: /*! [doc example start] */
         :end-before: /*! [doc example end] */
         :linenos:

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_hipsparse_csrsv2_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

hipsparseXhybmv()
=================

.. doxygenfunction:: hipsparseShybmv
  :outline:
.. doxygenfunction:: hipsparseDhybmv
  :outline:
.. doxygenfunction:: hipsparseChybmv
  :outline:
.. doxygenfunction:: hipsparseZhybmv

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_hipsparse_hybmv_cpp.cpp
         :language: cpp
         :start-after: //! [doc example start]
         :end-before: //! [doc example end]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_hipsparse_hybmv_c.c
         :language: c
         :start-after: /*! [doc example start] */
         :end-before: /*! [doc example end] */
         :linenos:

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_hipsparse_hybmv_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

hipsparseXbsrmv()
=================

.. doxygenfunction:: hipsparseSbsrmv
  :outline:
.. doxygenfunction:: hipsparseDbsrmv
  :outline:
.. doxygenfunction:: hipsparseCbsrmv
  :outline:
.. doxygenfunction:: hipsparseZbsrmv

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_hipsparse_bsrmv_cpp.cpp
         :language: cpp
         :start-after: //! [doc example start]
         :end-before: //! [doc example end]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_hipsparse_bsrmv_c.c
         :language: c
         :start-after: /*! [doc example start] */
         :end-before: /*! [doc example end] */
         :linenos:

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_hipsparse_bsrmv_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

hipsparseXbsrxmv()
==================

.. doxygenfunction:: hipsparseSbsrxmv
  :outline:
.. doxygenfunction:: hipsparseDbsrxmv
  :outline:
.. doxygenfunction:: hipsparseCbsrxmv
  :outline:
.. doxygenfunction:: hipsparseZbsrxmv

hipsparseXbsrsv2_zeroPivot()
============================

.. doxygenfunction:: hipsparseXbsrsv2_zeroPivot

hipsparseXbsrsv2_bufferSize()
=============================

.. doxygenfunction:: hipsparseSbsrsv2_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDbsrsv2_bufferSize
  :outline:
.. doxygenfunction:: hipsparseCbsrsv2_bufferSize
  :outline:
.. doxygenfunction:: hipsparseZbsrsv2_bufferSize

hipsparseXbsrsv2_bufferSizeExt()
================================

.. doxygenfunction:: hipsparseSbsrsv2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDbsrsv2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseCbsrsv2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseZbsrsv2_bufferSizeExt

hipsparseXbsrsv2_analysis()
===========================

.. doxygenfunction:: hipsparseSbsrsv2_analysis
  :outline:
.. doxygenfunction:: hipsparseDbsrsv2_analysis
  :outline:
.. doxygenfunction:: hipsparseCbsrsv2_analysis
  :outline:
.. doxygenfunction:: hipsparseZbsrsv2_analysis

hipsparseXbsrsv2_solve()
========================

.. doxygenfunction:: hipsparseSbsrsv2_solve
  :outline:
.. doxygenfunction:: hipsparseDbsrsv2_solve
  :outline:
.. doxygenfunction:: hipsparseCbsrsv2_solve
  :outline:
.. doxygenfunction:: hipsparseZbsrsv2_solve

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_hipsparse_bsrsv2_cpp.cpp
         :language: cpp
         :start-after: //! [doc example start]
         :end-before: //! [doc example end]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_hipsparse_bsrsv2_c.c
         :language: c
         :start-after: /*! [doc example start] */
         :end-before: /*! [doc example end] */
         :linenos:

hipsparseXgemvi_bufferSize()
============================

.. doxygenfunction:: hipsparseSgemvi_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDgemvi_bufferSize
  :outline:
.. doxygenfunction:: hipsparseCgemvi_bufferSize
  :outline:
.. doxygenfunction:: hipsparseZgemvi_bufferSize

hipsparseXgemvi()
=================

.. doxygenfunction:: hipsparseSgemvi
  :outline:
.. doxygenfunction:: hipsparseDgemvi
  :outline:
.. doxygenfunction:: hipsparseCgemvi
  :outline:
.. doxygenfunction:: hipsparseZgemvi

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_hipsparse_gemvi_cpp.cpp
         :language: cpp
         :start-after: //! [doc example start]
         :end-before: //! [doc example end]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level2/example_hipsparse_gemvi_c.c
         :language: c
         :start-after: /*! [doc example start] */
         :end-before: /*! [doc example end] */
         :linenos:
