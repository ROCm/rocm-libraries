.. meta::
  :description: hipSPARSE sparse extra functions API documentation
  :keywords: hipSPARSE, rocSPARSE, ROCm, API, documentation, extra functions

.. _hipsparse_extra_functions:

********************************************************************
Sparse extra functions
********************************************************************

This module contains all sparse extra routines.

The sparse extra routines describe operations that manipulate sparse matrices.

hipsparseXcsrgeamNnz()
======================

.. doxygenfunction:: hipsparseXcsrgeamNnz

hipsparseXcsrgeam()
======================

.. doxygenfunction:: hipsparseScsrgeam
  :outline:
.. doxygenfunction:: hipsparseDcsrgeam
  :outline:
.. doxygenfunction:: hipsparseCcsrgeam
  :outline:
.. doxygenfunction:: hipsparseZcsrgeam

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/extra/example_hipsparse_csrgeam_cpp.cpp
         :language: cpp
         :start-after: //! [doc example start]
         :end-before: //! [doc example end]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/extra/example_hipsparse_csrgeam_c.c
         :language: c
         :start-after: /*! [doc example start] */
         :end-before: /*! [doc example end] */
         :linenos:


   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/extra/example_hipsparse_csrgeam_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

hipsparseXcsrgeam2_bufferSizeExt()
==================================

.. doxygenfunction:: hipsparseScsrgeam2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDcsrgeam2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseCcsrgeam2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseZcsrgeam2_bufferSizeExt

hipsparseXcsrgeam2Nnz()
=======================

.. doxygenfunction:: hipsparseXcsrgeam2Nnz

hipsparseXcsrgeam2()
======================

.. doxygenfunction:: hipsparseScsrgeam2
  :outline:
.. doxygenfunction:: hipsparseDcsrgeam2
  :outline:
.. doxygenfunction:: hipsparseCcsrgeam2
  :outline:
.. doxygenfunction:: hipsparseZcsrgeam2

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/extra/example_hipsparse_csrgeam2_cpp.cpp
         :language: cpp
         :start-after: //! [doc example start]
         :end-before: //! [doc example end]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/extra/example_hipsparse_csrgeam2_c.c
         :language: c
         :start-after: /*! [doc example start] */
         :end-before: /*! [doc example end] */
         :linenos:


   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/extra/example_hipsparse_csrgeam2_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

hipsparseXcsrgemmNnz()
======================

.. doxygenfunction:: hipsparseXcsrgemmNnz

hipsparseXcsrgemm()
======================

.. doxygenfunction:: hipsparseScsrgemm
  :outline:
.. doxygenfunction:: hipsparseDcsrgemm
  :outline:
.. doxygenfunction:: hipsparseCcsrgemm
  :outline:
.. doxygenfunction:: hipsparseZcsrgemm

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/extra/example_hipsparse_csrgemm_cpp.cpp
         :language: cpp
         :start-after: //! [doc example start]
         :end-before: //! [doc example end]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/extra/example_hipsparse_csrgemm_c.c
         :language: c
         :start-after: /*! [doc example start] */
         :end-before: /*! [doc example end] */
         :linenos:


   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/extra/example_hipsparse_csrgemm_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:
         
hipsparseXcsrgemm2_bufferSizeExt()
==================================

.. doxygenfunction:: hipsparseScsrgemm2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDcsrgemm2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseCcsrgemm2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseZcsrgemm2_bufferSizeExt

hipsparseXcsrgemm2Nnz()
=======================

.. doxygenfunction:: hipsparseXcsrgemm2Nnz

hipsparseXcsrgemm2()
======================

.. doxygenfunction:: hipsparseScsrgemm2
  :outline:
.. doxygenfunction:: hipsparseDcsrgemm2
  :outline:
.. doxygenfunction:: hipsparseCcsrgemm2
  :outline:
.. doxygenfunction:: hipsparseZcsrgemm2

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/extra/example_hipsparse_csrgemm2_cpp.cpp
         :language: cpp
         :start-after: //! [doc example start]
         :end-before: //! [doc example end]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/extra/example_hipsparse_csrgemm2_c.c
         :language: c
         :start-after: /*! [doc example start] */
         :end-before: /*! [doc example end] */
         :linenos:

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/extra/example_hipsparse_csrgemm2_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:
