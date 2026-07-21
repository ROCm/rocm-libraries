.. meta::
  :description: hipSPARSE sparse level 1 functions API documentation
  :keywords: hipSPARSE, rocSPARSE, ROCm, API, documentation, level 1 functions

.. _hipsparse_level1_functions:

********************************************************************
Sparse level 1 functions
********************************************************************

The sparse level 1 routines describe operations between a vector in sparse format and a vector in dense format.
This section describes all hipSPARSE level 1 sparse linear algebra functions.

hipsparseXaxpyi()
=================

.. doxygenfunction:: hipsparseSaxpyi
  :outline:
.. doxygenfunction:: hipsparseDaxpyi
  :outline:
.. doxygenfunction:: hipsparseCaxpyi
  :outline:
.. doxygenfunction:: hipsparseZaxpyi

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_hipsparse_axpyi_cpp.cpp
         :language: cpp
         :start-after: //! [doc example start]
         :end-before: //! [doc example end]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_hipsparse_axpyi_c.c
         :language: c
         :start-after: /*! [doc example start] */
         :end-before: /*! [doc example end] */
         :linenos:

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_hipsparse_axpyi_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:




hipsparseXdoti()
=================

.. doxygenfunction:: hipsparseSdoti
  :outline:
.. doxygenfunction:: hipsparseDdoti
  :outline:
.. doxygenfunction:: hipsparseCdoti
  :outline:
.. doxygenfunction:: hipsparseZdoti

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_hipsparse_doti_cpp.cpp
         :language: cpp
         :start-after: //! [doc example start]
         :end-before: //! [doc example end]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_hipsparse_doti_c.c
         :language: c
         :start-after: /*! [doc example start] */
         :end-before: /*! [doc example end] */
         :linenos:

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_hipsparse_doti_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

hipsparseXdotci()
=================

.. doxygenfunction:: hipsparseCdotci
  :outline:
.. doxygenfunction:: hipsparseZdotci

hipsparseXgthr()
=================

.. doxygenfunction:: hipsparseSgthr
  :outline:
.. doxygenfunction:: hipsparseDgthr
  :outline:
.. doxygenfunction:: hipsparseCgthr
  :outline:
.. doxygenfunction:: hipsparseZgthr

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_hipsparse_gthr_cpp.cpp
         :language: cpp
         :start-after: //! [doc example start]
         :end-before: //! [doc example end]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_hipsparse_gthr_c.c
         :language: c
         :start-after: /*! [doc example start] */
         :end-before: /*! [doc example end] */
         :linenos:

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_hipsparse_gthr_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

hipsparseXgthrz()
=================

.. doxygenfunction:: hipsparseSgthrz
  :outline:
.. doxygenfunction:: hipsparseDgthrz
  :outline:
.. doxygenfunction:: hipsparseCgthrz
  :outline:
.. doxygenfunction:: hipsparseZgthrz

hipsparseXroti()
=================

.. doxygenfunction:: hipsparseSroti
  :outline:
.. doxygenfunction:: hipsparseDroti

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_hipsparse_roti_cpp.cpp
         :language: cpp
         :start-after: //! [doc example start]
         :end-before: //! [doc example end]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_hipsparse_roti_c.c
         :language: c
         :start-after: /*! [doc example start] */
         :end-before: /*! [doc example end] */
         :linenos:

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_hipsparse_roti_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:

hipsparseXsctr()
=================

.. doxygenfunction:: hipsparseSsctr
  :outline:
.. doxygenfunction:: hipsparseDsctr
  :outline:
.. doxygenfunction:: hipsparseCsctr
  :outline:
.. doxygenfunction:: hipsparseZsctr

.. tabs::

   .. tab:: C++

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_hipsparse_sctr_cpp.cpp
         :language: cpp
         :start-after: //! [doc example start]
         :end-before: //! [doc example end]
         :linenos:

   .. tab:: C

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_hipsparse_sctr_c.c
         :language: c
         :start-after: /*! [doc example start] */
         :end-before: /*! [doc example end] */
         :linenos:

   .. tab:: Fortran

      .. literalinclude:: ../../clients/samples/documentation_examples/level1/example_hipsparse_sctr_fortran.f90
         :language: fortran
         :start-after: ! [doc example start]
         :end-before: ! [doc example end]
         :linenos:
