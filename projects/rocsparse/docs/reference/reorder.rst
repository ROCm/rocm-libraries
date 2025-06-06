.. meta::
  :description: rocSPARSE reordering functions API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, reordering functions

.. _rocsparse_reordering_functions_:

********************************************************************
Sparse reordering functions
********************************************************************

This module contains all sparse reordering routines.

The sparse reordering routines describe algorithms for reordering sparse matrices.
These routines do not support execution in a ``hipGraph`` context.

rocsparse_csrcolor()
--------------------

.. doxygenfunction:: rocsparse_scsrcolor
  :outline:
.. doxygenfunction:: rocsparse_dcsrcolor
  :outline:
.. doxygenfunction:: rocsparse_ccsrcolor
  :outline:
.. doxygenfunction:: rocsparse_zcsrcolor
