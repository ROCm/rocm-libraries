.. meta::
  :description: rocSPARSE documentation and API reference library
  :keywords: rocSPARSE, ROCm, API, documentation

.. _rocsparse_reordering_functions_:

********************************************************************
Sparse reordering functions
********************************************************************

This module holds all sparse reordering routines.

The sparse reordering routines describe algorithm for reordering sparse matrices.

The routines in this module do not support execution in a hipGraph context.

rocsparse_csrcolor()
--------------------

.. doxygenfunction:: rocsparse_scsrcolor
  :outline:
.. doxygenfunction:: rocsparse_dcsrcolor
  :outline:
.. doxygenfunction:: rocsparse_ccsrcolor
  :outline:
.. doxygenfunction:: rocsparse_zcsrcolor
