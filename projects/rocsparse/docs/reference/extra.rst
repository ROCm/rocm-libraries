.. meta::
  :description: rocSPARSE extra functions API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, extra functions

.. _rocsparse_extra_functions_:

********************************************************************
Sparse extra functions
********************************************************************

This module contains all sparse extra routines.

The sparse extra routines describe operations that manipulate sparse matrices.

The routines in this module do not support execution in a ``hipGraph`` context.

rocsparse_bsrgeam_nnzb()
------------------------

.. doxygenfunction:: rocsparse_bsrgeam_nnzb

rocsparse_bsrgeam()
-------------------

.. doxygenfunction:: rocsparse_sbsrgeam
  :outline:
.. doxygenfunction:: rocsparse_dbsrgeam
  :outline:
.. doxygenfunction:: rocsparse_cbsrgeam
  :outline:
.. doxygenfunction:: rocsparse_zbsrgeam

rocsparse_bsrgemm_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_sbsrgemm_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dbsrgemm_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_cbsrgemm_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zbsrgemm_buffer_size

rocsparse_bsrgemm_nnzb()
------------------------

.. doxygenfunction:: rocsparse_bsrgemm_nnzb

rocsparse_bsrgemm()
-------------------

.. doxygenfunction:: rocsparse_sbsrgemm
  :outline:
.. doxygenfunction:: rocsparse_dbsrgemm
  :outline:
.. doxygenfunction:: rocsparse_cbsrgemm
  :outline:
.. doxygenfunction:: rocsparse_zbsrgemm

rocsparse_csrgeam_nnz()
-----------------------

.. doxygenfunction:: rocsparse_csrgeam_nnz

rocsparse_csrgeam()
-------------------

.. doxygenfunction:: rocsparse_scsrgeam
  :outline:
.. doxygenfunction:: rocsparse_dcsrgeam
  :outline:
.. doxygenfunction:: rocsparse_ccsrgeam
  :outline:
.. doxygenfunction:: rocsparse_zcsrgeam

rocsparse_csrgemm_buffer_size()
-------------------------------

.. doxygenfunction:: rocsparse_scsrgemm_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcsrgemm_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_ccsrgemm_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zcsrgemm_buffer_size

rocsparse_csrgemm_nnz()
-----------------------

.. doxygenfunction:: rocsparse_csrgemm_nnz

rocsparse_csrgemm_symbolic()
----------------------------

.. doxygenfunction:: rocsparse_csrgemm_symbolic

rocsparse_csrgemm()
-------------------

.. doxygenfunction:: rocsparse_scsrgemm
  :outline:
.. doxygenfunction:: rocsparse_dcsrgemm
  :outline:
.. doxygenfunction:: rocsparse_ccsrgemm
  :outline:
.. doxygenfunction:: rocsparse_zcsrgemm

rocsparse_csrgemm_numeric()
---------------------------

.. doxygenfunction:: rocsparse_scsrgemm_numeric
  :outline:
.. doxygenfunction:: rocsparse_dcsrgemm_numeric
  :outline:
.. doxygenfunction:: rocsparse_ccsrgemm_numeric
  :outline:
.. doxygenfunction:: rocsparse_zcsrgemm_numeric
