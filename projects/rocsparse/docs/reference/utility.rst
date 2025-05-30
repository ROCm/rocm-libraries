.. meta::
  :description: rocSPARSE utility functions API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, utility functions

.. _rocsparse_utility_functions_:

********************************************************************
Sparse utility functions
********************************************************************

This module contains all sparse utility routines.

The sparse utility routines allow for testing whether matrix data is valid for different matrix formats.
These routines do not support execution in a ``hipGraph`` context.

rocsparse_check_matrix_csr_buffer_size()
----------------------------------------

.. doxygenfunction:: rocsparse_scheck_matrix_csr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcheck_matrix_csr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_ccheck_matrix_csr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zcheck_matrix_csr_buffer_size

rocsparse_check_matrix_csr()
----------------------------

.. doxygenfunction:: rocsparse_scheck_matrix_csr
  :outline:
.. doxygenfunction:: rocsparse_dcheck_matrix_csr
  :outline:
.. doxygenfunction:: rocsparse_ccheck_matrix_csr
  :outline:
.. doxygenfunction:: rocsparse_zcheck_matrix_csr

rocsparse_check_matrix_csc_buffer_size()
----------------------------------------

.. doxygenfunction:: rocsparse_scheck_matrix_csc_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcheck_matrix_csc_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_ccheck_matrix_csc_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zcheck_matrix_csc_buffer_size

rocsparse_check_matrix_csc()
----------------------------

.. doxygenfunction:: rocsparse_scheck_matrix_csc
  :outline:
.. doxygenfunction:: rocsparse_dcheck_matrix_csc
  :outline:
.. doxygenfunction:: rocsparse_ccheck_matrix_csc
  :outline:
.. doxygenfunction:: rocsparse_zcheck_matrix_csc

rocsparse_check_matrix_coo_buffer_size()
----------------------------------------

.. doxygenfunction:: rocsparse_scheck_matrix_coo_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcheck_matrix_coo_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_ccheck_matrix_coo_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zcheck_matrix_coo_buffer_size

rocsparse_check_matrix_coo()
----------------------------

.. doxygenfunction:: rocsparse_scheck_matrix_coo
  :outline:
.. doxygenfunction:: rocsparse_dcheck_matrix_coo
  :outline:
.. doxygenfunction:: rocsparse_ccheck_matrix_coo
  :outline:
.. doxygenfunction:: rocsparse_zcheck_matrix_coo

rocsparse_check_matrix_gebsr_buffer_size()
------------------------------------------

.. doxygenfunction:: rocsparse_scheck_matrix_gebsr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcheck_matrix_gebsr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_ccheck_matrix_gebsr_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zcheck_matrix_gebsr_buffer_size

rocsparse_check_matrix_gebsr()
------------------------------

.. doxygenfunction:: rocsparse_scheck_matrix_gebsr
  :outline:
.. doxygenfunction:: rocsparse_dcheck_matrix_gebsr
  :outline:
.. doxygenfunction:: rocsparse_ccheck_matrix_gebsr
  :outline:
.. doxygenfunction:: rocsparse_zcheck_matrix_gebsr

rocsparse_check_matrix_gebsc_buffer_size()
------------------------------------------

.. doxygenfunction:: rocsparse_scheck_matrix_gebsc_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcheck_matrix_gebsc_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_ccheck_matrix_gebsc_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zcheck_matrix_gebsc_buffer_size

rocsparse_check_matrix_gebsc()
------------------------------

.. doxygenfunction:: rocsparse_scheck_matrix_gebsc
  :outline:
.. doxygenfunction:: rocsparse_dcheck_matrix_gebsc
  :outline:
.. doxygenfunction:: rocsparse_ccheck_matrix_gebsc
  :outline:
.. doxygenfunction:: rocsparse_zcheck_matrix_gebsc

rocsparse_check_matrix_ell_buffer_size()
----------------------------------------

.. doxygenfunction:: rocsparse_scheck_matrix_ell_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_dcheck_matrix_ell_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_ccheck_matrix_ell_buffer_size
  :outline:
.. doxygenfunction:: rocsparse_zcheck_matrix_ell_buffer_size

rocsparse_check_matrix_ell()
----------------------------

.. doxygenfunction:: rocsparse_scheck_matrix_ell
  :outline:
.. doxygenfunction:: rocsparse_dcheck_matrix_ell
  :outline:
.. doxygenfunction:: rocsparse_ccheck_matrix_ell
  :outline:
.. doxygenfunction:: rocsparse_zcheck_matrix_ell

rocsparse_check_matrix_hyb_buffer_size()
----------------------------------------

.. doxygenfunction:: rocsparse_check_matrix_hyb_buffer_size

rocsparse_check_matrix_hyb()
----------------------------

.. doxygenfunction:: rocsparse_check_matrix_hyb
