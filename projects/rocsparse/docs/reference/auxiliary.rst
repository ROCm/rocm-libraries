.. meta::
  :description: rocSPARSE auxiliary functions API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, auxiliary functions

.. _rocsparse_auxiliary_functions_:

********************************************************************
Sparse auxiliary functions
********************************************************************

This module contains all sparse auxiliary functions.

The functions that are contained in the auxiliary module describe all available helper functions that are required for subsequent library calls.
These functions do not support execution in a ``hipGraph`` context.

.. _rocsparse_create_handle_:

rocsparse_create_handle()
-------------------------

.. doxygenfunction:: rocsparse_create_handle

.. _rocsparse_destroy_handle_:

rocsparse_destroy_handle()
--------------------------

.. doxygenfunction:: rocsparse_destroy_handle

.. _rocsparse_set_stream_:

rocsparse_set_stream()
----------------------

.. doxygenfunction:: rocsparse_set_stream

rocsparse_get_stream()
----------------------

.. doxygenfunction:: rocsparse_get_stream

rocsparse_set_pointer_mode()
----------------------------

.. doxygenfunction:: rocsparse_set_pointer_mode

rocsparse_get_pointer_mode()
----------------------------

.. doxygenfunction:: rocsparse_get_pointer_mode

rocsparse_get_version()
-----------------------

.. doxygenfunction:: rocsparse_get_version

rocsparse_get_git_rev()
-----------------------

.. doxygenfunction:: rocsparse_get_git_rev

rocsparse_destroy_error()
-------------------------

.. doxygenfunction:: rocsparse_destroy_error

rocsparse_error_get_message()
-----------------------------

.. doxygenfunction:: rocsparse_error_get_message

rocsparse_create_mat_descr()
----------------------------

.. doxygenfunction:: rocsparse_create_mat_descr

rocsparse_destroy_mat_descr()
-----------------------------

.. doxygenfunction:: rocsparse_destroy_mat_descr

rocsparse_copy_mat_descr()
--------------------------

.. doxygenfunction:: rocsparse_copy_mat_descr

rocsparse_set_mat_index_base()
------------------------------

.. doxygenfunction:: rocsparse_set_mat_index_base

rocsparse_get_mat_index_base()
------------------------------

.. doxygenfunction:: rocsparse_get_mat_index_base

rocsparse_set_mat_type()
------------------------

.. doxygenfunction:: rocsparse_set_mat_type

rocsparse_get_mat_type()
------------------------

.. doxygenfunction:: rocsparse_get_mat_type

rocsparse_set_mat_fill_mode()
-----------------------------

.. doxygenfunction:: rocsparse_set_mat_fill_mode

rocsparse_get_mat_fill_mode()
-----------------------------

.. doxygenfunction:: rocsparse_get_mat_fill_mode

rocsparse_set_mat_diag_type()
-----------------------------

.. doxygenfunction:: rocsparse_set_mat_diag_type

rocsparse_get_mat_diag_type()
-----------------------------

.. doxygenfunction:: rocsparse_get_mat_diag_type

rocsparse_set_mat_storage_mode()
--------------------------------

.. doxygenfunction:: rocsparse_set_mat_storage_mode

rocsparse_get_mat_storage_mode()
--------------------------------

.. doxygenfunction:: rocsparse_get_mat_storage_mode

.. _rocsparse_create_hyb_mat_:

rocsparse_create_hyb_mat()
--------------------------

.. doxygenfunction:: rocsparse_create_hyb_mat

rocsparse_destroy_hyb_mat()
---------------------------

.. doxygenfunction:: rocsparse_destroy_hyb_mat

rocsparse_copy_hyb_mat()
------------------------

.. doxygenfunction:: rocsparse_copy_hyb_mat

rocsparse_create_mat_info()
---------------------------

.. doxygenfunction:: rocsparse_create_mat_info

rocsparse_copy_mat_info()
-------------------------

.. doxygenfunction:: rocsparse_copy_mat_info

.. _rocsparse_destroy_mat_info_:

rocsparse_destroy_mat_info()
----------------------------

.. doxygenfunction:: rocsparse_destroy_mat_info

rocsparse_create_color_info()
-----------------------------

.. doxygenfunction:: rocsparse_create_color_info

rocsparse_destroy_color_info()
------------------------------

.. doxygenfunction:: rocsparse_destroy_color_info

rocsparse_copy_color_info()
---------------------------

.. doxygenfunction:: rocsparse_copy_color_info

rocsparse_create_spvec_descr()
------------------------------

.. doxygenfunction:: rocsparse_create_spvec_descr

rocsparse_destroy_spvec_descr()
-------------------------------

.. doxygenfunction:: rocsparse_destroy_spvec_descr

rocsparse_spvec_get()
---------------------

.. doxygenfunction:: rocsparse_spvec_get

rocsparse_spvec_get_index_base()
--------------------------------

.. doxygenfunction:: rocsparse_spvec_get_index_base

rocsparse_spvec_get_values()
----------------------------

.. doxygenfunction:: rocsparse_spvec_get_values

rocsparse_spvec_set_values()
----------------------------

.. doxygenfunction:: rocsparse_spvec_set_values

rocsparse_create_coo_descr
--------------------------

.. doxygenfunction:: rocsparse_create_coo_descr

rocsparse_create_coo_aos_descr
------------------------------

.. doxygenfunction:: rocsparse_create_coo_aos_descr

rocsparse_create_csr_descr
--------------------------

.. doxygenfunction:: rocsparse_create_csr_descr

rocsparse_create_csc_descr
--------------------------

.. doxygenfunction:: rocsparse_create_csc_descr

rocsparse_create_bsr_descr
--------------------------

.. doxygenfunction:: rocsparse_create_bsr_descr

rocsparse_create_ell_descr
--------------------------

.. doxygenfunction:: rocsparse_create_ell_descr

rocsparse_create_bell_descr
---------------------------

.. doxygenfunction:: rocsparse_create_bell_descr

rocsparse_create_const_coo_descr
--------------------------------

.. doxygenfunction:: rocsparse_create_const_coo_descr

rocsparse_create_const_csr_descr
--------------------------------

.. doxygenfunction:: rocsparse_create_const_csr_descr

rocsparse_create_const_csc_descr
--------------------------------

.. doxygenfunction:: rocsparse_create_const_csc_descr

rocsparse_create_const_bell_descr
---------------------------------

.. doxygenfunction:: rocsparse_create_const_bell_descr

rocsparse_destroy_spmat_descr
-----------------------------

.. doxygenfunction:: rocsparse_destroy_spmat_descr

rocsparse_create_sparse_to_sparse_descr
---------------------------------------

.. doxygenfunction:: rocsparse_create_sparse_to_sparse_descr

rocsparse_destroy_sparse_to_sparse_descr
----------------------------------------

.. doxygenfunction:: rocsparse_destroy_sparse_to_sparse_descr

rocsparse_sparse_to_sparse_permissive
-------------------------------------

.. doxygenfunction:: rocsparse_sparse_to_sparse_permissive

rocsparse_create_extract_descr
------------------------------

.. doxygenfunction:: rocsparse_create_extract_descr

rocsparse_destroy_extract_descr
-------------------------------

.. doxygenfunction:: rocsparse_destroy_extract_descr

rocsparse_create_spgeam_descr
-----------------------------

.. doxygenfunction:: rocsparse_create_spgeam_descr

rocsparse_destroy_spgeam_descr
------------------------------

.. doxygenfunction:: rocsparse_destroy_spgeam_descr

rocsparse_spgeam_set_input
--------------------------

.. doxygenfunction:: rocsparse_spgeam_set_input

rocsparse_spgeam_get_output
---------------------------

.. doxygenfunction:: rocsparse_spgeam_get_output

rocsparse_create_spmv_descr
---------------------------

.. doxygenfunction:: rocsparse_create_spmv_descr

rocsparse_destroy_spmv_descr
----------------------------

.. doxygenfunction:: rocsparse_destroy_spmv_descr

rocsparse_spmv_set_input
------------------------

.. doxygenfunction:: rocsparse_spmv_set_input

rocsparse_coo_get
-----------------

.. doxygenfunction:: rocsparse_coo_get

rocsparse_coo_aos_get
---------------------

.. doxygenfunction:: rocsparse_coo_aos_get

rocsparse_csr_get
-----------------

.. doxygenfunction:: rocsparse_csr_get

rocsparse_ell_get
-----------------

.. doxygenfunction:: rocsparse_ell_get

rocsparse_bell_get
------------------

.. doxygenfunction:: rocsparse_bell_get

rocsparse_coo_set_pointers
--------------------------

.. doxygenfunction:: rocsparse_coo_set_pointers

rocsparse_coo_aos_set_pointers
------------------------------

.. doxygenfunction:: rocsparse_coo_aos_set_pointers

rocsparse_csr_set_pointers
--------------------------

.. doxygenfunction:: rocsparse_csr_set_pointers

rocsparse_csc_set_pointers
--------------------------

.. doxygenfunction:: rocsparse_csc_set_pointers

rocsparse_ell_set_pointers
--------------------------

.. doxygenfunction:: rocsparse_ell_set_pointers

rocsparse_bsr_set_pointers
--------------------------

.. doxygenfunction:: rocsparse_bsr_set_pointers

rocsparse_spmat_get_size
------------------------

.. doxygenfunction:: rocsparse_spmat_get_size

rocsparse_spmat_get_nnz
------------------------

.. doxygenfunction:: rocsparse_spmat_get_nnz

rocsparse_spmat_get_format
--------------------------

.. doxygenfunction:: rocsparse_spmat_get_format

rocsparse_spmat_get_index_base
------------------------------

.. doxygenfunction:: rocsparse_spmat_get_index_base

rocsparse_spmat_get_values
--------------------------

.. doxygenfunction:: rocsparse_spmat_get_values

rocsparse_spmat_set_values
--------------------------

.. doxygenfunction:: rocsparse_spmat_set_values

rocsparse_spmat_get_strided_batch
---------------------------------

.. doxygenfunction:: rocsparse_spmat_get_strided_batch

rocsparse_spmat_set_strided_batch
---------------------------------

.. doxygenfunction:: rocsparse_spmat_set_strided_batch

rocsparse_coo_set_strided_batch
-------------------------------

.. doxygenfunction:: rocsparse_coo_set_strided_batch

rocsparse_csr_set_strided_batch
-------------------------------

.. doxygenfunction:: rocsparse_csr_set_strided_batch

rocsparse_csc_set_strided_batch
-------------------------------

.. doxygenfunction:: rocsparse_csc_set_strided_batch

rocsparse_spmat_get_attribute
-----------------------------

.. doxygenfunction:: rocsparse_spmat_get_attribute

rocsparse_spmat_set_attribute
-----------------------------

.. doxygenfunction:: rocsparse_spmat_set_attribute

rocsparse_create_dnvec_descr
----------------------------

.. doxygenfunction:: rocsparse_create_dnvec_descr

rocsparse_create_const_dnvec_descr
----------------------------------

.. doxygenfunction:: rocsparse_create_const_dnvec_descr

rocsparse_destroy_dnvec_descr
-----------------------------

.. doxygenfunction:: rocsparse_destroy_dnvec_descr

rocsparse_dnvec_get
-------------------

.. doxygenfunction:: rocsparse_dnvec_get

rocsparse_dnvec_get_values
--------------------------

.. doxygenfunction:: rocsparse_dnvec_get_values

rocsparse_dnvec_set_values
--------------------------

.. doxygenfunction:: rocsparse_dnvec_set_values

rocsparse_create_dnmat_descr
----------------------------

.. doxygenfunction:: rocsparse_create_dnmat_descr

rocsparse_create_const_dnmat_descr
----------------------------------

.. doxygenfunction:: rocsparse_create_const_dnmat_descr

rocsparse_destroy_dnmat_descr
-----------------------------

.. doxygenfunction:: rocsparse_destroy_dnmat_descr

rocsparse_dnmat_get
-------------------

.. doxygenfunction:: rocsparse_dnmat_get

rocsparse_dnmat_get_values
--------------------------

.. doxygenfunction:: rocsparse_dnmat_get_values

rocsparse_dnmat_set_values
--------------------------

.. doxygenfunction:: rocsparse_dnmat_set_values

rocsparse_dnmat_get_strided_batch
---------------------------------

.. doxygenfunction:: rocsparse_dnmat_get_strided_batch

rocsparse_dnmat_set_strided_batch
---------------------------------

.. doxygenfunction:: rocsparse_dnmat_set_strided_batch
