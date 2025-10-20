.. meta::
  :description: rocSPARSE exported functions API documentation
  :keywords: rocSPARSE, ROCm, API, documentation, exported functions

.. _api:

Exported rocSPARSE functions
============================

This topic provides a list of the exported rocSPARSE functions in various categories.


ROC-TX functions
----------------

+-----------------------------------------------------+
|Function name                                        |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_enable_roctx`                   |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_disable_roctx`                  |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_state_roctx`                    |
+-----------------------------------------------------+

Debugging functions
-------------------
+-----------------------------------------------------+
|Function name                                        |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_enable_debug_kernel_launch`     |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_disable_debug_kernel_launch`    |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_state_debug_kernel_launch`      |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_enable_debug_arguments`         |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_disable_debug_arguments`        |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_state_debug_arguments`          |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_enable_debug_arguments_verbose` |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_disable_debug_arguments_verbose`|
+-----------------------------------------------------+
|:cpp:func:`rocsparse_state_debug_arguments_verbose`  |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_enable_debug`                   |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_disable_debug`                  |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_state_debug`                    |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_enable_debug_warnings`          |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_disable_debug_warnings`         |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_enable_debug_verbose`           |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_disable_debug_verbose`          |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_state_debug_verbose`            |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_enable_debug_force_host_assert` |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_disable_debug_force_host_assert`|
+-----------------------------------------------------+
|:cpp:func:`rocsparse_state_debug_force_host_assert`  |
+-----------------------------------------------------+


Auxiliary functions
-------------------

+-----------------------------------------------------+
|Function name                                        |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_create_handle`                  |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_destroy_handle`                 |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_set_stream`                     |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_get_stream`                     |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_set_pointer_mode`               |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_get_pointer_mode`               |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_get_version`                    |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_get_git_rev`                    |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_destroy_error`                  |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_error_get_message`              |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_create_mat_descr`               |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_destroy_mat_descr`              |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_copy_mat_descr`                 |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_set_mat_index_base`             |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_get_mat_index_base`             |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_set_mat_type`                   |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_get_mat_type`                   |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_set_mat_fill_mode`              |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_get_mat_fill_mode`              |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_set_mat_diag_type`              |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_get_mat_diag_type`              |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_set_mat_storage_mode`           |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_get_mat_storage_mode`           |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_create_hyb_mat`                 |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_destroy_hyb_mat`                |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_copy_hyb_mat`                   |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_create_mat_info`                |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_copy_mat_info`                  |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_destroy_mat_info`               |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_create_color_info`              |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_destroy_color_info`             |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_copy_color_info`                |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_create_spvec_descr`             |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_destroy_spvec_descr`            |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_spvec_get`                      |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_spvec_get_index_base`           |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_spvec_get_values`               |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_spvec_set_values`               |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_create_coo_descr`               |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_create_coo_aos_descr`           |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_create_csr_descr`               |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_create_csc_descr`               |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_create_ell_descr`               |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_create_bsr_descr`               |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_create_bell_descr`              |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_create_const_coo_descr`         |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_create_const_csr_descr`         |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_create_const_csc_descr`         |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_create_const_bell_descr`        |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_destroy_spmat_descr`            |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_create_extract_descr`           |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_destroy_extract_descr`          |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_create_sparse_to_sparse_descr`  |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_destroy_sparse_to_sparse_descr` |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_sparse_to_sparse_permissive`    |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_create_spgeam_descr`            |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_destroy_spgeam_descr`           |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_spgeam_set_input`               |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_spgeam_get_output`              |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_create_spmv_descr`              |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_destroy_spmv_descr`             |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_spmv_set_input`                 |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_coo_get`                        |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_coo_aos_get`                    |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_csr_get`                        |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_ell_get`                        |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_bell_get`                       |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_coo_set_pointers`               |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_coo_aos_set_pointers`           |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_csr_set_pointers`               |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_csc_set_pointers`               |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_ell_set_pointers`               |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_bsr_set_pointers`               |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_spmat_get_size`                 |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_spmat_get_nnz`                  |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_spmat_get_format`               |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_spmat_get_index_base`           |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_spmat_get_values`               |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_spmat_set_values`               |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_spmat_get_strided_batch`        |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_spmat_set_strided_batch`        |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_coo_set_strided_batch`          |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_csr_set_strided_batch`          |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_csc_set_strided_batch`          |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_spmat_get_attribute`            |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_spmat_set_attribute`            |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_create_dnvec_descr`             |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_create_const_dnvec_descr`       |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_destroy_dnvec_descr`            |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_dnvec_get`                      |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_dnvec_get_values`               |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_dnvec_set_values`               |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_create_dnmat_descr`             |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_create_const_dnmat_descr`       |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_destroy_dnmat_descr`            |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_dnmat_get`                      |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_dnmat_get_values`               |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_dnmat_set_values`               |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_dnmat_get_strided_batch`        |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_dnmat_set_strided_batch`        |
+-----------------------------------------------------+

Sparse level 1 functions
------------------------

================================================= ====== ====== ============== ==============
Function name                                     single double single complex double complex
================================================= ====== ====== ============== ==============
:cpp:func:`rocsparse_Xaxpyi() <rocsparse_saxpyi>` x      x      x              x
:cpp:func:`rocsparse_Xdoti() <rocsparse_sdoti>`   x      x      x              x
:cpp:func:`rocsparse_Xdotci() <rocsparse_cdotci>`               x              x
:cpp:func:`rocsparse_Xgthr() <rocsparse_sgthr>`   x      x      x              x
:cpp:func:`rocsparse_Xgthrz() <rocsparse_sgthrz>` x      x      x              x
:cpp:func:`rocsparse_Xroti() <rocsparse_sroti>`   x      x
:cpp:func:`rocsparse_Xsctr() <rocsparse_ssctr>`   x      x      x              x
================================================= ====== ====== ============== ==============

Sparse level 2 functions
------------------------

============================================================================= ====== ====== ============== ==============
Function name                                                                 single double single complex double complex
============================================================================= ====== ====== ============== ==============
:cpp:func:`rocsparse_Xbsrmv_analysis() <rocsparse_sbsrmv_analysis>`           x      x      x              x
:cpp:func:`rocsparse_bsrmv_clear`
:cpp:func:`rocsparse_Xbsrmv() <rocsparse_sbsrmv>`                             x      x      x              x
:cpp:func:`rocsparse_Xbsrxmv() <rocsparse_sbsrxmv>`                           x      x      x              x
:cpp:func:`rocsparse_Xbsrsv_buffer_size() <rocsparse_sbsrsv_buffer_size>`     x      x      x              x
:cpp:func:`rocsparse_Xbsrsv_analysis() <rocsparse_sbsrsv_analysis>`           x      x      x              x
:cpp:func:`rocsparse_bsrsv_zero_pivot`
:cpp:func:`rocsparse_bsrsv_clear`
:cpp:func:`rocsparse_Xbsrsv_solve() <rocsparse_sbsrsv_solve>`                 x      x      x              x
:cpp:func:`rocsparse_Xcoomv() <rocsparse_scoomv>`                             x      x      x              x
:cpp:func:`rocsparse_Xcsrmv_analysis() <rocsparse_scsrmv_analysis>`           x      x      x              x
:cpp:func:`rocsparse_csrmv_clear`
:cpp:func:`rocsparse_Xcsrmv() <rocsparse_scsrmv>`                             x      x      x              x
:cpp:func:`rocsparse_Xcsrsv_buffer_size() <rocsparse_scsrsv_buffer_size>`     x      x      x              x
:cpp:func:`rocsparse_Xcsrsv_analysis() <rocsparse_scsrsv_analysis>`           x      x      x              x
:cpp:func:`rocsparse_csrsv_zero_pivot`
:cpp:func:`rocsparse_csrsv_clear`
:cpp:func:`rocsparse_Xcsrsv_solve() <rocsparse_scsrsv_solve>`                 x      x      x              x
:cpp:func:`rocsparse_Xcsritsv_buffer_size() <rocsparse_scsritsv_buffer_size>` x      x      x              x
:cpp:func:`rocsparse_Xcsritsv_analysis() <rocsparse_scsritsv_analysis>`       x      x      x              x
:cpp:func:`rocsparse_csritsv_zero_pivot`
:cpp:func:`rocsparse_csritsv_clear`
:cpp:func:`rocsparse_Xcsritsv_solve() <rocsparse_scsritsv_solve>`             x      x      x              x
:cpp:func:`rocsparse_Xcsritsv_solve_ex() <rocsparse_scsritsv_solve_ex>`       x      x      x              x
:cpp:func:`rocsparse_Xellmv() <rocsparse_sellmv>`                             x      x      x              x
:cpp:func:`rocsparse_Xhybmv() <rocsparse_shybmv>`                             x      x      x              x
:cpp:func:`rocsparse_Xgebsrmv() <rocsparse_sgebsrmv>`                         x      x      x              x
:cpp:func:`rocsparse_Xgemvi_buffer_size() <rocsparse_sgemvi_buffer_size>`     x      x      x              x
:cpp:func:`rocsparse_Xgemvi() <rocsparse_sgemvi>`                             x      x      x              x
============================================================================= ====== ====== ============== ==============

Sparse level 3 functions
------------------------

========================================================================= ====== ====== ============== ==============
Function name                                                             single double single complex double complex
========================================================================= ====== ====== ============== ==============
:cpp:func:`rocsparse_Xbsrmm() <rocsparse_sbsrmm>`                         x      x      x              x
:cpp:func:`rocsparse_Xgebsrmm() <rocsparse_sgebsrmm>`                     x      x      x              x
:cpp:func:`rocsparse_Xcsrmm() <rocsparse_scsrmm>`                         x      x      x              x
:cpp:func:`rocsparse_Xcsrsm_buffer_size() <rocsparse_scsrsm_buffer_size>` x      x      x              x
:cpp:func:`rocsparse_Xcsrsm_analysis() <rocsparse_scsrsm_analysis>`       x      x      x              x
:cpp:func:`rocsparse_csrsm_zero_pivot`
:cpp:func:`rocsparse_csrsm_clear`
:cpp:func:`rocsparse_Xcsrsm_solve() <rocsparse_scsrsm_solve>`             x      x      x              x
:cpp:func:`rocsparse_Xbsrsm_buffer_size() <rocsparse_sbsrsm_buffer_size>` x      x      x              x
:cpp:func:`rocsparse_Xbsrsm_analysis() <rocsparse_sbsrsm_analysis>`       x      x      x              x
:cpp:func:`rocsparse_bsrsm_zero_pivot`
:cpp:func:`rocsparse_bsrsm_clear`
:cpp:func:`rocsparse_Xbsrsm_solve() <rocsparse_sbsrsm_solve>`             x      x      x              x
:cpp:func:`rocsparse_Xgemmi() <rocsparse_sgemmi>`                         x      x      x              x
========================================================================= ====== ====== ============== ==============

Sparse extra functions
----------------------

============================================================================= ====== ====== ============== ==============
Function name                                                                 single double single complex double complex
============================================================================= ====== ====== ============== ==============
:cpp:func:`rocsparse_bsrgeam_nnzb`
:cpp:func:`rocsparse_Xbsrgeam() <rocsparse_sbsrgeam>`                         x      x      x              x
:cpp:func:`rocsparse_Xbsrgemm_buffer_size() <rocsparse_sbsrgemm_buffer_size>` x      x      x              x
:cpp:func:`rocsparse_bsrgemm_nnzb`
:cpp:func:`rocsparse_Xbsrgemm() <rocsparse_sbsrgemm>`                         x      x      x              x
:cpp:func:`rocsparse_csrgeam_nnz`
:cpp:func:`rocsparse_Xcsrgeam() <rocsparse_scsrgeam>`                         x      x      x              x
:cpp:func:`rocsparse_Xcsrgemm_buffer_size() <rocsparse_scsrgemm_buffer_size>` x      x      x              x
:cpp:func:`rocsparse_csrgemm_nnz`
:cpp:func:`rocsparse_csrgemm_symbolic`
:cpp:func:`rocsparse_Xcsrgemm() <rocsparse_scsrgemm>`                         x      x      x              x
:cpp:func:`rocsparse_Xcsrgemm_numeric() <rocsparse_scsrgemm_numeric>`         x      x      x              x
============================================================================= ====== ====== ============== ==============

Preconditioner functions
------------------------

===================================================================================================================== ====== ====== ============== ==============
Function name                                                                                                         single double single complex double complex
===================================================================================================================== ====== ====== ============== ==============
:cpp:func:`rocsparse_Xbsric0_buffer_size() <rocsparse_sbsric0_buffer_size>`                                           x      x      x              x
:cpp:func:`rocsparse_Xbsric0_analysis() <rocsparse_sbsric0_analysis>`                                                 x      x      x              x
:cpp:func:`rocsparse_bsric0_zero_pivot`
:cpp:func:`rocsparse_bsric0_clear`
:cpp:func:`rocsparse_Xbsric0() <rocsparse_sbsric0>`                                                                   x      x      x              x
:cpp:func:`rocsparse_Xbsrilu0_buffer_size() <rocsparse_sbsrilu0_buffer_size>`                                         x      x      x              x
:cpp:func:`rocsparse_Xbsrilu0_analysis() <rocsparse_sbsrilu0_analysis>`                                               x      x      x              x
:cpp:func:`rocsparse_bsrilu0_zero_pivot`
:cpp:func:`rocsparse_Xbsrilu0_numeric_boost() <rocsparse_sbsrilu0_numeric_boost>`                                     x      x      x              x
:cpp:func:`rocsparse_bsrilu0_clear`
:cpp:func:`rocsparse_Xbsrilu0() <rocsparse_sbsrilu0>`                                                                 x      x      x              x
:cpp:func:`rocsparse_Xcsric0_buffer_size() <rocsparse_scsric0_buffer_size>`                                           x      x      x              x
:cpp:func:`rocsparse_Xcsric0_analysis() <rocsparse_scsric0_analysis>`                                                 x      x      x              x
:cpp:func:`rocsparse_csric0_zero_pivot`
:cpp:func:`rocsparse_csric0_clear`
:cpp:func:`rocsparse_Xcsric0() <rocsparse_scsric0>`                                                                   x      x      x              x
:cpp:func:`rocsparse_Xcsrilu0_buffer_size() <rocsparse_scsrilu0_buffer_size>`                                         x      x      x              x
:cpp:func:`rocsparse_Xcsrilu0_numeric_boost() <rocsparse_scsrilu0_numeric_boost>`                                     x      x      x              x
:cpp:func:`rocsparse_Xcsrilu0_analysis() <rocsparse_scsrilu0_analysis>`                                               x      x      x              x
:cpp:func:`rocsparse_csrilu0_zero_pivot`
:cpp:func:`rocsparse_csrilu0_clear`
:cpp:func:`rocsparse_Xcsrilu0() <rocsparse_scsrilu0>`                                                                 x      x      x              x
:cpp:func:`rocsparse_csritilu0_buffer_size`
:cpp:func:`rocsparse_csritilu0_preprocess`
:cpp:func:`rocsparse_Xcsritilu0_compute() <rocsparse_scsritilu0_compute>`                                             x      x      x              x
:cpp:func:`rocsparse_Xcsritilu0_compute_ex() <rocsparse_scsritilu0_compute_ex>`                                       x      x      x              x
:cpp:func:`rocsparse_Xcsritilu0_history() <rocsparse_scsritilu0_history>`                                             x      x      x              x
:cpp:func:`rocsparse_Xgtsv_buffer_size() <rocsparse_sgtsv_buffer_size>`                                               x      x      x              x
:cpp:func:`rocsparse_Xgtsv() <rocsparse_sgtsv>`                                                                       x      x      x              x
:cpp:func:`rocsparse_Xgtsv_no_pivot_buffer_size() <rocsparse_sgtsv_no_pivot_buffer_size>`                             x      x      x              x
:cpp:func:`rocsparse_Xgtsv_no_pivot() <rocsparse_sgtsv_no_pivot>`                                                     x      x      x              x
:cpp:func:`rocsparse_Xgtsv_no_pivot_strided_batch_buffer_size() <rocsparse_sgtsv_no_pivot_strided_batch_buffer_size>` x      x      x              x
:cpp:func:`rocsparse_Xgtsv_no_pivot_strided_batch() <rocsparse_sgtsv_no_pivot_strided_batch>`                         x      x      x              x
:cpp:func:`rocsparse_Xgtsv_interleaved_batch_buffer_size() <rocsparse_sgtsv_interleaved_batch_buffer_size>`           x      x      x              x
:cpp:func:`rocsparse_Xgtsv_interleaved_batch() <rocsparse_sgtsv_interleaved_batch>`                                   x      x      x              x
:cpp:func:`rocsparse_Xgpsv_interleaved_batch_buffer_size() <rocsparse_sgpsv_interleaved_batch_buffer_size>`           x      x      x              x
:cpp:func:`rocsparse_Xgpsv_interleaved_batch() <rocsparse_sgpsv_interleaved_batch>`                                   x      x      x              x
===================================================================================================================== ====== ====== ============== ==============

Conversion functions
--------------------

========================================================================================================================= ====== ====== ============== ==============
Function name                                                                                                             single double single complex double complex
========================================================================================================================= ====== ====== ============== ==============
:cpp:func:`rocsparse_csr2coo`
:cpp:func:`rocsparse_csr2csc_buffer_size`
:cpp:func:`rocsparse_Xcsr2csc() <rocsparse_scsr2csc>`                                                                     x      x      x              x
:cpp:func:`rocsparse_Xgebsr2gebsc_buffer_size() <rocsparse_sgebsr2gebsc_buffer_size>`                                     x      x      x              x
:cpp:func:`rocsparse_Xgebsr2gebsc() <rocsparse_sgebsr2gebsc>`                                                             x      x      x              x
:cpp:func:`rocsparse_csr2ell_width`
:cpp:func:`rocsparse_Xcsr2ell() <rocsparse_scsr2ell>`                                                                     x      x      x              x
:cpp:func:`rocsparse_Xcsr2hyb() <rocsparse_scsr2hyb>`                                                                     x      x      x              x
:cpp:func:`rocsparse_csr2bsr_nnz`
:cpp:func:`rocsparse_Xcsr2bsr() <rocsparse_scsr2bsr>`                                                                     x      x      x              x
:cpp:func:`rocsparse_csr2gebsr_nnz`
:cpp:func:`rocsparse_Xcsr2gebsr_buffer_size() <rocsparse_scsr2gebsr_buffer_size>`                                         x      x      x              x
:cpp:func:`rocsparse_Xcsr2gebsr() <rocsparse_scsr2gebsr>`                                                                 x      x      x              x
:cpp:func:`rocsparse_coo2csr`
:cpp:func:`rocsparse_ell2csr_nnz`
:cpp:func:`rocsparse_Xell2csr() <rocsparse_sell2csr>`                                                                     x      x      x              x
:cpp:func:`rocsparse_hyb2csr_buffer_size`
:cpp:func:`rocsparse_Xhyb2csr() <rocsparse_shyb2csr>`                                                                     x      x      x              x
:cpp:func:`rocsparse_Xbsr2csr() <rocsparse_sbsr2csr>`                                                                     x      x      x              x
:cpp:func:`rocsparse_Xgebsr2csr() <rocsparse_sgebsr2csr>`                                                                 x      x      x              x
:cpp:func:`rocsparse_Xgebsr2gebsr_buffer_size() <rocsparse_sgebsr2gebsr_buffer_size>`                                     x      x      x              x
:cpp:func:`rocsparse_gebsr2gebsr_nnz()`
:cpp:func:`rocsparse_Xgebsr2gebsr() <rocsparse_sgebsr2gebsr>`                                                             x      x      x              x
:cpp:func:`rocsparse_Xcsr2csr_compress() <rocsparse_scsr2csr_compress>`                                                   x      x      x              x
:cpp:func:`rocsparse_create_identity_permutation`
:cpp:func:`rocsparse_inverse_permutation`
:cpp:func:`rocsparse_cscsort_buffer_size`
:cpp:func:`rocsparse_cscsort`
:cpp:func:`rocsparse_csrsort_buffer_size`
:cpp:func:`rocsparse_csrsort`
:cpp:func:`rocsparse_coosort_buffer_size`
:cpp:func:`rocsparse_coosort_by_row`
:cpp:func:`rocsparse_coosort_by_column`
:cpp:func:`rocsparse_Xdense2csr() <rocsparse_sdense2csr>`                                                                 x      x      x              x
:cpp:func:`rocsparse_Xdense2csc() <rocsparse_sdense2csc>`                                                                 x      x      x              x
:cpp:func:`rocsparse_Xdense2coo() <rocsparse_sdense2coo>`                                                                 x      x      x              x
:cpp:func:`rocsparse_Xcsr2dense() <rocsparse_scsr2dense>`                                                                 x      x      x              x
:cpp:func:`rocsparse_Xcsc2dense() <rocsparse_scsc2dense>`                                                                 x      x      x              x
:cpp:func:`rocsparse_Xcoo2dense() <rocsparse_scoo2dense>`                                                                 x      x      x              x
:cpp:func:`rocsparse_Xnnz_compress() <rocsparse_snnz_compress>`                                                           x      x      x              x
:cpp:func:`rocsparse_Xnnz() <rocsparse_snnz>`                                                                             x      x      x              x
:cpp:func:`rocsparse_Xprune_dense2csr_buffer_size() <rocsparse_sprune_dense2csr_buffer_size>`                             x      x
:cpp:func:`rocsparse_Xprune_dense2csr_nnz() <rocsparse_sprune_dense2csr_nnz>`                                             x      x
:cpp:func:`rocsparse_Xprune_dense2csr() <rocsparse_sprune_dense2csr>`                                                     x      x
:cpp:func:`rocsparse_Xprune_csr2csr_buffer_size() <rocsparse_sprune_csr2csr_buffer_size>`                                 x      x
:cpp:func:`rocsparse_Xprune_csr2csr_nnz() <rocsparse_sprune_csr2csr_nnz>`                                                 x      x
:cpp:func:`rocsparse_Xprune_csr2csr() <rocsparse_sprune_csr2csr>`                                                         x      x
:cpp:func:`rocsparse_Xprune_dense2csr_by_percentage_buffer_size() <rocsparse_sprune_dense2csr_by_percentage_buffer_size>` x      x
:cpp:func:`rocsparse_Xprune_dense2csr_nnz_by_percentage() <rocsparse_sprune_dense2csr_nnz_by_percentage>`                 x      x
:cpp:func:`rocsparse_Xprune_dense2csr_by_percentage() <rocsparse_sprune_dense2csr_by_percentage>`                         x      x
:cpp:func:`rocsparse_Xprune_csr2csr_by_percentage_buffer_size() <rocsparse_sprune_csr2csr_by_percentage_buffer_size>`     x      x
:cpp:func:`rocsparse_Xprune_csr2csr_nnz_by_percentage() <rocsparse_sprune_csr2csr_nnz_by_percentage>`                     x      x
:cpp:func:`rocsparse_Xprune_csr2csr_by_percentage() <rocsparse_sprune_csr2csr_by_percentage>`                             x      x
:cpp:func:`rocsparse_Xbsrpad_value() <rocsparse_sbsrpad_value>`                                                           x      x      x              x
========================================================================================================================= ====== ====== ============== ==============

Reordering functions
--------------------

======================================================= ====== ====== ============== ==============
Function name                                           single double single complex double complex
======================================================= ====== ====== ============== ==============
:cpp:func:`rocsparse_Xcsrcolor() <rocsparse_scsrcolor>` x      x      x              x
======================================================= ====== ====== ============== ==============

Utility functions
-----------------

=================================================================================================== ====== ====== ============== ==============
Function name                                                                                       single double single complex double complex
=================================================================================================== ====== ====== ============== ==============
:cpp:func:`rocsparse_Xcheck_matrix_csr_buffer_size() <rocsparse_scheck_matrix_csr_buffer_size>`     x      x      x              x
:cpp:func:`rocsparse_Xcheck_matrix_csr() <rocsparse_scheck_matrix_csr>`                             x      x      x              x
:cpp:func:`rocsparse_Xcheck_matrix_csc_buffer_size() <rocsparse_scheck_matrix_csc_buffer_size>`     x      x      x              x
:cpp:func:`rocsparse_Xcheck_matrix_csc() <rocsparse_scheck_matrix_csc>`                             x      x      x              x
:cpp:func:`rocsparse_Xcheck_matrix_coo_buffer_size() <rocsparse_scheck_matrix_coo_buffer_size>`     x      x      x              x
:cpp:func:`rocsparse_Xcheck_matrix_coo() <rocsparse_scheck_matrix_coo>`                             x      x      x              x
:cpp:func:`rocsparse_Xcheck_matrix_gebsr_buffer_size() <rocsparse_scheck_matrix_gebsr_buffer_size>` x      x      x              x
:cpp:func:`rocsparse_Xcheck_matrix_gebsr() <rocsparse_scheck_matrix_gebsr>`                         x      x      x              x
:cpp:func:`rocsparse_Xcheck_matrix_gebsc_buffer_size() <rocsparse_scheck_matrix_gebsc_buffer_size>` x      x      x              x
:cpp:func:`rocsparse_Xcheck_matrix_gebsc() <rocsparse_scheck_matrix_gebsc>`                         x      x      x              x
:cpp:func:`rocsparse_Xcheck_matrix_ell_buffer_size() <rocsparse_scheck_matrix_ell_buffer_size>`     x      x      x              x
:cpp:func:`rocsparse_Xcheck_matrix_ell() <rocsparse_scheck_matrix_ell>`                             x      x      x              x
:cpp:func:`rocsparse_check_matrix_hyb_buffer_size() <rocsparse_check_matrix_hyb_buffer_size>`       x      x      x              x
:cpp:func:`rocsparse_check_matrix_hyb() <rocsparse_check_matrix_hyb>`                               x      x      x              x
=================================================================================================== ====== ====== ============== ==============

Sparse generic functions
------------------------

The generic API allows different combinations of data types and compute types to be used. See the specific routine
documentation for details about the supported data types and compute types.

+-----------------------------------------------------+
|Function name                                        |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_axpby()`                        |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_gather()`                       |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_scatter()`                      |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_rot()`                          |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_spvv()`                         |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_sparse_to_dense()`              |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_dense_to_sparse()`              |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_spmv()`                         |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_v2_spmv_buffer_size()`          |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_v2_spmv()`                      |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_spsv()`                         |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_sptrsv_buffer_size()`           |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_sptrsv()`                       |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_spmm()`                         |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_spsm()`                         |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_sptrsm_buffer_size()`           |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_sptrsm()`                       |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_spgemm()`                       |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_spgeam_buffer_size()`           |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_spgeam()`                       |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_sddmm_buffer_size()`            |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_sddmm_preprocess()`             |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_sddmm()`                        |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_sparse_to_sparse_buffer_size()` |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_sparse_to_sparse()`             |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_extract_buffer_size()`          |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_extract_nnz()`                  |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_extract()`                      |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_check_spmat()`                  |
+-----------------------------------------------------+
|:cpp:func:`rocsparse_spitsv()`                       |
+-----------------------------------------------------+
