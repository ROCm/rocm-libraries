.. meta::
  :description: rocSPARSE bitwise reproducibility API reference library documentation
  :keywords: rocSPARSE, ROCm, API, documentation, bitwise reproducibility

.. _reproducibility:

Bitwise reproducibility
=======================

Some routines do not produce deterministic results from run to run. This is typically the case when HIP atomics are used.
This page catalogues the run-to-run reproducibility of each routine.

Sparse level 1 functions
------------------------

================================================= === ==
Function name                                     yes no
================================================= === ==
:cpp:func:`rocsparse_Xaxpyi() <rocsparse_saxpyi>` x
:cpp:func:`rocsparse_Xdoti() <rocsparse_sdoti>`   x
:cpp:func:`rocsparse_Xdotci() <rocsparse_cdotci>` x
:cpp:func:`rocsparse_Xgthr() <rocsparse_sgthr>`   x
:cpp:func:`rocsparse_Xgthrz() <rocsparse_sgthrz>` x
:cpp:func:`rocsparse_Xroti() <rocsparse_sroti>`   x
:cpp:func:`rocsparse_Xsctr() <rocsparse_ssctr>`   x
================================================= === ==

Sparse level 2 functions
------------------------

============================================================================= === ==
Function name                                                                 yes no
============================================================================= === ==
:cpp:func:`rocsparse_Xbsrmv_analysis() <rocsparse_sbsrmv_analysis>`           x
:cpp:func:`rocsparse_bsrmv_clear`                                             x
:cpp:func:`rocsparse_Xbsrmv() <rocsparse_sbsrmv>`                             x
:cpp:func:`rocsparse_Xbsrxmv() <rocsparse_sbsrxmv>`                           x
:cpp:func:`rocsparse_Xbsrsv_buffer_size() <rocsparse_sbsrsv_buffer_size>`     x
:cpp:func:`rocsparse_Xbsrsv_analysis() <rocsparse_sbsrsv_analysis>`               x
:cpp:func:`rocsparse_bsrsv_zero_pivot`                                        x
:cpp:func:`rocsparse_bsrsv_clear`                                             x
:cpp:func:`rocsparse_Xbsrsv_solve() <rocsparse_sbsrsv_solve>`                     x
:cpp:func:`rocsparse_Xcsrmv_analysis() <rocsparse_scsrmv_analysis>`           x
:cpp:func:`rocsparse_csrmv_clear`                                             x
:cpp:func:`rocsparse_Xcsrsv_buffer_size() <rocsparse_scsrsv_buffer_size>`     x
:cpp:func:`rocsparse_Xcsrsv_analysis() <rocsparse_scsrsv_analysis>`               x
:cpp:func:`rocsparse_csrsv_zero_pivot`                                        x
:cpp:func:`rocsparse_csrsv_clear`                                             x
:cpp:func:`rocsparse_Xcsrsv_solve() <rocsparse_scsrsv_solve>`                     x
:cpp:func:`rocsparse_Xcsritsv_buffer_size() <rocsparse_scsritsv_buffer_size>` x
:cpp:func:`rocsparse_Xcsritsv_analysis() <rocsparse_scsritsv_analysis>`           x
:cpp:func:`rocsparse_csritsv_zero_pivot`                                      x
:cpp:func:`rocsparse_csritsv_clear`                                           x
:cpp:func:`rocsparse_Xcsritsv_solve() <rocsparse_scsritsv_solve>`                 x
:cpp:func:`rocsparse_Xcsritsv_solve_ex() <rocsparse_scsritsv_solve_ex>`           x
:cpp:func:`rocsparse_Xgemvi_buffer_size() <rocsparse_sgemvi_buffer_size>`     x
:cpp:func:`rocsparse_Xgemvi() <rocsparse_sgemvi>`                             x
============================================================================= === ==

The reproducibility of :cpp:func:`rocsparse_Xbsrmv() <rocsparse_sbsrmv>`, :cpp:func:`rocsparse_Xbsrxmv() <rocsparse_sbsrxmv>`,
:cpp:func:`rocsparse_Xcoomv() <rocsparse_scoomv>`, :cpp:func:`rocsparse_Xcsrmv() <rocsparse_scsrmv>`,
:cpp:func:`rocsparse_Xellmv() <rocsparse_sellmv>`, :cpp:func:`rocsparse_Xhybmv() <rocsparse_shybmv>`,
and :cpp:func:`rocsparse_Xgebsrmv() <rocsparse_sgebsrmv>` is more complicated, depending on whether A
is transposed or not. See the chart below to determine whether these routines are deterministic.

+-----------------------------------------------+-----------------+-----------------+
|                                               | A non-transpose | A transpose     |
|    Routine                                    +--------+--------+--------+--------+
|                                               |  Yes   |   No   |  Yes   |   No   |
+===============================================+========+========+========+========+
| rocsparse_Xbsrmv                              |   x    |        |  N/A   |  N/A   |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_Xbsrxmv                             |   x    |        |  N/A   |  N/A   |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_Xcoomv                              |   x    |        |        |   x    |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_Xcsrmv                              |   x    |        |        |   x    |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_Xcsrmv  (info != NULL)              |        |   x    |        |   x    |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_Xellmv                              |   x    |        |        |   x    |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_Xhybmv                              |   x    |        |        |   x    |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_Xgebsrmv                            |   x    |        |  N/A   |  N/A   |
+-----------------------------------------------+--------+--------+--------+--------+

Sparse level 3 functions
------------------------

========================================================================= === ==
Function name                                                             yes no
========================================================================= === ==
:cpp:func:`rocsparse_Xcsrsm_buffer_size() <rocsparse_scsrsm_buffer_size>` x
:cpp:func:`rocsparse_Xcsrsm_analysis() <rocsparse_scsrsm_analysis>`           x
:cpp:func:`rocsparse_csrsm_zero_pivot`                                    x
:cpp:func:`rocsparse_csrsm_clear`                                         x
:cpp:func:`rocsparse_Xcsrsm_solve() <rocsparse_scsrsm_solve>`                 x
:cpp:func:`rocsparse_Xbsrsm_buffer_size() <rocsparse_sbsrsm_buffer_size>` x
:cpp:func:`rocsparse_Xbsrsm_analysis() <rocsparse_sbsrsm_analysis>`           x
:cpp:func:`rocsparse_bsrsm_zero_pivot`                                    x
:cpp:func:`rocsparse_bsrsm_clear`                                         x
:cpp:func:`rocsparse_Xbsrsm_solve() <rocsparse_sbsrsm_solve>`                 x
:cpp:func:`rocsparse_Xgemmi() <rocsparse_sgemmi>`                         x
========================================================================= === ==

The reproducibility of :cpp:func:`rocsparse_Xbsrmm() <rocsparse_sbsrmm>`,
:cpp:func:`rocsparse_Xgebsrmm() <rocsparse_sgebsrmm>`, and
:cpp:func:`rocsparse_Xcsrmm() <rocsparse_scsrmm>` is more complicated, depending on
whether A is transposed or not. See the chart below to determine whether these routines
are deterministic.

+-----------------------------------------------+-----------------+-----------------+
|                                               | A non-transpose | A transpose     |
|    Routine                                    +--------+--------+--------+--------+
|                                               |  Yes   |   No   |  Yes   |   No   |
+===============================================+========+========+========+========+
| rocsparse_Xbsrmm                              |   x    |        |  N/A   |  N/A   |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_Xgebsrmm                            |   x    |        |  N/A   |  N/A   |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_Xcsrmm                              |   x    |        |        |   x    |
+-----------------------------------------------+--------+--------+--------+--------+

Sparse extra functions
----------------------

============================================================================= === ==
Function name                                                                 yes no
============================================================================= === ==
:cpp:func:`rocsparse_bsrgeam_nnzb`                                            x
:cpp:func:`rocsparse_Xbsrgeam() <rocsparse_sbsrgeam>`                         x
:cpp:func:`rocsparse_Xbsrgemm_buffer_size() <rocsparse_sbsrgemm_buffer_size>` x
:cpp:func:`rocsparse_bsrgemm_nnzb`                                            x
:cpp:func:`rocsparse_Xbsrgemm() <rocsparse_sbsrgemm>`                         x
:cpp:func:`rocsparse_csrgeam_nnz`                                             x
:cpp:func:`rocsparse_Xcsrgeam() <rocsparse_scsrgeam>`                         x
:cpp:func:`rocsparse_Xcsrgemm_buffer_size() <rocsparse_scsrgemm_buffer_size>` x
:cpp:func:`rocsparse_csrgemm_nnz`                                                 x
:cpp:func:`rocsparse_csrgemm_symbolic`                                            x
:cpp:func:`rocsparse_Xcsrgemm() <rocsparse_scsrgemm>`                             x
:cpp:func:`rocsparse_Xcsrgemm_numeric() <rocsparse_scsrgemm_numeric>`             x
============================================================================= === ==

Preconditioner functions
------------------------

===================================================================================================================== === ==
Function name                                                                                                         yes no
===================================================================================================================== === ==
:cpp:func:`rocsparse_Xbsric0_buffer_size() <rocsparse_sbsric0_buffer_size>`                                           x
:cpp:func:`rocsparse_Xbsric0_analysis() <rocsparse_sbsric0_analysis>`                                                     x
:cpp:func:`rocsparse_bsric0_zero_pivot`                                                                               x
:cpp:func:`rocsparse_bsric0_clear`                                                                                    x
:cpp:func:`rocsparse_Xbsric0() <rocsparse_sbsric0>`                                                                       x
:cpp:func:`rocsparse_Xbsrilu0_buffer_size() <rocsparse_sbsrilu0_buffer_size>`                                         x
:cpp:func:`rocsparse_Xbsrilu0_analysis() <rocsparse_sbsrilu0_analysis>`                                                   x
:cpp:func:`rocsparse_bsrilu0_zero_pivot`                                                                              x
:cpp:func:`rocsparse_Xbsrilu0_numeric_boost() <rocsparse_sbsrilu0_numeric_boost>`                                         x
:cpp:func:`rocsparse_bsrilu0_clear`                                                                                   x
:cpp:func:`rocsparse_Xbsrilu0() <rocsparse_sbsrilu0>`                                                                     x
:cpp:func:`rocsparse_Xcsric0_buffer_size() <rocsparse_scsric0_buffer_size>`                                           x
:cpp:func:`rocsparse_Xcsric0_analysis() <rocsparse_scsric0_analysis>`                                                     x
:cpp:func:`rocsparse_csric0_zero_pivot`                                                                               x
:cpp:func:`rocsparse_csric0_clear`                                                                                    x
:cpp:func:`rocsparse_Xcsric0() <rocsparse_scsric0>`                                                                       x
:cpp:func:`rocsparse_Xcsrilu0_buffer_size() <rocsparse_scsrilu0_buffer_size>`                                         x
:cpp:func:`rocsparse_Xcsrilu0_numeric_boost() <rocsparse_scsrilu0_numeric_boost>`                                         x
:cpp:func:`rocsparse_Xcsrilu0_analysis() <rocsparse_scsrilu0_analysis>`                                                   x
:cpp:func:`rocsparse_csrilu0_zero_pivot`                                                                              x
:cpp:func:`rocsparse_csrilu0_clear`                                                                                   x
:cpp:func:`rocsparse_Xcsrilu0() <rocsparse_scsrilu0>`                                                                     x
:cpp:func:`rocsparse_csritilu0_buffer_size`                                                                           x
:cpp:func:`rocsparse_csritilu0_preprocess`                                                                                x
:cpp:func:`rocsparse_Xcsritilu0_compute() <rocsparse_scsritilu0_compute>`                                                 x
:cpp:func:`rocsparse_Xcsritilu0_compute_ex() <rocsparse_scsritilu0_compute_ex>`                                           x
:cpp:func:`rocsparse_Xcsritilu0_history() <rocsparse_scsritilu0_history>`                                                 x
:cpp:func:`rocsparse_Xgtsv_buffer_size() <rocsparse_sgtsv_buffer_size>`                                               x
:cpp:func:`rocsparse_Xgtsv() <rocsparse_sgtsv>`                                                                       x
:cpp:func:`rocsparse_Xgtsv_no_pivot_buffer_size() <rocsparse_sgtsv_no_pivot_buffer_size>`                             x
:cpp:func:`rocsparse_Xgtsv_no_pivot() <rocsparse_sgtsv_no_pivot>`                                                     x
:cpp:func:`rocsparse_Xgtsv_no_pivot_strided_batch_buffer_size() <rocsparse_sgtsv_no_pivot_strided_batch_buffer_size>` x
:cpp:func:`rocsparse_Xgtsv_no_pivot_strided_batch() <rocsparse_sgtsv_no_pivot_strided_batch>`                         x
:cpp:func:`rocsparse_Xgtsv_interleaved_batch_buffer_size() <rocsparse_sgtsv_interleaved_batch_buffer_size>`           x
:cpp:func:`rocsparse_Xgtsv_interleaved_batch() <rocsparse_sgtsv_interleaved_batch>`                                   x
:cpp:func:`rocsparse_Xgpsv_interleaved_batch_buffer_size() <rocsparse_sgpsv_interleaved_batch_buffer_size>`           x
:cpp:func:`rocsparse_Xgpsv_interleaved_batch() <rocsparse_sgpsv_interleaved_batch>`                                   x
===================================================================================================================== === ==


Conversion functions
--------------------

========================================================================================================================= === ==
Function name                                                                                                             yes no
========================================================================================================================= === ==
:cpp:func:`rocsparse_csr2coo`                                                                                             x
:cpp:func:`rocsparse_csr2csc_buffer_size`                                                                                 x
:cpp:func:`rocsparse_Xcsr2csc() <rocsparse_scsr2csc>`                                                                     x
:cpp:func:`rocsparse_Xgebsr2gebsc_buffer_size() <rocsparse_sgebsr2gebsc_buffer_size>`                                     x
:cpp:func:`rocsparse_Xgebsr2gebsc() <rocsparse_sgebsr2gebsc>`                                                             x
:cpp:func:`rocsparse_csr2ell_width`                                                                                       x
:cpp:func:`rocsparse_Xcsr2ell() <rocsparse_scsr2ell>`                                                                     x
:cpp:func:`rocsparse_Xcsr2hyb() <rocsparse_scsr2hyb>`                                                                     x
:cpp:func:`rocsparse_csr2bsr_nnz`                                                                                         x
:cpp:func:`rocsparse_Xcsr2bsr() <rocsparse_scsr2bsr>`                                                                     x
:cpp:func:`rocsparse_csr2gebsr_nnz`                                                                                       x
:cpp:func:`rocsparse_Xcsr2gebsr_buffer_size() <rocsparse_scsr2gebsr_buffer_size>`                                         x
:cpp:func:`rocsparse_Xcsr2gebsr() <rocsparse_scsr2gebsr>`                                                                 x
:cpp:func:`rocsparse_coo2csr`                                                                                             x
:cpp:func:`rocsparse_ell2csr_nnz`                                                                                         x
:cpp:func:`rocsparse_Xell2csr() <rocsparse_sell2csr>`                                                                     x
:cpp:func:`rocsparse_hyb2csr_buffer_size`                                                                                 x
:cpp:func:`rocsparse_Xhyb2csr() <rocsparse_shyb2csr>`                                                                     x
:cpp:func:`rocsparse_Xbsr2csr() <rocsparse_sbsr2csr>`                                                                     x
:cpp:func:`rocsparse_Xgebsr2csr() <rocsparse_sgebsr2csr>`                                                                 x
:cpp:func:`rocsparse_Xgebsr2gebsr_buffer_size() <rocsparse_sgebsr2gebsr_buffer_size>`                                     x
:cpp:func:`rocsparse_gebsr2gebsr_nnz()`                                                                                   x
:cpp:func:`rocsparse_Xgebsr2gebsr() <rocsparse_sgebsr2gebsr>`                                                             x
:cpp:func:`rocsparse_Xcsr2csr_compress() <rocsparse_scsr2csr_compress>`                                                   x
:cpp:func:`rocsparse_create_identity_permutation`                                                                         x
:cpp:func:`rocsparse_inverse_permutation`                                                                                 x
:cpp:func:`rocsparse_cscsort_buffer_size`                                                                                 x
:cpp:func:`rocsparse_cscsort`                                                                                             x
:cpp:func:`rocsparse_csrsort_buffer_size`                                                                                 x
:cpp:func:`rocsparse_csrsort`                                                                                             x
:cpp:func:`rocsparse_coosort_buffer_size`                                                                                 x
:cpp:func:`rocsparse_coosort_by_row`                                                                                      x
:cpp:func:`rocsparse_coosort_by_column`                                                                                   x
:cpp:func:`rocsparse_Xdense2csr() <rocsparse_sdense2csr>`                                                                 x
:cpp:func:`rocsparse_Xdense2csc() <rocsparse_sdense2csc>`                                                                 x
:cpp:func:`rocsparse_Xdense2coo() <rocsparse_sdense2coo>`                                                                 x
:cpp:func:`rocsparse_Xcsr2dense() <rocsparse_scsr2dense>`                                                                 x
:cpp:func:`rocsparse_Xcsc2dense() <rocsparse_scsc2dense>`                                                                 x
:cpp:func:`rocsparse_Xcoo2dense() <rocsparse_scoo2dense>`                                                                 x
:cpp:func:`rocsparse_Xnnz_compress() <rocsparse_snnz_compress>`                                                           x
:cpp:func:`rocsparse_Xnnz() <rocsparse_snnz>`                                                                             x
:cpp:func:`rocsparse_Xprune_dense2csr_buffer_size() <rocsparse_sprune_dense2csr_buffer_size>`                             x
:cpp:func:`rocsparse_Xprune_dense2csr_nnz() <rocsparse_sprune_dense2csr_nnz>`                                             x
:cpp:func:`rocsparse_Xprune_dense2csr() <rocsparse_sprune_dense2csr>`                                                     x
:cpp:func:`rocsparse_Xprune_csr2csr_buffer_size() <rocsparse_sprune_csr2csr_buffer_size>`                                 x
:cpp:func:`rocsparse_Xprune_csr2csr_nnz() <rocsparse_sprune_csr2csr_nnz>`                                                 x
:cpp:func:`rocsparse_Xprune_csr2csr() <rocsparse_sprune_csr2csr>`                                                         x
:cpp:func:`rocsparse_Xprune_dense2csr_by_percentage_buffer_size() <rocsparse_sprune_dense2csr_by_percentage_buffer_size>` x
:cpp:func:`rocsparse_Xprune_dense2csr_nnz_by_percentage() <rocsparse_sprune_dense2csr_nnz_by_percentage>`                 x
:cpp:func:`rocsparse_Xprune_dense2csr_by_percentage() <rocsparse_sprune_dense2csr_by_percentage>`                         x
:cpp:func:`rocsparse_Xprune_csr2csr_by_percentage_buffer_size() <rocsparse_sprune_csr2csr_by_percentage_buffer_size>`     x
:cpp:func:`rocsparse_Xprune_csr2csr_nnz_by_percentage() <rocsparse_sprune_csr2csr_nnz_by_percentage>`                     x
:cpp:func:`rocsparse_Xprune_csr2csr_by_percentage() <rocsparse_sprune_csr2csr_by_percentage>`                             x
:cpp:func:`rocsparse_Xbsrpad_value() <rocsparse_sbsrpad_value>`                                                           x
========================================================================================================================= === ==

Reordering functions
--------------------

======================================================= === ==
Function name                                           yes no
======================================================= === ==
:cpp:func:`rocsparse_Xcsrcolor() <rocsparse_scsrcolor>` x
======================================================= === ==

Utility functions
-----------------

=================================================================================================== === ==
Function name                                                                                       yes no
=================================================================================================== === ==
:cpp:func:`rocsparse_Xcheck_matrix_csr_buffer_size() <rocsparse_scheck_matrix_csr_buffer_size>`     x
:cpp:func:`rocsparse_Xcheck_matrix_csr() <rocsparse_scheck_matrix_csr>`                             x
:cpp:func:`rocsparse_Xcheck_matrix_csc_buffer_size() <rocsparse_scheck_matrix_csc_buffer_size>`     x
:cpp:func:`rocsparse_Xcheck_matrix_csc() <rocsparse_scheck_matrix_csc>`                             x
:cpp:func:`rocsparse_Xcheck_matrix_coo_buffer_size() <rocsparse_scheck_matrix_coo_buffer_size>`     x
:cpp:func:`rocsparse_Xcheck_matrix_coo() <rocsparse_scheck_matrix_coo>`                             x
:cpp:func:`rocsparse_Xcheck_matrix_gebsr_buffer_size() <rocsparse_scheck_matrix_gebsr_buffer_size>` x
:cpp:func:`rocsparse_Xcheck_matrix_gebsr() <rocsparse_scheck_matrix_gebsr>`                         x
:cpp:func:`rocsparse_Xcheck_matrix_gebsc_buffer_size() <rocsparse_scheck_matrix_gebsc_buffer_size>` x
:cpp:func:`rocsparse_Xcheck_matrix_gebsc() <rocsparse_scheck_matrix_gebsc>`                         x
:cpp:func:`rocsparse_Xcheck_matrix_ell_buffer_size() <rocsparse_scheck_matrix_ell_buffer_size>`     x
:cpp:func:`rocsparse_Xcheck_matrix_ell() <rocsparse_scheck_matrix_ell>`                             x
:cpp:func:`rocsparse_check_matrix_hyb_buffer_size() <rocsparse_check_matrix_hyb_buffer_size>`       x
:cpp:func:`rocsparse_check_matrix_hyb() <rocsparse_check_matrix_hyb>`                               x
=================================================================================================== === ==

Sparse generic functions
------------------------

==================================================== === ==
Function name                                        yes no
==================================================== === ==
:cpp:func:`rocsparse_axpby()`                        x
:cpp:func:`rocsparse_gather()`                       x
:cpp:func:`rocsparse_scatter()`                      x
:cpp:func:`rocsparse_rot()`                          x
:cpp:func:`rocsparse_spvv()`                         x
:cpp:func:`rocsparse_sparse_to_dense()`              x
:cpp:func:`rocsparse_dense_to_sparse()`              x
:cpp:func:`rocsparse_spsv()`                             x
:cpp:func:`rocsparse_spsm()`                             x
:cpp:func:`rocsparse_sptrsv_buffer_size()`           x
:cpp:func:`rocsparse_sptrsv()`                           x
:cpp:func:`rocsparse_sptrsm_buffer_size()`           x
:cpp:func:`rocsparse_sptrsm()`                           x
:cpp:func:`rocsparse_spgemm()`                           x
:cpp:func:`rocsparse_v2_spmv_buffer_size()`          x
:cpp:func:`rocsparse_spgeam_buffer_size()`           x
:cpp:func:`rocsparse_spgeam()`                       x
:cpp:func:`rocsparse_sddmm_buffer_size()`            x
:cpp:func:`rocsparse_sddmm_preprocess()`             x
:cpp:func:`rocsparse_sddmm()`                        x
:cpp:func:`rocsparse_sparse_to_sparse_buffer_size()` x
:cpp:func:`rocsparse_sparse_to_sparse()`             x
:cpp:func:`rocsparse_extract_buffer_size()`          x
:cpp:func:`rocsparse_extract()`                      x
==================================================== === ==

The reproducibility of :cpp:func:`rocsparse_spmv()` and :cpp:func:`rocsparse_v2_spmv()` is more complicated because these generic routines
support multiple sparse matrix formats and algorithms. See the chart below to determine whether
a given algorithm is deterministic.

+-----------------------------------------------------------------------------------+
|                        Bit-wise reproducibility of SpMV/v2_SpMV                   |
+-----------------------------------------------+-----------------+-----------------+
|                                               | A non-transpose | A transpose     |
|            Algorithm                          +--------+--------+--------+--------+
|                                               |  Yes   |   No   |  Yes   |   No   |
+===============================================+========+========+========+========+
| rocsparse_spmv_alg_csr_stream                 |   x    |        |        |   x    |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_spmv_alg_csr_rowsplit               |   x    |        |        |   x    |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_spmv_alg_csr_adaptive               |        |   x    |        |   x    |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_spmv_alg_csr_lrb                    |        |   x    |        |   x    |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_spmv_alg_csr_stream (CSC FORMAT)    |        |   x    |   x    |        |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_spmv_alg_csr_rowsplit (CSC FORMAT)  |        |   x    |   x    |        |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_spmv_alg_csr_adaptive (CSC FORMAT)  |        |   x    |        |   x    |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_spmv_alg_csr_lrb (CSC FORMAT)       |        |   x    |        |   x    |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_spmv_alg_coo                        |   x    |        |        |   x    |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_spmv_alg_coo_atomic                 |        |   x    |        |   x    |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_spmv_alg_ell                        |   x    |        |  N/A   |  N/A   |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_spmv_alg_bsr                        |   x    |        |  N/A   |  N/A   |
+-----------------------------------------------+--------+--------+--------+--------+

The reproducibility of :cpp:func:`rocsparse_spmm()` is more complicated because this generic routine
supports multiple sparse matrix formats and algorithms. See the chart below to determine whether
a given algorithm is deterministic.

+-----------------------------------------------------------------------------------+
|                        Bit-wise reproducibility of SpMM                           |
+-----------------------------------------------+-----------------+-----------------+
|                                               | A non-transpose | A transpose     |
|            Algorithm                          +--------+--------+--------+--------+
|                                               |  Yes   |  No    |  Yes   |  No    |
+===============================================+========+========+========+========+
| rocsparse_spmm_alg_csr                        |   x    |        |        |   x    |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_spmm_alg_csr_row_split              |   x    |        |        |   x    |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_spmm_alg_csr_nnz_split              |        |   x    |        |   x    |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_spmm_alg_csr_merge_path             |        |   x    |        |   x    |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_spmm_alg_csr (CSC FORMAT)           |        |   x    |   x    |        |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_spmm_alg_csr_row_split (CSC FORMAT) |        |   x    |   x    |        |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_spmm_alg_csr_nnz_split (CSC FORMAT) |        |   x    |        |   x    |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_spmm_alg_csr_merge_path (CSC FORMAT)|        |   x    |        |   x    |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_spmm_alg_coo_segmented              |   x    |        |        |   x    |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_spmm_alg_coo_atomic                 |        |   x    |        |   x    |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_spmm_alg_coo_segmented_atomic       |        |   x    |        |   x    |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_spmm_alg_bell                       |   x    |        |  N/A   |  N/A   |
+-----------------------------------------------+--------+--------+--------+--------+
| rocsparse_spmm_alg_bsr                        |   x    |        |  N/A   |  N/A   |
+-----------------------------------------------+--------+--------+--------+--------+
