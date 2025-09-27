.. meta::
  :description: rocSPARSE user guide and documentation
  :keywords: rocSPARSE, ROCm, API, documentation, user guide

.. _rocsparse_docs:

********************************************************************
rocSPARSE user guide
********************************************************************

This topic discusses how to use rocSPARSE, including a discussion of device and stream management, storage formats, pointer mode,
and how hipSPARSE interacts with rocSPARSE.

HIP device management
=====================

Before starting a HIP kernel, you can call :cpp:func:`hipSetDevice` to set the device to run the kernel on,
for example, device ``2``. Unless you explicitly specify a different device, HIP kernels always run on device ``0``.
This HIP (and CUDA) device management approach is not specific to the rocSPARSE library.
rocSPARSE honors this approach and assumes you have already set the preferred device before a rocSPARSE routine call.

After you set the device, you can create a handle with :ref:`rocsparse_create_handle_`.
Subsequent rocSPARSE routines take this handle as an input parameter.
rocSPARSE only queries the specified device (using :cpp:func:`hipGetDevice`) and does not set the device for users.
It's your responsibility to provide a valid device to rocSPARSE and ensure device safety.
If it's not a valid device, rocSPARSE returns an error message.

The handle should be destroyed at the end using :ref:`rocsparse_destroy_handle_` to release the resources
consumed by the rocSPARSE library. You **cannot** switch devices
between :ref:`rocsparse_create_handle_` and :ref:`rocsparse_destroy_handle_`. To change the device,
you must destroy the current handle and create another rocSPARSE handle on a new device.

.. note::

   :cpp:func:`hipSetDevice` and :cpp:func:`hipGetDevice` are not part of the rocSPARSE API.
   They are part of the `HIP Device Management API <https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___device.html>`_.

HIP stream management
=====================

HIP kernels are always launched in a queue, which is also known as a stream. If you don't explicitly specify a stream,
the system provides and maintains a default stream, which you cannot create or destroy.
However, you can freely create a new stream using :cpp:func:`hipStreamCreate` and bind it to a rocSPARSE handle
using :ref:`rocsparse_set_stream_`. The rocSPARSE routines invoke HIP kernels.
A rocSPARSE handle is always associated with a stream, which rocSPARSE passes to the kernels inside the routine.
One rocSPARSE routine only takes one stream in a single invocation.
If you create a stream, you are responsible for destroying it.
See the `HIP Stream Management API <https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___stream.html>`_ for more information.

Asynchronous execution
======================

All rocSPARSE library functions are non-blocking and execute asynchronously with respect to the host,
except for functions which allocate memory themselves, preventing asynchronicity.
These functions might return immediately or before the actual computation has finished.
To force synchronization, use either :cpp:func:`hipDeviceSynchronize` or :cpp:func:`hipStreamSynchronize`.
This ensures all previously executed rocSPARSE functions on the device or the stream have completed.

Multiple streams and multiple devices
=====================================

If a system has multiple HIP devices, you can run multiple rocSPARSE handles concurrently.
However, you **cannot** run a single rocSPARSE handle concurrently on multiple discrete devices.
Each handle can only be associated with a single device, and a new handle should be created for each additional device.

Graph support for rocSPARSE
===========================

Many of the rocSPARSE functions can be captured into a graph node using the HIP Graph Management APIs. See :ref:`Functions supported with Graph Capture` to determine
whether a rocSPARSE routine is supported or not. For a list of graph-related HIP APIs, see the `HIP Graph Management API <https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___graph.html#graph-management>`_.

The following code creates a graph with ``rocsparse_function()`` as the graph node.

.. code-block:: c++

   CHECK_HIP_ERROR((hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
   rocsparse_<function>(<arguments>);
   CHECK_HIP_ERROR(hipStreamEndCapture(stream, &graph));

The captured graph can be launched as shown below:

.. code-block:: c++

   CHECK_HIP_ERROR(hipGraphInstantiate(&instance, graph, NULL, NULL, 0));
   CHECK_HIP_ERROR(hipGraphLaunch(instance, stream));

Graph support requires asynchronous HIP APIs.

.. _Functions Supported with Graph Capture:

Functions supported with graph capture
========================================

The following functions support graph capture:

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
:cpp:func:`rocsparse_Xbsrmv_analysis() <rocsparse_sbsrmv_analysis>`               x
:cpp:func:`rocsparse_bsrmv_clear`                                                 x
:cpp:func:`rocsparse_Xbsrmv() <rocsparse_sbsrmv>`                             x
:cpp:func:`rocsparse_Xbsrxmv() <rocsparse_sbsrxmv>`                           x
:cpp:func:`rocsparse_Xbsrsv_buffer_size() <rocsparse_sbsrsv_buffer_size>`     x
:cpp:func:`rocsparse_Xbsrsv_analysis() <rocsparse_sbsrsv_analysis>`               x
:cpp:func:`rocsparse_bsrsv_zero_pivot`                                            x
:cpp:func:`rocsparse_bsrsv_clear`
:cpp:func:`rocsparse_Xbsrsv_solve() <rocsparse_sbsrsv_solve>`                 x
:cpp:func:`rocsparse_Xcoomv() <rocsparse_scoomv>`                             x
:cpp:func:`rocsparse_Xcsrmv_analysis() <rocsparse_scsrmv_analysis>`               x
:cpp:func:`rocsparse_Xcsrmv() <rocsparse_scsrmv>`                             x
:cpp:func:`rocsparse_csrmv_clear`                                                 x
:cpp:func:`rocsparse_Xcsrsv_buffer_size() <rocsparse_scsrsv_buffer_size>`     x
:cpp:func:`rocsparse_Xcsrsv_analysis() <rocsparse_scsrsv_analysis>`               x
:cpp:func:`rocsparse_csrsv_zero_pivot`                                            x
:cpp:func:`rocsparse_csrsv_clear`                                                 x
:cpp:func:`rocsparse_Xcsrsv_solve() <rocsparse_scsrsv_solve>`                 x
:cpp:func:`rocsparse_Xcsritsv_buffer_size() <rocsparse_scsritsv_buffer_size>`     x
:cpp:func:`rocsparse_Xcsritsv_analysis() <rocsparse_scsritsv_analysis>`           x
:cpp:func:`rocsparse_csritsv_zero_pivot`                                          x
:cpp:func:`rocsparse_csritsv_clear`                                               x
:cpp:func:`rocsparse_Xcsritsv_solve() <rocsparse_scsritsv_solve>`                 x
:cpp:func:`rocsparse_Xcsritsv_solve() <rocsparse_scsritsvx_solve>`                x
:cpp:func:`rocsparse_Xellmv() <rocsparse_sellmv>`                             x
:cpp:func:`rocsparse_Xgebsrmv() <rocsparse_sgebsrmv>`                         x
:cpp:func:`rocsparse_Xgemvi_buffer_size() <rocsparse_sgemvi_buffer_size>`     x
:cpp:func:`rocsparse_Xgemvi() <rocsparse_sgemvi>`                             x
:cpp:func:`rocsparse_Xhybmv() <rocsparse_shybmv>`                             x
============================================================================= === ==

Sparse level 3 functions
------------------------

========================================================================= === ==
Function name                                                             yes no
========================================================================= === ==
:cpp:func:`rocsparse_Xcsrmm() <rocsparse_scsrmm>`                         x
:cpp:func:`rocsparse_Xcsrsm_buffer_size() <rocsparse_scsrsm_buffer_size>` x
:cpp:func:`rocsparse_Xcsrsm_analysis() <rocsparse_scsrsm_analysis>`           x
:cpp:func:`rocsparse_csrsm_zero_pivot`                                        x
:cpp:func:`rocsparse_csrsm_clear`                                             x
:cpp:func:`rocsparse_Xcsrsm_solve() <rocsparse_scsrsm_solve>`             x
:cpp:func:`rocsparse_Xbsrmm() <rocsparse_sbsrmm>`                         x
:cpp:func:`rocsparse_Xbsrsm_buffer_size() <rocsparse_sbsrsm_buffer_size>` x
:cpp:func:`rocsparse_Xbsrsm_analysis() <rocsparse_sbsrsm_analysis>`           x
:cpp:func:`rocsparse_bsrsm_zero_pivot`                                        x
:cpp:func:`rocsparse_bsrsm_clear`                                             x
:cpp:func:`rocsparse_Xbsrsm_solve() <rocsparse_sbsrsm_solve>`             x
:cpp:func:`rocsparse_Xgebsrmm() <rocsparse_sgebsrmm>`                     x
:cpp:func:`rocsparse_Xgemmi() <rocsparse_sgemmi>`                         x
========================================================================= === ==

Sparse extra functions
----------------------

============================================================================= === ==
Function name                                                                 yes no
============================================================================= === ==
:cpp:func:`rocsparse_bsrgeam_nnzb`                                                x
:cpp:func:`rocsparse_Xbsrgeam() <rocsparse_sbsrgeam>`                             x
:cpp:func:`rocsparse_Xbsrgemm_buffer_size() <rocsparse_sbsrgemm_buffer_size>`     x
:cpp:func:`rocsparse_bsrgemm_nnzb`                                                x
:cpp:func:`rocsparse_Xbsrgemm() <rocsparse_sbsrgemm>`                             x
:cpp:func:`rocsparse_csrgeam_nnz`                                                 x
:cpp:func:`rocsparse_Xcsrgeam() <rocsparse_scsrgeam>`                             x
:cpp:func:`rocsparse_Xcsrgemm_buffer_size() <rocsparse_scsrgemm_buffer_size>`     x
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
:cpp:func:`rocsparse_bsric0_zero_pivot`                                                                                   x
:cpp:func:`rocsparse_bsric0_clear`                                                                                        x
:cpp:func:`rocsparse_Xbsric0() <rocsparse_sbsric0>`                                                                   x
:cpp:func:`rocsparse_Xbsrilu0_buffer_size() <rocsparse_sbsrilu0_buffer_size>`                                         x
:cpp:func:`rocsparse_Xbsrilu0_analysis() <rocsparse_sbsrilu0_analysis>`                                                   x
:cpp:func:`rocsparse_bsrilu0_zero_pivot`                                                                                  x
:cpp:func:`rocsparse_Xbsrilu0_numeric_boost() <rocsparse_sbsrilu0_numeric_boost>`                                     x
:cpp:func:`rocsparse_bsrilu0_clear`                                                                                       x
:cpp:func:`rocsparse_Xbsrilu0() <rocsparse_sbsrilu0>`                                                                 x
:cpp:func:`rocsparse_Xcsric0_buffer_size() <rocsparse_scsric0_buffer_size>`                                           x
:cpp:func:`rocsparse_Xcsric0_analysis() <rocsparse_scsric0_analysis>`                                                     x
:cpp:func:`rocsparse_csric0_zero_pivot`                                                                                   x
:cpp:func:`rocsparse_csric0_clear`                                                                                        x
:cpp:func:`rocsparse_Xcsric0() <rocsparse_scsric0>`                                                                   x
:cpp:func:`rocsparse_Xcsrilu0_buffer_size() <rocsparse_scsrilu0_buffer_size>`                                         x
:cpp:func:`rocsparse_Xcsrilu0_numeric_boost() <rocsparse_scsrilu0_numeric_boost>`                                     x
:cpp:func:`rocsparse_Xcsrilu0_analysis() <rocsparse_scsrilu0_analysis>`                                                   x
:cpp:func:`rocsparse_csrilu0_zero_pivot`                                                                                  x
:cpp:func:`rocsparse_csrilu0_clear`                                                                                       x
:cpp:func:`rocsparse_Xcsrilu0() <rocsparse_scsrilu0>`                                                                 x
:cpp:func:`rocsparse_csritilu0_buffer_size`                                                                               x
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
:cpp:func:`rocsparse_csr2bsr_nnz`                                                                                             x
:cpp:func:`rocsparse_Xcsr2bsr() <rocsparse_scsr2bsr>`                                                                         x
:cpp:func:`rocsparse_csr2gebsr_nnz`                                                                                           x
:cpp:func:`rocsparse_Xcsr2gebsr_buffer_size() <rocsparse_scsr2gebsr_buffer_size>`                                         x
:cpp:func:`rocsparse_Xcsr2gebsr() <rocsparse_scsr2gebsr>`                                                                     x
:cpp:func:`rocsparse_coo2csr`                                                                                             x
:cpp:func:`rocsparse_ell2csr_nnz`                                                                                         x
:cpp:func:`rocsparse_Xell2csr() <rocsparse_sell2csr>`                                                                     x
:cpp:func:`rocsparse_hyb2csr_buffer_size`                                                                                 x
:cpp:func:`rocsparse_Xhyb2csr() <rocsparse_shyb2csr>`                                                                     x
:cpp:func:`rocsparse_Xbsr2csr() <rocsparse_sbsr2csr>`                                                                     x
:cpp:func:`rocsparse_Xgebsr2csr() <rocsparse_sgebsr2csr>`                                                                 x
:cpp:func:`rocsparse_Xgebsr2gebsr_buffer_size() <rocsparse_sgebsr2gebsr_buffer_size>`                                     x
:cpp:func:`rocsparse_gebsr2gebsr_nnz()`                                                                                       x
:cpp:func:`rocsparse_Xgebsr2gebsr() <rocsparse_sgebsr2gebsr>`                                                                 x
:cpp:func:`rocsparse_Xcsr2csr_compress() <rocsparse_scsr2csr_compress>`                                                       x
:cpp:func:`rocsparse_create_identity_permutation`                                                                         x
:cpp:func:`rocsparse_inverse_permutation`                                                                                 x
:cpp:func:`rocsparse_cscsort_buffer_size`                                                                                 x
:cpp:func:`rocsparse_cscsort`                                                                                                 x
:cpp:func:`rocsparse_csrsort_buffer_size`                                                                                 x
:cpp:func:`rocsparse_csrsort`                                                                                                 x
:cpp:func:`rocsparse_coosort_buffer_size`                                                                                 x
:cpp:func:`rocsparse_coosort_by_row`                                                                                          x
:cpp:func:`rocsparse_coosort_by_column`                                                                                       x
:cpp:func:`rocsparse_Xdense2csr() <rocsparse_sdense2csr>`                                                                     x
:cpp:func:`rocsparse_Xdense2csc() <rocsparse_sdense2csc>`                                                                     x
:cpp:func:`rocsparse_Xdense2coo() <rocsparse_sdense2coo>`                                                                     x
:cpp:func:`rocsparse_Xcsr2dense() <rocsparse_scsr2dense>`                                                                 x
:cpp:func:`rocsparse_Xcsc2dense() <rocsparse_scsc2dense>`                                                                 x
:cpp:func:`rocsparse_Xcoo2dense() <rocsparse_scoo2dense>`                                                                 x
:cpp:func:`rocsparse_Xnnz_compress() <rocsparse_snnz_compress>`                                                               x
:cpp:func:`rocsparse_Xnnz() <rocsparse_snnz>`                                                                                 x
:cpp:func:`rocsparse_Xprune_dense2csr_buffer_size() <rocsparse_sprune_dense2csr_buffer_size>`                             x
:cpp:func:`rocsparse_Xprune_dense2csr_nnz() <rocsparse_sprune_dense2csr_nnz>`                                                 x
:cpp:func:`rocsparse_Xprune_dense2csr() <rocsparse_sprune_dense2csr>`                                                         x
:cpp:func:`rocsparse_Xprune_csr2csr_buffer_size() <rocsparse_sprune_csr2csr_buffer_size>`                                 x
:cpp:func:`rocsparse_Xprune_csr2csr_nnz() <rocsparse_sprune_csr2csr_nnz>`                                                     x
:cpp:func:`rocsparse_Xprune_csr2csr() <rocsparse_sprune_csr2csr>`                                                             x
:cpp:func:`rocsparse_Xprune_dense2csr_by_percentage_buffer_size() <rocsparse_sprune_dense2csr_by_percentage_buffer_size>` x
:cpp:func:`rocsparse_Xprune_dense2csr_nnz_by_percentage() <rocsparse_sprune_dense2csr_nnz_by_percentage>`                     x
:cpp:func:`rocsparse_Xprune_dense2csr_by_percentage() <rocsparse_sprune_dense2csr_by_percentage>`                             x
:cpp:func:`rocsparse_Xprune_csr2csr_by_percentage_buffer_size() <rocsparse_sprune_csr2csr_by_percentage_buffer_size>`     x
:cpp:func:`rocsparse_Xprune_csr2csr_nnz_by_percentage() <rocsparse_sprune_csr2csr_nnz_by_percentage>`                         x
:cpp:func:`rocsparse_Xprune_csr2csr_by_percentage() <rocsparse_sprune_csr2csr_by_percentage>`                                 x
:cpp:func:`rocsparse_Xbsrpad_value() <rocsparse_sbsrpad_value>`                                                           x
========================================================================================================================= === ==

Reordering functions
--------------------

======================================================= === ==
Function name                                           yes no
======================================================= === ==
:cpp:func:`rocsparse_Xcsrcolor() <rocsparse_scsrcolor>`     x
======================================================= === ==

Utility functions
-----------------

=================================================================================================== === ==
Function name                                                                                       yes no
=================================================================================================== === ==
:cpp:func:`rocsparse_Xcheck_matrix_csr_buffer_size() <rocsparse_scheck_matrix_csr_buffer_size>`         x
:cpp:func:`rocsparse_Xcheck_matrix_csr() <rocsparse_scheck_matrix_csr>`                                 x
:cpp:func:`rocsparse_Xcheck_matrix_csc_buffer_size() <rocsparse_scheck_matrix_csc_buffer_size>`         x
:cpp:func:`rocsparse_Xcheck_matrix_csc() <rocsparse_scheck_matrix_csc>`                                 x
:cpp:func:`rocsparse_Xcheck_matrix_coo_buffer_size() <rocsparse_scheck_matrix_coo_buffer_size>`         x
:cpp:func:`rocsparse_Xcheck_matrix_coo() <rocsparse_scheck_matrix_coo>`                                 x
:cpp:func:`rocsparse_Xcheck_matrix_gebsr_buffer_size() <rocsparse_scheck_matrix_gebsr_buffer_size>`     x
:cpp:func:`rocsparse_Xcheck_matrix_gebsr() <rocsparse_scheck_matrix_gebsr>`                             x
:cpp:func:`rocsparse_Xcheck_matrix_gebsc_buffer_size() <rocsparse_scheck_matrix_gebsc_buffer_size>`     x
:cpp:func:`rocsparse_Xcheck_matrix_gebsc() <rocsparse_scheck_matrix_gebsc>`                             x
:cpp:func:`rocsparse_Xcheck_matrix_ell_buffer_size() <rocsparse_scheck_matrix_ell_buffer_size>`         x
:cpp:func:`rocsparse_Xcheck_matrix_ell() <rocsparse_scheck_matrix_ell>`                                 x
:cpp:func:`rocsparse_check_matrix_hyb_buffer_size() <rocsparse_check_matrix_hyb_buffer_size>`           x
:cpp:func:`rocsparse_check_matrix_hyb() <rocsparse_check_matrix_hyb>`                                   x
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
:cpp:func:`rocsparse_spvv()`                             x
:cpp:func:`rocsparse_sparse_to_dense()`                  x
:cpp:func:`rocsparse_dense_to_sparse()`                  x
:cpp:func:`rocsparse_spgemm()`                           x
:cpp:func:`rocsparse_v2_spmv_buffer_size()`              x
:cpp:func:`rocsparse_spgeam_buffer_size()`               x
:cpp:func:`rocsparse_spgeam()`                           x
:cpp:func:`rocsparse_sptrsv_buffer_size()`               x
:cpp:func:`rocsparse_sptrsm_buffer_size()`               x
:cpp:func:`rocsparse_sddmm_buffer_size()`                x
:cpp:func:`rocsparse_sddmm_preprocess()`                 x
:cpp:func:`rocsparse_sparse_to_sparse_buffer_size()`     x
:cpp:func:`rocsparse_sparse_to_sparse()`                 x
:cpp:func:`rocsparse_extract_buffer_size()`          x
:cpp:func:`rocsparse_extract_nnz()`                  x
:cpp:func:`rocsparse_extract()`                      x
==================================================== === ==

For :cpp:func:`rocsparse_spmv()`, :cpp:func:`rocsparse_spmm()`, :cpp:func:`rocsparse_spsv()`, and :cpp:func:`rocsparse_spsm()`,
``hipGraph`` is supported when passing the buffer size or compute stages but is not supported when passing the preprocess stage.

For :cpp:func:`rocsparse_v2_spmv()`, :cpp:func:`rocsparse_v2_sptrsv()`, and :cpp:func:`rocsparse_v2_sptrsm()`,
``hipGraph`` is supported when passing the compute stage but is not supported when passing the analysis stage.

For :cpp:func:`rocsparse_sddmm()`, ``hipGraph`` is supported only when using the default algorithm.

Storage formats
===============

This section describes the supported matrix storage formats.

.. note::
    The different storage formats support indexing from a base of 0 or 1 as described in :ref:`index_base`.

COO storage format
------------------

The Coordinate (COO) storage format represents an :math:`m \times n` matrix by:

=========== ==================================================================
m           Number of rows (integer).
n           Number of columns (integer).
nnz         Number of non-zero elements (integer).
coo_val     Array of ``nnz`` elements containing the data (floating point).
coo_row_ind Array of ``nnz`` elements containing the row indices (integer).
coo_col_ind Array of ``nnz`` elements containing the column indices (integer).
=========== ==================================================================

The COO matrix is expected to be sorted by row indices and column indices per row. Furthermore, each pair of indices should appear only once.
Consider the following :math:`3 \times 5` matrix and the corresponding COO structures,
with :math:`m = 3, n = 5`, and :math:`\text{nnz} = 8` using zero-based indexing:

.. math::

  A = \begin{pmatrix}
        1.0 & 2.0 & 0.0 & 3.0 & 0.0 \\
        0.0 & 4.0 & 5.0 & 0.0 & 0.0 \\
        6.0 & 0.0 & 0.0 & 7.0 & 8.0 \\
      \end{pmatrix}

where

.. math::

  \begin{array}{ll}
    \text{coo_val}[8] & = \{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0\} \\
    \text{coo_row_ind}[8] & = \{0, 0, 0, 1, 1, 2, 2, 2\} \\
    \text{coo_col_ind}[8] & = \{0, 1, 3, 1, 2, 0, 3, 4\}
  \end{array}

COO (AoS) storage format
------------------------

The Coordinate (COO) Array of Structure (AoS) storage format represents an :math:`m \times n` matrix by:

======= ==========================================================================================
m           Number of rows (integer).
n           Number of columns (integer).
nnz         Number of non-zero elements (integer).
coo_val     Array of ``nnz`` elements containing the data (floating point).
coo_ind     Array of ``2 * nnz`` elements containing alternating row and column indices (integer).
======= ==========================================================================================

The COO (AoS) matrix is expected to be sorted by row indices and column indices per row.
Each pair of indices should appear only once.
Consider the following :math:`3 \times 5` matrix and the corresponding COO (AoS) structures,
with :math:`m = 3, n = 5`, and :math:`\text{nnz} = 8` using zero-based indexing:

.. math::

  A = \begin{pmatrix}
        1.0 & 2.0 & 0.0 & 3.0 & 0.0 \\
        0.0 & 4.0 & 5.0 & 0.0 & 0.0 \\
        6.0 & 0.0 & 0.0 & 7.0 & 8.0 \\
      \end{pmatrix}

where

.. math::

  \begin{array}{ll}
    \text{coo_val}[8] & = \{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0\} \\
    \text{coo_ind}[16] & = \{0, 0, 0, 1, 0, 3, 1, 1, 1, 2, 2, 0, 2, 3, 2, 4\} \\
  \end{array}

CSR storage format
------------------

The Compressed Sparse Row (CSR) storage format represents an :math:`m \times n` matrix by:

=========== =========================================================================
m           Number of rows (integer).
n           Number of columns (integer).
nnz         Number of non-zero elements (integer).
csr_val     Array of ``nnz`` elements containing the data (floating point).
csr_row_ptr Array of ``m+1`` elements that point to the start of every row (integer).
csr_col_ind Array of ``nnz`` elements containing the column indices (integer).
=========== =========================================================================

The CSR matrix is expected to be sorted by column indices within each row. Each pair of indices should appear only once.
Consider the following :math:`3 \times 5` matrix and the corresponding CSR structures,
with :math:`m = 3, n = 5`, and :math:`\text{nnz} = 8` using one-based indexing:

.. math::

  A = \begin{pmatrix}
        1.0 & 2.0 & 0.0 & 3.0 & 0.0 \\
        0.0 & 4.0 & 5.0 & 0.0 & 0.0 \\
        6.0 & 0.0 & 0.0 & 7.0 & 8.0 \\
      \end{pmatrix}

where

.. math::

  \begin{array}{ll}
    \text{csr_val}[8] & = \{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0\} \\
    \text{csr_row_ptr}[4] & = \{1, 4, 6, 9\} \\
    \text{csr_col_ind}[8] & = \{1, 2, 4, 2, 3, 1, 4, 5\}
  \end{array}

CSC storage format
------------------

The Compressed Sparse Column (CSC) storage format represents an :math:`m \times n` matrix by:

=========== =========================================================================
m           Number of rows (integer).
n           Number of columns (integer).
nnz         Number of non-zero elements (integer).
csc_val     Array of ``nnz`` elements containing the data (floating point).
csc_col_ptr Array of ``n+1`` elements that point to the start of every column (integer).
csc_row_ind Array of ``nnz`` elements containing the row indices (integer).
=========== =========================================================================

The CSC matrix is expected to be sorted by row indices within each column. Each pair of indices should appear only once.
Consider the following :math:`3 \times 5` matrix and the corresponding CSC structures,
with :math:`m = 3, n = 5`, and :math:`\text{nnz} = 8` using one-based indexing:

.. math::

  A = \begin{pmatrix}
        1.0 & 2.0 & 0.0 & 3.0 & 0.0 \\
        0.0 & 4.0 & 5.0 & 0.0 & 0.0 \\
        6.0 & 0.0 & 0.0 & 7.0 & 8.0 \\
      \end{pmatrix}

where

.. math::

  \begin{array}{ll}
    \text{csc_val}[8] & = \{1.0, 6.0, 2.0, 4.0, 5.0, 3.0, 7.0, 8.0\} \\
    \text{csc_col_ptr}[6] & = \{1, 3, 5, 6, 8, 9\} \\
    \text{csc_row_ind}[8] & = \{1, 3, 1, 2, 2, 1, 3, 3\}
  \end{array}

BSR storage format
------------------

The Block Compressed Sparse Row (BSR) storage format represents an :math:`(mb \cdot \text{bsr_dim}) \times (nb \cdot \text{bsr_dim})` matrix by:

=========== ==============================================================================================================================================
mb          Number of block rows (integer).
nb          Number of block columns (integer).
nnzb        Number of non-zero blocks (integer).
bsr_val     Array of ``nnzb * bsr_dim * bsr_dim`` elements containing the data (floating point). Blocks can be stored in column-major or row-major format.
bsr_row_ptr Array of ``mb+1`` elements that point to the start of every block row (integer).
bsr_col_ind Array of ``nnzb`` elements containing the block column indices (integer).
bsr_dim     Dimension of each block (integer).
=========== ==============================================================================================================================================

The BSR matrix is expected to be sorted by column indices within each row.
If :math:`m` or :math:`n` are not evenly divisible by the block dimension, then zeros are padded to the matrix,
such that :math:`mb = (m + \text{bsr_dim} - 1) / \text{bsr_dim}` and :math:`nb = (n + \text{bsr_dim} - 1) / \text{bsr_dim}`.
Consider the following :math:`4 \times 3` matrix and the corresponding BSR structures,
with :math:`\text{bsr_dim} = 2, mb = 2, nb = 2`, and :math:`\text{nnzb} = 4` using zero-based indexing and column-major storage:

.. math::

  A = \begin{pmatrix}
        1.0 & 0.0 & 2.0 \\
        3.0 & 0.0 & 4.0 \\
        5.0 & 6.0 & 0.0 \\
        7.0 & 0.0 & 8.0 \\
      \end{pmatrix}

with the blocks :math:`A_{ij}`

.. math::

  A_{00} = \begin{pmatrix}
             1.0 & 0.0 \\
             3.0 & 0.0 \\
           \end{pmatrix},
  A_{01} = \begin{pmatrix}
             2.0 & 0.0 \\
             4.0 & 0.0 \\
           \end{pmatrix},
  A_{10} = \begin{pmatrix}
             5.0 & 6.0 \\
             7.0 & 0.0 \\
           \end{pmatrix},
  A_{11} = \begin{pmatrix}
             0.0 & 0.0 \\
             8.0 & 0.0 \\
           \end{pmatrix}

such that

.. math::

  A = \begin{pmatrix}
        A_{00} & A_{01} \\
        A_{10} & A_{11} \\
      \end{pmatrix}

with arrays represented as

.. math::

  \begin{array}{ll}
    \text{bsr_val}[16] & = \{1.0, 3.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0, 5.0, 7.0, 6.0, 0.0, 0.0, 8.0, 0.0, 0.0\} \\
    \text{bsr_row_ptr}[3] & = \{0, 2, 4\} \\
    \text{bsr_col_ind}[4] & = \{0, 1, 0, 1\}
  \end{array}

GEBSR storage format
--------------------

The General Block Compressed Sparse Row (GEBSR) storage format represents an :math:`(mb \cdot \text{bsr_row_dim}) \times (nb \cdot \text{bsr_col_dim})` matrix by:

=========== ======================================================================================================================================================
mb          Number of block rows (integer).
nb          Number of block columns (integer).
nnzb        Number of non-zero blocks (integer).
bsr_val     Array of ``nnzb * bsr_row_dim * bsr_col_dim`` elements containing the data (floating point). Blocks can be stored in column-major or row-major format.
bsr_row_ptr Array of ``mb+1`` elements that point to the start of every block row (integer).
bsr_col_ind Array of ``nnzb`` elements containing the block column indices (integer).
bsr_row_dim Row dimension of each block (integer).
bsr_col_dim Column dimension of each block (integer).
=========== ======================================================================================================================================================

The GEBSR matrix is expected to be sorted by column indices within each row.
If :math:`m` is not evenly divisible by the row block dimension or :math:`n` is not evenly
divisible by the column block dimension, then zeros are padded to the matrix,
such that :math:`mb = (m + \text{bsr_row_dim} - 1) / \text{bsr_row_dim}` and
:math:`nb = (n + \text{bsr_col_dim} - 1) / \text{bsr_col_dim}`. Consider the following :math:`4 \times 5` matrix
and the corresponding GEBSR structures,
with :math:`\text{bsr_row_dim} = 2`, :math:`\text{bsr_col_dim} = 3`, :math:`mb = 2`, :math:`nb = 2`,
and :math:`\text{nnzb} = 4` using zero-based indexing and column-major storage:

.. math::

  A = \begin{pmatrix}
        1.0 & 0.0 & 0.0 & 2.0 & 0.0 \\
        3.0 & 0.0 & 4.0 & 0.0 & 0.0 \\
        5.0 & 6.0 & 0.0 & 7.0 & 0.0 \\
        0.0 & 0.0 & 8.0 & 0.0 & 9.0 \\
      \end{pmatrix}

with the blocks :math:`A_{ij}`

.. math::

  A_{00} = \begin{pmatrix}
             1.0 & 0.0 & 0.0 \\
             3.0 & 0.0 & 4.0 \\
           \end{pmatrix},
  A_{01} = \begin{pmatrix}
             2.0 & 0.0 & 0.0 \\
             0.0 & 0.0 & 0.0 \\
           \end{pmatrix},
  A_{10} = \begin{pmatrix}
             5.0 & 6.0 & 0.0 \\
             0.0 & 0.0 & 8.0 \\
           \end{pmatrix},
  A_{11} = \begin{pmatrix}
             7.0 & 0.0 & 0.0 \\
             0.0 & 9.0 & 0.0 \\
           \end{pmatrix}

such that

.. math::

  A = \begin{pmatrix}
        A_{00} & A_{01} \\
        A_{10} & A_{11} \\
      \end{pmatrix}

with arrays represented as

.. math::

  \begin{array}{ll}
    \text{bsr_val}[24] & = \{1.0, 3.0, 0.0, 0.0, 0.0, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 6.0, 0.0, 0.0, 8.0, 7.0, 0.0, 0.0, 9.0, 0.0, 0.0\} \\
    \text{bsr_row_ptr}[3] & = \{0, 2, 4\} \\
    \text{bsr_col_ind}[4] & = \{0, 1, 0, 1\}
  \end{array}

ELL storage format
------------------

The Ellpack-Itpack (ELL) storage format represents an :math:`m \times n` matrix by:

=========== ================================================================================
m           Number of rows (integer).
n           Number of columns (integer).
ell_width   Maximum number of non-zero elements per row (integer).
ell_val     Array of ``m * ell_width`` elements containing the data (floating point).
ell_col_ind Array of ``m * ell_width`` elements containing the column indices (integer).
=========== ================================================================================

The ELL matrix is assumed to be stored in column-major format. Rows with less
than ``ell_width`` non-zero elements are padded with zeros (``ell_val``) and :math:`-1` (``ell_col_ind``).
Consider the following :math:`3 \times 5` matrix and the corresponding ELL structures,
with :math:`m = 3, n = 5` and :math:`\text{ell_width} = 3` using zero-based indexing:

.. math::

  A = \begin{pmatrix}
        1.0 & 2.0 & 0.0 & 3.0 & 0.0 \\
        0.0 & 4.0 & 5.0 & 0.0 & 0.0 \\
        6.0 & 0.0 & 0.0 & 7.0 & 8.0 \\
      \end{pmatrix}

where

.. math::

  \begin{array}{ll}
    \text{ell_val}[9] & = \{1.0, 4.0, 6.0, 2.0, 5.0, 7.0, 3.0, 0.0, 8.0\} \\
    \text{ell_col_ind}[9] & = \{0, 1, 0, 1, 2, 3, 3, -1, 4\}
  \end{array}

Blocked ELL storage format
--------------------------

The Blocked Ellpack (ELL) storage format represents an :math:`(mb \cdot \text{block_dim}) \times (nb \cdot \text{block_dim})` matrix by:

=========== ================================================================================
mb          Number of block rows (integer).
nb          Number of block columns (integer).
ell_width   Maximum number of non-zero block elements per row (integer).
ell_val     Array of ``mb * ell_width * block_dim * block_dim`` elements containing the data (floating point).
ell_col_ind Array of ``mb * ell_width`` elements containing the column indices (integer).
block_dim   Dimension of each block (integer).
=========== ================================================================================

The Blocked ELL is similar to the ELL format except that column entries now indicate the location of two dimensional blocks of size
``block_dim * block_dim`` instead of single matrix entries. The block values can be stored in either row or column ordering.
Rows with less than ``ell_width`` non-zero blocks are padded with zero blocks (``ell_val``) and :math:`-1` (``ell_col_ind``).
Consider the following :math:`6 \times 6` matrix and the corresponding Blocked ELL structures,
with :math:`mb = 3, nb = 3, block_dim = 2` and :math:`\text{ell_width} = 2` using zero-based indexing and row ordering for the blocks:

.. math::

  A = \begin{pmatrix}
        1.0 & 2.0 & 0.0 & 0.0 & 3.0 & 1.0 \\
        2.0 & 4.0 & 0.0 & 0.0 & 4.0 & 3.0 \\
        0.0 & 0.0 & 6.0 & 4.0 & 7.0 & 8.0 \\
        0.0 & 0.0 & 4.0 & 5.0 & 3.0 & 2.0 \\
        1.0 & 2.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
        2.0 & 1.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
      \end{pmatrix}

with the blocks :math:`A_{ij}`

.. math::

  A_{00} = \begin{pmatrix}
             1.0 & 2.0 \\
             2.0 & 4.0 \\
           \end{pmatrix},
  A_{02} = \begin{pmatrix}
             3.0 & 1.0 \\
             4.0 & 3.0 \\
           \end{pmatrix},
  A_{11} = \begin{pmatrix}
             6.0 & 4.0 \\
             4.0 & 5.0 \\
           \end{pmatrix},
  A_{12} = \begin{pmatrix}
             7.0 & 8.0 \\
             3.0 & 2.0 \\
           \end{pmatrix},
  A_{21} = \begin{pmatrix}
             1.0 & 2.0 \\
             2.0 & 1.0 \\
           \end{pmatrix}

such that

.. math::

  A = \begin{pmatrix}
        A_{00} & 0      & A_{02} \\
        0      & A_{11} & A_{12} \\
        A_{21} & 0      & 0      \\
      \end{pmatrix}

where

.. math::

  \begin{array}{ll}
    \text{ell_val}[20] & = \{1.0, 2.0, 2.0, 4.0, 6.0, 4.0, 4.0, 5.0, 1.0, 2.0, 2.0, 1.0, 3.0, 1.0, 4.0, 3.0, 7.0, 8.0, 3.0, 2.0, 0.0, 0.0, 0.0, 0.0\} \\
    \text{ell_col_ind}[6] & = \{0, 1, 0, 2, 2, -1\}
  \end{array}

.. _HYB storage format:

HYB storage format
------------------

The Hybrid (HYB) storage format represents an :math:`m \times n` matrix by:

=========== =========================================================================================
m           Number of rows (integer).
n           Number of columns (integer).
nnz         Number of non-zero elements of the COO part (integer).
ell_width   Maximum number of non-zero elements per row of the ELL part (integer).
ell_val     Array of ``m * ell_width`` elements containing the data for the ELL part (floating point).
ell_col_ind Array of ``m * ell_width`` elements containing the column indices for the ELL part (integer).
coo_val     Array of ``nnz`` elements containing the data for the COO part (floating point).
coo_row_ind Array of ``nnz`` elements containing the row indices for the COO part (integer).
coo_col_ind Array of ``nnz`` elements containing the column indices for the COO part (integer).
=========== =========================================================================================

The HYB format is a combination of the ELL and COO sparse matrix formats.
Typically, the regular part of the matrix is stored in
ELL storage format, and the irregular part of the matrix is stored
in COO storage format. Three different partitioning schemes can
be applied when converting a CSR matrix to a matrix in
HYB storage format. For further details on the partitioning schemes,
see :ref:`rocsparse_hyb_partition_`.

.. _index_base:

Storage schemes and indexing base
=================================

rocSPARSE supports 0-based and 1-based indexing.
The index base is selected by the :cpp:enum:`rocsparse_index_base` type,
which is either passed as a standalone parameter or as part of the :cpp:type:`rocsparse_mat_descr` type.

Dense vectors are represented with a 1D array, stored linearly in memory.
Sparse vectors are represented by a 1D data array that holds all non-zero elements
and a 1D indexing array that holds the positions of the corresponding non-zero elements,
both stored linearly in memory.

Pointer mode
============

The auxiliary functions :cpp:func:`rocsparse_set_pointer_mode` and :cpp:func:`rocsparse_get_pointer_mode`
are used to set and get the value of the state variable :cpp:enum:`rocsparse_pointer_mode`.
If :cpp:enum:`rocsparse_pointer_mode` is equal to :cpp:enumerator:`rocsparse_pointer_mode_host`,
then scalar parameters must be allocated on the host.
If :cpp:enum:`rocsparse_pointer_mode` is equal to :cpp:enumerator:`rocsparse_pointer_mode_device`,
then scalar parameters must be allocated on the device.

There are two types of scalar parameter:

#. Scaling parameters, such as ``alpha`` and ``beta``, used, for example, in :cpp:func:`rocsparse_scsrmv` and :cpp:func:`rocsparse_scoomv`.
#. Scalar results from functions such as :cpp:func:`rocsparse_sdoti` or :cpp:func:`rocsparse_cdotci`.

For scalar parameters such as ``alpha`` and ``beta``, memory can be allocated on the host heap or stack
when :cpp:enum:`rocsparse_pointer_mode` is equal to :cpp:enumerator:`rocsparse_pointer_mode_host`.
The kernel launch is asynchronous, and if the scalar parameter is on the heap, it can be freed after the kernel launch returns.
When :cpp:enum:`rocsparse_pointer_mode` is equal to :cpp:enumerator:`rocsparse_pointer_mode_device`,
the scalar parameter must not be changed until the kernel completes.

For scalar results, when :cpp:enum:`rocsparse_pointer_mode` is equal to :cpp:enumerator:`rocsparse_pointer_mode_host`,
the function blocks the CPU until the GPU has copied the result back to the host.
When :cpp:enum:`rocsparse_pointer_mode` is equal to :cpp:enumerator:`rocsparse_pointer_mode_device`,
the function returns after the asynchronous launch.
Similar to the vector and matrix results, the scalar result is only available when the kernel has completed execution.

.. _rocsparse_logging:

Activity logging [Deprecated]
=============================

Four different environment variables can be set to enable logging in rocSPARSE:
``ROCSPARSE_LAYER``, ``ROCSPARSE_LOG_TRACE_PATH``, ``ROCSPARSE_LOG_BENCH_PATH``, and ``ROCSPARSE_LOG_DEBUG_PATH``.

``ROCSPARSE_LAYER`` is a bit mask that enables logging, where several logging modes for :ref:`rocsparse_layer_mode_`
can be specified as follows:

================================  ==============================================================
``ROCSPARSE_LAYER`` not set       Logging is disabled.
``ROCSPARSE_LAYER`` set to ``1``  Trace logging is enabled.
``ROCSPARSE_LAYER`` set to ``2``  Bench logging is enabled.
``ROCSPARSE_LAYER`` set to ``3``  Trace logging and bench logging are enabled.
``ROCSPARSE_LAYER`` set to ``4``  Debug logging is enabled.
``ROCSPARSE_LAYER`` set to ``5``  Trace logging and debug logging are enabled.
``ROCSPARSE_LAYER`` set to ``6``  Bench logging and debug logging are enabled.
``ROCSPARSE_LAYER`` set to ``7``  Trace logging and bench logging and debug logging are enabled.
================================  ==============================================================

When logging is enabled, each rocSPARSE function call writes the function name and function arguments to the logging stream.
The default logging output is streamed to ``stderr``.

.. note::

   Performance will degrade when logging is enabled. By default, the environment variable ``ROCSPARSE_LAYER`` is not set and
   logging is disabled.

To capture activity logging in a file, set the following environment variables as required:

*  ``ROCSPARSE_LOG_TRACE_PATH`` specifies a path and file name to capture trace logging streamed to that file.
*  ``ROCSPARSE_LOG_BENCH_PATH`` specifies a path and file name to capture bench logging.
*  ``ROCSPARSE_LOG_DEBUG_PATH`` specifies a path and file name to capture debug logging.

.. note::

   If the file cannot be opened, the logging output is streamed to ``stderr``.

.. warning::
  Trace, debug, and bench logging is deprecated and will be removed in a future release

ROC-TX support in rocSPARSE
============================

The `ROC-TX <https://rocm.docs.amd.com/projects/roctracer/en/latest/reference/roctx-spec.html>`_ library contains application code
instrumentation APIs to support the high-level correlation of runtime API or activity events. When integrated with rocSPARSE, ROC-TX
enables users to view the call stack of rocSPARSE and HIP API functions in profiling tools such as :doc:`rocProfiler <rocprofiler:index>`, offering better insights
into runtime behavior and performance bottlenecks.

To enable ROC-TX profiling, set the environment variable ``ROCSPARSE_ROCTX=1`` when running the program with rocProf:

.. code-block:: shell

   ROCSPARSE_ROCTX=1 /opt/rocm/bin/rocprofv3 --kernel-trace --marker-trace --hip-trace --output-format pftrace -- ./example_program

This will generate a ``.pftrace`` file which can then be viewed using the `Perfetto UI <https://ui.perfetto.dev/>`_.

.. note::

   ROC-TX support in rocSPARSE is unavailable on Windows and is not supported in the static library version on Linux.

hipSPARSE
=========

:doc:`hipSPARSE <hipsparse:index>` is a SPARSE marshalling library with multiple supported backends.
It sits between the application and a "worker"
SPARSE library, marshalling inputs into the backend library and marshalling results back to the application. hipSPARSE exports
an interface that does not require the client to change, regardless of the chosen backend.
hipSPARSE supports rocSPARSE and NVIDIA CUDA cuSPARSE as backends.

hipSPARSE focuses on convenience and portability.
If performance outweighs these factors, then it's best to use rocSPARSE itself.
hipSPARSE can be found on `GitHub <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipsparse>`_.
