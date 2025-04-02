.. meta::
  :description: rocSPARSE documentation and API reference library
  :keywords: rocSPARSE, ROCm, API, documentation

.. _hipsparse-docs:

********************************************************************
Using hipSPARSE
********************************************************************

HIP Device Management
=====================

Before starting a HIP kernel you can call :cpp:func:`hipSetDevice` to set a device. If you do not call the function, the system uses the default device. Unless you explicitly call :cpp:func:`hipSetDevice` to specify another device, HIP kernels are always launched on device 0.
This is a HIP (and CUDA) device management approach and is not specific to the hipSPARSE library. hipSPARSE honors this approach and assumes you have already set the preferred device before a hipSPARSE routine call.

Once you set the device, you can create a handle with :ref:`hipsparse_create_handle_`.
Subsequent hipSPARSE routines take this handle as an input parameter. hipSPARSE ONLY queries (by :cpp:func:`hipGetDevice`) the specified device. If hipSPARSE does not see a valid device, it returns an error message. It is you responsibility to provide a valid device to hipSPARSE and ensure the device safety.

If you want to change device, you must destroy the current handle using :ref:`hipsparse_destroy_handle_`, and create another handle using :ref:`hipsparse_create_handle_` specifying another device.

.. note::
  :cpp:func:`hipSetDevice` and :cpp:func:`hipGetDevice` are NOT part of the hipSPARSE API. They are part of the `HIP Runtime API - Device Management <https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___device.html>`_.

HIP Stream Management
=====================

HIP kernels are always launched in a queue (also known as stream). If you do not explicitly specify a stream, the system provides and maintains a default stream. You cannot create or destroy the default stream. However, you can freely create new streams (with :cpp:func:`hipStreamCreate`) and bind it to the hipSPARSE handle using :ref:`hipsparse_set_stream_`. HIP kernels are invoked in hipSPARSE routines. The hipSPARSE handle is always associated with a stream, and hipSPARSE passes its stream to the kernels inside the routine. One hipSPARSE routine only takes one stream in a single invocation. If you create a stream, you are responsible for destroying it. Refer to `HIP Runtime API - Stream Management <https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___stream.html>`_ for more information. 

Asynchronous Execution
======================

Except functions having memory allocation inside preventing asynchronicity, all hipSPARSE library functions are non-blocking and executed asynchronously with respect to the host, unless otherwise stated. The function may return before the actual computation has finished. To force synchronization, use either :cpp:func:`hipDeviceSynchronize` or :cpp:func:`hipStreamSynchronize`. This will ensure that all previously executed hipSPARSE functions on the device, or in the particular stream, have completed.

Multiple Streams and Multiple Devices
=====================================

If a system has multiple HIP devices, you can run multiple hipSPARSE handles concurrently. However, you can NOT run a single hipSPARSE handle on different discrete devices. Each handle is associated with a particular single device, and a new handle should be created for each additional device.

Storage Formats
===============

The following describes supported matrix storage formats.  

.. note::
    The different storage formats support indexing from a base of 0 or 1 as described in :ref:`index_base`. 

COO storage format
------------------
The Coordinate (COO) storage format represents a :math:`m \times n` matrix by

=========== ==================================================================
m           number of rows (integer).
n           number of columns (integer).
nnz         number of non-zero elements (integer).
coo_val     array of ``nnz`` elements containing the data (floating point).
coo_row_ind array of ``nnz`` elements containing the row indices (integer).
coo_col_ind array of ``nnz`` elements containing the column indices (integer).
=========== ==================================================================

The COO matrix is expected to be sorted by row indices and column indices per row. Furthermore, each pair of indices should appear only once.
Consider the following :math:`3 \times 5` matrix and the corresponding COO structures, with :math:`m = 3, n = 5` and :math:`\text{nnz} = 8` using zero based indexing:

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
The Coordinate (COO) Array of Structure (AoS) storage format represents a :math:`m \times n` matrix by

======= ==========================================================================================
m           number of rows (integer).
n           number of columns (integer).
nnz         number of non-zero elements (integer).
coo_val     array of ``nnz`` elements containing the data (floating point).
coo_ind     array of ``2 * nnz`` elements containing alternating row and column indices (integer).
======= ==========================================================================================

The COO (AoS) matrix is expected to be sorted by row indices and column indices per row. Furthermore, each pair of indices should appear only once.
Consider the following :math:`3 \times 5` matrix and the corresponding COO (AoS) structures, with :math:`m = 3, n = 5` and :math:`\text{nnz} = 8` using zero based indexing:

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
The Compressed Sparse Row (CSR) storage format represents a :math:`m \times n` matrix by

=========== =========================================================================
m           number of rows (integer).
n           number of columns (integer).
nnz         number of non-zero elements (integer).
csr_val     array of ``nnz`` elements containing the data (floating point).
csr_row_ptr array of ``m+1`` elements that point to the start of every row (integer).
csr_col_ind array of ``nnz`` elements containing the column indices (integer).
=========== =========================================================================

The CSR matrix is expected to be sorted by column indices within each row. Furthermore, each pair of indices should appear only once.
Consider the following :math:`3 \times 5` matrix and the corresponding CSR structures, with :math:`m = 3, n = 5` and :math:`\text{nnz} = 8` using one based indexing:

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
The Compressed Sparse Column (CSC) storage format represents a :math:`m \times n` matrix by

=========== =========================================================================
m           number of rows (integer).
n           number of columns (integer).
nnz         number of non-zero elements (integer).
csc_val     array of ``nnz`` elements containing the data (floating point).
csc_col_ptr array of ``n+1`` elements that point to the start of every column (integer).
csc_row_ind array of ``nnz`` elements containing the row indices (integer).
=========== =========================================================================

The CSC matrix is expected to be sorted by row indices within each column. Furthermore, each pair of indices should appear only once.
Consider the following :math:`3 \times 5` matrix and the corresponding CSC structures, with :math:`m = 3, n = 5` and :math:`\text{nnz} = 8` using one based indexing:

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
The Block Compressed Sparse Row (BSR) storage format represents a :math:`(mb \cdot \text{bsr_dim}) \times (nb \cdot \text{bsr_dim})` matrix by

=========== ====================================================================================================================================
mb          number of block rows (integer)
nb          number of block columns (integer)
nnzb        number of non-zero blocks (integer)
bsr_val     array of ``nnzb * bsr_dim * bsr_dim`` elements containing the data (floating point). Blocks can be stored column-major or row-major.
bsr_row_ptr array of ``mb+1`` elements that point to the start of every block row (integer).
bsr_col_ind array of ``nnzb`` elements containing the block column indices (integer).
bsr_dim     dimension of each block (integer).
=========== ====================================================================================================================================

The BSR matrix is expected to be sorted by column indices within each row. If :math:`m` or :math:`n` are not evenly divisible by the block dimension, then zeros are padded to the matrix, such that :math:`mb = (m + \text{bsr_dim} - 1) / \text{bsr_dim}` and :math:`nb = (n + \text{bsr_dim} - 1) / \text{bsr_dim}`.
Consider the following :math:`4 \times 3` matrix and the corresponding BSR structures, with :math:`\text{bsr_dim} = 2, mb = 2, nb = 2` and :math:`\text{nnzb} = 4` using zero based indexing and column-major storage:

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

with arrays representation

.. math::

  \begin{array}{ll}
    \text{bsr_val}[16] & = \{1.0, 3.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0, 5.0, 7.0, 6.0, 0.0, 0.0, 8.0, 0.0, 0.0\} \\
    \text{bsr_row_ptr}[3] & = \{0, 2, 4\} \\
    \text{bsr_col_ind}[4] & = \{0, 1, 0, 1\}
  \end{array}

GEBSR storage format
--------------------
The General Block Compressed Sparse Row (GEBSR) storage format represents a :math:`(mb \cdot \text{bsr_row_dim}) \times (nb \cdot \text{bsr_col_dim})` matrix by

=========== ====================================================================================================================================
mb          number of block rows (integer)
nb          number of block columns (integer)
nnzb        number of non-zero blocks (integer)
bsr_val     array of ``nnzb * bsr_row_dim * bsr_col_dim`` elements containing the data (floating point). Blocks can be stored column-major or row-major.
bsr_row_ptr array of ``mb+1`` elements that point to the start of every block row (integer).
bsr_col_ind array of ``nnzb`` elements containing the block column indices (integer).
bsr_row_dim row dimension of each block (integer).
bsr_col_dim column dimension of each block (integer).
=========== ====================================================================================================================================

The GEBSR matrix is expected to be sorted by column indices within each row. If :math:`m` is not evenly divisible by the row block dimension or :math:`n` is not evenly
divisible by the column block dimension, then zeros are padded to the matrix, such that :math:`mb = (m + \text{bsr_row_dim} - 1) / \text{bsr_row_dim}` and
:math:`nb = (n + \text{bsr_col_dim} - 1) / \text{bsr_col_dim}`. Consider the following :math:`4 \times 5` matrix and the corresponding GEBSR structures,
with :math:`\text{bsr_row_dim} = 2`, :math:`\text{bsr_col_dim} = 3`, mb = 2, nb = 2` and :math:`\text{nnzb} = 4` using zero based indexing and column-major storage:

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

with arrays representation

.. math::

  \begin{array}{ll}
    \text{bsr_val}[24] & = \{1.0, 3.0, 0.0, 0.0, 0.0, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 6.0, 0.0, 0.0, 8.0, 7.0, 0.0, 0.0, 9.0, 0.0, 0.0\} \\
    \text{bsr_row_ptr}[3] & = \{0, 2, 4\} \\
    \text{bsr_col_ind}[4] & = \{0, 1, 0, 1\}
  \end{array}

ELL storage format
------------------
The Ellpack-Itpack (ELL) storage format represents a :math:`m \times n` matrix by

=========== ================================================================================
m           number of rows (integer).
n           number of columns (integer).
ell_width   maximum number of non-zero elements per row (integer)
ell_val     array of ``m times ell_width`` elements containing the data (floating point).
ell_col_ind array of ``m times ell_width`` elements containing the column indices (integer).
=========== ================================================================================

The ELL matrix is assumed to be stored in column-major format. Rows with less than ``ell_width`` non-zero elements are padded with zeros (``ell_val``) and :math:`-1` (``ell_col_ind``).
Consider the following :math:`3 \times 5` matrix and the corresponding ELL structures, with :math:`m = 3, n = 5` and :math:`\text{ell_width} = 3` using zero based indexing:

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

.. _HYB storage format:

HYB storage format
------------------
The Hybrid (HYB) storage format represents a :math:`m \times n` matrix by

=========== =========================================================================================
m           number of rows (integer).
n           number of columns (integer).
nnz         number of non-zero elements of the COO part (integer)
ell_width   maximum number of non-zero elements per row of the ELL part (integer)
ell_val     array of ``m times ell_width`` elements containing the ELL part data (floating point).
ell_col_ind array of ``m times ell_width`` elements containing the ELL part column indices (integer).
coo_val     array of ``nnz`` elements containing the COO part data (floating point).
coo_row_ind array of ``nnz`` elements containing the COO part row indices (integer).
coo_col_ind array of ``nnz`` elements containing the COO part column indices (integer).
=========== =========================================================================================

The HYB format is a combination of the ELL and COO sparse matrix formats. Typically, the regular part of the matrix is stored in ELL storage format, and the irregular part of the matrix is stored in COO storage format. Three different partitioning schemes can be applied when converting a CSR matrix to a matrix in HYB storage format. For further details on the partitioning schemes, see :ref:`hipsparse_hyb_partition_`.

.. _index_base:

Storage schemes and indexing base
=================================
hipSPARSE supports 0 and 1 based indexing.
The index base is selected by the :cpp:enum:`hipsparseIndexBase_t` type which is either passed as standalone parameter or as part of the :cpp:type:`hipsparseMatDescr_t` type.

Furthermore, dense vectors are represented with a 1D array, stored linearly in memory.
Sparse vectors are represented by a 1D data array stored linearly in memory that hold all non-zero elements and a 1D indexing array stored linearly in memory that hold the positions of the corresponding non-zero elements.

Pointer mode
============
The auxiliary functions :cpp:func:`hipsparseSetPointerMode` and :cpp:func:`hipsparseGetPointerMode` are used to set and get the value of the state variable :cpp:enum:`hipsparsePointerMode_t`.
If :cpp:enum:`hipsparsePointerMode_t` is equal to :cpp:enumerator:`HIPSPARSE_POINTER_MODE_HOST`, then scalar parameters must be allocated on the host.
If :cpp:enum:`hipsparsePointerMode_t` is equal to :cpp:enumerator:`HIPSPARSE_POINTER_MODE_DEVICE`, then scalar parameters must be allocated on the device.

There are two types of scalar parameter:

  1. Scaling parameters, such as `alpha` and `beta` used for example in :cpp:func:`hipsparseScsrmv` and :cpp:func:`hipsparseSbsrmv`
  2. Scalar results from functions such as :cpp:func:`hipsparseSdoti` or :cpp:func:`hipsparseCdotci` 

For scalar parameters such as alpha and beta, memory can be allocated on the host heap or stack, when :cpp:enum:`hipsparsePointerMode_t` is equal to :cpp:enumerator:`HIPSPARSE_POINTER_MODE_HOST`.
The kernel launch is asynchronous, and if the scalar parameter is on the heap, it can be freed after the return from the kernel launch.
When :cpp:enum:`hipsparsePointerMode_t` is equal to :cpp:enumerator:`HIPSPARSE_POINTER_MODE_DEVICE`, the scalar parameter must not be changed till the kernel completes.

For scalar results, when :cpp:enum:`hipsparsePointerMode_t` is equal to :cpp:enumerator:`HIPSPARSE_POINTER_MODE_HOST`, the function blocks the CPU till the GPU has copied the result back to the host.
Using :cpp:enum:`hipsparsePointerMode_t` equal to :cpp:enumerator:`HIPSPARSE_POINTER_MODE_DEVICE`, the function will return after the asynchronous launch.
Similarly to vector and matrix results, the scalar result is only available when the kernel has completed execution.
