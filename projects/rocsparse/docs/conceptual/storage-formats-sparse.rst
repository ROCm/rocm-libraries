.. meta::
  :description: rocSPARSE sparse matrix storage format documentation
  :keywords: rocSPARSE, ROCm, API, documentation, design, storage format, sparse matrices
  
*******************************************
rocSPARSE storage formats
*******************************************

A sparse matrix is a matrix in which most of the items are zero,
although there is no strict criteria for the number of non-zero items. The
*sparsity* of a matrix refers to the number of non-zero elements divided by the total
number of elements. For example, if an 8 x 8 matrix with 64 elements has 16 non-zero elements,
it has a sparsity value of 0.25.

The main reason for storing and processing sparse matrices differently is to take advantage of lower memory
requirements and potentially faster processing times. It is inefficient or impractical to store
every element of a sparse matrix in memory (as a dense matrix) because most of the elements are zero.
Instead, sparse matrices are compressed in storage using
multiple vectors that map the individual non-zero values to their position in the
original matrix. However, more complex algorithms are required to store the
values in, and retrieve them from, these compound data structures.

rocSPARSE offers several storage formats for sparse matrices, each with specialized algorithms for
matrix storage, retrieval, and manipulation. For additional information about the
storage formats and their associated algorithms, see the
`Sparse matrix vector multiplication blog post <https://rocm.blogs.amd.com/high-performance-computing/spmv/part-1/README.html>`_.

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

The COO matrix is sorted by row indices and column indices per row. Furthermore, each pair of indices should appear only once.
The following :math:`3 \times 5` matrix and corresponding COO structures,
with :math:`m = 3`, :math:`n = 5`, and :math:`\text{nnz} = 8`, use zero-based indexing:

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

The COO (AoS) matrix is sorted by row indices and column indices per row.
Each pair of indices should appear only once.
The following :math:`3 \times 5` matrix and corresponding COO (AoS) structures,
with :math:`m = 3`, :math:`n = 5`, and :math:`\text{nnz} = 8`, use zero-based indexing:

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

The CSR matrix is sorted by column indices within each row. Each pair of indices should appear only once.
The following :math:`3 \times 5` matrix and corresponding CSR structures,
with :math:`m = 3`, :math:`n = 5`, and :math:`\text{nnz} = 8`, use one-based indexing:

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

The CSC matrix is sorted by row indices within each column. Each pair of indices should appear only once.
The following :math:`3 \times 5` matrix and corresponding CSC structures,
with :math:`m = 3`, :math:`n = 5`, and :math:`\text{nnz} = 8`, use one-based indexing:

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

The BSR matrix is sorted by column indices within each row.
This matrix is defined as having a number of rows equivalent to :math:`\text{block_dim} \times \text{number_of_row_blocks}`.
The following :math:`4 \times 3` matrix and corresponding BSR structures,
with :math:`\text{bsr_dim} = 2`, :math:`mb = 2`, :math:`nb = 2`, and :math:`\text{nnzb} = 4`, use zero-based indexing and column-major storage:

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

The GEBSR matrix is sorted by column indices within each row.
If :math:`m` is not evenly divisible by the row block dimension or :math:`n` is not evenly
divisible by the column block dimension, then zeros are padded to the matrix,
such that :math:`mb = (m + \text{bsr_row_dim} - 1) / \text{bsr_row_dim}` and
:math:`nb = (n + \text{bsr_col_dim} - 1) / \text{bsr_col_dim}`. The following :math:`4 \times 5` matrix
and corresponding GEBSR structures,
with :math:`\text{bsr_row_dim} = 2`, :math:`\text{bsr_col_dim} = 3`, :math:`mb = 2`, :math:`nb = 2`,
and :math:`\text{nnzb} = 4`, use zero-based indexing and column-major storage:

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
The following :math:`3 \times 5` matrix and corresponding ELL structures,
with :math:`m = 3`, :math:`n = 5`, and :math:`\text{ell_width} = 3`, use zero-based indexing:

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
The following :math:`6 \times 6` matrix and corresponding Blocked ELL structures,
with :math:`mb = 3`, :math:`nb = 3`, :math:`block_dim = 2`, and :math:`\text{ell_width} = 2`, use zero-based indexing and row ordering for the blocks:

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

rocSPARSE supports zero-based and one-based indexing.
The index base is selected by the :cpp:enum:`rocsparse_index_base` type,
which is either passed as a standalone parameter or as part of the :cpp:type:`rocsparse_mat_descr` type.

Dense vectors are represented with a 1D array, stored linearly in memory.
Sparse vectors are represented by a 1D data array that holds all non-zero elements
and a 1D indexing array that holds the positions of the corresponding non-zero elements,
both stored linearly in memory.