.. meta::
  :description: An introduction to BLAS operations
  :keywords: rocBLAS, ROCm, API, Linear Algebra, documentation, BLAS, vector, matrix, operations

.. _blas-operations-intro:

********************************************************************
BLAS operations introduction
********************************************************************

rocBLAS is an implementation of the BLAS (Basic Linear Algebra Subprograms) operations
based on the BLAS reference specification in Netlib.
The BLAS specification defines basic, low-level routines for matrix and vector operations.

Here are some of the advantages of the BLAS reference operations:

*  The BLAS reference specification is freely available to use (with attribution).
*  The base routines can be optimized and adapted for different architectures and specific use cases.
*  They provide efficient, validated, hardened, and well-accepted solutions for common matrix operations.
   Developers don't have to reimplement these algorithms.
*  The BLAS reference provides a common naming convention and syntax for the different operations.
*  The routines are platform independent, so they provide interoperability and promote code sharing.
*  Online support and discussion groups are available.

The BLAS routines are included in the LAPACK library on NetLib,
which provides solutions for more complex algebraic operations.

The BLAS levels
============================

BLAS routines are divided into three subcategories: level-1, level-2, and level-3.
The categories indicate the operations that are supported and the type of inputs
to the operations.

*  **Level-1**: Scalar, vector, and vector-vector operations
*  **Level-2**: Matrix-vector operations
*  **Level-3**: Matrix-matrix operations

The levels also provide a general indication of the algorithmic complexity. For example,
level-1 routines run in approximately linear time (:math:`O(n)`), level-2 functions are
quadratic (:math:`O(n^2)`), and level-3 functions are cubic (:math:`O(n^3)`).

.. note::

   Not all levels support all functions. See the BLAS reference specification in Netlib
   for more information.

Core BLAS operations
============================

This section summarizes the most common BLAS operations and the naming conventions
that apply to the functions.

BLAS type prefixes
-------------------

Most BLAS operations indicate the data type they are designed to manipulate. The data types
are as follows.


.. csv-table::
   :header: "Prefix","Data type"
   :widths: 30, 100

   "``s``","real/float"
   "``d``","double"
   "``c``","complex"
   "``z``","double complex"

The data type is appended before the function name, so ``daxpy`` is the ``axpy``
(vector update) function for double-precision inputs.

Level-1 BLAS operations
-----------------------

The BLAS operations include the following level-1 routines.

.. csv-table::
   :header: "Command","Description"
   :widths: 30, 100

   "``axpy``","Updates a vector"
   "``scal``","Scales a vector"
   "``copy``","Copies a vector"
   "``swap``","Swaps two vectors"
   "``dot``","Calculates the vector dot product"
   "``asum``","Calculates the L1-norm of a vector"
   "``nrm2``","Calculates the L2-norm of a vector"
   "``i_amax``","Calculates the infinity-norm of a vector"
   "``rot``","Applies a plane rotation"

.. note::

   Some of these operations are available in variant forms, which are indicated by suffixes attached
   to the function name. For example, ``dotu`` indicates a "complex dot product" that operates on complex
   data types.

Level-2 and level-3 BLAS operations
-----------------------------------

The names of the level-2 and level-3 functions are composite abbreviations.
Typically, the first letter indicates the data type the function operates on,
as described above. The next two letters indicate the matrix type, and the final
two letters describe the high-level operation. For example, the name of the ``sgemm`` function
is assembled from the ``s`` (real) data type, the ``ge`` (general) matrix type,
and the ``mm`` (matrix-matrix multiply) operation. The following table lists the
most frequently used matrix types for level-2 and level-3 operations.

.. csv-table::
   :header: "Matrix type","Description"
   :widths: 20, 30

   "``ge``","General"
   "``sy``","Symmetric"
   "``he``","Hermitian"
   "``tr``","Triangular"
   "``gb``","General banded"
   "``sb``","Symmetric banded"
   "``hb``","Hermitian banded"
   "``tb``","Triangular banded"
   "``sp``","Symmetric packed"
   "``hp``","Hermitian packed"
   "``tp``","Triangular packed"

The operations differ somewhat for level-2 and level-3 operations. The next table lists the abbreviations for
the level-2 operations.

.. csv-table::
   :header: "Level-2 operation","Description"
   :widths: 20, 30

   "``mv``","Matrix-vector multiply"
   "``s``","Solve"
   "``r``/``ru``/``rc``","Rank-1 update"
   "``r2``","Rank-2 update"

The following table lists the abbreviations for the level-3 operations.

.. csv-table::
   :header: "Level-3 operation","Description"
   :widths: 20, 30

   "``mm``","Matrix-matrix multiply"
   "``sm``","Solve matrix"
   "``rk``","Rank-k update"
   "``r2k``","Rank-2k update"

BLAS operation options
-----------------------------------

Many BLAS operations accept options, although several restrictions apply.
The following table lists the main options.
See the BLAS reference specification in Netlib for
more information.

.. csv-table::
   :header: "Option","Description"
   :widths: 20, 40

   "``trans``","No transpose/transpose"
   "``uplo``","Upper/lower triangular"
   "``diag``","Non-unit/unit diagonal"
   "``side``","Left/right"

BLAS versus BLASLt
=======================

The BLASLt routines extend the BLAS routines, specifically on
general matrix-to-matrix multiply (GEMM) operations.
It provides extra flexibility and lets you control the
matrix data layouts, input types, and compute types and choose the algorithm implementations and heuristics.
The options set for a GEMM operation can be reused for different inputs.
For more information, see the :doc:`hipBLASLt <hipblaslt:index>` documentation.
