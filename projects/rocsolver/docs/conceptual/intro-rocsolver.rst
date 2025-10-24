.. meta::
  :description: introduction to rocSOLVER and LAPACK guide
  :keywords: rocSOLVER, ROCm, documentation, solvers, BLAS, LAPACK

.. _intro-to-solvers:

*********************************
Introduction to rocSOLVER
*********************************

rocSOLVER is an implementation of the `LAPACK routines <https://www.netlib.org/lapack/index.html>`_.
LAPACK (Linear Algebra PACKage) is a software package that contains libraries for decompositions and linear system solutions.

The LAPACK routines reorganize algorithms to make them more efficient. The high-level routines are structured to
perform block operations in the innermost loops, where they can be optimized for each architecture.
The routines are therefore transportable (requiring some optimization for the target system), rather than portable.

The LAPACK routines support both real and complex matrices, for single and double precision.
They provide solutions for a wide range of complex mathematical problems, including:

*  Solving systems of simultaneous linear equations
*  Least-squares solutions for linear systems of equations (LLS)
*  Eigenvalue problems (singular and generalized)
*  Singular value problems, including singular value decomposition (SVD)
*  Orthogonal factorization, including QR, LQ, and other factorizations
*  Estimating condition numbers
*  Triangular solve routines

Routine categorization
==========================

The LAPACK routines can be divided into several categories, based on the details of the implementation. The categories are:

*  **Driver routines**: High-level routines for solving complete problems, for instance, solving a system of equations.
   A typical driver routine calls several computational routines. It's easiest to directly call a driver routine, provided
   there's one that solves the precise problem.
*  **Computational routines**: Mid-level routines that perform a distinct computational task, such as reducing a matrix
   to tridiagonal form. If you can't find a driver routine that meets your needs, you can build your own algorithm out of
   these routines.
*  **Auxiliary routines**: Lower-level routines that perform subtasks or basic computations. Some of these routines
   might be considered for inclusion in BLAS in the future.

Many of the computational routines call BLAS routines for the lower-level implementation, 
especially Level 3 BLAS routines for matrix multiplication, rather than a LAPACK auxiliary routine.
The rocSOLVER implementation is integrated with rocBLAS for this purpose.
For information about the BLAS routines, see the :doc:`BLAS operations introduction <rocblas:conceptual/blas-operations-intro>`.

Naming conventions
==========================

The name of each routine encodes several key pieces of information about the routine, as follows:

*  **Data type**: The first letter of the routine indicates the data type that the routine can handle.
*  **Matrix type**: The next two letters indicate the type of matrix or the most significant matrix if there are several types.
   Most of these two-letter codes apply to both real and complex matrices, but a few apply specifically to one or the other.
*  **Computation**: The next three letters (sometimes two) indicate the basic operation to be performed.

The three components are combined using the format *XYYZZZ*, where X indicates the data type, YY the matrix type,
and ZZZ the operation. For example, ``SGEBRD`` is a single-precision (``S``) routine that operates on a 
general (``GE``) matrix to perform a bidiagonal reduction (``BRD``) operation.

Data types
------------------------------

The first letter (X) maps to one of the following data types:

*  ``S``: Real
*  ``D``: Double precision
*  ``C``: Complex
*  ``Z``: Double complex (Complex*16)

.. note::

   For more detailed information on the numerical precision types supported by the rocSOLVER library,
   see :doc:`rocSOLVER precision support <../reference/precision>`.

Matrix types
------------------------------

The next two letters (YY) indicate the type of matrix (or of the most significant matrix).
Here are some common LAPACK matrix types that rocSOLVER supports.

*  ``BD``:	Bidiagonal
*  ``DI``:	Diagonal
*  ``GE``:	General (for instance, unsymmetric, or, in some cases, rectangular)
*  ``GG``:	General matrices, generalized problem (pair of general matrices)
*  ``GT``:	General tridiagonal
*  ``HE``:	(complex) Hermitian
*  ``HG``:	Upper Hessenberg matrix, generalized problem (for instance, a Hessenberg and a triangular matrix)
*  ``HS``:	Upper Hessenberg
*  ``OR``:	(real) Orthogonal
*  ``PO``:	Symmetric or Hermitian positive definite
*  ``PT``:	Symmetric or Hermitian positive definite tridiagonal
*  ``ST``:	(real) Symmetric tridiagonal
*  ``SY``:	Symmetric
*  ``TG``:	Triangular matrices, generalized problem (a pair of triangular matrices)
*  ``TR``:	Triangular (or, in some cases, quasi-triangular)
*  ``TZ``:	Trapezoidal
*  ``UN``:	(complex) Unitary

Operation types
------------------------------

The last three letters, (ZZZ) indicate the computation performed.

Here are the main LAPACK driver routines that rocSOLVER supports:

*  ``SV``: Linear equation: simple variant (Solves :math:`AX = B`)
*  ``SVX``: Linear equation: expert variant (Solves more advanced variations, including :math:`A^{T}X = B` or :math:`A^{H}X = B`,
   estimating the condition number of ``A``, refining the solution, and computing forward and backward error bounds)
*  ``LS``, ``LSY``, ``LSS``, and ``LSD``: Linear least squares (LLS) problems
*  ``LSE`` and ``GLM``: Generalized linear least squares problems
*  ``EVx``: Symmetric eigenproblems (SEP)
*  ``SVD`` and ``SDD``: Singular value decomposition (SVD)
*  ``GV``: Generalized symmetric definite eigenproblems (GSEP)

Here's a list of some of the common LAPACK computational routines that rocSOLVER supports:

*  ``TRF``: Linear equations: factorize
*  ``TRS``: Linear equations: use the factorization to solve by forward or backward substitution
*  ``RFS``: Linear equations: compute the error bounds for the solution
*  ``TRI``: Linear equations: invert using factorization
*  ``EQU``: Linear equations: equilibrate
*  ``QRF``: QR factorization
*  ``LQF``: LQ factorization
*  ``RQF``: Complete orthogonal factorization
