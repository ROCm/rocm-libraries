.. meta::
  :description: introduction to rocSOLVER and LAPACK guide
  :keywords: rocSOLVER, ROCm, documentation, solvers, BLAS, LAPACK

.. _intro-to-solvers:

*********************************
Introduction to rocSOLVER
*********************************

rocSOLVER is an implementation of a subset of the
`Linear Algebra PACKage (LAPACK) routines <https://www.netlib.org/lapack/index.html>`_.

These routines support both real and complex matrices, in single and double precision.
They provide solutions for a wide range of complex linear algebra problems, including:

*  Solving systems of linear equations and matrix inversion
*  Least-squares problems
*  Symmetric Eigenvalue and Generalized Eigenvalue problems
*  Singular value decomposition
*  Matrix factorizations, including QR, LU, Cholesky, and others
*  Matrix and problem reductions, including tridiagonalization, bidiagonalization, and others

Many of the computational routines call BLAS routines for the lower-level implementation of basic
matrix/vector operations (for example, Level 3 BLAS routines for matrix multiplication).
rocSOLVER is integrated with rocBLAS for this purpose.
For information about rocBLAS and the BLAS routines, see the
:doc:`BLAS operations introduction <rocblas:conceptual/blas-operations-intro>`.

The use of optimized BLAS functionality allows rocSOLVER to structure algorithms to perform block operations
in a more efficient way. In addition to these blocking mechanisms, there are other features in the rocSOLVER
library that can be optimized for each architecture. The routines are therefore transportable,
requiring some optimization and tuning for the target system, rather than portable.

.. note::

   For more detailed information on tuning rocSOLVER for better performance,
   see :doc:`rocSOLVER performance tuning <../reference/tuning>`.

rocSOLVER follows the LAPACK naming convention and routine categorization as closely as possible,
but there are some differences resulting from specific requirements and design choices.

Routine categorization
==========================

The rocSOLVER routines are divided into several categories, based on their visibility from a user or application
perspective. The categories are:

*  **LAPACK functions**: These are the high-level functions that LAPACK users are most likely to call.
   These routines normally solve a complete, well-known problem (such as the Symmetric Eigenvalue problem) or
   perform a computational task that users might need to build their own workflows when solving specific problems
   for their applications (for example, reducing a matrix to tridiagonal form).
   This category corresponds to what LAPACK classifies as *driver* and 
   *computational* routines and includes all flavors of the same algorithm (for example, the unblocked QR factorization,
   GEQR2, and the blocked version, GEQRF).
*  **LAPACK-like functions**: These are also high-level functions that provide similar functionality
   but are not included in the standard LAPACK routines. Some examples of rocSOLVER LAPACK-like routines include
   the LU factorization without pivoting and the Jacobi Eigensolvers.
*  **Auxiliary functions**: These are lower-level functions that perform subtasks or basic computations. They are normally
   called internally by other high-level LAPACK routines. Consequently, their interfaces can be convoluted and
   less user friendly. rocSOLVER exposes a limited set of these functions in the public API,  
   but they are less likely to be called by most users. 

Naming conventions
==========================

The name of each function in the rocSOLVER API encodes several key pieces of information about the 
routine and the problem it tackles:

*  **Data type**: The first letter of the function name indicates the data type (precision) that the routine can handle.
*  **Matrix type**: The next two letters indicate the type of matrix (or the most significant type if there are several) that
   the routine is expecting. Most of these two-letter codes apply to both real and complex precision, 
   but a few apply specifically to one or the other.
*  **Computation**: The remaining letters (normally two or three) indicate the basic operation/computation to be performed.

The three components are combined using the format *XYYZZZ*, where X indicates the data type, YY the matrix type,
and ZZZ the operation. For example, ``SGEBRD`` is a single-precision (``S``) routine that operates on a 
general (``GE``) matrix to perform a bidiagonal reduction (``BRD``) operation.


Data types
------------------------------

rocSOLVER supports the following data precision types:

*  ``S``: Real single precision
*  ``D``: Real double precision
*  ``C``: Complex single precision
*  ``Z``: Complex double precision

.. note::

   For more detailed information on the numerical precision types supported by the rocSOLVER library,
   see :doc:`rocSOLVER precision support <../reference/precision>`.

Matrix types
------------------------------

Here are the LAPACK matrix types that rocSOLVER currently supports:

*  ``GE``:	General (for instance, unsymmetric, or, in some cases, rectangular)
*  ``SY``:	Symmetric
*  ``HE``:	Hermitian (only complex)
*  ``OR``:	Orthogonal (only real)
*  ``UN``:	Unitary (only complex)
*  ``PO``:	Symmetric or Hermitian Positive definite
*  ``ST``:	Symmetric Tridiagonal (only real)
*  ``BD``:	Bidiagonal
*  ``TR``:	Triangular (or, in some cases, quasi-triangular)
*  ``LA``:	Used for some auxiliary functions where the matrix type is not relevant

Operation types
------------------------------

It's less obvious how the operation type is encoded in the function name. 
This section lists some of the computations performed by the rocSOLVER functions. For more details, see
the :doc:`API reference <../reference/intro>` documents.

Some main LAPACK computations that rocSOLVER performs are:

*  ``SVD`` and ``SDD``: Singular value decomposition and SVD with divide and conquer approach
*  ``TRF``: Triangular factorization (LU factorization)
*  ``TRS``: Linear system solver based on triangular factorization
*  ``QRF``: QR factorization (blocked version)
*  ``QR2``: QR factorization (unblocked version)
*  ``GQR``: Generation of orthogonal matrix from QR factorization
*  ``RFG``: Generation of Householder reflection
*  ``EV`` and ``EVD``: Eigenvalue solver and Eigenvalue solver with divide and conquer approach

Other examples of LAPACK-like computations performed in rocSOLVER include:

*  ``EVJ``: Jacobi iteration for Eigenvalue solver
*  ``BLTTRF``: Triangular factorization of block tridiagonal matrices
