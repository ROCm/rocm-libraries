.. meta::
   :description: rocALUTION preconditioners
   :keywords: rocALUTION, ROCm, library, API, preconditioners

.. _preconditioners:

***************************
rocALUTION Preconditioners
***************************

This document describes the theory and usage of rocALUTION preconditioners. All preconditioners support local operators. They can be used as a global preconditioner via a block-Jacobi scheme, which works locally on each interior matrix. To provide fast application, all preconditioners require extra memory to keep the approximated operator.

Member documentation, configuration routines, and usage notes for each class are on the :ref:`api` page.

Code structure
==============

The preconditioners provide a solution to the system :math:`Mz = r`, where the solution :math:`z` is either directly computed by the approximation scheme or iteratively obtained with :math:`z = 0` initial guess.

:cpp:class:`rocalution::Preconditioner`

Jacobi method
=============

The Jacobi method is for solving a diagonally dominant system of linear equations :math:`Ax=b`. It solves for each diagonal element iteratively until convergence, such that

.. math::

   x_{i}^{(k+1)} = (1 - \omega)x_{i}^{(k)} + \frac{\omega}{a_{ii}}
   \left(
     b_{i} - \sum\limits_{j=1}^{i-1}{a_{ij}x_{j}^{(k)}} -
     \sum\limits_{j=i}^{n}{a_{ij}x_{j}^{(k)}}
   \right)

.. note::

   To adjust the damping parameter :math:`\omega`, use :cpp:func:`rocalution::FixedPoint::SetRelaxation`.

:cpp:class:`rocalution::Jacobi`

(Symmetric) Gauss-Seidel or (S)SOR method
==========================================

The Gauss-Seidel / SOR method is for solving system of linear equations :math:`Ax=b`. It approximates the solution iteratively with

.. math::

   x_{i}^{(k+1)} = (1 - \omega) x_{i}^{(k)} + \frac{\omega}{a_{ii}}
   \left(
     b_{i} - \sum\limits_{j=1}^{i-1}{a_{ij}x_{j}^{(k+1)}} -
     \sum\limits_{j=i}^{n}{a_{ij}x_{j}^{(k)}}
   \right),

with :math:`\omega \in (0,2)`.

The Symmetric Gauss-Seidel / SSOR method is for solving system of linear equations :math:`Ax=b`. It approximates the solution iteratively.

.. note::

   To adjust the relaxation parameter :math:`\omega`, use :cpp:func:`rocalution::FixedPoint::SetRelaxation`.

:cpp:class:`rocalution::GS`, :cpp:class:`rocalution::SGS`

Incomplete factorizations
=========================

ILU
---

The Incomplete LU Factorization based on levels computes a sparse lower and sparse upper triangular matrix such that :math:`A = LU - R`.

:cpp:class:`rocalution::ILU`

ILUT
----

The Incomplete LU Factorization based on threshold computes a sparse lower and sparse upper triangular matrix such that :math:`A = LU - R`. Fill-in values are dropped depending on a threshold and number of maximal fill-ins per row.

:cpp:class:`rocalution::ILUT`

IC
---

The Incomplete Cholesky Factorization computes a sparse lower triangular matrix such that :math:`A=LL^{T} - R`. Additional fill-ins are dropped and the sparsity pattern of the original matrix is preserved.

:cpp:class:`rocalution::IC`

AI Chebyshev
============

The Approximate Inverse - Chebyshev Preconditioner is an inverse matrix preconditioner with values from a linear combination of matrix-valued Chebyshev polynomials.

:cpp:class:`rocalution::AIChebyshev`

FSAI
====

The Factorized Sparse Approximate Inverse preconditioner computes a direct approximation of :math:`M^{-1}` by minimizing the Frobenius norm :math:`||I - GL||_{F}`, where :math:`L` denotes the exact lower triangular part of :math:`A` and :math:`G:=M^{-1}`. The FSAI preconditioner is initialized by :math:`q`, based on the sparsity pattern of :math:`|A^{q}|`. However, it is also possible to supply external sparsity patterns in form of the LocalMatrix class.

.. note::

   The FSAI preconditioner is only suited for symmetric positive definite matrices.

:cpp:class:`rocalution::FSAI`

SPAI
====

The SParse Approximate Inverse algorithm is an explicitly computed preconditioner for general sparse linear systems. In its current implementation, only the sparsity pattern of the system matrix is supported. The SPAI computation is based on the minimization of the Frobenius norm :math:`||AM - I||_{F}`.

:cpp:class:`rocalution::SPAI`

TNS
===

The Truncated Neumann Series (TNS) preconditioner is based on :math:`M^{-1} = K^{T} D^{-1} K`, where :math:`K=(I-LD^{-1}+(LD^{-1})^{2})`, with the diagonal :math:`D` of :math:`A` and the strictly lower triangular part :math:`L` of :math:`A`. The preconditioner can be computed in two forms - explicitly and implicitly. In the explicit form, the full construction of :math:`M` is performed via matrix-matrix operations, whereas in the implicit form, the application of the preconditioner is based on matrix-vector operations only. The matrix format for the stored matrices can be specified.

:cpp:class:`rocalution::TNS`

MultiColored preconditioners
============================

Multi-colored preconditioners reorder the unknowns to expose parallelism in forward and backward substitution. Derived multi-colored preconditioners can change the preconditioner matrix format with :cpp:func:`rocalution::MultiColored::SetPrecondMatrixFormat`.

:cpp:class:`rocalution::MultiColored`

MultiColored (symmetric) Gauss-Seidel / (S)SOR
----------------------------------------------

The Multi-Colored Symmetric Gauss-Seidel / SSOR preconditioner is based on the splitting of the original matrix. Higher parallelism in solving the forward and backward substitution is obtained by performing a multi-colored decomposition. Details on the Symmetric Gauss-Seidel / SSOR algorithm can be found in the :cpp:class:`rocalution::SGS` preconditioner.

The Multi-Colored Gauss-Seidel / SOR preconditioner is based on the splitting of the original matrix. Higher parallelism in solving the forward substitution is obtained by performing a multi-colored decomposition. Details on the Gauss-Seidel / SOR algorithm can be found in the :cpp:class:`rocalution::GS` preconditioner.

.. note::

   To change the preconditioner matrix format, use :cpp:func:`rocalution::MultiColored::SetPrecondMatrixFormat`.

:cpp:class:`rocalution::MultiColoredGS`, :cpp:class:`rocalution::MultiColoredSGS`

MultiColored power(q)-pattern method ILU(p,q)
---------------------------------------------

Multi-Colored Incomplete LU Factorization based on the ILU(p) factorization with a power(q)-pattern method. This method provides a higher degree of parallelism of forward and backward substitution compared to the standard ILU(p) preconditioner.

.. note::

   To change the preconditioner matrix format, use :cpp:func:`rocalution::MultiColored::SetPrecondMatrixFormat`.

:cpp:class:`rocalution::MultiColoredILU`

Multi-elimination incomplete LU
===============================

The Multi-Elimination Incomplete LU preconditioner is based on the following decomposition

.. math::

   A = \begin{pmatrix} D & F \\ E & C \end{pmatrix}
     = \begin{pmatrix} I & 0 \\ ED^{-1} & I \end{pmatrix} \times
       \begin{pmatrix} D & F \\ 0 & \hat{A} \end{pmatrix},

where :math:`\hat{A} = C - ED^{-1} F`. To make the inversion of :math:`D` easier, we permute the preconditioning before the factorization with a permutation :math:`P` to obtain only diagonal elements in :math:`D`. The permutation here is based on a maximal independent set. This procedure can be applied to the block matrix :math:`\hat{A}`, in this way we can perform the factorization recursively. In the last level of the recursion, we need to provide a solution procedure. By the design of the library, this can be any kind of solver.

:cpp:class:`rocalution::MultiElimination`

Diagonal preconditioner for saddle-point problems
=================================================

Consider the following saddle-point problem

.. math::

   A = \begin{pmatrix} K & F \\ E & 0 \end{pmatrix}.

For such problems we can construct a diagonal Jacobi-type preconditioner of type

.. math::

   P = \begin{pmatrix} K & 0 \\ 0 & S \end{pmatrix},

with :math:`S=ED^{-1}F`, where :math:`D` are the diagonal elements of :math:`K`. The matrix :math:`S` is fully constructed (via sparse matrix-matrix multiplication). The preconditioner needs to be initialized with two external solvers/preconditioners - one for the matrix :math:`K` and one for the matrix :math:`S`.

:cpp:class:`rocalution::DiagJacobiSaddlePointPrecond`

(Restricted) Additive Schwarz preconditioner
============================================

The Additive Schwarz preconditioner relies on a preconditioning technique, where the linear system :math:`Ax=b` can be decomposed into small sub-problems based on :math:`A_{i} = R_{i}^{T}AR_{i}`, where :math:`R_{i}` are restriction operators. Those restriction operators produce sub-matrices which overlap. This leads to contributions from two preconditioners on the overlapped area which are scaled by :math:`1/2`.

The Restricted Additive Schwarz preconditioner relies on a preconditioning technique, where the linear system :math:`Ax=b` can be decomposed into small sub-problems based on :math:`A_{i} = R_{i}^{T}AR_{i}`, where :math:`R_{i}` are restriction operators. The RAS method is a mixture of block Jacobi and the AS scheme. In this case, the sub-matrices contain overlapped areas from other blocks, too.

See the overlapped area in the figure below:

.. _AS:
.. figure:: ../data/AS.png
  :alt: 4 block additive schwarz
  :align: center

  Example of a 4 block-decomposed matrix - Additive Schwarz with overlapping preconditioner (left) and Restricted Additive Schwarz preconditioner (right).

:cpp:class:`rocalution::AS`, :cpp:class:`rocalution::RAS`

Block-Jacobi (MPI) preconditioner
=================================

The Block-Jacobi preconditioner is designed to wrap any local preconditioner and apply it in a global block fashion locally on each interior matrix.

See the Block-Jacobi (MPI) preconditioner in the figure below:

.. _BJ:
.. figure:: ../data/BJ.png
  :alt: 4 block jacobi
  :align: center

  Example of a 4 block-decomposed matrix - Block-Jacobi preconditioner.

:cpp:class:`rocalution::BlockJacobi`

Block preconditioner
====================

When handling vector fields, typically one can try to use different preconditioners and/or solvers for the different blocks. For such problems, the library provides a block-type preconditioner. This preconditioner builds the following block-type matrix

.. math::

   P = \begin{pmatrix}
         A_{d} & 0     & . & 0     \\
         B_{1} & B_{d} & . & 0     \\
         .     & .     & . & .     \\
         Z_{1} & Z_{2} & . & Z_{d}
       \end{pmatrix}

The solution of :math:`P` can be performed in two ways. It can be solved by block-lower-triangular sweeps with inversion of the blocks :math:`A_{d} \ldots Z_{d}` and with a multiplication of the corresponding blocks. This is set by :cpp:func:`rocalution::BlockPreconditioner::SetLSolver` (which is the default solution scheme). Alternatively, it can be used only with an inverse of the diagonal :math:`A_{d} \ldots Z_{d}` (Block-Jacobi type) by using :cpp:func:`rocalution::BlockPreconditioner::SetDiagonalSolver`.

:cpp:class:`rocalution::BlockPreconditioner`

Variable preconditioner
=======================

The Variable Preconditioner can hold a selection of preconditioners. Thus, any type of preconditioners can be combined. As example, the variable preconditioner can combine Jacobi, GS and ILU - then, the first iteration of the iterative solver will apply Jacobi, the second iteration will apply GS and the third iteration will apply ILU. After that, the solver will start again with Jacobi, GS, ILU.

:cpp:class:`rocalution::VariablePreconditioner`
