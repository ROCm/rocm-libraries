.. meta::
   :description: rocALUTION solvers
   :keywords: rocALUTION, ROCm, library, API, tool, solvers

.. _solver-class:

********************
rocALUTION solvers
********************

This document describes the theory and usage of rocALUTION solvers. Member documentation for each solver class is on the :ref:`api` page. The sections below group solvers by category, describe the underlying mathematics, and explain how to use them in practice.

Code structure
==============

Most solvers can be performed on linear operators LocalMatrix, LocalStencil and GlobalMatrix - i.e. the solvers can be performed locally (on a shared memory system) or in a distributed manner (on a cluster) via MPI. The only exception is the AMG (Algebraic Multigrid) solver which has two versions (one for LocalMatrix and one for GlobalMatrix class). The only pure local solvers (which do not support global/MPI operations) are the mixed-precision defect-correction solver and all direct solvers.

All solvers need three template parameters - Operators, Vectors and Scalar type.

The :cpp:class:`rocalution::Solver` class is purely virtual and provides an interface for:

- :cpp:func:`rocalution::Solver::SetOperator` to set the operator :math:`A`, i.e. the user can pass the matrix here.
- :cpp:func:`rocalution::Solver::Build` to build the solver (including preconditioners, sub-solvers, etc.). The user need to specify the operator first before calling Build().
- :cpp:func:`rocalution::Solver::Solve` to solve the system :math:`Ax = b`. The user need to pass a right-hand-side :math:`b` and a vector :math:`x`, where the solution will be obtained.
- :cpp:func:`rocalution::Solver::Print` to show solver information.
- :cpp:func:`rocalution::Solver::ReBuildNumeric` to only re-build the solver numerically (if possible).
- :cpp:func:`rocalution::Solver::MoveToHost` and :cpp:func:`rocalution::Solver::MoveToAccelerator` to offload the solver (including preconditioners and sub-solvers) to the host/accelerator.

:cpp:class:`rocalution::Solver`

Iterative linear solvers
========================

The iterative solvers are controlled by an iteration control object, which monitors the convergence properties of the solver, i.e. maximum number of iteration, relative tolerance, absolute tolerance and divergence tolerance. The iteration control can also record the residual history and store it in an ASCII file.

All iterative solvers are controlled based on

- Absolute stopping criteria, when :math:`|r_{k}|_{L_{p}} < \epsilon_{abs}`
- Relative stopping criteria, when :math:`|r_{k}|_{L_{p}} / |r_{1}|_{L_{p}} \leq \epsilon_{rel}`
- Divergence stopping criteria, when :math:`|r_{k}|_{L_{p}} / |r_{1}|_{L_{p}} \geq \epsilon_{div}`
- Maximum number of iteration :math:`N`, when :math:`k = N`

where :math:`k` is the current iteration, :math:`r_{k}` the residual for the current iteration :math:`k` (i.e. :math:`r_{k} = b - Ax_{k}`) and :math:`r_{1}` the starting residual (i.e. :math:`r_{1} = b - Ax_{init}`). In addition, the minimum number of iterations :math:`M` can be specified. In this case, the solver will not stop to iterate, before :math:`k \geq M`.

The :math:`L_{p}` norm is used for the computation, where :math:`p` could be 1, 2 and :math:`\infty`. The norm computation can be set with :cpp:func:`rocalution::IterativeLinearSolver::SetResidualNorm` with 1 for :math:`L_{1}`, 2 for :math:`L_{2}` and 3 for :math:`L_{\infty}`. For the computation with :math:`L_{\infty}`, the index of the maximum value can be obtained with :cpp:func:`rocalution::IterativeLinearSolver::GetAmaxResidualIndex`. If this function is called and :math:`L_{\infty}` was not selected, this function will return -1.

The reached criteria can be obtained with :cpp:func:`rocalution::IterativeLinearSolver::GetSolverStatus`, returning

- 0, if no criteria has been reached yet
- 1, if absolute tolerance has been reached
- 2, if relative tolerance has been reached
- 3, if divergence tolerance has been reached
- 4, if maximum number of iteration has been reached

:cpp:class:`rocalution::IterativeLinearSolver`

Building and solving phase
==========================
Each iterative solver consists of a building step and a solving step. During the building step all necessary auxiliary data is allocated and the preconditioner is constructed. You can now call the solving procedure, which can be called several times.

When the initial matrix associated with the solver is on the accelerator, the solver tries to build everything on the accelerator. However, some preconditioners and solvers (such as FSAI and AMG) must be constructed on the host before being transferred to the accelerator. If the initial matrix is on the host and you want to run the solver on the accelerator, then you need to move the solver to the accelerator, matrix, right-hand side, and solution vector.

.. note::

   If you have a preconditioner associated with the solver, it is moved automatically to the accelerator when you move the solver.

.. code-block:: cpp

  // CG solver
  CG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType> ls;
  // Multi-Colored ILU preconditioner
  MultiColoredILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType> p;

  // Move matrix and vectors to the accelerator
  mat.MoveToAccelerator();
  rhs.MoveToAccelerator();
  x.MoveToAccelerator();

  // Set mat to be the operator
  ls.SetOperator(mat);
  // Set p as the preconditioner of ls
  ls.SetPreconditioner(p);

  // Build the solver and preconditioner on the accelerator
  ls.Build();

  // Compute the solution on the accelerator
  ls.Solve(rhs, &x);

.. code-block:: cpp

  // CG solver
  CG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType> ls;
  // Multi-Colored ILU preconditioner
  MultiColoredILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType> p;

  // Set mat to be the operator
  ls.SetOperator(mat);
  // Set p as the preconditioner of ls
  ls.SetPreconditioner(p);

  // Build the solver and preconditioner on the host
  ls.Build();

  // Move matrix and vectors to the accelerator
  mat.MoveToAccelerator();
  rhs.MoveToAccelerator();
  x.MoveToAccelerator();

  // Move linear solver to the accelerator
  ls.MoveToAccelerator();

  // Compute the solution on the accelerator
  ls.Solve(rhs, &x);

Clear function and destructor
=============================

The :cpp:func:`rocalution::Solver::Clear` function clears all the data which is in the solver, including the associated preconditioner. Thus, the solver is not anymore associated with this preconditioner.

.. note::

   The preconditioner is not deleted (via destructor), only a :cpp:func:`rocalution::Preconditioner::Clear` is called.

.. note::

   When the destructor of the solver class is called, it automatically calls the *Clear()* function. Be careful, when declaring your solver and preconditioner in different places - we highly recommend to manually call the *Clear()* function of the solver and not rely on the destructor of the solver.

Numerical update
================

Some preconditioners require two phases in the their construction: an algebraic (e.g. compute a pattern or structure) and a numerical (compute the actual values) phase. In cases, where the structure of the input matrix is a constant (e.g. Newton-like methods), it is not necessary to fully reconstruct the preconditioner. In this case, the user can apply a numerical update to the current preconditioner and pass the new operator with :cpp:func:`rocalution::Solver::ReBuildNumeric`. If the preconditioner/solver does not support the numerical update, then a full :cpp:func:`rocalution::Solver::Clear` and :cpp:func:`rocalution::Solver::Build` is performed.

Fixed-Point iteration
=====================

The Fixed-Point iteration scheme is based on additive splitting of the matrix :math:`A = M + N`. The scheme reads

.. math::

   x_{k+1} = M^{-1} (b - N x_{k}).

It can also be reformulated as a weighted defect correction scheme

.. math::

   x_{k+1} = x_{k} - \omega M^{-1} (Ax_{k} - b).

The inversion of :math:`M` can be performed by preconditioners (Jacobi, Gauss-Seidel, ILU, etc.) or by any type of solvers.

:cpp:class:`rocalution::FixedPoint`

Krylov subspace solvers
=======================

CG
--
The Conjugate Gradient method is the best known iterative method for solving sparse symmetric positive definite (SPD) linear systems :math:`Ax=b`. It is based on orthogonal projection onto the Krylov subspace :math:`\mathcal{K}_{m}(r_{0}, A)`, where :math:`r_{0}` is the initial residual. The method can be preconditioned, where the approximation should also be SPD.

:cpp:class:`rocalution::CG`

CR
--
The Conjugate Residual method is an iterative method for solving sparse symmetric semi-positive definite linear systems :math:`Ax=b`. It is a Krylov subspace method and differs from the much more popular Conjugate Gradient method that the system matrix is not required to be positive definite. The method can be preconditioned where the approximation should also be SPD or semi-positive definite.

:cpp:class:`rocalution::CR`

GMRES
-----
The Generalized Minimum Residual method (GMRES) is a projection method for solving sparse (non) symmetric linear systems :math:`Ax=b`, based on restarting technique. The solution is approximated in a Krylov subspace :math:`\mathcal{K}=\mathcal{K}_{m}` and :math:`\mathcal{L}=A\mathcal{K}_{m}` with minimal residual, where :math:`\mathcal{K}_{m}` is the :math:`m`-th Krylov subspace with :math:`v_{1} = r_{0}/||r_{0}||_{2}`.

The Krylov subspace basis size can be set using :cpp:func:`rocalution::GMRES::SetBasisSize`. The default size is 30.

:cpp:class:`rocalution::GMRES`

FGMRES
------
The Flexible Generalized Minimum Residual method (FGMRES) is a projection method for solving sparse (non) symmetric linear systems :math:`Ax=b`. It is similar to the GMRES method with the only difference, the FGMRES is based on a window shifting of the Krylov subspace and thus allows the preconditioner :math:`M^{-1}` to be not a constant operator. This can be especially helpful if the operation :math:`M^{-1}x` is the result of another iterative process and not a constant operator.

The Krylov subspace basis size can be set using :cpp:func:`rocalution::FGMRES::SetBasisSize`. The default size is 30.

:cpp:class:`rocalution::FGMRES`

BiCGStab
--------
The Bi-Conjugate Gradient Stabilized method is a variation of CGS and solves sparse (non) symmetric linear systems :math:`Ax=b`.

:cpp:class:`rocalution::BiCGStab`

IDR
---
The Induced Dimension Reduction method is a Krylov subspace method for solving sparse (non) symmetric linear systems :math:`Ax=b`. IDR(s) generates residuals in a sequence of nested subspaces.

The dimension of the shadow space can be set by :cpp:func:`rocalution::IDR::SetShadowSpace`. The default size of the shadow space is 4.

:cpp:class:`rocalution::IDR`

FCG
---
The Flexible Conjugate Gradient method is an iterative method for solving sparse symmetric positive definite linear systems :math:`Ax=b`. It is similar to the Conjugate Gradient method with the only difference, that it allows the preconditioner :math:`M^{-1}` to be not a constant operator. This can be especially helpful if the operation :math:`M^{-1}x` is the result of another iterative process and not a constant operator.

:cpp:class:`rocalution::FCG`

QMRCGStab
---------
The Quasi-Minimal Residual Conjugate Gradient Stabilized method is a variant of the Krylov subspace BiCGStab method for solving sparse (non) symmetric linear systems :math:`Ax=b`.

:cpp:class:`rocalution::QMRCGStab`

BiCGStab(l)
-----------
The Bi-Conjugate Gradient Stabilized (l) method is a generalization of BiCGStab for solving sparse (non) symmetric linear systems :math:`Ax=b`. It minimizes residuals over :math:`l`-dimensional Krylov subspaces. The degree :math:`l` can be set with :cpp:func:`rocalution::BiCGStabl::SetOrder`.

:cpp:class:`rocalution::BiCGStabl`

Chebyshev iteration scheme
==========================

The Chebyshev Iteration scheme (also known as acceleration scheme) is similar to the CG method but requires minimum and maximum eigenvalues of the operator.

:cpp:class:`rocalution::Chebyshev`

Mixed-precision defect correction scheme
========================================

The Mixed-Precision solver is based on a defect-correction scheme. The current implementation of the library is using host based correction in double precision and accelerator computation in single precision. The solver is implementing the scheme

.. math::

   x_{k+1} = x_{k} + A^{-1} r_{k},

where the computation of the residual :math:`r_{k} = b - Ax_{k}` and the update :math:`x_{k+1} = x_{k} + d_{k}` are performed on the host in double precision. The computation of the residual system :math:`Ad_{k} = r_{k}` is performed on the accelerator in single precision. In addition to the setup functions of the iterative solver, the user need to specify the inner (:math:`Ad_{k} = r_{k}`) solver.

:cpp:class:`rocalution::MixedPrecisionDC`

MultiGrid solvers
=================

The library provides algebraic multigrid and a skeleton for geometric multigrid methods. The ``BaseMultigrid`` class itself doesn't construct data for the method. It contains the solution procedure for V, W and K-cycles. The AMG has two different versions for Local (non-MPI) and for Global (MPI) type of computations.

:cpp:class:`rocalution::BaseMultiGrid`

Geometric multiGrid
-------------------

The MultiGrid method can be used with external data, such as externally computed restriction, prolongation and operator hierarchy. The user need to pass all this information for each level and for its construction. This includes smoothing step, prolongation/restriction, grid traversing and coarse grid solver. This data need to be passed to the solver.

- Restriction and prolongation operations can be performed in two ways, based on Restriction() and Prolongation() of the LocalVector class, or by matrix-vector multiplication. This is configured by a set function.
- Smoothers can be of any iterative linear solver. Valid options are Jacobi, Gauss-Seidel, ILU, etc. using a FixedPoint iteration scheme with pre-defined number of iterations. The smoothers could also be a solver such as CG, BiCGStab, etc.
- Coarse grid solver could be of any iterative linear solver type. The class also provides mechanisms to specify, where the coarse grid solver has to be performed, on the host or on the accelerator. The coarse grid solver can be preconditioned.
- Grid scaling based on a :math:`L_2` norm ratio.
- Operator matrices need to be passed on each grid level.

:cpp:class:`rocalution::MultiGrid`

Algebraic multiGrid
-------------------

The Algebraic MultiGrid solver is based on the BaseMultiGrid class. The coarsening is obtained by different aggregation techniques. The smoothers can be constructed inside or outside of the class. All parameters in the Algebraic MultiGrid class can be set externally, including smoothers and coarse grid solver.

:cpp:class:`rocalution::BaseAMG`

Unsmoothed aggregation AMG
==========================

The Unsmoothed Aggregation Algebraic MultiGrid method is based on unsmoothed aggregation based interpolation scheme.

:cpp:class:`rocalution::UAAMG`

Smoothed aggregation AMG
========================

The Smoothed Aggregation Algebraic MultiGrid method is based on smoothed aggregation based interpolation scheme.

:cpp:class:`rocalution::SAAMG`

Ruge-stueben AMG
================

The Ruge-Stueben Algebraic MultiGrid method is based on the classic Ruge-Stueben coarsening with direct interpolation. The solver provides high-efficiency in terms of complexity of the solver (i.e. number of iterations). However, most of the time it has a higher building step and requires higher memory usage.

:cpp:class:`rocalution::RugeStuebenAMG`

Pairwise AMG
============

The Pairwise Aggregation Algebraic MultiGrid method is based on a pairwise aggregation matching scheme. It delivers very efficient building phase which is suitable for Poisson-like equation. Most of the time it requires K-cycle for the solving phase to provide low number of iterations. This version has multi-node support.

:cpp:class:`rocalution::PairwiseAMG`

Direct linear solvers
=====================

The library provides three direct methods - LU, QR and Inversion (based on QR decomposition). The user can pass a sparse matrix, internally it will be converted to dense and then the selected method will be applied. These methods are not very optimal and due to the fact that the matrix is converted to a dense format, these methods should be used only for very small matrices.

.. note::

   These methods can only be used with local-type problems.

LU
--
Lower-Upper Decomposition factors a given square matrix into lower and upper triangular matrix, such that :math:`A = LU`.

:cpp:class:`rocalution::LU`

QR
--
The QR Decomposition decomposes a given matrix into :math:`A = QR`, such that :math:`Q` is an orthogonal matrix and :math:`R` an upper triangular matrix.

:cpp:class:`rocalution::QR`

Inversion
---------
Full matrix inversion based on QR decomposition.

:cpp:class:`rocalution::Inversion`

:cpp:class:`rocalution::DirectLinearSolver`
