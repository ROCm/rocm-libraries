.. meta::
   :description: rocALUTION solvers
   :keywords: rocALUTION, ROCm, library, API, tool, solvers

.. _solver-class:

********************
rocALUTION solvers
********************

This document provides a category-wise listing of the solver APIs along with the information required to use them.

Code structure
==============

.. doxygenclass:: rocalution::Solver

It provides an interface for:

.. doxygenfunction:: rocalution::Solver::SetOperator
.. doxygenfunction:: rocalution::Solver::Build
.. doxygenfunction:: rocalution::Solver::Clear
.. doxygenfunction:: rocalution::Solver::Solve
.. doxygenfunction:: rocalution::Solver::Print
.. doxygenfunction:: rocalution::Solver::ReBuildNumeric
.. doxygenfunction:: rocalution::Solver::MoveToHost
.. doxygenfunction:: rocalution::Solver::MoveToAccelerator

Iterative linear solvers
========================

.. doxygenclass:: rocalution::IterativeLinearSolver

It provides an interface for:

.. doxygenfunction:: rocalution::IterativeLinearSolver::Init(double, double, double, int)
.. doxygenfunction:: rocalution::IterativeLinearSolver::Init(double, double, double, int, int)
.. doxygenfunction:: rocalution::IterativeLinearSolver::InitMinIter
.. doxygenfunction:: rocalution::IterativeLinearSolver::InitMaxIter
.. doxygenfunction:: rocalution::IterativeLinearSolver::InitTol
.. doxygenfunction:: rocalution::IterativeLinearSolver::RecordResidualHistory
.. doxygenfunction:: rocalution::IterativeLinearSolver::RecordHistory
.. doxygenfunction:: rocalution::IterativeLinearSolver::Verbose
.. doxygenfunction:: rocalution::IterativeLinearSolver::SetPreconditioner
.. doxygenfunction:: rocalution::IterativeLinearSolver::SetResidualNorm
.. doxygenfunction:: rocalution::IterativeLinearSolver::GetAmaxResidualIndex
.. doxygenfunction:: rocalution::IterativeLinearSolver::GetSolverStatus

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

.. doxygenclass:: rocalution::FixedPoint
.. doxygenfunction:: rocalution::FixedPoint::SetRelaxation

Krylov subspace solvers
=======================

CG
--
.. doxygenclass:: rocalution::CG

CR
--
.. doxygenclass:: rocalution::CR

GMRES
-----
.. doxygenclass:: rocalution::GMRES
.. doxygenfunction:: rocalution::GMRES::SetBasisSize

FGMRES
------
.. doxygenclass:: rocalution::FGMRES
.. doxygenfunction:: rocalution::FGMRES::SetBasisSize

BiCGStab
--------
.. doxygenclass:: rocalution::BiCGStab

IDR
---
.. doxygenclass:: rocalution::IDR
.. doxygenfunction:: rocalution::IDR::SetShadowSpace

FCG
---
.. doxygenclass:: rocalution::FCG

QMRCGStab
---------
.. doxygenclass:: rocalution::QMRCGStab

BiCGStab(l)
-----------
.. doxygenclass:: rocalution::BiCGStabl
.. doxygenfunction:: rocalution::BiCGStabl::SetOrder

Chebyshev iteration scheme
==========================

.. doxygenclass:: rocalution::Chebyshev

Mixed-precision defect correction scheme
========================================

.. doxygenclass:: rocalution::MixedPrecisionDC

MultiGrid solvers
=================

The library provides algebraic multigrid and a skeleton for geometric multigrid methods. The ``BaseMultigrid`` class itself doesn't construct data for the method. It contains the solution procedure for V, W and K-cycles. The AMG has two different versions for Local (non-MPI) and for Global (MPI) type of computations.

.. doxygenclass:: rocalution::BaseMultiGrid

Geometric multiGrid
-------------------

.. doxygenclass:: rocalution::MultiGrid

Algebraic multiGrid
-------------------

.. doxygenclass:: rocalution::BaseAMG
.. doxygenfunction:: rocalution::BaseAMG::BuildHierarchy
.. doxygenfunction:: rocalution::BaseAMG::BuildSmoothers
.. doxygenfunction:: rocalution::BaseAMG::SetCoarsestLevel
.. doxygenfunction:: rocalution::BaseAMG::SetManualSmoothers
.. doxygenfunction:: rocalution::BaseAMG::SetManualSolver
.. doxygenfunction:: rocalution::BaseAMG::SetDefaultSmootherFormat
.. doxygenfunction:: rocalution::BaseAMG::SetOperatorFormat
.. doxygenfunction:: rocalution::BaseAMG::GetNumLevels

Unsmoothed aggregation AMG
==========================

.. doxygenclass:: rocalution::UAAMG
.. doxygenfunction:: rocalution::UAAMG::SetCouplingStrength
.. doxygenfunction:: rocalution::UAAMG::SetOverInterp

Smoothed aggregation AMG
========================

.. doxygenclass:: rocalution::SAAMG
.. doxygenfunction:: rocalution::SAAMG::SetCouplingStrength
.. doxygenfunction:: rocalution::SAAMG::SetInterpRelax

Ruge-stueben AMG
================

.. doxygenclass:: rocalution::RugeStuebenAMG
.. doxygenfunction:: rocalution::RugeStuebenAMG::SetStrengthThreshold

Pairwise AMG
============

.. doxygenclass:: rocalution::PairwiseAMG
.. doxygenfunction:: rocalution::PairwiseAMG::SetBeta
.. doxygenfunction:: rocalution::PairwiseAMG::SetOrdering
.. doxygenfunction:: rocalution::PairwiseAMG::SetCoarseningFactor

Direct linear solvers
=====================
.. doxygenclass:: rocalution::DirectLinearSolver
.. doxygenclass:: rocalution::LU
.. doxygenclass:: rocalution::QR
.. doxygenclass:: rocalution::Inversion

.. note:: 

  These methods can only be used with local-type problems.
