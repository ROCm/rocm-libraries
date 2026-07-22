.. meta::
   :description: rocALUTION solvers
   :keywords: rocALUTION, ROCm, library, API, tool, solvers

.. _solver-class:

********************
rocALUTION solvers
********************

This document provides a category-wise listing of the solver APIs along with the information required to use them.

Member documentation for each solver class is on the :ref:`api` page. The sections below group solvers by category and describe how to use them in practice.

Code structure
==============

:cpp:class:`rocalution::Solver`

Iterative linear solvers
========================

:cpp:class:`rocalution::IterativeLinearSolver`

Building and solving phase
==========================
Each iterative solver consists of a building step and a solving step. During the building step all necessary auxiliary data is allocated and the preconditioner is constructed. You can now call the solving procedure, which can be called several times.

When the initial matrix associated with the solver is on the accelerator, the solver tries to build everything on the accelerator. However, some preconditioners and solvers (such as FSAI and AMG) must be constructed on the host before being transferred to the accelerator. If the initial matrix is on the host and you want to run the solver on the accelerator, then you need to move the solver to the accelerator, matrix, right-hand side, and solution vector.

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

See :cpp:func:`rocalution::Solver::Clear` and the :cpp:class:`rocalution::Solver` class documentation on the :ref:`api` page.

Numerical update
================

Some preconditioners require two phases in the their construction: an algebraic (e.g. compute a pattern or structure) and a numerical (compute the actual values) phase. In cases, where the structure of the input matrix is a constant (e.g. Newton-like methods), it is not necessary to fully reconstruct the preconditioner. In this case, the user can apply a numerical update to the current preconditioner and pass the new operator with :cpp:func:`rocalution::Solver::ReBuildNumeric`. If the preconditioner/solver does not support the numerical update, then a full :cpp:func:`rocalution::Solver::Clear` and :cpp:func:`rocalution::Solver::Build` is performed.

Fixed-Point iteration
=====================

:cpp:class:`rocalution::FixedPoint`

Krylov subspace solvers
=======================

:cpp:class:`rocalution::CG`

:cpp:class:`rocalution::CR`

:cpp:class:`rocalution::GMRES`

:cpp:class:`rocalution::FGMRES`

:cpp:class:`rocalution::BiCGStab`

:cpp:class:`rocalution::IDR`

:cpp:class:`rocalution::FCG`

:cpp:class:`rocalution::QMRCGStab`

:cpp:class:`rocalution::BiCGStabl`

Chebyshev iteration scheme
==========================

:cpp:class:`rocalution::Chebyshev`

Mixed-precision defect correction scheme
========================================

:cpp:class:`rocalution::MixedPrecisionDC`

MultiGrid solvers
=================

The library provides algebraic multigrid and a skeleton for geometric multigrid methods. The ``BaseMultigrid`` class itself doesn't construct data for the method. It contains the solution procedure for V, W and K-cycles. The AMG has two different versions for Local (non-MPI) and for Global (MPI) type of computations.

:cpp:class:`rocalution::BaseMultiGrid`

Geometric multiGrid
-------------------

:cpp:class:`rocalution::MultiGrid`

Algebraic multiGrid
-------------------

:cpp:class:`rocalution::BaseAMG`

Unsmoothed aggregation AMG
==========================

:cpp:class:`rocalution::UAAMG`

Smoothed aggregation AMG
========================

:cpp:class:`rocalution::SAAMG`

Ruge-stueben AMG
================

:cpp:class:`rocalution::RugeStuebenAMG`

Pairwise AMG
============

:cpp:class:`rocalution::PairwiseAMG`

Direct linear solvers
=====================

:cpp:class:`rocalution::DirectLinearSolver`, :cpp:class:`rocalution::LU`, :cpp:class:`rocalution::QR`, :cpp:class:`rocalution::Inversion`
