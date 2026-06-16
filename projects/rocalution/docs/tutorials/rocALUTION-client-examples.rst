.. meta::
   :description: rocALUTION examples
   :keywords: rocALUTION, ROCm, library, API, tool, examples

.. _clients:

****************************
rocALUTION client examples
****************************

rocALUTION clients host a variety of different examples as well as a unit test package.
For detailed instructions on how to build rocALUTION with clients, see :ref:`linux-installation` or :ref:`windows-installation`.

Examples
========
The examples collection offers different possible set-ups of solvers and preconditioners.
The following tables provide a quick overview of various examples:

===================== ====
Example               Description
===================== ====
``amg``               Algebraic Multigrid solver (smoothed aggregation scheme, GS smoothing)
``as-precond``        GMRES solver with Additive Schwarz preconditioning
``async``             Asynchronous rocALUTION object transfer
``benchmark``         Benchmarking important sparse functions
``bicgstab``          BiCGStab solver with multicolored Gauss-Seidel preconditioning
``block-precond``     GMRES solver with blockwise multicolored ILU preconditioning
``cg-amg``            CG solver with Algebraic Multigrid (smoothed aggregation scheme) preconditioning
``cg``                CG solver with Jacobi preconditioning
``cmk``               CG solver with ILU preconditioning using Cuthill McKee ordering
``direct``            Matrix inversion
``fgmres``            Flexible GMRES solver with multicolored Gauss-Seidel preconditioning
``fixed-point``       Fixed-Point iteration scheme using Jacobi relaxation
``gmres``             GMRES solver with multicolored Gauss-Seidel preconditioning
``idr``               Induced Dimension Reduction solver with Jacobi preconditioning
``key``               Sparse matrix unique key computation
``me-preconditioner`` CG solver with multi-elimination preconditioning
``mixed-precision``   Mixed-precision CG solver with multicolored ILU preconditioning
``power-method``      CG solver using Chebyshev preconditioning and power method for eigenvalue approximation
``simple-spmv``       Sparse Matrix Vector multiplication
``sp-precond``        BiCGStab solver with multicolored ILU preconditioning for saddle point problems
``stencil``           CG solver using stencil as operator
``tns``               CG solver with Truncated Neumann Series preconditioning
``var-precond``       FGMRES solver with variable preconditioning
===================== ====

================= ====
Example (MPI)     Description
================= ====
``benchmark_mpi`` Benchmarking important sparse functions
``bicgstab_mpi``  BiCGStab solver with multicolored Gauss-Seidel preconditioning
``cg-amg_mpi``    CG solver with Algebraic Multigrid (pairwise aggregation scheme) preconditioning
``cg_mpi``        CG solver with Jacobi preconditioning
``fcg_mpi``       Flexible CG solver with ILU preconditioning
``fgmres_mpi``    Flexible GMRES solver with SParse Approximate Inverse preconditioning
``global-io_mpi`` File I/O with CG solver and Factorized Sparse Approximate Inverse preconditioning
``idr_mpi``       IDR solver with Factorized Sparse Approximate Inverse preconditioning
``qmrcgstab_mpi`` QMRCGStab solver with ILU-T preconditioning
================= ====

Unit Tests
==========
There are multiple unit tests available to test for bad arguments, invalid parameters, and solver and preconditioner functionality.
These unit tests are based on google test.
The tests cover a variety of solver, preconditioning, and matrix format combinations and can be performed on all available backends.