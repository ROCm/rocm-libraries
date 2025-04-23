.. meta::
   :description: rocALUTION multi-node computation
   :keywords: rocALUTION, ROCm, library, API, tool, API library.

.. _api:

************************
rocALUTION API library
************************

This document provides the detailed API list.

Host Utility Functions
======================
.. doxygenfunction:: rocalution::allocate_host
.. doxygenfunction:: rocalution::free_host
.. doxygenfunction:: rocalution::set_to_zero_host
.. doxygenfunction:: rocalution::rocalution_time

Backend Manager
===============
.. doxygenfunction:: rocalution::init_rocalution
.. doxygenfunction:: rocalution::stop_rocalution
.. doxygenfunction:: rocalution::set_device_rocalution
.. doxygenfunction:: rocalution::set_omp_threads_rocalution
.. doxygenfunction:: rocalution::set_omp_affinity_rocalution
.. doxygenfunction:: rocalution::set_omp_threshold_rocalution
.. doxygenfunction:: rocalution::info_rocalution(void)
.. doxygenfunction:: rocalution::info_rocalution(const struct Rocalution_Backend_Descriptor& backend_descriptor)
.. doxygenfunction:: rocalution::disable_accelerator_rocalution
.. doxygenfunction:: rocalution::_rocalution_sync

Base Rocalution
===============
.. doxygenclass:: rocalution::BaseRocalution
   :members:

Operator
========
.. doxygenclass:: rocalution::Operator
   :members:

Vector
======
.. doxygenclass:: rocalution::Vector
   :members:

Local Matrix
============
.. doxygenclass:: rocalution::LocalMatrix
   :members:

Local Stencil
=============
.. doxygenclass:: rocalution::LocalStencil
   :members:

Global Matrix
=============
.. doxygenclass:: rocalution::GlobalMatrix
   :members:

Local Vector
============
.. doxygenclass:: rocalution::LocalVector
   :members:

Global Vector
=============
.. doxygenclass:: rocalution::GlobalVector
   :members:

Base Classes
============
.. doxygenclass:: rocalution::BaseMatrix
   :members:

.. doxygenclass:: rocalution::BaseStencil
   :members:

.. doxygenclass:: rocalution::BaseVector
   :members:

.. doxygenclass:: rocalution::HostMatrix
   :members:

.. doxygenclass:: rocalution::HostStencil
   :members:

.. doxygenclass:: rocalution::HostVector
   :members:

.. doxygenclass:: rocalution::AcceleratorMatrix
   :members:

.. doxygenclass:: rocalution::AcceleratorStencil
   :members:

.. doxygenclass:: rocalution::AcceleratorVector
   :members:


Parallel Manager
================
.. doxygenclass:: rocalution::ParallelManager
   :members:

Solvers
=======
.. doxygenclass:: rocalution::Solver
   :members:

Iterative Linear Solvers
------------------------
.. doxygenclass:: rocalution::IterativeLinearSolver
   :members:

.. doxygenclass:: rocalution::FixedPoint
   :members:

.. doxygenclass:: rocalution::MixedPrecisionDC
   :members:

.. doxygenclass:: rocalution::Chebyshev
   :members:

Krylov Subspace Solvers
```````````````````````
.. doxygenclass:: rocalution::BiCGStab
   :members:

.. doxygenclass:: rocalution::BiCGStabl
   :members:

.. doxygenclass:: rocalution::CG
   :members:

.. doxygenclass:: rocalution::CR
   :members:

.. doxygenclass:: rocalution::FCG
   :members:

.. doxygenclass:: rocalution::GMRES
   :members:

.. doxygenclass:: rocalution::FGMRES
   :members:

.. doxygenclass:: rocalution::IDR
   :members:

.. doxygenclass:: rocalution::QMRCGStab
   :members:

MultiGrid Solvers
`````````````````
.. doxygenclass:: rocalution::BaseMultiGrid
   :members:

.. doxygenclass:: rocalution::MultiGrid
   :members:

.. doxygenclass:: rocalution::BaseAMG
   :members:

.. doxygenclass:: rocalution::UAAMG
   :members:

.. doxygenclass:: rocalution::SAAMG
   :members:

.. doxygenclass:: rocalution::RugeStuebenAMG
   :members:

.. doxygenclass:: rocalution::PairwiseAMG
   :members:

Direct Solvers
--------------
.. doxygenclass:: rocalution::DirectLinearSolver
   :members:

.. doxygenclass:: rocalution::Inversion
   :members:

.. doxygenclass:: rocalution::LU
   :members:

.. doxygenclass:: rocalution::QR
   :members:


Preconditioners
===============
.. doxygenclass:: rocalution::Preconditioner
   :members:

.. doxygenclass:: rocalution::AIChebyshev
   :members:

.. doxygenclass:: rocalution::FSAI
   :members:

.. doxygenclass:: rocalution::SPAI
   :members:

.. doxygenclass:: rocalution::TNS
   :members:

.. doxygenclass:: rocalution::AS
   :members:

.. doxygenclass:: rocalution::RAS
   :members:

.. doxygenclass:: rocalution::BlockJacobi
   :members:

.. doxygenclass:: rocalution::BlockPreconditioner
   :members:

.. doxygenclass:: rocalution::Jacobi
   :members:

.. doxygenclass:: rocalution::GS
   :members:

.. doxygenclass:: rocalution::SGS
   :members:

.. doxygenclass:: rocalution::ILU
   :members:

.. doxygenclass:: rocalution::ILUT
   :members:

.. doxygenclass:: rocalution::IC
   :members:

.. doxygenclass:: rocalution::VariablePreconditioner
   :members:

.. doxygenclass:: rocalution::MultiColored
   :members:

.. doxygenclass:: rocalution::MultiColoredSGS
   :members:

.. doxygenclass:: rocalution::MultiColoredGS
   :members:

.. doxygenclass:: rocalution::MultiColoredILU
   :members:

.. doxygenclass:: rocalution::MultiElimination
   :members:

.. doxygenclass:: rocalution::DiagJacobiSaddlePointPrecond
   :members:
