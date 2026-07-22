.. meta::
   :description: rocALUTION preconditioners
   :keywords: rocALUTION, ROCm, library, API, preconditioners

.. _preconditioners:

***************************
rocALUTION Preconditioners
***************************

This document provides a category-wise listing of the preconditioners. All preconditioners support local operators. They can be used as a global preconditioner via block-jacobi scheme, which works locally on each interior matrix. To provide fast application, all preconditioners require extra memory to keep the approximated operator.

Member documentation, configuration routines, and usage notes for each class are on the :ref:`api` page.

Code structure
==============

The preconditioners provide a solution to the system :math:`Mz = r`, where the solution :math:`z` is either directly computed by the approximation scheme or iteratively obtained with :math:`z = 0` initial guess.

:cpp:class:`rocalution::Preconditioner`

Jacobi method
=============

:cpp:class:`rocalution::Jacobi`

(Symmetric) Gauss-Seidel or (S)SOR method
==========================================

:cpp:class:`rocalution::GS`, :cpp:class:`rocalution::SGS`

Incomplete factorizations
=========================

ILU
---

:cpp:class:`rocalution::ILU`

ILUT
----

:cpp:class:`rocalution::ILUT`

IC
---

:cpp:class:`rocalution::IC`

AI Chebyshev
============

:cpp:class:`rocalution::AIChebyshev`

FSAI
====

:cpp:class:`rocalution::FSAI`

SPAI
====

:cpp:class:`rocalution::SPAI`

TNS
===

:cpp:class:`rocalution::TNS`

MultiColored preconditioners
============================

:cpp:class:`rocalution::MultiColored`

MultiColored (symmetric) Gauss-Seidel / (S)SOR
----------------------------------------------

:cpp:class:`rocalution::MultiColoredGS`, :cpp:class:`rocalution::MultiColoredSGS`

MultiColored power(q)-pattern method ILU(p,q)
---------------------------------------------

:cpp:class:`rocalution::MultiColoredILU`

Multi-elimination incomplete LU
===============================

:cpp:class:`rocalution::MultiElimination`

Diagonal preconditioner for saddle-point problems
=================================================

:cpp:class:`rocalution::DiagJacobiSaddlePointPrecond`

(Restricted) Additive Schwarz preconditioner
============================================

:cpp:class:`rocalution::AS`, :cpp:class:`rocalution::RAS`

Block-Jacobi (MPI) preconditioner
=================================

:cpp:class:`rocalution::BlockJacobi`

Block preconditioner
====================

:cpp:class:`rocalution::BlockPreconditioner`

Variable preconditioner
=======================

:cpp:class:`rocalution::VariablePreconditioner`
