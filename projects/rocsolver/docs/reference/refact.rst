.. meta::
  :description: rocSOLVER refactorization and direct solvers API documentation
  :keywords: rocSOLVER, ROCm, API, documentation, refactorization, direct solvers

.. _refactor:

**********************************************
rocSOLVER refactorization and direct solvers
**********************************************

These functions implement direct solvers for sparse systems with
different coefficient matrices that share the same sparsity pattern.
The refactorization functions are divided into the following categories:

* :ref:`rfinit`: Basic functions to initialize and destroy meta data.
* :ref:`rfrefact`: Refactorization of new matrices given a known sparsity pattern.
* :ref:`rfsolver`: Based on triangular refactorization.

.. note::
   
   The API descriptions use the following notations:

   *  ``i``, ``j``, and ``k`` are used as general purpose indices. In some legacy LAPACK APIs, ``k`` can be
      a parameter indicating some problem or matrix dimension.
   *  Depending on the context, when it is necessary to index rows, columns, and blocks or submatrices,
      ``i`` is assigned to rows, ``j`` to columns, and ``k`` to blocks. ``l`` is always used to index
      matrices or problems in a batch.
   *  ``x[i]`` stands for the i-th element of vector x, while ``A[i,j]`` represents the element
      in the i-th row and j-th column of matrix ``A``. Indices are 1-based, for instance, ``x[1]`` is the first
      element of ``x``.
   *  To identify a block in a matrix or a matrix in the batch, ``k`` and ``l`` are used as sub-indices
   *  ``x_i`` :math:`=x_i`. Both notations are used, :math:`x_i` when displaying mathematical
      equations and ``x_i`` in the text describing the function parameters.
   *  If ``X`` is a real vector or matrix, :math:`X^T` indicates its transpose. If ``X`` is complex, then
      :math:`X^H` represents its conjugate transpose. When ``X`` could be real or complex, the descriptions use ``X'`` to
      indicate ``X`` transposed or ``X`` conjugate transposed, accordingly.
   *  When a matrix ``A`` is formed as the product of several matrices, the following notation is used:
      ``A=M(1)M(2)...M(t)``.


.. _rfinit:

Initialization and meta data
==================================

.. contents:: List of initialization functions
   :local:
   :backlinks: top


.. _rfinfocreate:

rocsolver_create_rfinfo()
---------------------------------------
.. doxygenfunction:: rocsolver_create_rfinfo


.. _rfinfodestroy:

rocsolver_destroy_rfinfo()
---------------------------------------
.. doxygenfunction:: rocsolver_destroy_rfinfo


.. _rfinfoset:

rocsolver_set_rfinfo_mode()
---------------------------------------
.. doxygenfunction:: rocsolver_set_rfinfo_mode


.. _rfinfoget:

rocsolver_get_rfinfo_mode()
---------------------------------------
.. doxygenfunction:: rocsolver_get_rfinfo_mode


.. _rfanalysis:

rocsolver_csrrf_analysis()
--------------------------------------
.. doxygenfunction:: rocsolver_dcsrrf_analysis
   :outline:
.. doxygenfunction:: rocsolver_scsrrf_analysis



.. _rfrefact:

Triangular refactorization
==================================

.. contents:: List of refactorization functions
   :local:
   :backlinks: top

.. _rfsumlu:

rocsolver_<type>csrrf_sumlu()
----------------------------------
.. doxygenfunction:: rocsolver_dcsrrf_sumlu
   :outline:
.. doxygenfunction:: rocsolver_scsrrf_sumlu


.. _rfsplitlu:

rocsolver_<type>csrrf_splitlu()
------------------------------------
.. doxygenfunction:: rocsolver_dcsrrf_splitlu
   :outline:
.. doxygenfunction:: rocsolver_scsrrf_splitlu


.. _rfrefactlu:

rocsolver_<type>csrrf_refactlu()
------------------------------------
.. doxygenfunction:: rocsolver_dcsrrf_refactlu
   :outline:
.. doxygenfunction:: rocsolver_scsrrf_refactlu


.. _rfrefactchol:

rocsolver_<type>csrrf_refactchol()
------------------------------------
.. doxygenfunction:: rocsolver_dcsrrf_refactchol
   :outline:
.. doxygenfunction:: rocsolver_scsrrf_refactchol



.. _rfsolver:

Direct sparse solvers
==================================

.. contents:: List of direct solvers
   :local:
   :backlinks: top

.. _rfsolve:

rocsolver_<type>csrrf_solve()
----------------------------------
.. doxygenfunction:: rocsolver_dcsrrf_solve
   :outline:
.. doxygenfunction:: rocsolver_scsrrf_solve
