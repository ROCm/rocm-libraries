.. meta::
   :description: Extending rocALUTION functionality
   :keywords: rocALUTION, ROCm, library, API, tool, rocALUTION functionality

.. _functionality-extension:

***********************************************
Extending rocALUTION functionality
***********************************************

The rocALUTION library is open source and can be extended with user-specific routines, solvers, and preconditioners.

You can choose to extend rocALUTION 
In the following sections, different levels of functionality enhancements are illustrated.
These examples can be used as guidelines to extend rocALUTION step by step with your own routines.
Please note, that user-added routines can also be added to the main GitHub repository using pull requests.

``LocalMatrix`` functionality extension
========================================

This section demonstrates how to extend the :cpp:class:`LocalMatrix <rocalution::LocalMatrix>` class with an additional routine.
The routine supports both Host and Accelerator backend.
Furthermore, the routine requires the matrix to be in CSR format. 
Here are the steps to extend the :cpp:class:`LocalMatrix <rocalution::LocalMatrix>` functionality:

1.  API enhancement
--------------------

To make the new routine available through the API, modify the :cpp:class:`LocalMatrix <rocalution::LocalMatrix>` class.
The corresponding header file ``local_matrix.hpp`` is located in ``src/base/``.
The new routines can be added as public member functions as shown below:

.. code-block:: cpp

  ...
  void ConvertTo(unsigned int matrix_format, int blockdim);

  void MyNewFunctionality(void);

  virtual void Apply(const LocalVector<ValueType>& in, LocalVector<ValueType>* out) const;
  virtual void ApplyAdd(const LocalVector<ValueType>& in,
  ...

For the implementation of the new API function, it is important to know the location of the availability of this functionality.
To add support for any backend and matrix format, format conversions are required if ``MyNewFunctionality()`` is only supported for CSR matrices.
This is subject to the API function implementation:

.. code-block:: cpp

  template <typename ValueType>
  void LocalMatrix<ValueType>::MyNewFunctionality(void)
  {
      // Debug logging
      log_debug(this, "LocalMatrix::MyNewFunctionality()");

  #ifdef DEBUG_MODE
      // If we are in debug mode, perform an additional matrix sanity check
      this->Check();
  #endif

      // If no non-zero entries, do nothing
      if(this->GetNnz() > 0)
      {
          // As we want to implement our function only for CSR, we first need to convert
          // the matrix to CSR format
          unsigned int format = this->GetFormat();
          int blockdim = this->GetBlockDimension();
          this->ConvertToCSR();

          // Call the corresponding base matrix implementation
          bool err = this->matrix_->MyNewFunctionality();

          // Check its return type
          if((err == false) && (this->is_host_() == true))
          {
              // If our matrix is on the host, the function call failed.
              LOG_INFO("Computation of LocalMatrix::MyNewFunctionality() failed");
              this->Info();
              FATAL_ERROR(__FILE__, __LINE__);
          }

          // Run backup algorithm on host, in case the accelerator version failed
          if(err == false)
          {
              // Move matrix to host
              bool is_accel = this->is_accel_();
              this->MoveToHost();

              // Try again
              if(this->matrix_->MyNewFunctionality() == false)
              {
                  LOG_INFO("Computation of LocalMatrix::MyNewFunctionality() failed");
                  this->Info();
                  FATAL_ERROR(__FILE__, __LINE__);
              }

              // On a successful host call, move the data back to the accelerator
              // if initial data was on the accelerator
              if(is_accel == true)
              {
                  // Print a warning, that the algorithm was performed on the host
                  // even though the initial data was on the device
                  LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::MyNewFunctionality() was performed on the host");

                  this->MoveToAccelerator();
              }
          }

          // Convert the matrix back to CSR format
          if(format != CSR)
          {
              // Print a warning, that the algorithm was performed in CSR format
              // even though the initial matrix format was different
              LOG_VERBOSE_INFO(2, "*** warning: LocalMatrix::MyNewFunctionality() was performed in CSR format");

              this->ConvertTo(format, blockdim);
          }
      }

  #ifdef DEBUG_MODE
      // Perform additional sanity check in debug mode, because this is a non-const function
      this->Check();
  #endif
  }

Similarly, you can implement host-only functions.
In this case, initial data explicitly needs to be moved to the host backend using the API implementation.

The next step is to implement the actual functionality in the :cpp:class:`BaseMatrix <rocalution::BaseMatrix>` class.

2.  Enhancement of the ``BaseMatrix`` class
---------------------------------------------

To make the new routine available in the base class, first modify the :cpp:class:`BaseMatrix <rocalution::BaseMatrix>` class.
The corresponding header file ``base_matrix.hpp`` is located in ``src/base/``.
The new routines can be added as public member functions, e.g.

.. code-block:: cpp

  ...
  virtual bool ILU0Factorize(void);

  /// Perform MyNewFunctionality algorithm
  virtual bool MyNewFunctionality(void);

  /// Perform LU factorization
  ...

We don't implement the purely virtual ``MyNewFunctionality()`` as we don't supply an implementation for all base classes.
We decided to implement it only for CSR format and hence need to return an error flag, so that the :cpp:class:`LocalMatrix <rocalution::LocalMatrix>` class is aware of the failure and can convert it to CSR.

.. code-block:: cpp

  template <typename ValueType>
  bool MyNewFunctionality(void)
  {
      return false;
  }

3.  Platform-specific host implementation
-------------------------------------------

To satisfy the rocALUTION host backup philosophy, there must be a host implementation available.
Hence, for the new function to succeed, there must be backend implementation available.
Place the host implementation in ``src/base/host/host_matrix_csr.cpp`` as we decided to make it available for CSR format.

.. code-block:: cpp

  ...
  virtual bool ILUTFactorize(double t, int maxrow);

  virtual bool MyNewFunctionality(void);

  virtual void LUAnalyse(void);
  ...

.. code-block:: cpp

  template <typename ValueType>
  bool HostMatrixCSR<ValueType>::MyNewFunctionality(void)
  {
      // Place some asserts to verify sanity of input data

      // Our algorithm works only for squared matrices
      assert(this->nrow_ == this->ncol_);
      assert(this->nnz_ > 0);

      // place the actual host based algorithm here:
      // for illustration, we scale the matrix by its inverse diagonal
      for(int i = 0; i < this->nrow_; ++i)
      {
          int row_begin = this->mat_.row_offset[i];
          int row_end   = this->mat_.row_offset[i + 1];

          bool diag_found = false;
          ValueType inv_diag;

          // Find the diagonal entry
          for(int j = row_begin; j < row_end; ++j)
          {
              if(this->mat_.col[j] == i)
              {
                  diag_found = true;
                  inv_diag = static_cast<ValueType>(1) / this->mat_.val[j];
              }
          }

          // Our algorithm works only with full rank
          assert(diag_found == true);

          // Scale the row
          for(int j = row_begin; j < row_end; ++j)
          {
              this->mat_.val[j] *= inv_diag;
          }
      }

      return true;
  }

4.  Platform-specific HIP implementation
------------------------------------------

You can now add an additional implementation for the HIP backend using HIP programming framework.
This is required to make your algorithm available on accelerators so that rocALUTION doesn't need to switch to the host backend on function calls anymore.
Add the HIP implementation ``src/base/hip/hip_matrix_csr.cpp`` in this case.

.. code-block:: cpp

  ...
  virtual bool ILU0Factorize(void);

  virtual bool MyNewFunctionality(void);

  virtual bool ICFactorize(BaseVector<ValueType>* inv_diag = NULL);
  ...

.. code-block:: cpp

  template <typename ValueType>
  bool HIPAcceleratorMatrixCSR<ValueType>::MyNewFunctionality(void)
  {
      // Place some asserts to verify sanity of input data

      // Our algorithm works only for squared matrices
      assert(this->nrow_ == this->ncol_);
      assert(this->nnz_ > 0);

      // Enqueue the HIP kernel
      hipLaunchKernelGGL((kernel_csr_mynewfunctionality),
                         dim3((this->nrow_ - 1) / this->local_backend_.HIP_block_size + 1),
                         dim3(this->local_backend_.HIP_block_size),
                         0,
                         0,
                         this->mat_.row_offset,
                         this->mat_.col,
                         this->mat_.val);

      // Check for HIP execution error before successfully returning
      CHECK_HIP_ERROR(__FILE__, __LINE__);

      return true;
  }

Place the corresponding HIP kernel in ``src/base/hip/hip_kernels_csr.hpp``.

Adding a solver
===============

This section demonstrates how to add a new solver to rocALUTION. Here are the steps:

1.  Define the API for the new solver

As an example, we add a new :cpp:class:`IterativeLinearSolver <rocalution::IterativeLinearSolver>`.
To achieve this, we use :cpp:class:`CG <rocalution::CG>` as a template.
Thus, we first copy ``src/solvers/krylov/cg.hpp`` to ``src/solvers/krylov/mysolver.hpp`` and ``src/solvers/krylov.cg.cpp`` to ``src/solvers/krylov/mysolver.cpp`` (assuming we add a krylov subspace solvers).

2.  Modify the `cg.hpp` and `cg.cpp` as per your requirement (e.g. change the solver name from `CG` to `MySolver`)

Implement each of the following virtual functions present in the class. Follow the implementation details given below:

- ``MySolver()``: The constructor of the new solver class.
- ``~MySolver()``: The destructor of the new solver class. It calls the ``Clear()`` function.
- ``void Print(void) const``: Prints some informations about the solver.
- ``void Build(void)``: Creates all required structures of the solver, e.g. allocates memory and sets the backend of temporary objects.
- ``void BuildMoveToAcceleratorAsync(void)``: Moves all solver-related objects asynchronously to the accelerator device.
- ``void Sync(void)``: Synchronizes all solver related objects.
- ``void ReBuildNumeric(void)``: Rebuilds the solver only numerically.
- ``void Clear(void)``: Cleans up all solver-relevant structures that have been created using ``Build()``.
- ``void SolveNonPrecond_(const VectorType& rhs, VectorType* x)``: Performs the solving phase ``Ax=y`` without the use of a preconditioner.
- ``void SolvePrecond_(const VectorType& rhs, VectorType* x)``: Performs the solving phase ``Ax=y`` with the use of a preconditioner.
- ``void PrintStart_(void) const``: Protected function. Called when the solver starts.
- ``void PrintEnd_(void) const``: Protected function. Called when the solver ends.
- ``void MoveToHostLocalData_(void)``: Protected function. Moves all local solver objects to the host.
- ``void MoveToAcceleratorLocalData_(void)``: Protected function. Moves all local solver objects to the accelerator.

You can also introduce any additional solver-specific member functions.

3.  Make the new solver visible

To make the new solver visible, add it to the ``src/rocalution.hpp`` header:

.. code-block:: cpp

  ...
  #include "solvers/krylov/cg.hpp"
  #include "solvers/krylov/mysolver.hpp"
  #include "solvers/krylov/cr.hpp"
  ...

4.  Add the new solver to the CMake compilation list

The CMake compilation list is found in ``src/solvers/CMakeLists.txt``:

.. code-block:: cpp

  ...
  set(SOLVERS_SOURCES
    solvers/krylov/cg.cpp
    solvers/krylov/mysolver.cpp
    solvers/krylov/fcg.cpp
  ...
