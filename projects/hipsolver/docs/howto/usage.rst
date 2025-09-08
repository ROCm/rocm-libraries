.. meta::
  :description: How to use the hipSOLVER library and API
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation, usage

.. _usage_label:

*****************
Using hipSOLVER
*****************

hipSOLVER is an open-source marshalling library for `LAPACK routines <https://www.netlib.org/lapack/index.html>`_ on the GPU.
It sits between a backend library and the user application, marshalling inputs to and outputs from the backend library so that the user
application doesn't have to change when using different backends. hipSOLVER supports two backend libraries: The NVIDIA CUDA `cuSOLVER
library <https://developer.nvidia.com/cusolver>`_ and the open-source AMD ROCm :doc:`rocSOLVER library <rocsolver:index>`.

The :ref:`regular hipSOLVER API <library_api>` is a thin wrapper layer around the different backends that does not typically introduce
significant overhead. However, its main purpose is portability, so when performance is critical, it's recommended to
use the library backend directly.

After it is installed, hipSOLVER can be used like any other library with a C API. The header file has to be included
in the user code, which means the shared library becomes a link-time and run-time dependency for the user application. The
user application code can be ported with no changes to any system with hipSOLVER installed, regardless of the backend library.

For more details on how to use the API methods, see the client code samples on the
`hipSOLVER GitHub <https://github.com/ROCm/hipSOLVER/tree/develop/clients/samples>`_ or
the documentation for the corresponding backend libraries.

.. _porting:

Porting cuSOLVER applications to hipSOLVER
============================================

The hipSOLVER design facilitates the translation of cuSOLVER applications to the AMD open-source
:doc:`ROCm <rocm:index>` platform and ecosystem.
This makes it easy for you to port your existing cuSOLVER applications to hipSOLVER. hipSOLVER provides two
separate but interchangeable API patterns to facilitate a two-stage transition process. Users are encouraged to start with
the hipSOLVER compatibility APIs, which use the :ref:`hipsolverDn <library_dense>`, :ref:`hipsolverSp <library_sparse>`, and
:ref:`hipsolverRf <library_refactor>` prefixes and have method signatures that are fully consistent with cuSOLVER functions.

However, the compatibility APIs might introduce some performance drawbacks, especially when using the rocSOLVER backend. So, as a second
stage, it's best to switch to using the hipSOLVER :ref:`regular API <library_api>` when possible. The regular API uses the ``hipsolver`` prefix.
It makes minor adjustments to the API to get the best performance out of the rocSOLVER backend. In most cases, switching to
the regular API is as simple as removing ``Dn`` from the ``hipsolverDn`` prefix.

.. note::

   Methods with the ``hipsolverSp`` and ``hipsolverRf`` prefixes are not currently supported by the regular API.

No matter which API is used, a hipSOLVER application can be executed without modifications to the code on systems with cuSOLVER or
rocSOLVER installed. However, using the regular API ensures the best performance out of both backends.

.. _dense_api_differences:

Using the hipsolverDn API
====================================================

The hipsolverDn API is intended as a 1:1 translation of the cusolverDn API, but not all functionality is equally supported in
rocSOLVER. The following considerations apply when using this compatibility API.


Arguments not referenced by rocSOLVER
--------------------------------------

* Unlike cuSOLVER, rocSOLVER functions do not provide information on invalid arguments in the ``info`` parameter, although they
  do provide info on singularities and algorithm convergence. Therefore, when using the rocSOLVER backend, ``info`` always
  returns a value >= 0. In cases where a rocSOLVER function does not accept ``info`` as an argument, hipSOLVER
  sets it to zero.

* The ``niters`` argument for :ref:`hipsolverDnXXgels <dense_gels>` and :ref:`hipsolverDnXXgesv <dense_gesv>` is not referenced
  by the rocSOLVER backend. rocSOLVER does not implement an iterative refinement.

* The ``hRnrmF`` argument for :ref:`hipsolverDnXgesvdaStridedBatched <dense_gesvda_strided_batched>` is not referenced by the
  rocSOLVER backend.

.. _dense_performance:

Performance implications of the hipsolverDn API
------------------------------------------------

*  To calculate the workspace required by the ``gesvd`` function in rocSOLVER, the values of ``jobu`` and ``jobv`` are needed. However,
   the function :ref:`hipsolverDnXgesvd_bufferSize <dense_gesvd_bufferSize>` does not accept these arguments. When using
   the rocSOLVER backend, ``hipsolverDnXgesvd_bufferSize`` has to internally calculate the workspace for all possible values of ``jobu`` and ``jobv``
   and return the maximum.

   .. note::

      ``hipsolverDnXgesvd_bufferSize`` is slower than ``hipsolverXgesvd_bufferSize``, and the workspace size it returns might be slightly larger than
      what is actually required.

*  To properly use a user-provided workspace, rocSOLVER requires both the allocated pointer and its size. However, the function
   :ref:`hipsolverDnXgetrf <dense_getrf>` does not accept ``lwork`` as an argument. Consequently, when using the rocSOLVER backend,
   ``hipsolverDnXgetrf`` has to internally call ``hipsolverDnXgetrf_bufferSize`` to obtain the workspace size.

   .. note::

      In practice, ``hipsolverDnXgetrf_bufferSize`` is called twice, once by the user before allocating the workspace and once
      by hipSOLVER internally when executing the ``hipsolverDnXgetrf`` function. ``hipsolverDnXgetrf`` can be slightly slower than ``hipsolverXgetrf``
      because of the extra call to the ``bufferSize`` helper.

*  The functions :ref:`hipsolverDnXgetrs <dense_getrs>`, :ref:`hipsolverDnXpotrs <dense_potrs>`,
   :ref:`hipsolverDnXpotrsBatched <dense_potrs_batched>`, and
   :ref:`hipsolverDnXpotrfBatched <dense_potrf_batched>` do not accept ``work`` and ``lwork`` as arguments.
   However, this functionality does require a non-zero workspace
   in rocSOLVER. As a result, these functions switch to
   the automatic workspace management model when using the rocSOLVER backend. For more information, see the :ref:`memory model information <mem_model>`.

   .. note::

      Even though the compatibility API does not provide ``bufferSize`` helpers for these functions, the functions still require
      a workspace to use rocSOLVER. This workspace is automatically managed, but it might result in device memory reallocations with a corresponding overhead.

.. _sparse_api_differences:

Using the hipsolverSp API
====================================================

The hipsolverSp API is intended as a 1:1 translation of the cusolverSp API, but not all functionality is equally supported in
rocSOLVER. The following considerations apply when using this compatibility API.

Unsupported methods
--------------------

*  RCM reordering is not supported by rocSOLVER, rocSPARSE, and SuiteSparse. The following methods use AMD
   reordering instead when RCM is requested.

   *  :ref:`hipsolverSpXcsrlsvcholHost <sparse_csrlsvcholHost>` with ``reorder = 1``
   *  :ref:`hipsolverSpXcsrlsvchol <sparse_csrlsvchol>` with ``reorder = 1``

*  The function :ref:`hipsolverSpScsrlsvqr <sparse_csrlsvqr>` is implemented by converting the sparse input matrix to a dense
   matrix. It therefore does not support any reordering method. The host path is also unsupported.

Arguments not referenced by rocSOLVER
--------------------------------------

*  The ``reorder`` and ``tolerance`` arguments of :ref:`hipsolverSpScsrlsvqr <sparse_csrlsvqr>` are not referenced by the rocSOLVER
   backend.

.. _sparse_performance:

Performance implications of the hipsolverSp API
------------------------------------------------

*  The third-party SuiteSparse library is used to provide host-side functionality for :ref:`hipsolverSpXcsrlsvchol <sparse_csrlsvchol>`
   when using the rocSOLVER backend. SuiteSparse does not support single-precision arrays, so hipSOLVER must allocate
   temporary double-precision arrays and copy the values one-by-one to and from the user-provided arguments.

   .. note::

      Single precision :ref:`hipsolverSpScsrlsvchol <sparse_csrlsvchol>` is expected to have slower performance and use more memory than the
      double-precision version.

*  A fully-featured, GPU-accelerated Cholesky factorization for sparse matrices is not implemented in either rocSOLVER or
   rocSPARSE. These components rely on SuiteSparse to provide this functionality. The :ref:`hipsolverSpXcsrlsvchol <sparse_csrlsvchol>` functions
   allocate space for sparse matrices on the host, copy the data to the host, use SuiteSparse to perform the symbolic factorization, and
   then copy the resulting data back to the device.

   .. note::

      :ref:`hipsolverSpXcsrlsvchol <sparse_csrlsvchol>` might show slower performance and use more memory than
      :ref:`hipsolverSpXcsrlsvcholHost <sparse_csrlsvcholHost>`.

*  The function :ref:`hipsolverSpScsrlsvqr <sparse_csrlsvqr>` converts the sparse input matrix to a dense
   matrix, then runs the dense factorization and linear solver on the result. This might result in slower-than-expected performance and
   significant memory usage for large matrices.

   .. note::

      :ref:`hipsolverSpXcsrlsvqr <sparse_csrlsvqr>` must allocate enough memory to hold a dense matrix. It performs similarly
      to :ref:`hipsolverXXgels <gels>`.

.. _refactor_api_differences:

Using the hipsolverRf API
====================================================

The hipsolverRf API is intended as a 1:1 translation of the cusolverRf API, but not all functionality is equally supported in
rocSOLVER. The following considerations apply when using this compatibility API.

Unsupported methods
--------------------

*  Batched refactorization methods are currently unsupported with the rocSOLVER backend. They return a ``HIPSOLVER_STATUS_NOT_SUPPORTED``
   status code.

   *  :ref:`hipsolverRfBatchSetupHost <refactor_batch_setup_host>`
   *  :ref:`hipsolverRfBatchAnalyze <refactor_batch_analyze>`
   *  :ref:`hipsolverRfBatchResetValues <refactor_batch_reset_values>`
   *  :ref:`hipsolverRfBatchZeroPivot <refactor_batch_zero_pivot>`
   *  :ref:`hipsolverRfBatchRefactor <refactor_batch_refactor>`
   *  :ref:`hipsolverRfBatchSolve <refactor_batch_solve>`

*  Parameter-setting methods are currently unsupported with the rocSOLVER backend. They return a ``HIPSOLVER_STATUS_NOT_SUPPORTED``
   status code.

   *  :ref:`hipsolverRfSetAlgs <refactor_set_algs>`
   *  :ref:`hipsolverRfSetMatrixFormat <refactor_set_matrix_format>`
   *  :ref:`hipsolverRfSetNumericProperties <refactor_set_numeric_properties>`
   *  :ref:`hipsolverRfSetResetValuesFastMode <refactor_set_reset_values_fast_mode>`

.. _api_differences:

Using the regular hipSOLVER API
==========================================================

hipSOLVER's regular API is similar to cuSOLVER. However, due to differences in the implementation and design between
cuSOLVER and rocSOLVER, some minor adjustments were introduced to ensure the best performance out of both backends.

Different signatures and additional API methods
------------------------------------------------

*  The methods to obtain the size of the workspace needed by the ``gels`` and ``gesv`` functions in cuSOLVER require ``dwork`` as
   an argument. However, this argument is never used and can be null. On the rocSOLVER side, ``dwork`` is not needed to calculate the
   workspace size. As a consequence, :ref:`hipsolverXXgels_bufferSize <gels_bufferSize>` and
   :ref:`hipsolverXXgesv_bufferSize <gesv_bufferSize>` do not require ``dwork`` as an argument.

   .. note::

      These wrappers pass ``dwork = nullptr`` when calling cuSOLVER.

*  To calculate the workspace required by the function ``gesvd`` in rocSOLVER, the values of ``jobu`` and ``jobv`` are needed. As a result,
   :ref:`hipsolverXgesvd_bufferSize <gesvd_bufferSize>` requires ``jobu`` and ``jobv`` as arguments.

   .. note::

      These arguments are ignored when the wrapper calls cuSOLVER because they are not needed.

*  To properly use a user-provided workspace, rocSOLVER requires both the allocated pointer and its size. Consequently,
   :ref:`hipsolverXgetrf <getrf>` requires ``lwork`` as an argument.

   .. note::

      ``lwork`` is ignored when the wrapper calls cuSOLVER because it is not needed.

*  All rocSOLVER functions called by hipSOLVER require a workspace. To allow the user to specify one, 
   :ref:`hipsolverXgetrs <getrs>`, :ref:`hipsolverXpotrfBatched <potrf_batched>`, :ref:`hipsolverXpotrs <potrs>`, and
   :ref:`hipsolverXpotrsBatched <potrs_batched>` require ``work`` and ``lwork`` as arguments.

   .. note::

      These arguments are ignored when these wrappers call cuSOLVER because they are not needed.

   To support these changes, the regular API adds the following functions:

   *  :ref:`hipsolverXgetrs_bufferSize <getrs_bufferSize>`
   *  :ref:`hipsolverXpotrfBatched_bufferSize <potrf_batched_bufferSize>`
   *  :ref:`hipsolverXpotrs_bufferSize <potrs_bufferSize>`
   *  :ref:`hipsolverXpotrsBatched_bufferSize <potrs_batched_bufferSize>`

   .. note::

      These methods return ``lwork = 0`` when using the cuSOLVER backend, because the corresponding functions
      in cuSOLVER do not need a workspace.

Arguments not referenced by rocSOLVER
--------------------------------------

*  Unlike cuSOLVER, rocSOLVER functions do not provide information on invalid arguments in the ``info`` parameter, although they
   provide info on singularities and algorithm convergence. Therefore, when using the rocSOLVER backend, ``info`` always
   returns a value >= 0. In cases where a rocSOLVER function does not accept ``info`` as an argument, hipSOLVER
   sets it to zero.

*  The ``niters`` argument for :ref:`hipsolverXXgels <gels>` and :ref:`hipsolverXXgesv <gesv>` is not referenced by the rocSOLVER
   backend. rocSOLVER does not implement any type of iterative refinement.

.. _mem_model:

Using the rocSOLVER memory model
---------------------------------

Most hipSOLVER functions take a workspace pointer and size as arguments, allowing the user to manage the device memory used
internally by the backends. rocSOLVER, however, can maintain the device workspace automatically by default
(see the :doc:`rocSOLVER memory model <rocsolver:howto/memory>` for more details). To take
advantage of this feature, users can pass a null pointer for the ``work`` argument or a zero size for the ``lwork`` argument of any function
when using the rocSOLVER backend. The workspace will then be automatically managed behind-the-scenes. However, it's best to use
a consistent workspace management strategy because performance issues might arise if the internal workspace is forced to frequently switch between
user-provided and automatically allocated workspaces.

.. warning::

   This feature should not be used with the cuSOLVER backend. hipSOLVER does not guarantee a defined behavior when passing
   a null workspace to cuSOLVER functions that require one.

Using the rocSOLVER in-place functions
--------------------------------------
In cuSOLVER, the solvers ``gesv`` and ``gels`` are out-of-place in the sense that the solution vectors ``X`` do not overwrite the input matrix ``B``.
In rocSOLVER, this is not the case. When ``hipsolverXXgels`` or ``hipsolverXXgesv`` call rocSOLVER, some data
movements must be done internally to restore ``B`` and copy the results back to ``X``. These copies might introduce noticeable
overhead, depending on the size of the matrices. To avoid this potential problem, pass ``X = B`` to ``hipsolverXXgels``
or ``hipsolverXXgesv`` when using the rocSOLVER backend. In this case, no data movements are required, and the solution
vectors can be retrieved using either ``B`` or ``X``.

.. warning::

   This feature should not be used with the cuSOLVER backend. hipSOLVER does not guarantee a defined behavior when passing
   ``X = B`` to these functions in cuSOLVER.
