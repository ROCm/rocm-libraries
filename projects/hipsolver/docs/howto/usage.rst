.. meta::
  :description: hipSOLVER documentation and API reference library
  :keywords: hipSOLVER, rocSOLVER, ROCm, API, documentation

.. _usage_label:

*****************
Using hipSOLVER
*****************

hipSOLVER is an open-source marshalling library for `LAPACK routines <https://www.netlib.org/lapack/index.html>`_ on the GPU.
It sits between a backend library and the user application, marshalling inputs to and outputs from the backend library so that the user
application remains unchanged when using different backends. Currently, two backend libraries are supported by hipSOLVER: NVIDIA's `cuSOLVER
library <https://developer.nvidia.com/cusolver>`_ and AMD's open-source `rocSOLVER library <https://github.com/ROCm/rocSOLVER>`_.

The :ref:`regular hipSOLVER API <library_api>` is a thin wrapper layer around the different backends. As such, it is not expected to introduce
significant overhead. However, its main purpose is portability, so when performance is critical directly using the library backend is recommended.

Once installed, hipSOLVER can be used just like any other library with a C API. The header file will need to be included
in the user code, and the shared library will become link-time and run-time dependencies for the user application. The
user code can be ported, with no changes, to any system with hipSOLVER installed regardless of the backend library.

For more details on how to use the API methods, see the code samples on
`hipSOLVER's clients' GitHub page <https://github.com/ROCm/hipSOLVER/tree/develop/clients/samples>`_, or
the documentation of the corresponding backend libraries.

.. _porting:

Porting cuSOLVER applications to hipSOLVER
============================================

Another purpose of hipSOLVER is to facilitate the translation of cuSOLVER applications to
:doc:`AMD's open-source ROCm platform <rocm:index>` ecosystem.
hipSOLVER is designed to make it easy for users of cuSOLVER to port their existing applications to hipSOLVER, and provides two
separate but interchangeable API patterns in order to facilitate a two-stage transition process. Users are encouraged to start with
hipSOLVER's compatibility APIs, which use the :ref:`hipsolverDn <library_dense>`, :ref:`hipsolverSp <library_sparse>`, and
:ref:`hipsolverRf <library_refactor>` prefixes and have method signatures that are fully consistent with cuSOLVER functions.

However, the compatibility APIs may introduce some performance drawbacks, especially when using the rocSOLVER backend. So, as a second
stage, it is recommended to begin the switch to hipSOLVER's :ref:`regular API <library_api>` when possible. The regular API  uses the `hipsolver` prefix and
introduces minor adjustments to the API in order to get the best performance out of the rocSOLVER backend. In most cases, switching to
the regular API is as simple as removing `Dn` from the `hipsolverDn` prefix (methods with the `hipsolverSp` and `hipsolverRf` prefixes
are not currently supported by the regular API).

No matter which API is used, a hipSOLVER application can be executed, without modifications to the code, in systems with cuSOLVER or
rocSOLVER installed. However, using the regular API ensures the best performance out of both backends.


.. _dense_api_differences:

Some considerations when using the hipsolverDn API
====================================================

The hipsolverDn API is intended as a 1:1 translation of the cusolverDn API, but not all functionality is equally supported in
rocSOLVER. Keep in mind the following considerations when using this compatibility API.


Arguments not referenced by rocSOLVER
--------------------------------------

- Unlike cuSOLVER, rocSOLVER functions do not provide information on invalid arguments in the `info` parameter, though they
  will provide info on singularities and algorithm convergence. Hence, when using the rocSOLVER backend, `info` will always
  return a value >= 0. In those cases where a rocSOLVER function does not accept `info` as an argument, hipSOLVER will
  set it to zero.

- The `niters` argument of :ref:`hipsolverDnXXgels <dense_gels>` and :ref:`hipsolverDnXXgesv <dense_gesv>` is not referenced
  by the rocSOLVER backend; there is no iterative refinement currently implemented in rocSOLVER.

- The `hRnrmF` argument of :ref:`hipsolverDnXgesvdaStridedBatched <dense_gesvda_strided_batched>` is not referenced by the
  rocSOLVER backend.

.. _dense_performance:

Performance implications of the hipsolverDn API
------------------------------------------------

- To calculate the workspace required by function `gesvd` in rocSOLVER, the values of `jobu` and `jobv` are needed, however,
  the function :ref:`hipsolverDnXgesvd_bufferSize <dense_gesvd_bufferSize>` does not accept these arguments. So, when using
  the rocSOLVER backend, `hipsolverDnXgesvd_bufferSize` has to calculate internally the workspace for all possible values of `jobu` and `jobv`,
  and return the maximum.

  (`hipsolverDnXgesvd_bufferSize` is slower than `hipsolverXgesvd_bufferSize`, and its returned workspace size could be slightly larger than
  what is actually needed).

- To properly use a user-provided workspace, rocSOLVER requires both the allocated pointer and its size. However, the function
  :ref:`hipsolverDnXgetrf <dense_getrf>` does not accept `lwork` as an argument. In consequence, when using the rocSOLVER backend,
  `hipsolverDnXgetrf` has to call internally `hipsolverDnXgetrf_bufferSize` to know the size of the workspace.

  (`hipsolverDnXgetrf_bufferSize` will be called twice in practice, once by the user before allocating the workspace, and once
  by hipSOLVER internally when executing the `hipsolverDnXgetrf` function. `hipsolverDnXgetrf` could be slightly slower than `hipsolverXgetrf`
  because of the extra call to the bufferSize helper).

- The functions :ref:`hipsolverDnXgetrs <dense_getrs>`, :ref:`hipsolverDnXpotrs <dense_potrs>`, :ref:`hipsolverDnXpotrsBatched <dense_potrs_batched>`, and
  :ref:`hipsolverDnXpotrfBatched <dense_potrf_batched>` do not accept `work` and `lwork` as arguments. However, this functionality does require a non-zero workspace
  in rocSOLVER. As a result, when using the rocSOLVER backend, these functions will switch to the automatic workspace management model (see :ref:`here <mem_model>`).

  (Users must keep in mind that even if the compatibility API does not have bufferSize helpers for the mentioned functions, these functions do require
  workspace when using rocSOLVER, and it will be automatically managed. This may imply device memory reallocations with corresponding overheads).


.. _sparse_api_differences:

Some considerations when using the hipsolverSp API
====================================================

The hipsolverSp API is intended as a 1:1 translation of the cusolverSp API, but not all functionality is equally supported in
rocSOLVER. Keep in mind the following considerations when using this compatibility API.

Unsupported methods
--------------------

- RCM reordering is currently not supported by rocSOLVER, rocSPARSE, and SuiteSparse. The following methods will instead use AMD
  reordering when RCM is requested.

  * :ref:`hipsolverSpXcsrlsvcholHost <sparse_csrlsvcholHost>` with `reorder = 1`
  * :ref:`hipsolverSpXcsrlsvchol <sparse_csrlsvchol>` with `reorder = 1`

- The function :ref:`hipsolverSpScsrlsvqr <sparse_csrlsvqr>` is currently implemented by converting the sparse input matrix to a dense
  matrix, and therefore does not support any reordering method. The host path is also currently unsupported.

Arguments not referenced by rocSOLVER
--------------------------------------

- The `reorder` and `tolerance` arguments of :ref:`hipsolverSpScsrlsvqr <sparse_csrlsvqr>` are not referenced by the rocSOLVER
  backend.

.. _sparse_performance:

Performance implications of the hipsolverSp API
------------------------------------------------

- The third-party SuiteSparse library is used to provide host-side functionality for :ref:`hipsolverSpXcsrlsvchol <sparse_csrlsvchol>`
  when using the rocSOLVER backend. At present, SuiteSparse does not support single precision arrays, therefore hipSOLVER must allocate
  temporary double precision arrays and copy the values one-by-one to and from the user-provided arguments.

  (Single precision :ref:`hipsolverSpScsrlsvchol <sparse_csrlsvchol>` is expected to perform slower and require more memory usage than the
  double precision version.)

- A fully-featured, GPU-accelerated Cholesky factorization for sparse matrices has not yet been implemented in either rocSOLVER or
  rocSPARSE. Therefore, we rely on SuiteSparse to provide this functionality. The functions :ref:`hipsolverSpXcsrlsvchol <sparse_csrlsvchol>`
  will allocate space for sparse matrices on the host, copy the data to the host, use SuiteSparse to perform the symbolic factorization, and
  then copy the resulting data back to the device.

  (:ref:`hipsolverSpXcsrlsvchol <sparse_csrlsvchol>` might perform slower and will require more memory usage than
  :ref:`hipsolverSpXcsrlsvcholHost <sparse_csrlsvcholHost>`.)

- The function :ref:`hipsolverSpScsrlsvqr <sparse_csrlsvqr>` is currently implemented by converting the sparse input matrix to a dense
  matrix, and then running the dense factorization and linear solver on the result. This might result in slower-than-expected performance and
  significant memory usage for large matrices.

  (:ref:`hipsolverSpXcsrlsvqr <sparse_csrlsvqr>` must allocate enough memory to hold a dense matrix, and will have similar performance
  to :ref:`hipsolverXXgels <gels>`)


.. _refactor_api_differences:

Some considerations when using the hipsolverRf API
====================================================

The hipsolverRf API is intended as a 1:1 translation of the cusolverRf API, but not all functionality is equally supported in
rocSOLVER. Keep in mind the following considerations when using this compatibility API.

Unsupported methods
--------------------

- Batched refactorization methods are currently unsupported with the rocSOLVER backend and will return a `HIPSOLVER_STATUS_NOT_SUPPORTED`
  status code.

  * :ref:`hipsolverRfBatchSetupHost <refactor_batch_setup_host>`
  * :ref:`hipsolverRfBatchAnalyze <refactor_batch_analyze>`
  * :ref:`hipsolverRfBatchResetValues <refactor_batch_reset_values>`
  * :ref:`hipsolverRfBatchZeroPivot <refactor_batch_zero_pivot>`
  * :ref:`hipsolverRfBatchRefactor <refactor_batch_refactor>`
  * :ref:`hipsolverRfBatchSolve <refactor_batch_solve>`

- Parameter setting methods are currently unsupported with the rocSOLVER backend and will return a `HIPSOLVER_STATUS_NOT_SUPPORTED`
  status code.

  * :ref:`hipsolverRfSetAlgs <refactor_set_algs>`
  * :ref:`hipsolverRfSetMatrixFormat <refactor_set_matrix_format>`
  * :ref:`hipsolverRfSetNumericProperties <refactor_set_numeric_properties>`
  * :ref:`hipsolverRfSetResetValuesFastMode <refactor_set_reset_values_fast_mode>`


.. _api_differences:

Some considerations when using the regular hipSOLVER API
==========================================================

hipSOLVER's regular API is similar to cuSOLVER; however, due to differences in the implementation and design between
cuSOLVER and rocSOLVER, some minor adjustments were introduced to ensure the best performance out of both backends.

Different signatures and additional API methods
------------------------------------------------

- The methods to obtain the size of the workspace needed by functions `gels` and `gesv` in cuSOLVER require `dwork` as
  an argument; however, it is never used and can be null. On the rocSOLVER side, `dwork` is not needed to calculate the
  workspace size. In consequence:

  * :ref:`hipsolverXXgels_bufferSize <gels_bufferSize>` does not require `dwork` as an argument, and
  * :ref:`hipsolverXXgesv_bufferSize <gesv_bufferSize>` does not require `dwork` as an argument.

  (These wrappers pass `dwork = nullptr` when calling cuSOLVER).

- To calculate the workspace required by function `gesvd` in rocSOLVER, the values of `jobu` and `jobv` are needed. As a result,

  * :ref:`hipsolverXgesvd_bufferSize <gesvd_bufferSize>` requires `jobu` and `jobv` as arguments.

  (These arguments are ignored when the wrapper calls cuSOLVER, as they are not needed).

- To properly use a user-provided workspace, rocSOLVER requires both the allocated pointer and its size. Consequently:

  * :ref:`hipsolverXgetrf <getrf>` requires `lwork` as an argument.

  (`lwork` is ignored when the wrapper calls cuSOLVER, as it is not needed).

- All rocSOLVER functions called by hipSOLVER require a workspace. To allow the user to specify one,

  * :ref:`hipsolverXgetrs <getrs>` requires `work` and `lwork` as arguments,
  * :ref:`hipsolverXpotrfBatched <potrf_batched>` requires `work` and `lwork` as arguments,
  * :ref:`hipsolverXpotrs <potrs>` requires `work` and `lwork` as arguments, and
  * :ref:`hipsolverXpotrsBatched <potrs_batched>` requires `work` and `lwork` as arguments.

  (These arguments are ignored when these wrappers call cuSOLVER, as they are not needed).

  In order to support these changes, the regular API adds the following functions as well:

  * :ref:`hipsolverXgetrs_bufferSize <getrs_bufferSize>`
  * :ref:`hipsolverXpotrfBatched_bufferSize <potrf_batched_bufferSize>`
  * :ref:`hipsolverXpotrs_bufferSize <potrs_bufferSize>`
  * :ref:`hipsolverXpotrsBatched_bufferSize <potrs_batched_bufferSize>`

  (These methods return `lwork = 0` when using the cuSOLVER backend, as the corresponding functions
  in cuSOLVER do not need workspace).

Arguments not referenced by rocSOLVER
--------------------------------------

- Unlike cuSOLVER, rocSOLVER functions do not provide information on invalid arguments in the `info` parameter, though they
  will provide info on singularities and algorithm convergence. Hence, when using the rocSOLVER backend, `info` will always
  return a value >= 0. In those cases where a rocSOLVER function does not accept `info` as an argument, hipSOLVER will
  set it to zero.

- The `niters` argument of :ref:`hipsolverXXgels <gels>` and :ref:`hipsolverXXgesv <gesv>` is not referenced by the rocSOLVER
  backend; there is no iterative refinement currently implemented in rocSOLVER.

.. _mem_model:

Using rocSOLVER's memory model
---------------------------------

Most hipSOLVER functions take a workspace pointer and size as arguments, allowing the user to manage the device memory used
internally by the backends. rocSOLVER, however, can maintain the device workspace automatically by default
(see :doc:`rocSOLVER's memory model <rocsolver:howto/memory>` for more details). In order to take
advantage of this feature, users may pass a null pointer for the `work` argument or a zero size for the `lwork` argument of any function
when using the rocSOLVER backend, and the workspace will be automatically managed behind-the-scenes. It is recommended, however, to use
a consistent strategy for workspace management, as performance issues may arise if the internal workspace is made to flip-flop between
user-provided and automatically allocated workspaces.

.. warning::
    This feature should not be used with the cuSOLVER backend; hipSOLVER does not guarantee a defined behavior when passing
    a null workspace to cuSOLVER functions that require one.

Using rocSOLVER's in-place functions
--------------------------------------
The solvers `gesv` and `gels` in cuSOLVER are out-of-place in the sense that the solution vectors `X` do not overwrite the input matrix `B`.
In rocSOLVER this is not the case; when `hipsolverXXgels` or `hipsolverXXgesv` call rocSOLVER, some data
movements must be done internally to restore `B` and copy the results back to `X`. These copies could introduce noticeable
overhead depending on the size of the matrices. To avoid this potential problem, users can pass `X = B` to `hipsolverXXgels`
or `hipsolverXXgesv` when using the rocSOLVER backend; in this case, no data movements will be required, and the solution
vectors can be retrieved using either `B` or `X`.

.. warning::
    This feature should not be used with the cuSOLVER backend; hipSOLVER does not guarantee a defined behavior when passing
    `X = B` to the mentioned functions in cuSOLVER.

