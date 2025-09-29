.. meta::
  :description: rocSPARSE design documentation
  :keywords: rocSPARSE, ROCm, API, documentation, design

.. _design:

******************************
rocSPARSE design notes
******************************

This topic is intended for advanced developers who want to understand, modify, or extend the functionality of the rocSPARSE library.

The rocSPARSE library is developed using the "Hourglass API" approach.
This provides a thin C89 API which retains all the convenience of C++.
As a side effect, this avoids API-related binary compatibility issues.
This approach also allows rocSPARSE routines to be used by other programming languages.

In public API header files, rocSPARSE only relies on functions, pointers, forward declared structs, enumerations, and type definitions.
rocSPARSE introduces multiple library and object handles by using opaque types to hide layout and implementation details from the user.

Temporary device memory
=======================

Many routines exposed by the rocSPARSE API require a temporary storage buffer on the device.
You are responsible for buffer allocation and deallocation.
The allocated buffers can be reused and do not need to be regularly freed and re-allocated for every single API call.
For this purpose, routines that require a temporary storage buffer offer a special API function
to query for the storage buffer size, for example, :cpp:func:`rocsparse_scsrsv_buffer_size`.

Library source code organization
================================

This section discusses the structure of the rocSPARSE library in the `rocSPARSE GitHub repository <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocsparse>`_.

The library/include directory
-----------------------------

The ``library/include`` directory contains all files that are exposed to the user.
The rocSPARSE API is declared here.

============================= ===========
File                          Description
============================= ===========
``rocsparse.h``               Includes all other API-related rocSPARSE header files.
``rocsparse-auxiliary.h``     Declares all rocSPARSE auxiliary functions, such as handle and descriptor management.
``rocsparse-complex-types.h`` Defines the rocSPARSE complex data types ``rocsparse_float_complex`` and ``rocsparse_double_complex``.
``rocsparse-functions.h``     Declares all rocSPARSE Sparse Linear Algebra Subroutines of types level 1, level 2, level 3, extra, preconditioner, format conversion, reordering, generic, and utility. This is achieved by including headers from the ``library/include/internal`` directory.
``rocsparse-types.h``         Defines all data types used by rocSPARSE.
``rocsparse-version.h.in``    Provides the configured version and settings that are initially set by CMake during compilation.
============================= ===========

The ``library/include/internal`` directory contains the public API for all rocSPARSE sparse linear algebra subroutines,
organized into ``level1``, ``level2``, ``level3``, ``extra``, ``precond``, ``conversion``, ``reordering``, ``generic``, and ``utility`` directories.

The library/src directory
--------------------------

This directory contains all rocSPARSE library source files.
The root of the ``library/src/`` directory hosts the implementation of the library handle and auxiliary functions.
Each subdirectory is responsible for a specific class of sparse linear algebra subroutines.
The ``library/src/include`` directory defines the following categories:

* :ref:`rocsparse_common`
* :ref:`rocsparse_macros`
* :ref:`rocsparse_mat_struct`
* :ref:`rocsparse_logging`

=========================== ===========
File                        Description
=========================== ===========
``handle.cpp``              Implementation of opaque handle structures.
``rocsparse_auxiliary.cpp`` Implementation of auxiliary functions, for example, create and destroy handles.
``status.cpp``              Implementation of the :cpp:enum:`hipError_t` to :cpp:enum:`rocsparse_status` conversion function.
``include/common.h``        Commonly used functions among several rocSPARSE routines. See :ref:`rocsparse_common`.
``include/definitions.h``   Definition of status-flag macros. See :ref:`rocsparse_macros`.
``include/handle.h``        Declaration of opaque handle structures.
``include/logging.h``       Implementation of the different rocSPARSE logging helper functions.
``include/status.h``        Declaration of the :cpp:enum:`hipError_t` to :cpp:enum:`rocsparse_status` conversion function.
``include/utility.h``       Implementation of the different rocSPARSE logging functions.
=========================== ===========

The clients directory
----------------------

The ``clients`` directory contains all clients, for example, samples, unit tests, and benchmarks.
For more details, see :ref:`rocsparse_clients`.

Sparse linear algebra subroutines
---------------------------------

Each sparse linear algebra subroutine is implemented in a set of source files in the
corresponding directory: ``rocsparse_<subroutine>.cpp``, ``rocsparse_<subroutine>.hpp``, and ``<subroutine>_device.h``,
where ``<subroutine>`` indicates any of the rocSPARSE library functions.

``rocsparse_<subroutine>.cpp`` implements the C wrapper and API functionality for each supported precision.
``rocsparse_<subroutine>.hpp`` implements the API functionality, using the precision as template parameter.
Finally, ``<subroutine>_device.h`` implements the device code required for the computation of the subroutine.

.. note::

   Each subroutine exposed in the API is expected to return a :cpp:type:`rocsparse_status`.
   Additionally, each device function is expected to use a specified stream which is accessible through the libraries handle.

The following code block contains a sample for ``rocsparse_<subroutine>.cpp``, ``rocsparse_<subroutine>.hpp``,
and ``<subroutine>_device.h``.

.. code-block:: cpp
   :caption: rocsparse_subroutine.cpp

   #include "rocsparse.h"
   #include "rocsparse_subroutine.hpp"

   /*
    * ===========================
    *    C wrapper
    * ===========================
    */

   extern "C" rocsparse_status rocsparse_ssubroutine(rocsparse_handle handle,
                                                     rocsparse_int    m,
                                                     const float*     alpha,
                                                     float*           val)
   {
       return rocsparse_subroutine_template(handle, m, alpha, val);
   }

   extern "C" rocsparse_status rocsparse_dsubroutine(rocsparse_handle handle,
                                                     rocsparse_int    m,
                                                     const double*    alpha,
                                                     double*          val)
   {
       return rocsparse_subroutine_template(handle, m, alpha, val);
   }

   extern "C" rocsparse_status rocsparse_csubroutine(rocsparse_handle               handle,
                                                     rocsparse_int                  m,
                                                     const rocsparse_float_complex* alpha,
                                                     rocsparse_float_complex*       val)
   {
       return rocsparse_subroutine_template(handle, m, alpha, val);
   }

   extern "C" rocsparse_status rocsparse_zsubroutine(rocsparse_handle                handle,
                                                     rocsparse_int                   m,
                                                     const rocsparse_double_complex* alpha,
                                                     rocsparse_double_complex*       val)
   {
       return rocsparse_subroutine_template(handle, m, alpha, val);
   }

.. code-block:: cpp
   :caption: rocsparse_subroutine.hpp

   #pragma once
   #ifndef ROCSPARSE_SUBROUTINE_HPP
   #define ROCSPARSE_SUBROUTINE_HPP

   #include "definitions.h"
   #include "handle.h"
   #include "rocsparse.h"
   #include "subroutine_device.h"
   #include "utility.h"

   #include <hip/hip_runtime.h>

   template <typename T>
   __global__ void subroutine_kernel_host_pointer(rocsparse_int m, T alpha, T* val)
   {
       subroutine_device(m, alpha, val);
   }

   template <typename T>
   __global__ void subroutine_kernel_device_pointer(rocsparse_int m, const T* alpha, T* val)
   {
       subroutine_device(m, *alpha, val);
   }

   template <typename T>
   rocsparse_status rocsparse_subroutine_template(rocsparse_handle handle,
                                                  rocsparse_int    m,
                                                  const T*         alpha,
                                                  T*               val)
   {
       // Check for valid handle
       if(handle == nullptr)
       {
           return rocsparse_status_invalid_handle;
       }

       // Check size
       if(m < 0)
       {
           return rocsparse_status_invalid_size;
       }

       // Quick return if possible
       if(m == 0)
       {
           return rocsparse_status_success;
       }

       // Check pointer arguments
       if(alpha == nullptr || val == nullptr)
       {
           return rocsparse_status_invalid_pointer;
       }

       // Differentiate between the pointer modes
       if(handle->pointer_mode == rocsparse_pointer_mode_device)
       {
           // Launch kernel
           hipLaunchKernelGGL((subroutine_kernel_device_pointer<T>),
                              dim3(...),
                              dim3(...),
                              0,
                              handle->stream,
                              m,
                              alpha,
                              val);
       }
       else
       {
           // Launch kernel
           hipLaunchKernelGGL((subroutine_kernel_host_pointer<T>),
                              dim3(...),
                              dim3(...),
                              0,
                              handle->stream,
                              m,
                              *alpha,
                              val);
       }

       return rocsparse_status_success;
   }

   #endif // ROCSPARSE_SUBROUTINE_HPP

.. code-block:: cpp
   :caption: subroutine_device.h

   #pragma once
   #ifndef SUBROUTINE_DEVICE_H
   #define SUBROUTINE_DEVICE_H

   #include <hip/hip_runtime.h>

   template <typename T>
   __device__ void subroutine_device(rocsparse_int m, T alpha, T* val)
   {
       ...
   }

   #endif // SUBROUTINE_DEVICE_H


Important functions and data structures
=======================================

This section describes the important rocSPARSE functions and data structures.

.. _rocsparse_common:

Commonly shared device code
---------------------------

The following table lists many of the device functions that are shared among several rocSPARSE functions.

================================== ===========
Device function                    Description
================================== ===========
``rocsparse::clz()``               Computes the leftmost significant bit position for ``int32_t`` and ``int64_t`` types.
``rocsparse::one()``               Returns a pointer to ``1`` for the specified precision.
``rocsparse::ldg()``               Wrapper to ``__ldg()`` for ``int32_t``, ``int64_t``, ``float``, ``double``, and complex types.
``rocsparse::nontemporal_load()``  Non-temporal memory load access for ``int32_t``, ``int64_t``, ``float``, ``double``, and complex types.
``rocsparse::nontemporal_store()`` Non-temporal memory store access for ``int32_t``, ``int64_t``, ``float``, ``double``, and complex types.
``rocsparse::mul24()``             Multiply 24-bit integer values.
``rocsparse::mad24()``             Multiply 24-bit integers and add a 32-bit value.
``rocsparse::blockreduce_sum()``   Block-wide reduction sum for ``int32_t``, ``int64_t``, ``float``, ``double``, and complex types.
``rocsparse::blockreduce_max()``   Block-wide reduction max for ``int32_t``, ``int64_t``, ``float``, ``double``, and complex types.
``rocsparse::blockreduce_min()``   Block-wide reduction min for ``int32_t``, ``int64_t``, ``float``, ``double``, and complex types.
``rocsparse::wfreduce_max()``      DPP-based wavefront reduction max for the ``int32_t`` type.
``rocsparse::wfreduce_min()``      DPP-based wavefront reduction min for the ``int32_t`` and ``int64_t`` types.
``rocsparse::wfreduce_sum()``      DPP-based wavefront reduction sum for ``int32_t``, ``int64_t``, ``float``, ``double``, and complex types.
================================== ===========

.. _rocsparse_macros:

Status-flag macros
------------------

The following table lists the status-flag macros available in rocSPARSE and their purpose.

=================================== ===========
Macro                               Description
=================================== ===========
``RETURN_IF_HIP_ERROR(stat)``       Returns if ``stat`` is not equal to :cpp:enumerator:`hipSuccess`.
``THROW_IF_HIP_ERROR(stat)``        Throws an exception if ``stat`` is not equal to :cpp:enumerator:`hipSuccess`.
``PRINT_IF_HIP_ERROR(stat)``        Prints an error message if ``stat`` is not equal to :cpp:enumerator:`hipSuccess`.
``RETURN_IF_ROCSPARSE_ERROR(stat)`` Returns if ``stat`` is not equal to :cpp:enumerator:`rocsparse_status_success`.
=================================== ===========

.. _rocsparse_mat_struct:

The rocsparse_mat_info structure
----------------------------------

The rocSPARSE :cpp:type:`rocsparse_mat_info` structure contains all matrix meta information that is collected during the analysis routines.

The following table lists all the internal metadata structures:

========================== ===========
Meta data structure        Description
========================== ===========
``rocsparse_csrmv_info``   Structure to hold analysis metadata for sparse matrix and vector multiplication in CSR format.
``rocsparse_csrtr_info``   Structure to hold analysis metadata for operations on sparse triangular matrices, for example, the dependency graph.
``rocsparse_csrgemm_info`` Structure to hold analysis metadata for sparse matrix and sparse matrix multiplication in CSR format.
========================== ===========

Cross-Routine Data Sharing
^^^^^^^^^^^^^^^^^^^^^^^^^^

Metadata that has already been collected, such as the dependency graph of a sparse matrix, can be shared among multiple routines.
For example, if the incomplete LU factorization of a sparse matrix is computed,
the gathered analysis data can be shared for subsequent lower triangular solves of the same matrix.
This behavior can be specified by the :ref:`rocsparse_analysis_policy_` parameter.

The following table lists subroutines that can, in some cases, share metadata:

================================== ====
Subroutine                         Shares metadata with
================================== ====
:cpp:func:`rocsparse_scsrsv_solve` :cpp:func:`rocsparse_scsric0`, :cpp:func:`rocsparse_scsrilu0`
:cpp:func:`rocsparse_dcsrsv_solve` :cpp:func:`rocsparse_dcsric0`, :cpp:func:`rocsparse_dcsrilu0`
:cpp:func:`rocsparse_ccsrsv_solve` :cpp:func:`rocsparse_ccsric0`, :cpp:func:`rocsparse_ccsrilu0`
:cpp:func:`rocsparse_zcsrsv_solve` :cpp:func:`rocsparse_zcsric0`, :cpp:func:`rocsparse_zcsrilu0`
:cpp:func:`rocsparse_scsric0`      :cpp:func:`rocsparse_scsrilu0`, :cpp:func:`rocsparse_scsrsv_solve`
:cpp:func:`rocsparse_dcsric0`      :cpp:func:`rocsparse_dcsrilu0`, :cpp:func:`rocsparse_dcsrsv_solve`
:cpp:func:`rocsparse_ccsric0`      :cpp:func:`rocsparse_ccsrilu0`, :cpp:func:`rocsparse_ccsrsv_solve`
:cpp:func:`rocsparse_zcsric0`      :cpp:func:`rocsparse_zcsrilu0`, :cpp:func:`rocsparse_zcsrsv_solve`
:cpp:func:`rocsparse_scsrilu0`     :cpp:func:`rocsparse_scsric0`, :cpp:func:`rocsparse_scsrsv_solve`
:cpp:func:`rocsparse_dcsrilu0`     :cpp:func:`rocsparse_dcsric0`, :cpp:func:`rocsparse_dcsrsv_solve`
:cpp:func:`rocsparse_ccsrilu0`     :cpp:func:`rocsparse_ccsric0`, :cpp:func:`rocsparse_ccsrsv_solve`
:cpp:func:`rocsparse_zcsrilu0`     :cpp:func:`rocsparse_zcsric0`, :cpp:func:`rocsparse_zcsrsv_solve`
================================== ====

.. note::

   This functionality can be further expanded on rocSPARSE extensions to significantly improve metadata collection performance.

.. _rocsparse_clients:

Clients
=======

rocSPARSE clients host a variety of different examples, as well as a unit test and benchmarking package.
For detailed instructions on how to build rocSPARSE with clients, see the :doc:`Linux Install <../install/Linux_Install_Guide>`
or :doc:`Windows Install <../install/Windows_Install_Guide>` guides.

Samples
-------

The ``clients/samples`` collection offers sample implementations of the rocSPARSE API.
The following table lists the available examples along with a description.

================== ===========
Sample             Description
================== ===========
``example_coomv``  Performs sparse matrix vector multiplication in COO format.
``example_csrmv``  Performs sparse matrix vector multiplication in CSR format.
``example_ellmv``  Performs sparse matrix vector multiplication in ELL format.
``example_handle`` Demonstrates rocSPARSE handle initialization and finalization.
``example_hybmv``  Performs sparse matrix vector multiplication in HYB format.
================== ===========

Unit tests
----------

Multiple unit tests are available to test for bad arguments, invalid parameters, and sparse routine functionality.
The unit tests are based on `GoogleTest <https://github.com/google/googletest>`_.
The tests cover all routines that are exposed by the API, including all available floating-point precision.

Benchmarks
----------

rocSPARSE offers a benchmarking tool that can be compiled with the clients package.
The benchmark tool can perform time measurements for any routine exposed by the API.
To set up a benchmark run, multiple options are available.

======================== ===========
Command-line option      Description
======================== ===========
``help``, ``h``          Prints the help message
``sizem``, ``m``         Specify the ``m`` parameter, for example, the number of rows of a sparse matrix.
``sizen``, ``n``         Specify the ``n`` parameter, for example, the number of columns of a sparse matrix or the length of a dense vector.
``sizek``, ``k``         Specify the ``k`` parameter, for example, the number of rows of a dense matrix.
``sizennz``, ``z``       Specify the ``nnz`` parameter, for example, the number of non-zero entries of a sparse vector.
``blockdim``             Specify the ``blockdim`` parameter, for example, the block dimension in BSR matrices.
``row-blockdimA``        Specify the ``row-blockdimA``  parameter, for example, the row block dimension in GEBSR matrices.
``col-blockdimA``        Specify the ``col-blockdimA`` parameter, for example, the column block dimension in GEBSR matrices.
``row-blockdimB``        Specify the ``row-blockdimB`` parameter, for example, the row block dimension in GEBSR matrices.
``col-blockdimB``        Specify the ``col-blockdimB`` parameter, for example, the column block dimension in GEBSR matrices.
``mtx``                  Read from `MatrixMarket (.mtx) format <https://math.nist.gov/MatrixMarket/formats.html>`_. This overrides parameters ``m``, ``n``, and ``z``.
``rocalution``           Read from `rocALUTION format <https://github.com/ROCm/rocALUTION>`_. This overrides parameters ``m``, ``n``, ``z``, ``mtx``, and ``laplacian-dim``.
``laplacian-dim``        Assemble a 2D/3D Laplacian matrix with dimensions ``dimx``, ``dimy``, and ``dimz``. ``dimz`` is optional. This overrides parameters ``m``, ``n``, ``z``, and ``mtx``.
``alpha``                Specify the scalar :math:`\alpha`.
``beta``                 Specify the scalar :math:`\beta`.
``transposeA``           Specify whether matrix A is (conjugate) transposed or not. See :ref:`rocsparse_operation_`.
``transposeB``           Specify whether matrix B is (conjugate) transposed or not. See :ref:`rocsparse_operation_`.
``indexbaseA``           Specify the index base of matrix A. See :ref:`rocsparse_index_base_`.
``indexbaseB``           Specify the index base of matrix B. See :ref:`rocsparse_index_base_`.
``indexbaseC``           Specify the index base of matrix C. See :ref:`rocsparse_index_base_`.
``indexbaseD``           Specify the index base of matrix D. See :ref:`rocsparse_index_base_`.
``action``               Specify whether the operation is performed symbolically or numerically. See :ref:`rocsparse_action_`.
``hybpart``              Specify the HYB partitioning type. See :ref:`rocsparse_hyb_partition_`.
``diag``                 Specify the diagonal type of a sparse matrix. See :ref:`rocsparse_diag_type_`.
``uplo``                 Specify the fill mode of a sparse matrix. See :ref:`rocsparse_fill_mode_`.
``storage``              Specify the storage mode of a sparse matrix. See :ref:`rocsparse_storage_mode_`.
``apolicy``              Specify the analysis policy. See :ref:`rocsparse_analysis_policy_`.
``function``, ``f``      Specify the API-exposed subroutine to benchmark.
``indextype``            Index precision: integer 32 bit or integer 64 bit.
``precision``, ``r``     Floating-point precision: single real, double real, single complex, or double complex.
``verify``, ``v``        Specify whether the results should be validated with the host reference implementation.
``iters``, ``i``         Iterations to run inside the timing loop.
``device``, ``d``        Set the device to be used for subsequent benchmark runs.
``direction``            Specify whether BSR blocks should be laid out in row-major storage or column-major storage.
``order``                Specify whether a dense matrix is laid out in column-major or row-major storage.
``format``               Specify whether a sparse matrix is laid out in coo, coo_aos, csr, csc, or ell format.
``denseld``              Specify the leading dimension of a dense matrix.
``batch_count``          Specify the batch count for batched routines.
``batch_count_A``        Specify the batch count for batched routines.
``batch_count_B``        Specify the batch count for batched routines.
``batch_count_C``        Specify the batch count for batched routines.
``batch_stride``         Specify the batch stride for batched routines.
``memstat-report``       Specify the output filename for the memory report.
``spmv_alg``             Specify the algorithm to use when running SpMV.
``spmm_alg``             Specify the algorithm to use when running SpMM.
``gtsv_interleaved_alg`` Specify the algorithm to use when running the gtsv interleaved batch routine.
======================== ===========

For example, to benchmark the ``csrmv`` routine using double precision, run the following command:

.. code-block:: shell

   ./rocsparse-bench -f csrmv --precision d --alpha 1 --beta 0 --iters 1000 --rocalution <path to .csr matrix file>

Python plotting scripts
-----------------------

rocSPARSE also contains some useful Python plotting scripts that work in conjunction with the ``rocsparse-bench`` executable. To use these
plotting scripts to, for example, plot the performance of the csrmv routine with multiple matrices, first call:

.. code-block:: shell

   ./rocsparse-bench -f csrmv --precision d --alpha 1 --beta 0 --iters 1000 --bench-x --rocalution /path/to/matrix/files/*.csr --bench-o name_of_output_file.json

This produces the JSON file ``name_of_output_file.json``, which contains all the performance data. This file can be passed to the Python plotting script
``rocSPARSE/scripts/rocsparse-bench-plot.py`` using the following command:

.. code-block:: shell

   python rocsparse-bench-plot.py /path/to/json/file/name_of_output_file.json

This generates pdf files that plot the following information:

* GB/s
* GFLOPS/s
* milliseconds

.. note::

   The parameter that follows the ``--bench-x`` option for ``rocsparse-bench`` specifies the values to use on the x-axis.
   An earlier example passed ``--bench-x --rocalution /path/to/matrix/files/*.csr``, which means that each entry in the
   x-axis of the generated plot is a matrix found in the directory ``/path/to/matrix/files/``.

rocSPARSE also provides plotting scripts that let you generate plots comparing two or more rocsparse-bench performance
runs. For example, to compare the performance of csrmv with single precision and double precision,
first run:

.. code-block:: shell

   ./rocsparse-bench -f csrmv --precision s --alpha 1 --beta 0 --iters 1000 --bench-x --rocalution /path/to/matrix/files/*.csr --bench-o scsrmv_output_file.json

   ./rocsparse-bench -f csrmv --precision d --alpha 1 --beta 0 --iters 1000 --bench-x --rocalution /path/to/matrix/files/*.csr --bench-o dcsrmv_output_file.json

Doing this generates two JSON output files: ``scsrmv_output_file.json`` and ``dcsrmv_output_file.json``. These files can then be
passed to the Python plotting script ``rocSPARSE/scripts/rocsparse-bench-compare.py``:

.. code-block:: shell

   python rocsparse-bench-compare.py /path/to/json/file/scsrmv_output_file.json /path/to/json/file/dcsrmv_output_file.json

This generates pdf files plotting the following comparisons between the two runs:

* GB/s
* GFLOPS/s
* milliseconds
* GB/s ratio
* GFLOPS/s ratio

In both Python scripts, the y axis defaults to log scaling. To use linear scaling for the y axis, pass
the ``--linear`` option to either of the Python plotting scripts. To see a full list of options, use the ``-h`` or ``--help`` option.

Helper scripts for downloading matrices
---------------------------------------

rocSPARSE contains some helper scripts for downloading matrices from the `sparse suite collection <http://sparse.tamu.edu/>`_.
These matrices can be useful for additional testing and performance measurement. The scripts are found in the
``rocSPARSE/scripts/performance/matrices`` directory. To use these scripts to download matrices, run the following commands:

.. code-block:: shell

   ./build_convert.sh

.. code-block:: shell

   ./get_matrices_1.sh

.. code-block:: shell

   ./get_matrices_2.sh

.. code-block:: shell

   ./get_matrices_3.sh

.. code-block:: shell

   ./get_matrices_4.sh

.. code-block:: shell

   ./get_matrices_5.sh

.. code-block:: shell

   ./get_matrices_6.sh

.. code-block:: shell

   ./get_matrices_7.sh

.. code-block:: shell

   ./get_matrices_8.sh

This downloads the matrices and converts them to .csr format so rocsparse-bench can use them when
the ``--rocalution`` option is provided.
