.. meta::
   :description: Information about the clients for the hipBLASLt library
   :keywords: hipBLASLt, ROCm, library, API, clients, test, benchmark

.. _clients:

============================
hipBLASLt clients
============================

The following client executables are available for use with hipBLASLt:

*  ``hipblaslt-test``

*  ``hipblaslt-bench``

To build these clients, follow the instructions in :doc:`Building and installing hipBLASLt <../install/building-installing-hipblaslt>`.
After building the hipBLASLt clients, you can find them in the ``hipBLASLt/build/release/clients`` directory.

Here is a brief explanation of each hipBLASLt client and how to use it.

hipblaslt-test
============================

``hipblaslt-test`` is the main GoogleTest-based regression suite for hipBLASLt. All test items must pass.

To run the full selection of test items, use this command:

.. code-block:: bash

   ./hipblaslt-test

To run a partial subsection of the test items using a filter, run this command:

.. code-block:: bash

   ./hipblaslt-test --gtest_filter=<test pattern>

For a "quick"-level demo test, use the following command:

.. code-block:: bash

   ./hipblaslt-test --gtest_filter=*quick*

hipblaslt-bench
============================

``hipblaslt-bench`` measures the performance and verifies the correctness of the hipBLASLt functions.
It includes a command line interface for ease of use.

For example, to run ``fp32`` GEMM with validation, use this command:

.. code-block:: bash

   ./hipblaslt-bench --precision f32_r -v
   transA,transB,M,N,K,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,d_type,compute_type,activation_type,bias_vector,hipblaslt-Gflops,us
   N,N,128,128,128,1,128,16384,0,128,16384,128,16384,128,16384,f32_r,f32_r,none,0, 415.278, 10.

``-v`` compares the GPU result with a CPU reference computed from the same inputs and adds three columns to the output:

* ``norm_error`` is the Frobenius norm of the difference divided by the Frobenius norm of the reference, summed over batches and gemms. It averages over each whole matrix, so a few wrong elements in a large result can still give a small value.
* ``atol`` and ``rtol`` come from an element-wise check. It tries the pairs 1e-6, 1e-5, ..., 1e-1 for each, tightest ``atol`` first, and reports the first pair under which every element satisfies ``|reference - result| <= atol + rtol * |result|``. With several batches, or the gemms of a grouped GEMM, the columns show the loosest pair any one matrix needed. ``failed`` means at least one element is outside even ``atol 0.1 + rtol 0.1 * |result|``; the bench then also prints the number of such elements and the worst one to stderr.

The norm check of D fails when any one batch's ``norm_error`` reaches the same limit ``hipblaslt-test`` uses for the GEMM. The limit is the largest tolerance among the output and both input types (1e-5 for f32, 0.01 for f16, 0.1 for bf16); f32 output with ``c_f32_fast_bf16_r`` uses at least 0.5. For example, f16 inputs with f32 output use 0.01. The limit applies to each batch; ``hipblaslt-test`` applies the same limit to the sum of the batches instead, so with many batches it can fail a result the bench passes. The bench then prints the failing batch and its error to stderr, and the solution it was for. If D's type has no norm limit, or isn't one the norm check can compare, the bench prints that D was not verified and counts it as a failed check. By default the exit code is 0 whether or not a check fails. With ``--verify_strict``, the bench exits with code 2 when any check of D failed. It checks D only: the checks of E, the bias gradient and amaxD, and ``--ulp``, don't affect the exit code. Only the first launch of each solution is compared with the reference; the launches timed by ``--iters`` and ``--cold_iters`` are not checked.

Verification computes a CPU reference for each problem and copies and compares D on the host for each solution, including every selected datafile record. This work is outside the timed GEMM launches and can substantially increase the total runtime of ``--algo_method all`` sweeps.

For more information, run the command with the ``--help`` option. The output of this command is shown below.

.. code-block:: bash

   ./hipblaslt-bench --help
   --sizem |-m <value>                Specific matrix size: the number of rows or columns in matrix.                      (Default value is: 128)
   --sizen |-n <value>                Specific matrix the number of rows or columns in matrix                             (Default value is: 128)
   --sizek |-k <value>                Specific matrix size: the number of columns in A and rows in B.                     (Default value is: 128)
   --lda <value>                      Leading dimension of matrix A.
   --ldb <value>                      Leading dimension of matrix B.
   --ldc <value>                      Leading dimension of matrix C.
   --ldd <value>                      Leading dimension of matrix D.
   --lde <value>                      Leading dimension of matrix E.
   --any_stride                       Do not modify input strides based on leading dimensions
   --stride_a <value>                 Specific stride of strided_batched matrix A, second dimension * leading dimension.
   --stride_b <value>                 Specific stride of strided_batched matrix B, second dimension * leading dimension.
   --stride_c <value>                 Specific stride of strided_batched matrix C, second dimension * leading dimension.
   --stride_d <value>                 Specific stride of strided_batched matrix D, second dimension * leading dimension.
   --stride_e <value>                 Specific stride of strided_batched matrix E, second dimension * leading dimension.
   --alpha <value>                    Specifies the scalar alpha                                                          (Default value is: 1)
   --beta <value>                     Specifies the scalar beta                                                           (Default value is: 0)
   --function |-f <value>             BLASLt function to test. Options: matmul                                            (Default value is: matmul)
   --precision |-r <value>            Precision of matrix A,B,C,D  Options: f32_r,f16_r,bf16_r,f64_r,i32_r,i8_r           (Default value is: f16_r)
   --a_type <value>                   Precision of matrix A. Options: f32_r,f16_r,bf16_r,i8_r
   --b_type <value>                   Precision of matrix B. Options: f32_r,f16_r,bf16_r,i8_r
   --c_type <value>                   Precision of matrix C. Options: f32_r,f16_r,bf16_r,i8_r
   --d_type <value>                   Precision of matrix D. Options: f32_r,f16_r,bf16_r,i8_r
   --compute_type <value>             Precision of computation. Options: s,f32_r,x,xf32_r,f64_r,i32_r                     (Default value is: f32_r)
   --compute_input_typeA <value>      Precision of computation input A. Options: f32_r, f16_r, bf16_r, f8_r, bf8_r, f8_fnuz_r, bf8_fnuz_r, The default value indicates that the compute_input_typeA has no effect.
   --compute_input_typeB <value>      Precision of computation input B. Options: f32_r, f16_r, bf16_r, f8_r, bf8_r, f8_fnuz_r, bf8_fnuz_r, The default value indicates that the compute_input_typeA has no effect.
   --scale_type <value>               Precision of scalar. Options: f16_r,bf16_r
   --initialization <value>           Initialize matrix data.Options: rand_int, trig_float, hpl(floating), special, zero, norm_dist, uniform_01  (Default value is: hpl)
   --transA <value>                   N = no transpose, T = transpose                                                     (Default value is: N)
   --transB <value>                   N = no transpose, T = transpose                                                     (Default value is: N)
   --swizzleA                         Enable tensor swizzling for A
   --swizzleB                         Enable tensor swizzling for B
   --batch_count <value>              Number of matrices. Only applicable to batched and strided_batched routines         (Default value is: 1)
   --HMM                              Parameter requesting the use of HipManagedMemory
   --verify |-v                       Validate GPU results with CPU?
   --verify_strict                    Exit with code 2 if a --verify check of D fails (norm_error above its limit, or allclose failed). Checks D only. Implies --verify.
   --iters |-i <value>                Iterations to run inside timing loop                                                (Default value is: 10)
   --cold_iters |-j <value>           Cold Iterations to run before entering the timing loop                              (Default value is: 2)
   --algo_method <value>              Use different algorithm search API. Options: heuristic, all, index.                 (Default value is: heuristic)
   --solution_index <value>           Used with --algo_method 2.  Specify solution index to use in benchmark.             (Default value is: -1)
   --requested_solution <value>       Requested solution num. Set to -1 to get all solutions. Only valid when algo_method is set to heuristic.  (Default value is: 1)
   --activation_type <value>          Options: none, gelu, relu, swish, clamp                                             (Default value is: none)
   --activation_arg1 <value>          First extra argument for activation function if needed.                             (Default value is: 0)
   --activation_arg2 <value>          Second extra argument for activation function if needed.                            (Default value is: inf)
   --bias_type <value>                Precision of bias vector.Options: f16_r,bf16_r,f32_r,default(same with D type)
   --bias_source <value>              Choose bias source: a, b, d                                                         (Default value is: d)
   --bias_vector                      Apply bias vector
   --scaleA <value>                   Apply scale for A buffer. 0 = None, 1 = scalar, 2 = vector.                         (Default value is: 0)
   --scaleB <value>                   Apply scale for B buffer. 0 = None, 1 = scalar, 2 = vector.                         (Default value is: 0)
   --scaleC <value>                   Apply scale for C buffer. 0 = None, 1 = scalar                                      (Default value is: 0)
   --scaleD <value>                   Apply scale for D buffer. 0 = None, 1 = scalar                                      (Default value is: 0)
   --scaleAlpha_vector                Apply scaleAlpha vector
   --amaxScaleA                       Apply scale for A buffer by abs max of A buffer
   --amaxScaleB                       Apply scale for B buffer by abs max of B buffer
   --amaxD                            Output Amax of intermediate D matrix
   --use_e                            Apply AUX output/ gradient input
   --gradient                         Enable gradient
   --grouped_gemm                     Use grouped_gemm.
   --use_user_args                    Use UserArguments located in device memory for grouped gemm.
   --device <value>                   Set default device to be used for subsequent program runs                           (Default value is: 0)
   --c_equal_d                        C and D are stored in same memory
   --workspace <value>                Set fixed workspace memory size (bytes) instead of using hipblaslt managed memory   (Default value is: 134217728)
   --log_function_name                Function name precedes other items.
   --function_filter <value>          Simple strstr filter on function name only without wildcards
   --api_method <value>               Use extension API. c: C style API. mix: declaration with C hipblasLtMatmul Layout/Desc but set, initialize, and run the problem with C++ extension API. cpp: Using C++ extension API only. Options: c, mix, cpp.  (Default value is: c)
   --print_kernel_info                Print solution, kernel name and solution index.
   --rotating <value>                 Use rotating memory blocks for each iteration, size in MB.                          (Default value is: 0)
   --use_gpu_timer                    Use hipEventElapsedTime to profile elapsed time.
   --skip_slow_solution_ratio <value> Specifies a ratio to skip slow solution when warm up stage. Skip condition: (current solution's warm up time * ratio) > best solution's warm up time. Ratio range: 0 ~ 1. 0 means no skip.  (Default value is: 0)
   --splitk <value>                   [Tuning parameter] Set split K for a solution, 0 is use solution's default value. (Only support GEMM + api_method mix or cpp)
   --wgm <value>                      [Tuning parameter] Set workgroup mapping for a solution, 0 is use solution's default value. (Only support GEMM + api_method mix or cpp)
   --flush                            Flush icache, only works for gemm.
   --uniform_summation_order <value>  Set the HIPBLASLT_MATMUL_DESC_UNIFORM_SUMMATION_ORDER_EXT extension attribute. Accepts off|0, on|1 (case-insensitive). When on, the matmul returns HIPBLAS_STATUS_INVALID_VALUE if no configuration honoring the guarantee exists. When omitted the bench leaves the attribute unset so the library default (off) applies.
   --help |-h                         Produces this help message
   --version <value>                  Prints the version number


Building clients with prebuilt libraries
========================================

Sometimes it is desirable to build or rebuild the clients without having to conduct a full build of the library. This can be done by passing ``--no-tensile`` to ``inv build``. For example, ``inv build --clients --architecture gfx942 --no-tensile`` will build the clients and host code, but will not build Tensile libraries.

Internally, this passes the ``-DHIPBLASLT_ENABLE_DEVICE=OFF`` option to CMake. If you prefer to build hipBLASLt with CMake directly instead of through invoke, the same effect can be achieved with the following steps:

.. code-block:: bash

   mkdir build && cd build
   cmake [other options]
       -DBUILD_CLIENTS_SAMPLES=ON
       -DBUILD_CLIENTS_TESTS=ON
       -DBUILD_CLIENTS_BENCHMARKS=ON
       -DTensile_SKIP_BUILD=ON ..
   make

To run the clients with existing Tensile libraries, you must set the environment variable ``HIPBLASLT_TENSILE_LIBPATH`` to the path of the Tensile libraries. For example, if the Tensile libraries are in ``/mnt/build/release/Tensile/library/``, you would run the clients as follows:

.. code-block:: bash

   HIPBLASLT_TENSILE_LIBPATH=/mnt/build/release/Tensile/library/ ./hipblaslt-test

If ``HIPBLASLT_TENSILE_LIBPATH`` is not set, the clients will attempt to find the libraries in ``/opt/rocm/lib``. If libraries cannot be found in that location, the following diagnostics will be printed:

.. code-block:: bash

   rocblaslt info: HIPBLASLT_TENSILE_LIBPATH not set. Using /opt/rocm/lib

   rocblaslt error: Cannot read /mnt/build/release/library/../Tensile/library/TensileLibrary_lazy_gfx90a.dat: No such file or directory
