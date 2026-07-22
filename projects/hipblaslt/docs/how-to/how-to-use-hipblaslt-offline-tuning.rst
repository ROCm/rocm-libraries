.. meta::
   :description: How to use the hipBLASLt offline tuning utility
   :keywords: hipBLASLt, ROCm, library, API, tuning, GEMM, offline tuning, utility

.. _how-to-use-hipblaslt-offline-tuning:

********************************
Using hipBLASLt offline tuning
********************************

``hipblaslt-bench`` can help find the best-performing GEMM kernel for a given set of GEMM problems
and provide the best solution index for a given problem size.
This index can be used directly in future GEMM calls through the User Offline Tuning mechanism.

Solution indices are not portable across library releases or device architectures on their own.
An index is only a position in the solution library, and that position can shift whenever kernels
are added, removed, or reordered. Current tuning files therefore record the winning kernel's
``solution_name`` alongside its index (see :ref:`validated-tuning-entries`). On replay, hipBLASLt
uses the entry only when the recorded index still resolves to the recorded name. Otherwise, that
problem falls back safely to default kernel selection instead of silently running a different
kernel.

Use the command line interface to access this functionality. See :ref:`clients` for more details.

Using hipblaslt-bench to run the tuning with the best GEMM kernel
=================================================================

To find and use the best GEMM kernel for a problem, follow these steps:

#. Generate the tuning command line by setting the environment variable ``HIPBLASLT_LOG_MASK=32`` before calling any hipBLASLt APIs. For more details on how to use ``hipblaslt-bench``, see :ref:`Logging and heuristics <logging-heuristics>`.
   In the Bash shell, set the following environment variable:

   .. code-block:: bash

      export HIPBLASLT_LOG_MASK=32

#. The following command uses `sample_hipblaslt_gemm.cpp <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipblaslt/clients/samples/01_hipblaslt_gemm/sample_hipblaslt_gemm.cpp>`_ as an example:

   .. code-block:: bash

      ./sample_hipblaslt_gemm

   The tuning command displays the following log entry:

   .. code-block:: bash

      hipblaslt-bench --api_method c -m 1024 -n 512 -k 1024 --lda 1024 --ldb 1024 --ldc 1024 --ldd 1024  --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0  --alpha 1.000000 --beta 1.000000 --transA N --transB N --batch_count 1  --a_type f16_r --b_type f16_r --c_type f16_r --d_type f16_r --scale_type f32_r --bias_type f32_r   --compute_type f32_r --algo_method index --solution_index 56073

#. Set the environment variable ``HIPBLASLT_TUNING_FILE=<file_name>`` to tune and store the tuning result, which indicates the best solution
   indices for the GEMM problems. The ``<file_name>`` points to the tuning file.

   In the Bash shell, set the following environment variable:

   .. code-block:: bash

      export HIPBLASLT_TUNING_FILE=tuning.txt

   Additionally, you can set the environment variable to specify that the solution found in the tuning stage is under the constraint of the max workspace size setting:

   .. code-block:: bash

      export HIPBLASLT_TUNING_USER_MAX_WORKSPACE=<value> (Default value is: 128 * 1024 * 1024)

   The default settings for the following parameters in ``hipblaslt-bench`` are changed in the tuning environment.

   .. code-block:: bash

      --iters |-i <value>             (Default value is: 1000)
      --cold_iters |-j <value>        (Default value is: 1000)
      --requested_solution <value>    (Default value is: -1)
      --rotating <value>              (Default value is: 512)

   After the tuning completes, the expected output is displayed as follows:

   .. code-block:: bash

      ./hipblaslt-bench --api_method c -m 1024 -n 512 -k 1024 --lda 1024 --ldb 1024 --ldc 1024 --ldd 1024  --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0  --alpha 1.000000 --beta 1.000000 --transA N --transB N --batch_count 1  --a_type f16_r --b_type f16_r --c_type f16_r --d_type f16_r --scale_type f32_r --bias_type f32_r   --compute_type f32_r --algo_method index --solution_index 56073

      Winner:
      transA,transB,grouped_gemm,batch_count,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,a_type,b_type,c_type,d_type,compute_type,scaleA,scaleB,scaleC,scaleD,amaxD,activation_type,bias_vector,bias_type,rotating_buffer,hipblaslt-Gflops,hipblaslt-GB/s,us,soulution_index
      N,N,0,1,1024,512,1024,1,1024,1048576,1,1024,524288,1024,524288,1024,524288,f16_r,f16_r,f16_r,f16_r,f32_r,0,0,0,0,0,none,0,f32_r,512,66613.8,363.509,16.1189,56537


#. Set the environment variable ``HIPBLASLT_TUNING_OVERRIDE_FILE=<file_name>`` to load the tuning file and override
   the default kernel selection with the optimal kernel choices, where ``<file_name>`` points to the tuning file.

   In the Bash shell, set the following environment variable:

   .. code-block:: bash

      export HIPBLASLT_TUNING_OVERRIDE_FILE=tuning.txt

   For example, you can use ``hipblaslt-bench`` with ``algo_method`` set to ``heuristic`` to obtain the solutions for a problem,
   which include the best tuning solution index.

   .. code-block:: bash

      ./hipblaslt-bench --api_method c -m 1024 -n 512 -k 1024 --lda 1024 --ldb 1024 --ldc 1024 --ldd 1024  --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0  --alpha 1.000000 --beta 1.000000 --transA N --transB N --batch_count 1  --a_type f16_r --b_type f16_r --c_type f16_r --d_type f16_r --scale_type f32_r --bias_type f32_r   --compute_type f32_r --algo_method heuristic --requested_solution 1 --print_kernel_info

      transA,transB,grouped_gemm,batch_count,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,a_type,b_type,c_type,d_type,compute_type,scaleA,scaleB,scaleC,scaleD,amaxD,activation_type,bias_vector,bias_type,rotating_buffer,hipblaslt-Gflops,hipblaslt-GB/s,us,soulution_index
      [0]:
      N,N,0,1,1024,512,1024,1,1024,1048576,1,1024,524288,1024,524288,1024,524288,f16_r,f16_r,f16_r,f16_r,f32_r,0,0,0,0,0,none,0,f32_r,512,37575.2,205.047,28.5758,56537

.. _validated-tuning-entries:

Tuning entries are validated before use
=======================================

Tuning files written by a current ``hipblaslt-bench`` include ``solution_name`` next to
``solution_index``:

.. code-block:: text

   ...,solution_index,solution_name
   ...,56537,Cijk_Alik_Bljk_HHS_BH_MT128x128x16

When ``HIPBLASLT_TUNING_OVERRIDE_FILE`` is loaded, the index is treated as a lookup hint and the
name as its check. hipBLASLt resolves the recorded index in the current solution library and
compares the resolved name with the recorded name:

* If the names match and the solution supports the current problem, hipBLASLt uses the tuned
  solution.
* If the index is missing, the names differ, or the solution is unsupported, hipBLASLt ignores
  that entry and uses default kernel selection for that problem. Other entries remain eligible.

Phase 1 does not search for the recorded name at a different index and does not retune
automatically. Therefore, an entry can fall back after a rebuild or ROCm upgrade even when a
similarly named kernel still exists elsewhere in the library. This is intentional: fallback is
safe, although it can be slower than the previously tuned result. To recover tuned performance,
run ``hipblaslt-bench`` again and replace the affected tuning entry.

Older tuning files without ``solution_name`` remain accepted for compatibility. Because those
entries cannot validate kernel identity, hipBLASLt trusts them only when the file's recorded git
version matches the running build and the index still resolves. On a build-version mismatch,
legacy entries fall back to default selection.
