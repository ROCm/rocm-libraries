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
However, these indices cannot be reused across library releases or across different device architectures.

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
      transA,transB,grouped_gemm,batch_count,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,a_type,b_type,c_type,d_type,compute_type,scaleA,scaleB,scaleC,scaleD,amaxD,activation_type,bias_vector,bias_type,rotating_buffer,hipblaslt-Gflops,hipblaslt-GB/s,us,solution_index,kernel_name
      N,N,0,1,1024,512,1024,1,1024,1048576,1,1024,524288,1024,524288,1024,524288,f16_r,f16_r,f16_r,f16_r,f32_r,0,0,0,0,0,none,0,f32_r,512,66613.8,363.509,16.1189,56537,<kernel_name>


#. Set the environment variable ``HIPBLASLT_TUNING_OVERRIDE_FILE=<file_name>`` to load the tuning file and override
   the default kernel selection with the optimal kernel choices, where ``<file_name>`` points to the tuning file.

   In the Bash shell, set the following environment variable:

   .. code-block:: bash

      export HIPBLASLT_TUNING_OVERRIDE_FILE=tuning.txt

   For example, you can use ``hipblaslt-bench`` with ``algo_method`` set to ``heuristic`` to obtain the solutions for a problem,
   which include the best tuning solution index.

   .. code-block:: bash

      ./hipblaslt-bench --api_method c -m 1024 -n 512 -k 1024 --lda 1024 --ldb 1024 --ldc 1024 --ldd 1024  --stride_a 0 --stride_b 0 --stride_c 0 --stride_d 0  --alpha 1.000000 --beta 1.000000 --transA N --transB N --batch_count 1  --a_type f16_r --b_type f16_r --c_type f16_r --d_type f16_r --scale_type f32_r --bias_type f32_r   --compute_type f32_r --algo_method heuristic --requested_solution 1 --print_kernel_info

      transA,transB,grouped_gemm,batch_count,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,a_type,b_type,c_type,d_type,compute_type,scaleA,scaleB,scaleC,scaleD,amaxD,activation_type,bias_vector,bias_type,rotating_buffer,hipblaslt-Gflops,hipblaslt-GB/s,us,solution_index,kernel_name
      [0]:
      N,N,0,1,1024,512,1024,1,1024,1048576,1,1024,524288,1024,524288,1024,524288,f16_r,f16_r,f16_r,f16_r,f32_r,0,0,0,0,0,none,0,f32_r,512,37575.2,205.047,28.5758,56537,<kernel_name>

How tuned entries are validated
===============================

A tuning file records a ``kernel_name`` next to each ``solution_index``. When the file is replayed,
the index is treated as a lookup hint: hipBLASLt resolves it in the current library and uses it only
if it still names the same kernel. If it does not, that entry is rejected and the problem falls back
to normal heuristic selection.

Validation is per entry, so upgrading hipBLASLt thins a tuning file rather than discarding it, and
only the shapes that failed need re-tuning. A row that carries a ``solution_name`` instead is
validated the same way. Files written by an older hipBLASLt carry no name at all, and those are used
only when the file's ``Git Version`` line matches the running build. A build made outside a git
checkout has no version to match, so it uses none of them.

``kernel_name`` names the compiled kernel, not the whole solution. The index still selects the
solution, including its GSU, WGM and StaggerU defaults, and several solutions can share one compiled
kernel, which is why the name is paired with the index rather than used on its own. What validation
guarantees is that the index resolves to the same compiled kernel as when the row was written. It
does not detect a rebuild that changes only that solution's defaults, whose row is used with the new
defaults, and it cannot find a kernel that moved to a different index: that row is rejected and the
shape falls back to heuristic selection.
