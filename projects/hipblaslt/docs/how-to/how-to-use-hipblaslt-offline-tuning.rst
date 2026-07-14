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

Solution indices are not portable across library releases or device architectures on their own -
the index is just a position in the solution library, and that position shifts whenever kernels are
added, removed, or reordered. To avoid losing tuning work on every rebuild or ROCm upgrade, tuning
files also record each winning kernel's stable solution name alongside its index (see
:ref:`self-healing-tuning-cache` below). When the recorded index no longer points at the tuned
kernel, hipBLASLt looks it up by name instead of silently using whichever kernel now occupies that
index, and falls back to the default kernel selection for that specific problem size if the tuned
kernel is no longer available at all - without disabling the rest of the tuning file.

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

.. _self-healing-tuning-cache:

Tuning files survive rebuilds automatically
============================================

Tuning files written by a current ``hipblaslt-bench`` include a ``solution_name`` column
in addition to ``solution_index``. ``solution_name`` is a stable identifier for the tuned kernel
that stays the same as long as that kernel (with the same tuning parameters) still exists in the
library, even if a rebuild changes its numeric index.

When ``HIPBLASLT_TUNING_OVERRIDE_FILE`` is loaded, hipBLASLt uses the stored index as a fast-path
guess, then confirms it still names the tuned kernel before trusting it. If the index has shifted,
hipBLASLt looks the kernel up by ``solution_name`` among the kernels the default heuristic would
already consider for that problem size, and transparently uses its new index. If the tuned kernel
is no longer present in the library at all, that specific problem size falls back to the default
kernel selection - other entries in the same tuning file are unaffected.

In practice this means you do not need to regenerate your tuning file after every hipBLASLt build or
ROCm upgrade: entries whose tuned kernel is still selectable by the default kernel-selection logic
keep working (healed automatically if their index moved), and only entries whose tuned kernel is no
longer selectable at all lose their tuning and fall back to the default choice, which you can address
by re-tuning just that shape if desired.

Healing looks the tuned kernel up among the same candidates the default kernel selection would
already consider for that problem size - it does not scan the full solution library. This keeps
healing cheap, but it means a kernel that offline tuning found only through its exhaustive benchmark
sweep (rather than one the default selection logic would also have proposed on its own) may not be
found by healing after that kernel's index changes, even though the tuning file still records the
right identifier. When that happens, hipBLASLt falls back to the default kernel choice for that shape
rather than an incorrect one - the same safety guarantee as for kernels that can be healed - but you
will need to re-tune that specific shape to recover its previous performance.

Tuning files written by older hipBLASLt releases (without the ``solution_name`` column) continue to
be accepted, but their entries have no stable identifier to heal by: they are trusted only when the
running build's git version still matches the version recorded when the file was written. As soon as
that version differs - i.e. after any rebuild or ROCm upgrade - every entry without a
``solution_name`` falls back to the default kernel selection, the same fail-safe behavior these files
had before self-healing existed. Recreating the tuning file with a current ``hipblaslt-bench`` is
recommended so that its entries get ``solution_name`` and gain per-entry self-healing instead.
