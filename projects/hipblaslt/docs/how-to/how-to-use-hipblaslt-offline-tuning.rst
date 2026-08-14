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
only the shapes that failed need re-tuning. Files written by an older hipBLASLt may carry a
``solution_name`` instead, which is validated the same way, and files older still carry no name at
all; those last are used only when the file was produced by the running build.

Runtime tuning
==============

Instead of tuning offline with ``hipblaslt-bench``, hipBLASLt can benchmark candidate kernels itself
the first time it sees a GEMM shape and remember the winner. This is off by default and is enabled
with two environment variables:

.. code-block:: bash

   export HIPBLASLT_TUNING_MODE=<off|cache|tune>
   export HIPBLASLT_TUNING_CACHE_PATH=<file_name>

``tune`` benchmarks supported candidates for each shape it has not seen before, at that shape's first
matmul, and appends the winner to the cache file. ``cache`` only replays what the file already
contains, validating each entry as described above. ``off`` is the default and changes nothing.

The cache file uses the same header/value-row format as an offline tuning file, so a file produced by
one can be read by the other. Runtime-tuned rows include a schema version and use the complete problem
key. Rows produced by the current ``hipblaslt-bench`` writer have no schema version and retain the
historical matching behavior: they distinguish transpose, shape and the principal datatypes, but not
leading dimensions, batch strides, epilogue details or device identity. Do not put offline-tuned
problems that differ only in one of those omitted fields in the same file.

``HIPBLASLT_TUNING_CACHE_PATH`` and ``HIPBLASLT_TUNING_OVERRIDE_FILE`` are mutually exclusive; set
only one.

A typical run writes the cache once and reuses it afterwards:

.. code-block:: bash

   export HIPBLASLT_TUNING_CACHE_PATH=tuning.txt
   HIPBLASLT_TUNING_MODE=tune ./your_application    # benchmark and record
   HIPBLASLT_TUNING_MODE=cache ./your_application   # replay only

The fastest measured candidate wins outright, with no minimum improvement over the default choice,
matching how ``hipblaslt-bench`` selects. The benchmarking effort can be adjusted:

.. code-block:: bash

   export HIPBLASLT_TUNING_ALL_KERNELS=<0|1>            (Default value is: 1)
   export HIPBLASLT_TUNING_MAX_CANDIDATES=<value>       (Default value is: 128)
   export HIPBLASLT_TUNING_COLD_ITERS=<value>           (Default value is: 1000)
   export HIPBLASLT_TUNING_HOT_ITERS=<value>            (Default value is: 1000)
   export HIPBLASLT_TUNING_ROTATING_MB=<value>          (Default value is: 512)
   export HIPBLASLT_TUNING_FLUSH_ICACHE=<0|1>           (Default value is: 1)
   export HIPBLASLT_TUNING_BUDGET_MS_PER_SHAPE=<value>  (Default value is: 0, no limit)

Each candidate is measured the way ``hipblaslt-bench`` measures one: ``HIPBLASLT_TUNING_COLD_ITERS``
untimed launches, then ``HIPBLASLT_TUNING_HOT_ITERS`` launches timed as a single span and reported as
the per-launch mean. Every candidate gets the full measurement, so there is no cheap ranking round
that could discard the real winner before it has been timed properly.

Candidates are measured against rotating copies of the tensors, so that successive launches do not
read and write the same cache lines. Without this every candidate is timed with its inputs already
resident, which is not how a production GEMM runs and which measurably changes which kernel appears
fastest. ``HIPBLASLT_TUNING_ROTATING_MB`` is the memory the rotation may use, and 0 disables it.

Rotation is declined for problems whose buffer sizes cannot be established from the problem
description alone, such as broadcast inputs with a zero batch stride and swizzled ``A`` or ``B``.
Those shapes are still tuned, just with their inputs cache-resident. Very small problems rotate over
fewer blocks than the memory budget would allow, since each block costs setup work per candidate.

The instruction cache is invalidated between timed launches, the same way and with the same kernel
the bench client uses, so a candidate is not timed with its own code already resident. It costs
roughly 5% of tuning time and measurably improves agreement with ``hipblaslt-bench``.
``HIPBLASLT_TUNING_FLUSH_ICACHE=0`` turns it off. The per-flush cost is measured once per device and
subtracted, so the recorded time stays the GEMM time and remains comparable to entries tuned without
it.

Where a shape's leading kernels are separated by more than measurement noise, this is enough to make
tuning land on the same kernel run after run. Where they are within a couple of percent of each
other, the choice stays unstable and tuning may pick a kernel marginally slower than the heuristic
default; that is a property of the measurement, not of the cache, and ``hipblaslt-bench`` behaves the
same way on those shapes.

By default every kernel that can run the problem is measured, not just the ones the selection
heuristic ranks highest, because the fastest kernel is not always inside that ranked prefix. Setting
``HIPBLASLT_TUNING_ALL_KERNELS`` to 0 restricts the search to the top ``HIPBLASLT_TUNING_MAX_CANDIDATES``
ranked kernels instead, which is faster but can miss the best one.

Tuning one shape this thoroughly takes time. An exhaustive search over several hundred candidates
takes minutes, and it is paid once per shape in the first process that runs with ``tune``.
``HIPBLASLT_TUNING_BUDGET_MS_PER_SHAPE`` puts a wall-clock ceiling on one shape, covering enumeration
and setup as well as the timed measurements. A search the ceiling cuts short is discarded rather than
recorded: candidates are not measured in order of expected performance, so the best of an arbitrary
prefix is not the best kernel, and caching it would freeze that arbitrary choice in place. Use the
ceiling to bound how long tuning may run, not as a way to tune faster; to tune faster, reduce the
candidate list or the iteration counts.

The ceiling is checked between candidates, since a batch of launches already submitted cannot be
recalled. One candidate may therefore overrun it by as much as its own measurement takes. A search
that measures every candidate is kept however long it took, because a complete search gives the right
answer regardless of its duration.

Runtime tuning is close to how ``hipblaslt-bench`` measures, but not identical, so the two can
disagree between kernels that are within noise of each other. Runtime tuning caps how many rotation
blocks it uses, leaves the scale vectors and a forward bias unrotated, and enumerates candidates
through a different Tensile entry point. Both flush the instruction cache. Runtime tuning always
times with HIP events; ``hipblaslt-bench`` does the same only with ``--use_gpu_timer`` and otherwise
uses a synchronized CPU timer. Treat ``hipblaslt-bench`` as the reference when a shape matters
enough to check by hand.

Benchmarking never writes to your buffers. Candidates run against memory the library allocates for
the purpose, and only the winner runs on the real output.

Limitations
-----------

* Tuning happens on the C API execution path. Callers of the C++ extension API can replay a cache but do not tune.
* Grouped GEMM, pointer-array batch, RocRoller and HIP graph capture are excluded from tuning.
* An uncached in-place problem where ``C`` and ``D`` alias and ``beta`` is nonzero runs with normal selection instead of being tuned. Repeated benchmark launches would otherwise overwrite and reuse C rather than measuring the caller's input.
* One process should write a given cache file at a time.
* ``alpha``, ``beta`` and whether ``C`` and ``D`` alias are not part of the lookup key, so an entry tuned at one value of ``beta`` can serve a caller using another.
* Passing ``algo=nullptr`` to ``hipblasLtMatmul`` does not replay cached winners.
