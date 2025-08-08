.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _tuning:

********************************************************************
rocPRIM Performance tuning
********************************************************************

Algorithms often perform better if their launch parameters (number of blocks, block size, items per thread, etc.) are tailored for the particular architecture they are run on.
rocPRIM achieves this by passing structs called configs to algorithms as template parameters. A config struct encapsulates all the information that's needed to run a particular algorithm in the most performant way for a specific device. Default configurations (non custom-defined by users) can be selected by device code at compile time, since the architecture is known, while device algorithms detect at runtime which configuration should be used.

What we call *autotuning* is a method of generating the above-mentioned architecture-optimized default configurations for algorithms. The process to run the autotune is described below, as well as all the templates and scripts used.

1. Configure the project for autotuning. Autotune is an extension on top of the regular benchmarking process and it is enabled with a CMake option ``BENCHMARK_CONFIG_TUNING``, which doubles as a C++ macro to determine whether autotuning is enabled.
2. When the project is configured, a large amount of C++ benchmark files are generated with variation in parameters such as block size, items per thread, and method. The files are generated based on a template (``benchmark/benchmark_*.parallel.cpp.in``) and arguments defined in ``ConfigAutotuneSettings.cmake``. CMake will automatically detect when the input template changes and will reconfigure the required files as necessary.
3. Compile results in one executable based on all generated files for an algorithm.
4. Run the executable and gather the JSON output files. The generation of output files is triggered by the use of ``--benchmark_out=<output_file_name>.json`` when running the executable.
5. Convert the benchmark results into a config with ``scripts/autotune/create_optimization.py``. This python script injects the optimal configurations into the templates in ``scripts/autotune/templates``.

  * The option ``--out_basedir`` can be used to place the output config(s) in a specific path, otherwise the config(s) will be placed in the current directory.

6. If ``--out_basedir rocprim/include/device/detail/config`` was not used in the previous step, place the generated config(s) from the output path to ``rocprim/include/device/detail/config``.

Device-level algorithm dependencies
===================================

Due to the modularity of rocPRIM, some device-level algorithms depend on other device-level algorithms. The implication is that when an algorithm is changed, the performance of algorithms that depend on it must be checked as well. This also applies to configuration tuning. Below is a list of device-level algorithm dependencies and additional considerations for the tuning of these algorithms.

* ``lower_bound``, ``upper_bound``, and ``binary_search`` depend on ``transform``. However, all these algorithms are all tuned separately.
* ``merge_sort`` has two stages that are tuned separately: ``merge_sort_block_sort`` and ``merge_sort_block_merge``. Since the latter algorithm depends on the sorted block size, the best ``merge_sort`` configurations are obtained by tuning ``merge_sort_block_sort`` first, adding the configurations, and then tuning ``merge_sort_block_merge``.
* ``partition_two_way``, ``partition``, ``partition_three_way``, ``select``, ``unique``, and ``unique_by_key`` all use the same underlying implementation. However, all these algorithms have separate tuning.
* ``radix_sort`` has three sub-algorithms: ``radix_sort_block_sort``, ``radix_sort_onesweep``, and a merge sort for small sizes. ``radix_sort_block_sort`` and ``radix_sort_onesweep`` are tuned. The threshold for radix sort that determines whether to perform merge sort or onesweep is manually set.
* ``segmented_radix_sort`` depends on ``partition`` and ``partition_three_way`` but does not use the tuned configurations.
* ``segmented_reduce`` does not depend on ``reduce`` but uses the same tuned configurations.
* ``segmented_scan`` does not depend on ``scan`` but uses the same tuned configurations.
* ``run_length_encode`` depends on ``reduce_by_key``, but uses separate tuned configurations. ``run_length_encode_non_trivial_runs`` depends on ``reduce_by_key`` and ``select``, but does not use the tuned configurations from ``reduce_by_key`` and ``select``.
