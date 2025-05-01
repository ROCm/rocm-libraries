.. meta::
   :description: rocRAND documentation and API reference library
   :keywords: rocRAND, ROCm, API, documentation

.. _dynamic-ordering-configuration:

=============================================================
Kernel configurations for dynamic ordering
=============================================================

Overview
========

When dynamic ordering (``ROCRAND_ORDERING_PSEUDO_DYNAMIC``) is set, the number of blocks and threads launched on the GPU is selected such that it best accommodates the specific GPU model. As a consequence, the number of allocated generators and thereby the sequence of the generated numbers can also vary.

The tuning, i.e. the selection of the most performant configuration for each GPU architecture can be performed in an automated manner. The necessary tools and benchmarks for the tuning are provided in the rocRAND repository. In the following, the process of the tuning is described.

.. _tuning-benchmark-build:

Building the tuning benchmarks
==============================

The principle of the tuning is very simple: the random number generation kernel is run for a list of kernel block size / kernel grid size combinations, and the fastest combination is selected as the dynamic ordering configuration for the particular device. rocRAND provides an executable target that runs the benchmarks with all these combinations: `benchmark_rocrand_tuning`. This target is disabled by default, and can be enabled and built by the following snippet.

Use the `GPU_TARGETS` variable to specify the comma-separated list of GPU architectures to build the benchmarks for. To acquire the architecture of the GPU(s) installed, run `rocminfo`, and look for `gfx` in the "ISA Info" section. ::

    $ cd rocRAND
    $ cmake -S . -B ./build
        -D BUILD_BENCHMARK=ON
        -D BUILD_BENCHMARK_TUNING=ON
        -D CMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++
        -D GPU_TARGETS=gfx908
    $ cmake --build build --target benchmark_rocrand_tuning

Additionally, the following CMake cache variables control the generation of the benchmarked matrix:

========================================== ===============================================================
Variable name                              Explanation
========================================== ===============================================================
``BENCHMARK_TUNING_THREAD_OPTIONS``        Comma-separated list of benchmarked block sizes
``BENCHMARK_TUNING_BLOCK_OPTIONS``         Comma-separated list of benchmarked grid sizes
``BENCHMARK_TUNING_MIN_GRID_SIZE``         Configurations with fewer total number of threads are omitted
========================================== ===============================================================

Note, that currently the benchmark tuning is only supported for AMD GPUs. 

Using the number of multiprocessors as candidates
-------------------------------------------------

Multiples of the number of multiprocessors of the GPU at hand are good candidates for ``BENCHMARK_TUNING_BLOCK_OPTIONS``. Running `rocRAND/scripts/config-tuning/get_tuned_grid_sizes.py` executes `rocminfo` to acquire the number of multiprocessors, and prints a comma-separated list of grid size candidates to the standard output.

.. _tuning-benchmark-run:

Running the tuning benchmarks
=============================

When the `benchmark_rocrand_tuning` target is built, the benchmarks can be run and the results can be collected for further processing. Since the benchmarks run for a longer time period, it is crucial that the GPU in use is thermally stable, i.e. the cooling must be adequate enough to keep the GPU at the preset clock rates without throttling. Additionally, make sure that no other workload is dispatched on the GPU concurrently. Otherwise the resulting dynamic ordering configs might not be the optimal ones. The full benchmark suite can be run with the following command: ::

    $ cd ./build/benchmark/tuning
    $ ./benchmark_rocrand_tuning --benchmark_out_format=json --benchmark_out=rocrand_tuning_gfx908.json

This executes the benchmarks and saves the benchmark results into the JSON file at `rocrand_tuning_gfx908.json`. If only a subset of the benchmarks needs to be run, e.g. for a single generator, the `--benchmark_filter=<regex>` option can be used. For example: `--benchmark_filter=".*philox.*"`.

.. _tuning-benchmark-process:

Processing the benchmark results
================================

Once the benchmark results in JSON format from all architectures are present, the best configs are selected using the `rocRAND/scripts/config-tuning/select_best_config.py` script. Make sure that the prerequisite libraries are installed, by running ``pip install -r rocRAND/scripts/config-tuning/requirements.txt``.

Each rocRAND generator is capable of generating a multitude of output types and distributions. However, a single configuration is selected for each GPU architecture, which applies uniformly to all types and distributions. It is possible that the configuration that performs the best for one distribution is not the fastest for another. `select_best_config.py` selects the configuration that performs best **on average**. If, under the selected configuration, any type/distribution performs worse than ``ROCRAND_ORDERING_PSEUDO_DEFAULT``, a warning is printed to the standard output. The eventual decision about applying the configuration or not have to be made by the library's maintainers.

The main output of running `select_best_config.py` is a number of C++ header files that contain the definitions of the dynamic ordering config for the benchmarked architectures. These files are intended to be copied to the `rocRAND/library/src/rng/config` directory of the source tree to be checked in to the version control. The directory, to which the header files are written, can be specified with the `--out-dir` option.

To help humans comprehend the results, `select_best_config.py` can generate colorized diagrams to visually compare the performance of the configuration candidates. This can be invoked by passing the optional `--plot-out` argument, e.g. `--plot-out rocrand-tuning.svg`. This generates an SVG image for each GPU architecture the script has processed.

To put it all together, a potential invocation of the `select_best_config.py` script: ::

    $ ./rocRAND/scripts/config-tuning/select_best_config.py --plot-out ./rocrand-tuning.svg --out-dir ./rocRAND/library/src/rng/config/ ./rocRAND/build/benchmark/tuning/rocrand_tuning_gfx908.json ./rocRAND/build/benchmark/tuning/rocrand_tuning_gfx1030.json

Adding support for a new GPU architecture
=========================================

The intended audience of this section is the developer, who is adding support to rocRAND for a new GPU architecture.

1. The list of the recognized architectures are hard-coded in source file `library/src/rng/config_types.hpp`. The following symbols have to be updated accordingly:
    * Enum class ``target_arch`` - lists the recognized architectures as an enumeration.
    * Function ``get_device_arch`` - recognizes the device that we compile to in device code.
    * Function ``parse_gcn_arch`` - dispatches from the name of the architecture to the ``target_arch`` enum in host code.
2. The tuning benchmarks has to be compiled and run for the new architecture. See :ref:`tuning-benchmark-build` and :ref:`tuning-benchmark-run`.
3. The benchmark results have to be processed by the provided `select_best_config.py` script. See :ref:`tuning-benchmark-process`.
4. The resulting header files have to be merged with the ones that are checked in the version control in directory `rocRAND/library/src/rng/config`.
