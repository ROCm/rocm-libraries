.. meta::
   :description: rocRAND documentation for dynamic ordering configuration
   :keywords: rocRAND, ROCm, API, documentation, dynamic ordering

.. _dynamic-ordering-configuration:

=============================================================
Kernel configurations for dynamic ordering
=============================================================

When dynamic ordering (``ROCRAND_ORDERING_PSEUDO_DYNAMIC``) is set, rocRAND selects the number of blocks and threads
to launch on the GPU to accommodate the specific GPU model best.
Consequently, the number of allocated generators and the sequence of the generated numbers can also vary.

The tuning, which is the selection of the most performant configuration for each GPU architecture,
can be performed in an automated manner. The necessary tools and benchmarks for the tuning are provided
in the rocRAND repository. The following sections provide additional details about the tuning process.

.. _tuning-benchmark-build:

Building the tuning benchmarks
==============================

The principle behind the tuning is straightforward. The random number generation kernel is run
for a list of kernel block size and kernel grid size combinations. The fastest combination
is then selected as the dynamic ordering configuration for that particular device.
rocRAND provides an executable target named ``benchmark_rocrand_tuning`` that runs the benchmarks with all these
combinations.

This target is disabled by default, but it can be enabled and built using the following snippet.
Use the ``GPU_TARGETS`` variable to specify a comma-separated list of GPU architectures to build the benchmarks for.
To determine the architecture of the installed GPU(s), run the ``rocminfo`` command
and look for ``gfx`` in the "ISA Info" section.

.. code-block:: shell

   cd rocm-libraries/projects/rocrand
   cmake -S . -B ./build
      -D BUILD_BENCHMARK=ON
      -D BUILD_BENCHMARK_TUNING=ON
      -D CMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++
      -D GPU_TARGETS=gfx908
   cmake --build build --target benchmark_rocrand_tuning

The following CMake cache variables control the generation of the benchmarked matrix:

========================================== ===============================================================
Variable name                              Explanation
========================================== ===============================================================
``BENCHMARK_TUNING_THREAD_OPTIONS``        Comma-separated list of benchmarked block sizes
``BENCHMARK_TUNING_BLOCK_OPTIONS``         Comma-separated list of benchmarked grid sizes
``BENCHMARK_TUNING_MIN_GRID_SIZE``         Configurations with fewer total threads are omitted
========================================== ===============================================================

.. note::

   The benchmark tuning is only supported for AMD GPUs. 

Using the number of multiprocessors as candidates
-------------------------------------------------

Multiples of the number of multiprocessors on the GPU being benchmarked are
good candidate values for ``BENCHMARK_TUNING_BLOCK_OPTIONS``. 
The ``rocm-libraries/projects/rocrand/scripts/config-tuning/get_tuned_grid_sizes.py`` executable
runs ``rocminfo`` to acquire the number of multiprocessors and prints a comma-separated list
of grid size candidates to the standard output.

.. _tuning-benchmark-run:

Running the tuning benchmarks
=============================

After building the ``benchmark_rocrand_tuning`` target, you can run the benchmarks
and collect the results for further processing.
The benchmarks can run for a long time, so it is crucial that the GPU in use is thermally stable.
For instance, there must be adequate cooling to keep the GPU at the preset clock rates without throttling.
Additionally, ensure that no other workload is concurrently dispatched to the GPU.
Otherwise, the resulting dynamic ordering configurations might not be the optimal ones.
Run the full benchmark suite using the following command:

.. code-block:: shell

   cd ./build/benchmark/tuning
   ./benchmark_rocrand_tuning --benchmark_out_format=json --benchmark_out=rocrand_tuning_gfx908.json

This executes the benchmarks and saves the benchmark results to the ``rocrand_tuning_gfx908.json`` JSON file.
To only run a subset of the benchmarks, such as for a single generator, use the ``--benchmark_filter=<regex>`` option,
for example, ``--benchmark_filter=".*philox.*"``.

.. _tuning-benchmark-process:

Processing the benchmark results
================================

After the benchmark results from all architectures in JSON format are available, the best configurations
are selected using the ``rocm-libraries/projects/rocrand/scripts/config-tuning/select_best_config.py`` script.
Ensure the prerequisite libraries are installed by running the following command:

.. code-block:: shell

   pip install -r rocm-libraries/projects/rocrand/scripts/config-tuning/requirements.txt.

Each rocRAND generator can generate a multitude of output types and distributions.
However, a single configuration is selected for each GPU architecture, which applies uniformly to all types
and distributions. It's possible that the best performing configuration for one distribution
isn't the fastest for another. ``select_best_config.py`` selects the configuration that performs best **on average**.
If any type or distribution performs worse than ``ROCRAND_ORDERING_PSEUDO_DEFAULT`` under the selected configuration,
a warning is printed to the standard output.
The eventual decision about whether to apply the configuration is made by the library's maintainers.

The ``select_best_config.py``  script produces a set of C++ header files as output
that contain the definitions of the dynamic ordering configuration for the benchmarked architectures.
These files are intended to be copied to the ``rocm-libraries/projects/rocrand/library/src/rng/config`` directory of the source tree
and checked in to the version control system. The directory where the header files are written to
can be specified using the ``--out-dir`` option.

For more readable results, ``select_best_config.py`` can generate colorized diagrams to visually
compare the performance of the configuration candidates. To select this option, use the
optional ``--plot-out`` argument, for example, ``--plot-out rocrand-tuning.svg``.
This generates an SVG image for each GPU architecture processed by the script.

The following invokation of the ``select_best_config.py`` script demonstrates all these options:

.. code-block:: shell

   ./rocm-libraries/projects/rocrand/scripts/config-tuning/select_best_config.py --plot-out ./rocrand-tuning.svg --out-dir ./rocm-libraries/projects/rocrand/library/src/rng/config/ ./rocm-libraries/projects/rocrand/build/benchmark/tuning/rocrand_tuning_gfx908.json ./rocm-libraries/projects/rocrand/build/benchmark/tuning/rocrand_tuning_gfx1030.json

Adding support for a new GPU architecture
=========================================

This section is intended for developers who want to add rocRAND support for a new GPU architecture.
To add support, follow this checklist:

#. Update the hard-coded list of recognized architectures in the ``library/src/rng/config_types.hpp`` file. The following symbols must be updated accordingly:

   *  Enum class ``target_arch``: Lists the recognized architectures as an enumeration.
   *  Function ``get_device_arch``: The device to compile to in the device code.
   *  Function ``parse_gcn_arch``: Translates from the name of the architecture to the ``target_arch`` enum in the host code.

#. The tuning benchmarks must be compiled and run for the new architecture. See :ref:`tuning-benchmark-build` and :ref:`tuning-benchmark-run`.
#. The benchmark results must be processed by the ``select_best_config.py`` script. See :ref:`tuning-benchmark-process`.
#. The resulting header files must be added to version control in the ``rocm-libraries/projects/rocrand/library/src/rng/config`` directory.
