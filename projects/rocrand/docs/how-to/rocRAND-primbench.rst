.. meta::
  :description: Using primbench with rocRAND
  :keywords: ROCm libraries, rocRAND, ROCm, benchmarking, tools

*****************************
Benchmarking with primbench
*****************************

primbench is a single-header `HIP <https://rocm.docs.amd.com/projects/HIP/en/latest/index.html>`_ benchmarking library for rocRAND that outputs detailed benchmarking information in JSON format or, optionally, in CSV format.

primbench requires `HIP <https://rocm.docs.amd.com/projects/HIP/en/latest/index.html>`_ and C++17 or later. `AMD SMI <https://rocm.docs.amd.com/projects/amdsmi/en/latest/index.html>`_ is required for temperature monitoring and control.

.. note::

  Because AMD SMI is only supported on Linux, temperature monitoring and control isn't available on Windows.

To use primbench, import |primbench.hpp|_ into your benchmarking code.

Use ``PRIMBENCH_REGISTER_TYPE`` to register a name for each variable type, or specialization, that will be benchmarked. This name is used to identify the type in the output.

For example, in |copy_benchmark.cpp|_ the ``char`` and ``long long`` types are given the names "char" and "long long", respectively:

.. code:: cpp

  PRIMBENCH_REGISTER_TYPE(char, "char")
  PRIMBENCH_REGISTER_TYPE(long long, "long long")

Registering also lets you provide alternate names for your types. For example, you could register ``long long`` as "longx2":

.. code:: cpp

  PRIMBENCH_REGISTER_TYPE(long long, "longx2")

Both the ``meta()`` and ``run()`` functions in ``primbench::benchmark_interface`` must be implemented.

The ``meta`` function returns metadata as a JSON object.

The returned JSON object must include a value for the ``algo`` key. The ``algo`` key sets the name of the algorithm being benchmarked. This will be the name used in the JSON output.

For example, from ``copy_benchmark.cpp``:

.. code:: cpp

  template<typename T>
  struct copy_benchmark : public primbench::benchmark_interface
  {
    primbench::json meta() const override
    {
      return primbench::json{}.add("algo", "copy").add("type", primbench::name<T>());
    }

You will need to define the algorithm to benchmark. This will be passed to ``state.run()`` in the implementation of the ``primbench::benchmark_interface`` ``run()`` function.

For example, the ``copy_kernel`` algorithm is defined in ``copy_benchmark.cpp``:

.. code:: cpp

  template<typename T, unsigned int BlockSize, unsigned int ItemsPerThread>
  __global__ __launch_bounds__(BlockSize)
  void copy_kernel(const T* input, T* output)
  {
    unsigned int idx = threadIdx.x + blockIdx.x * BlockSize * ItemsPerThread;
    #pragma unroll
    for(unsigned int i = 0; i < ItemsPerThread; ++i)
        output[idx + i * BlockSize] = input[idx + i * BlockSize];
  }

The ``run()`` function runs the benchmark. ``run()`` must include a call to ``state.set_items()``. ``state.set_items()`` sets the number of items processed per iteration. The number of items must be greater than 0.

The ``state`` class saves the state of the benchmarking run, including the number of reads and writes.

Depending on the algorithm being benchmarked, ``run()`` may call ``state.add_reads()``, ``state.add_writes()``, or both. These functions are used to calculate the number of items or bytes processed per second.

``set_items()`` must be called before ``add_reads()`` or ``add_writes()``. If you call both, call ``add_reads()`` before ``add_writes()``. Call ``state.run()`` after ``set_items()`` and any read or write counters you need.

For example, from ``copy_benchmark.cpp``:

.. code:: cpp

  state.set_items(items);
  state.add_reads<T>(items);
  state.add_writes<T>(items);

The kernel call is wrapped in a lambda and passed to ``state.run()``. ``state.run()`` will run the kernel as many times as required.

.. code:: cpp

  state.run(
          [&] {
              copy_kernel<T, BlockSize, ItemsPerThread>
                  <<<grid, block, 0, stream>>>(d_input, d_output);
          });

Benchmark settings and flags can be passed to the ``executor`` class constructor as optional parameters. The ``executor`` queues and runs the benchmarks.

For more information on settings, see `the primbench README file <https://github.com/ROCm/rocm-libraries/blob/develop/shared/primbench/README.md#passing-settings-programmatically>`_.

``executor.queue()`` is called to queue the benchmark for each specialization. When ``executor.run()`` is called, the queued benchmark specializations will be run in alphabetical order.

For example, from ``copy_benchmark.cpp``:

.. code:: cpp

  int main(int argc, char* argv[])
  {
    primbench::executor executor(argc, argv);

    executor.queue<copy_benchmark<char>>();
    executor.queue<copy_benchmark<long long>>();

    executor.run();
  }

Compile the benchmark using hipcc. For example, on Linux:

.. code:: shell

  hipcc -o copy_benchmark copy_benchmark.cpp -lamd_smi
  ./copy_benchmark

And in the Windows PowerShell:

.. code:: shell

  hipcc -o copy_benchmark.exe examples/hip/copy_benchmark.cpp -I. -DPRIMBENCH_NO_MONITORING -std=c++17 -g --offload-arch=$(amdgpu-arch) ; ./copy_benchmark.exe

For the complete list of command-line options, see `the primbench README file <https://github.com/ROCm/rocm-libraries/blob/develop/shared/primbench/README.md#command-line-options>`_.

The output will be written to the terminal and to ``results.json``.

.. |primbench.hpp| replace:: ``primbench.hpp``
.. _primbench.hpp: https://github.com/ROCm/rocm-libraries/blob/develop/shared/primbench/primbench.hpp

.. |copy_benchmark.cpp| replace:: ``copy_benchmark.cpp``
.. _copy_benchmark.cpp: https://github.com/ROCm/rocm-libraries/blob/develop/shared/primbench/examples/hip/copy_benchmark.cpp
