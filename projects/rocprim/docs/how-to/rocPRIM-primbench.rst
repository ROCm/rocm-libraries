.. meta::
  :description: Using primbench with rocPRIM
  :keywords: ROCm libraries, rocPRIM, ROCm, benchmarking, tools

*****************************
Benchmarking with primbench 
*****************************

primbench is a single-header `HIP <https://rocm.docs.amd.com/projects/HIP/en/latest/index.html>`_ benchmarking library for rocPRIM that outputs detailed benchmarking information in JSON format or, optionally, in CSV format.

primbench requires `HIP <https://rocm.docs.amd.com/projects/HIP/en/latest/index.html>`_, `AMD SMI <https://rocm.docs.amd.com/projects/amdsmi/en/latest/index.html>`_, and C++17 or later. 

To use primbench, import |primbench.hpp|_ into your benchmarking code.

Use ``PRIMBENCH_REGISTER_TYPE`` to register a name for each variable type that will be benchmarked. This name is used to identify the type in the output. 

For example, in |copy_benchmark.cpp|_ the ``char`` and ``long long`` types are given the names "char" and "long long", respectively:

.. code:: cpp

    PRIMBENCH_REGISTER_TYPE(char, "char")
    PRIMBENCH_REGISTER_TYPE(long long, "long long")

Registering also lets you provide alternate names for your types. For example, you could register ``long long`` to "longx2":

.. code:: cpp

    PRIMBENCH_REGISTER_TYPE(long long, "longx2")

You will need to define the algorithm to benchmark. This will be passed to ``state.run()`` in the implementation of the ``primbench::benchmark_interface`` ``run()`` function.

Only one algorithm can be benchmarked.

For example, ``copy_benchmark.cpp`` benchmarks a copy algorithm:

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

Both the ``meta()`` and ``run()`` functions in ``primbench::benchmark_interface`` must be implemented. 

The ``meta`` function returns metadata as a JSON object. 

The returned JSON object must include a value for the ``algo`` key. The ``algo`` key sets the name of the algorithm being benchmarked. 

For example, from ``copy_benchmark.cpp``:

.. code:: cpp

  template<typename T>
  struct copy_benchmark : public primbench::benchmark_interface
  {
    primbench::json meta() const override
    {
      return primbench::json{}.add("algo", "copy").add("type", primbench::name<T>());
    }

The ``run()`` function runs the benchmark. ``run()`` must include a call to ``state.set_items()``. ``state.set_items()`` sets the number of items processed per iteration. The number of items must be greater than 0.

The ``state`` class saves the state of the benchmarking run, including the number of reads and writes. 

Depending on the algorithm being benchmarked, ``run()`` must call ``state.add_reads()``, ``state.add_writes()``, or both. These functions are used to calculate the number of items or bytes processed per second.

``set_items()`` must be called before ``add_reads()``, and ``add_reads()`` must be called before ``add_writes()``.  ``state.run()`` is called after both ``add_writes()`` and ``add_reads()``. 

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

The ``executor`` class is instantiated in ``main()``. Benchmark settings and flags can be passed to the constructor as optional parameters.

For more information on settings, see `the primbench README file <https://github.com/ROCm/rocm-libraries/blob/develop/shared/primbench/README.md#passing-settings-programmatically>`_.

Call ``executor.queue()`` for each specialization of the benchmark. When ``executor.run()`` is called, the queued benchmark specializations will be run in alphabetical order.

For example, from ``copy_benchmark.cpp``:

.. code:: cpp

  int main(int argc, char* argv[])
  {
    primbench::executor executor(argc, argv);

    executor.queue<copy_benchmark<char>>();
    executor.queue<copy_benchmark<long long>>();

    executor.run();
  }

Compile the benchmark using hipcc. For example:

.. code:: shell

  hipcc -o copy_benchmark copy_benchmark.cpp -lamd_smi
  ./copy_benchmark
  
For the complete list of command line options, see `the primbench README file <https://github.com/ROCm/rocm-libraries/blob/develop/shared/primbench/README.md#command-line-options>`_.

The output will be written to the terminal and to ``results.json``.

.. |primbench.hpp| replace:: ``primbench.hpp``
.. _primbench.hpp: https://github.com/ROCm/rocm-libraries/blob/develop/shared/primbench/primbench.hpp

.. |copy_benchmark.cpp| replace:: ``copy_benchmark.cpp``
.. _copy_benchmark.cpp: https://github.com/ROCm/rocm-libraries/blob/develop/shared/primbench/examples/hip/copy_benchmark.cpp