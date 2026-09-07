.. meta::
   :description: How to validate benchmark output in Primbench using correctness assertions, placeholder values, floating-point tolerances, and input restoration.
   :keywords: Primbench, ROCm, benchmark, validation, correctness, PRIMBENCH_ASSERT, state::test, GPU

**************************
Validate benchmark output
**************************

Primbench includes built-in correctness checking so you can verify that your GPU kernel produces the expected results alongside performance measurements.

Register a correctness test
============================

Call ``state.test()`` inside your ``run()`` method and pass a lambda that contains your validation logic. Primbench calls this lambda once during warmup, before timed iterations begin.

.. code:: cpp

   state.test(
       [&]
       {
           std::vector<T> h_output(3);
           PRIMBENCH_CHECK(
               hipMemcpy(h_output.data(), d_output, 3 * sizeof(T), hipMemcpyDeviceToHost));

           PRIMBENCH_ASSERT(h_output, {0, 1, 2});
       });

Because the test runs after the warmup kernel launch, the GPU output buffers already contain results that you can copy back to the host for inspection.

Use the ``PRIMBENCH_ASSERT`` macro to compare actual values against expected values. It works with:

- Scalar arithmetic types: Compares a single value against an expected scalar.
- Iterable containers: Compares element-wise against another container or a brace-enclosed initializer list.

On mismatch, ``PRIMBENCH_ASSERT`` prints the file name, line number, and the mismatched values to ``stderr``, then exits the program.

When containers differ in size, the assertion reports the size mismatch. When an element differs, the assertion reports the index and the mismatching values.


When you don't yet know the correct output values, pass placeholder values such as ``{0, 0, 0}``. When the assertion fails, Primbench prints the actual values to ``stderr``. You can then copy those values back into your test as the correct expectations. For example:

.. code:: cpp

   PRIMBENCH_ASSERT(h_output, {0, 0, 0});


For floating-point types, including ``__half``, pass a tolerance as the third argument to ``PRIMBENCH_ASSERT``. The tolerance defaults to ``0.0``. When the absolute difference between a pair of values exceeds the tolerance, the assertion fails and reports the difference alongside the tolerance. For example:

.. code:: cpp

   std::vector<float> h_output(3);
   PRIMBENCH_ASSERT(h_output, {1.0f, 2.0f, 3.0f}, 1e-5);

The tolerance also works for scalar comparisons:

.. code:: cpp

   double result = 3.14159;
   PRIMBENCH_ASSERT(result, 3.14159, 1e-10);

To skip all correctness checks, such as in production benchmark runs where only throughput matters, compile with the ``PRIMBENCH_NO_TEST`` preprocessor define:

.. code:: shell

   hipcc -o my_benchmark my_benchmark.cpp -I. -DPRIMBENCH_NO_TEST -lamd_smi

When ``PRIMBENCH_NO_TEST`` is defined, ``state.test()`` lambdas aren't executed.


If your kernel modifies its input data in place, subsequent kernel calls during timed iterations operate on already-mutated data, which can produce incorrect results. Use ``state.run_before_every_iteration()`` to register a lambda that restores the input before each kernel call.

This registration is independent of correctness testing. It keeps every timed iteration deterministic regardless of whether ``state.test()`` is used.
