.. meta::
  :description: rocThrust documentation and API reference
  :keywords: rocThrust, ROCm, API, reference

.. _bitwise-repro:

************************************
rocThrust bitwise reproducibility
************************************

When the HIP backend is used, the default ``thrust::device`` policy dispatches to the ``thrust::hip::par`` policy.

Not all rocThrust functions are bitwise reproducible under the ``thrust::hip::par`` policy. The following functions are bitwise reproducible for associative operators but not for for pseudo-associative floating point operators: 

* ``inclusive_scan``
* ``exclusive_scan``
* ``inclusive_scan_by_key``
* ``exclusive_scan_by_key``
* ``transform_inclusive_scan``
* ``transform_exclusive_scan``
* ``reduce_by_key``

Bitwise reproducible versions of these functions for pseudo-associative floating point operators are available under the ``thrust::hip::par_det`` deterministic parallel execution policy. 

When using these functions with ``thrust::hip::par_det``, their bitwise reproducibility must be verified regardless of the type of operator. Because of this overhead, these functions should only use ``thrust::hip::par`` with associative operators. 

.. note::

    The behavior of other bitwise reproducible functions under the ``thrust::hip::par_det`` policy will be identical to their behavior under the default policy, and they can be used under either policy without the need for testing.

To run the bitwise reproducibility tests, first build the ``reproducibility.hip`` target:

.. code:: shell

       cmake --build build --target reproducibility.hip

.. note::

    rocThrust must have been built with ``-DBUILD_TEST=ON`` to build ``reproducibility.hip``.

This target tests bitwise reproducibility either by issuing multiple calls to the functions or by running multiple iterations of the same test.

In the first case, where multiple calls are made, a special scan operator inserts a random amount of delay into calculations to create variations in the internal timing of operations. The test then verifies that the results for each call are the same. All calls are issued within a single run of the test program.

In the second case, several runs of the test are done and compared to each other. On the first run, the test stores all the input and output pairs for each function in a database file. On subsequent runs, the test compares the input and output pairs to those in the database. If identical pairs for a function are found, the test has succeeded. If no matching pair is found, the test has failed.

To enable this test, set ``ROCTHRUST_BWR_PATH`` to the path to the database file. Set ``ROCTHRUST_BWR_GENERATE`` to 1 on the first run of the test, and to 0 for all subsequent runs.

.. code:: shell

    ROCTHRUST_BWR_PATH=/path/to/repro.db ROCTHRUST_BWR_GENERATE=1 reproducibility.hip
    ROCTHRUST_BWR_PATH=/path/to/repro.db reproducibility.hip


A new first iteration of the test must be run if the ROCm version, rocThrust version, or GPU architecture changes.

