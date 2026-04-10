.. meta::
  :description: rocThrust documentation and API reference
  :keywords: rocThrust, ROCm, API, reference

.. _bitwise-repro:

************************************
rocThrust bitwise reproducibility
************************************

Not all rocThrust functions are bitwise reproducible under the ``thrust::hip::par`` policy. The following functions are bitwise reproducible for associative operators but not for pseudo-associative floating point operators: 

* ``inclusive_scan``
* ``exclusive_scan``
* ``inclusive_scan_by_key``
* ``exclusive_scan_by_key``
* ``transform_inclusive_scan``
* ``transform_exclusive_scan``
* ``reduce_by_key``

.. note:: 

    Under the :doc:`HIP backend <../how-to/rocThrust-build-for-backends>`, the default ``thrust::device`` policy is an alias for the ``thrust::hip::par`` policy.

Bitwise reproducible versions of these functions for pseudo-associative floating point operators are available under the ``thrust::hip::par_det`` deterministic parallel execution policy. 

When using these functions with ``thrust::hip::par_det``, operations are forced to complete in a fixed order. This ensures that the result is bitwise-identical on every run. Because of this overhead, the ``thrust::hip::par_det`` policy should only be used with pseudo-associative floating point operators.  The ``thrust::hip::par`` policy should be used otherwise.

.. note::

    The behavior of other bitwise reproducible functions under the ``thrust::hip::par_det`` policy will be identical to their behavior under the ``thrust::hip::par`` policy.

To run bitwise reproducibility tests, first build the ``reproducibility.hip`` target:

.. code:: shell

       cmake --build build --target reproducibility.hip

.. note::

    rocThrust must have been built with ``-DBUILD_TEST=ON`` to build ``reproducibility.hip``.

This target tests bitwise reproducibility either by issuing multiple calls to the functions or by running multiple iterations of the same test.

In the first case, where multiple calls are made, a special scan operator inserts a random amount of delay into calculations to create variations in the internal timing of operations. The test then verifies that the results for each call are the same. All calls are issued within a single run of the test program.

In the second case, several test runs are performed and compared. On the first run, the test stores input-output pairs for each function in a database file. In subsequent runs, the test compares the input-output pairs to those in the database. If identical pairs for a function are found, the test has succeeded. If no matching pair is found, the test has failed.

On the first run, set ``ROCTHRUST_BWR_PATH`` to the database file path and ``ROCTHRUST_BWR_GENERATE`` to 1. This will create the database file and populate it with input-output pairs.

.. code:: shell

    ROCTHRUST_BWR_PATH=/path/to/repro.db ROCTHRUST_BWR_GENERATE=1 reproducibility.hip

On subsequent iterations, point  ``ROCTHRUST_BWR_PATH`` to the database file, and set ``ROCTHRUST_BWR_GENERATE`` to 0 or leave it undefined. This will compare the results of the run with the values in the database.
    
.. code:: shell

    ROCTHRUST_BWR_PATH=/path/to/repro.db ROCTHRUST_BWR_GENERATE=0 reproducibility.hip

If the ROCm version, rocThrust version, or the GPU architecture changes, the test needs to be reset. Set ``ROCTHRUST_BWR_PATH`` to point to a new database file and set ``ROCTHRUST_BWR_GENERATE`` to 1 to run the initial test and and populate the database file with new data. After running the initial test, run the subsequent tests as usual.

