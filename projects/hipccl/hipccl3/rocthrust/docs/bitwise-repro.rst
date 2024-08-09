.. meta::
  :description: rocThrust documentation and API reference
  :keywords: rocThrust, ROCm, API, reference, data type, support

.. _bitwise-repro:

***********************
Bitwise reproducibility
***********************

The default device execution policy, ``thrust::device`` (``thrust::hip::par``) does not guarantee run-to-run bitwise reproducibility: that is, given identical inputs and device, the function will return the exact same results in repeated invocations. In practise, all rocThrust API functions are bitwise reproducible, with exception of the following:

* scan (inclusive & exclusive)
* scan_by_key (inclusive & exclusive)
* reduce_by_key
* transform_scan

In particular, the above operations are only bitwise reprodicible for **associative scan and reduce operators**. Notably, this does not include the pseudo-associative floating point operators.

An alternative version of the above operations that *is* bitwise reproducible with non-associative operators may be selected by using the *deterministic parallel* execution policy, ``thrust::hip::par_det``. Note that this implies a performance overhead, required to ensure that the results are run-to-run reproducible. There is no automatic detection for operator and input type pairs for which the default execution policy, that is ``thrust::hip::par``, is already bitwise reproducible. It is advised to only use ``thrust::hip::par_det`` for non-associative operators. ``thrust::hip::par_det`` may also be used with any of the other rocThrust API functions which are already bitwise reprodicible. In this case the behavior is the same as ``thrust::hip::par``.

=====
Tests
=====
To run the bitwise reproduciblity tests, you'll need to build the reproducibility.hip target. 
This target provides bitwise reproduciblity test coverage in two forms:

1. The first form runs tests by issuing multiple calls to the bitwise-reproducible versions of the algorithms mentioned in the section above using the deterministic parallel execution policy.
A special scan operator that inserts a random amount of delay into calculations is used to create variation in the internal timing of operations within the algorithm.
We then check to make sure the results for each call are the same. In this approach, calls are all issued within a single run of the test program.

2. The second form tests bitwise reproducibility across runs of the test program. On the initial run, information about the calls being made to the deterministic algorithms (all inputs and outputs)
is stored in a database file. On subsequent runs, when a deterministic algorithm is called, we look for an corresponding entry in the database (a call to the same algorithm with the same inputs that
produced the same output) and, if such an entry is found, the test succeeds. If no entry is found, the test fails.

Because the second form of the tests requires disk accesses, it can be very time consuming to run. For this reason, it is disabled by default. To enable it, define an environment variable called
``ROCTHRUST_BWR_PATH`` and set it to the path to the database file (or the path where you'd like it created if it doesn't already exist).

It is also necessary to distinguish between the initial run (in which information about calls is inserted into the database), and subsequent runs (in which the output of calls is compared
against existing entries in the database). You can use the ``ROCTHRUST_BWR_GENERATE`` environment variable to do this.
A value of:

* ``1`` indicates that this is the inital test run, and information about calls should be inserted into the database. In this mode, bitwise reproducibility tests will not fail.
* ``0`` (or if the variable is undefined) indicates that this is a subsequent run, and the results of calls should be compared to existing database entries. In this mode, no information is inserted into the database, and tests will fail if no matching database entry is found.

Note that bitwise reproduciblity is only guarenteed within a given combination of ROCm version, rocThrust version, and GPU architecture.
This means that if any of these factors changes, additional database entries need to be generated. To do this, you can run the tests with ``ROCTHRUST_GENERATE=1`` a second time and the database will append additional entries for the new environment.

For example, suppose we are running tests on gfx1030. On the first run, we use the environment variables like this to generate the database file:

``ROCTHRUST_BWR_PATH=/path/to/repro.db ROCTHRUST_BWR_GENERATE=1 reproducibility.hip``

As long as the ROCm version, rocThrust version, and GPU architecture remain the same, we can now run the tests using the database file like this:

``ROCTHRUST_BWR_PATH=/path/to/repro.db reproducibility.hip``

If one or more of the three factors changes - suppose we now want to run on gfx1100 - using the same database file, we must do another inital run with ``ROCTHRUST_BWR_GENERATE=1`` to append new entries to the database for the new environment:

``ROCTHRUST_BWR_PATH=/path/to/repro.db ROCTHRUST_BWR_GENERATE=1 reproducibility.hip``

After that we can test in the same manner as before:

``ROCTHRUST_BWR_PATH=/path/to/repro.db reproducibility.hip``