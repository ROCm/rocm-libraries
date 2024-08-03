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
