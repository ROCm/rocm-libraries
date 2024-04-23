.. meta::
  :description: rocThrust documentation and API reference
  :keywords: rocThrust, ROCm, API, reference, data type, support
  
.. _bitwise-repro:

******************************************
Bitwise reproducibility
******************************************

With the exception of the following functions, all rocThrust API functions are bitwise reproducible.  That is, given identical inputs, the function will return the exact same result in repeated invocations.

* scan (inclusive & exclusive)
* scan_by_key (inclusive & exclusive)
* reduce_by_key
* transform_scan
