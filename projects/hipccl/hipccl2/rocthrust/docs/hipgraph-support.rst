:orphan:

.. meta::
    :description: rocThrust documentation and API reference
    :keywords: rocThrust, ROCm, API, reference, hipGraph

.. _hipgraph-support:

******************************************
hipGraph Support
******************************************
Currently, rocThrust does not support the use of ``hipGraphs``. ``hipGraphs`` are not allowed to contain any synchronous
function calls or barriers. Thrust API functions are blocking (synchronous with respect to the host) by default.

Thrust does provide asynchronous versions of a number of algorithms. These are contained in the ``thrust::async`` namespace
(see the headers in ``rocThrust/thrust/async/``). However, these algorithms operate asynchronously by returning futures.
This approach is different from the form of asynchronous execution required within ``hipGraphs``, which must be achieved by
issuing calls into a user-defined ``hipStream``.

While it is possible to create execution policies that encourage Thrust API algorithms to execute within a user-defined stream,
(eg. ``thrust::hip_rocprim::par.on(stream)``), this is not enough to guarentee that synchronization will not occur within
a given algorithm. This is because some Thrust functions require execution policies to be passed in at compile time (via template
arguments) rather than at runtime. Since streams must be created at runtime, there is no way to pass these functions a stream.
Adding a stream argument to such functions breaks compatibility with the Thrust API.

For these reasons, we recommend that you do not use hipGraphs together with rocThrust code.