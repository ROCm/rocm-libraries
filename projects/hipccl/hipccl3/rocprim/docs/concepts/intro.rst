.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _rocprim-intro:

********************************************************************
 Introduction to rocPRIM
********************************************************************

Operations and Sequences
========================

A rocPRIM operation is a computation over a sequence of objects. A rocPRIM operation can return a single value like the ``reduce`` operation; return another sequence like the ``sort`` operation; or return multiple sequences like the ``partition`` operation. The elements of the sequence could be of any type or class, although template specialization allows rocPRIM to optimize the computations over the usual numerical datatypes. Operations accept input in the form of ``iterators`` that point to a sequence of objects to process, and write output to a mutable ``iterator``.

A high level view of the available operations can be found on :ref:`ops-summary`. rocPRIM includes a variety of generic operations that are frequently very useful.

.. note::
  Refer to :ref:`data-type-support` for information on supported datatypes.

Scope
======

An important property of a rocPRIM operation is its scope, which determines the level of the computing model used for processing the operation. The scope determines which parts of the GPU will cooperate to compute the result. The scope has a direct influence on how the data will be subdivided and processed by the computing units or VALUs. The rocPRIM operation scopes are:  

* *Device/Grid* the operation and data will be split and dispatched to all the CUs.
* :term:`Block` The operation should take place within the same block by the same CU.
* :term:`Warp` as above but with a warp and a VALU.
* *Thread* The operation will take place sequentially in the same thread. Thread-wide operations are also called *Utilities* since they coincide with utility functions used on a CPU.

The scope has an impact on how the operation is initiated:

* *Device/Grid* it is a kernel, thus it is dispatched with its own grid/block dimensions.
* *Block/Warp/Thread* it is a function call, and inherits the dimensions of the current kernel.

This also dictates how synchronization should be done to wait for completion:

* *Device/Grid* Synchronization is done via wait lists and queue barriers (``stream``).
* *Block/Warp/Thread* it is in the same control flow of the caller threads. Synchronization is done via memory barriers.
