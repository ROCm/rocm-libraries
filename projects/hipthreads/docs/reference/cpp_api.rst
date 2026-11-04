.. meta::
  :description: hipThreads C++ API reference
  :keywords: hipThreads, ROCm, API, reference, thread, mutex, condition_variable

.. _api-reference:

******************************************
API reference guide
******************************************

Threading
=========

The threading groups cover threads, mutexes, and condition variables.

.. doxygengroup:: threading

Thread management
-----------------

.. doxygengroup:: thread
    :inner:

Mutexes
-------

.. doxygengroup:: mutex
    :inner:

Condition variables
-------------------

.. doxygengroup:: condition_variable
    :inner:

C library utilities
===================

The C library groups cover memory allocation and byte and string manipulation.

.. doxygengroup:: c_library

Memory allocation
-----------------

.. doxygengroup:: c_memory
    :inner:

Byte and string manipulation
----------------------------

.. doxygengroup:: c_bytestring
    :inner:
