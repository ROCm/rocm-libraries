.. meta::
  :description: hipThreads C++ API reference
  :keywords: hipThreads, ROCm, API, reference, thread, mutex, condition_variable

.. _api-reference:

******************************************
API reference guide
******************************************

Doxygen generates the hipThreads C++ API reference from the in-source documentation.

Threading
=========

The threading group covers ``hip::wthread`` and the ``hip::this_thread`` helpers.

.. doxygengroup:: threading

Thread management
-----------------

The thread group documents construction, joining, and identification.

.. doxygengroup:: thread
    :inner:

Mutexes
-------

The mutex group documents spin and pseudo mutex types and lock helpers.

.. doxygengroup:: mutex
    :inner:

Condition variables
-------------------

The condition-variable group documents spin and pseudo condition-variable types.

.. doxygengroup:: condition_variable
    :inner:

C library utilities
===================

The C library group covers device memory helpers in the ``hip::`` namespace.

.. doxygengroup:: c_library

Memory allocation
-----------------

The memory group documents ``hip::malloc()`` and ``hip::free()``.

.. doxygengroup:: c_memory
    :inner:

Byte and string manipulation
----------------------------

The byte-string group documents ``hip::memcpy()`` and related helpers.

.. doxygengroup:: c_bytestring
    :inner:
