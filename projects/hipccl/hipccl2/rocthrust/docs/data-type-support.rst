.. meta::
  :description: rocThrust API data type support
  :keywords: rocThrust, ROCm, API, reference, data type, support

.. _data-type-support:

******************************************
Data type support
******************************************

Fundamental types
=================


 .. list-table:: Comparison of supported fundamental types of rocThrust and Thrust
    :header-rows: 1
    :name: supported-fundamental-types-rocthrust-vs-thrust

    *
      - Type
      - rocThrust support
      - Thrust support
    *
      - :code:`int8`
      - ✅
      - ✅
    *
      - :code:`int16`
      - ✅
      - ✅
    *
      - :code:`int32`
      - ✅
      - ✅
    *
      - :code:`int64`
      - ✅
      - ✅
    *
      - :code:`half` [1]_
      - ⚠️
      - ⚠️
    *
      - :code:`bfloat16` [1]_
      - ⚠️
      - ⚠️
    *
      - :code:`float`
      - ✅
      - ✅
    *
      - :code:`double`
      - ✅
      - ✅

Custom types
============

rocThrust and Thrust support custom, user-defined types, if they provide the interface required by the used functions.

.. rubric:: Footnotes
.. [1] These types are supported in rocThrust and Thrust, however the host-side hip-implementations of these types miss some functionality, and are mostly intended as storage types to be passed between functions.
