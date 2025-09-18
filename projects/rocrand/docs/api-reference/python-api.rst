.. meta::
  :description: rocRAND Python API reference
  :keywords: rocRAND, ROCm, API, documentation, Python
  
.. _python-api:

====================
Python API reference
====================

This chapter describes the rocRAND Python module API.

API index
------------

To search the API, see the API :ref:`genindex`.

.. default-domain:: py
.. py:currentmodule:: rocrand

class PRNG
----------

.. autoclass:: rocrand.PRNG
   :inherited-members:
   :members:

class QRNG
----------

.. autoclass:: rocrand.QRNG
   :inherited-members:
   :members:

Exceptions
----------

.. autoexception:: rocrand.RocRandError
   :members:

.. autoexception:: rocrand.HipError
   :members:

Utilities
---------

.. autoclass:: rocrand.DeviceNDArray
   :members:

.. autofunction:: rocrand.empty

.. autofunction:: rocrand.get_version

To search the API, see the :ref:`genindex` for all rocRAND APIs.
