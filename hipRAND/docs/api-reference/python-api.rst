.. meta::
   :description: hipRAND Python API reference
   :keywords: hipRAND, ROCm, library, API, tool, Python

.. _python-api:

====================
Python API reference
====================

This document describes the hipRAND APIs in Python.

The APIs in this wrapper are similar to `pyculib.rand <http://pyculib.readthedocs.io/en/latest/curand.html>`_.

.. default-domain:: py
.. py:currentmodule:: hiprand

class PRNG
----------

.. autoclass:: hiprand.PRNG
   :inherited-members:
   :members:


class QRNG
----------

.. autoclass:: hiprand.QRNG
   :inherited-members:
   :members:

Exceptions
----------

.. autoexception:: hiprand.HipRandError
   :members:

.. autoexception:: hiprand.HipError
   :members:

Utilities
---------

.. autoclass:: hiprand.DeviceNDArray
   :members:

.. autofunction:: hiprand.empty

.. autofunction:: hiprand.get_version
