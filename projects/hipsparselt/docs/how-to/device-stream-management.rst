.. meta::
   :description: HIP device and stream management with hipSPARSELt
   :keywords: hipSPARSELt, ROCm, API library, API reference, stream
      management, device management

.. _device-stream-manage:

*******************************************
HIP device and stream management
*******************************************

``hipSetDevice`` and ``hipGetDevice`` are HIP device management APIs. They are not part of the
hipSPARSELt API.

Device management
===============================

hipSPARSELt assumes that you've already set the device before making a hipSPARSELt call.

To set a device, use ``hipSetDevice`` before making a HIP kernel invocation. If you don't
explicitly set a device, the system uses ``device 0`` as the default. HIP kernels are launched on ``device 0`` by
default.

After setting a device, you can create a handle using ``hipsparselt_init``. Subsequent hipSPARSELt
routines take this handle as an input parameter. hipSPARSELt only queries the
device  (using ``hipGetDevice``). It doesn't set the device. If hipSPARSELt doesn't recognize a valid device,
it returns an error message.
To ensure device safety, it's your responsibility to provide hipSPARSELt with a valid device.

You can't switch devices between ``hipsparselt_init`` and ``hipsparselt_destroy``. If you want to change
devices, you must first destroy the current handle and then create a new one.

Stream management
================================

HIP kernels are always launched in a queue (also known as a stream).

If you don't explicitly specify a stream, the system provides a default stream that is maintained by the
system. You can't create or destroy the default stream. However, you can create new streams
using ``hipStreamCreate`` and bind them to hipSPARSELt operations, such as ``hipsparselt_spmma_prune``
and ``hipsparselt_matmul``. HIP kernels are invoked in hipSPARSELt routines

.. note::

   If you create a stream, you're also responsible for destroying it.

Multiple streams and devices
=====================================

If the system under test has multiple HIP devices, you can run multiple hipSPARSELt handles
concurrently. Each handle is associated with a specific device, so a new handle must be created
for each additional device. You can't run a single hipSPARSELt handle on different discrete devices.
