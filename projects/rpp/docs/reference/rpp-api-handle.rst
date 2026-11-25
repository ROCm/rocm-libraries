.. meta::
  :description: ROCm Performance Primitives (RPP) handle API reference
  :keywords: RPP, ROCm, Performance Primitives, API, reference, handle, rppCreate, rppDestroy

********************************************************************
RPP handle API
********************************************************************

.. doxygengroup:: group_rpp
   :content-only:
   :members:

Kernel timing
=============

RPP does not expose a profiling API of its own. Earlier releases declared
``rppEnableProfiling()`` and ``rppGetKernelTime()`` for HIP builds, but neither
function was ever defined, so they have been removed from ``rpp.h``. To measure
kernel time, capture a trace with ``rocprofv3`` (part of the ROCm profiling
tooling) and read the kernel durations from its output.
