.. meta::
  :description: rocThrust and atomic path
  :keywords: rocThrust, rocPRIM, atomic path, ROCPRIM_USE_ATOMIC_BLOCK_ID, HIP

.. _atomic-path:

************************************
rocThrust and atomic path
************************************

When using the atomic path, block IDs are assigned by atomically incrementing a counter in global device memory. This produces a predictable ordering of block IDs for kernels that require path correctness. 

The default rocThrust setting uses the atomic path only with gfx942 or gfx950 architectures.

This behavior can be changed through the ``ROCPRIM_USE_ATOMIC_BLOCK_ID`` environment variable. Set ``ROCPRIM_USE_ATOMIC_BLOCK_ID`` to 0, to never use the atomic path. Set ``ROCPRIM_USE_ATOMIC_BLOCK_ID`` to 2 to always use the atomic path. Set ``ROCPRIM_USE_ATOMIC_BLOCK_ID`` to 1 to return to the default behavior where the atomic path is used automatically only on gfx942 or gfx950 architectures.

Because using the atomic path can come at a performance cost, ``ROCPRIM_USE_ATOMIC_BLOCK_ID`` should be set to 2 only if there is a noticeable performance degradation or a deadlock in the rocThrust and rocPRIM kernels regardless of the underlying architecture.

``ROCPRIM_USE_ATOMIC_BLOCK_ID`` should be set to 0 only in the case where neither the gfx942 or gfx950 architectures are being used.