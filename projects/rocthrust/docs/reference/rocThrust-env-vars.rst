.. meta::
  :description: rocThrust environment variables for developers and testing
  :keywords: rocThrust, ROCm, environment variables, HIP, testing, bitwise reproducibility

.. _env-var:

************************************
rocThrust environment variables
************************************

These rocThrust environment variables affect how the HIP backend dispatches to rocPRIM kernels.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Environment variable
     - Values
   * - ``ROCPRIM_USE_HMM``
     - | ``1``:  requests managed memory allocation.
       | Ordinary device allocation is used if left unset or if set to a value other than 1.
   * - ``ROCPRIM_USE_ATOMIC_BLOCK_ID``
     - | ``0``: Never use the atomic path
       | ``1``: Use the atomic path if the GPU architecture requires the atomic path. For example, MI300 architectures running rocPRIM algorithms that use Lookback Scan concurrently might require the atomic path under certain workloads.
       | ``2``: Always use the atomic path.
