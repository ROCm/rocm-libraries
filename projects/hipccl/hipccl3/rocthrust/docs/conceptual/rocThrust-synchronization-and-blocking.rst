.. meta::
  :description: Definitions of synchronous and asynchronous behavior in rocThrust
  :keywords: rocThrust, ROCm, synchronous, asynchronous, blocking, host, device, HIP

.. _synchronization-and-blocking:

**********************************************
Synchronization and blocking in rocThrust
**********************************************
 
Operations that are synchronous with respect to the host (CPU) are operations that run on the GPU and block the CPU. The CPU will remain idle until the operation completes on the GPU. Once the operation on the GPU finishes, the CPU will launch the next instruction. 

Operations that are asynchronous with respect to the host are operations that don't block the CPU. The CPU can continue launching more operations and doing other work while the GPU runs its workload. The GPU and the CPU can run concurrently. The host and device must be synchronized before the host can access the results of the operations. 

rocThrust functions are synchronous with respect to the host by default. 

The ``hip_rocprim::par_nosync`` policy can be used to avoid synchronization barriers. This policy communicates to the runtime that synchronization isn't necessary and that it can be skipped if the implementation allows it. This policy doesn't guarantee non-blocking calls because some operations require internal synchronization. There is no way to guarantee asynchronization in rocThrust.

