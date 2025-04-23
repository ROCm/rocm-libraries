.. meta::
   :description: rocALUTION design and philosophy
   :keywords: rocALUTION, ROCm, library, API, design, philosophy

.. _design-philosophy:

**********************************
rocALUTION design and philosophy
**********************************

rocALUTION is written in C++ and HIP.

The rocALUTION objects are designed to be separate from the actual hardware specification.
Once you declare a matrix, a vector, or a solver, these rocALUTION objects are initially allocated on the host (CPU).
Then, every object can be moved to a selected accelerator using a simple function call.
The whole execution mechanism is based on the Run-Time Type Information (RTTI), which allows you to select the location and method for performing the operations at run-time.
This is in contrast to the template-based libraries that require this information at compile-time.

The philosophy of the library is to abstract the hardware-specific functions and routines from the actual program that describes the algorithm.
It is difficult and almost impossible for most of the large simulation softwares based on sparse computation to adapt and port their implementation to suit every new technology.
On the other hand, the new high performance accelerators and devices can decrease the computational time significantly in many critical parts.

This abstraction layer of the hardware-specific routines is the core of the rocALUTION design.
It is built to explore fine-grained level of parallelism suited for multi/many-core devices.
This is in contrast to most of the parallel sparse libraries that are based mainly on domain decomposition techniques.
That's why the design of the iterative solvers and preconditioners is very different.
Another cornerstone of rocALUTION is the native support for accelerators where the memory allocation, transfers, and specific hardware functions are handled internally in the library.

rocALUTION doesn't make the use of accelerator technologies mandatory.
Even if you offload your algorithms and solvers on the accelerator device, the same source code can be compiled and executed on a system without an accelerator.

Naturally, not all routines and algorithms can be performed efficiently on many-core systems (i.e. on accelerators).
To provide full functionality, the library has internal mechanisms to check if a particular routine is implemented on the accelerator.
If not, the object is moved to the host and the routine is computed there.
This ensures that your code runs on any accelerator, regardless of the available functionality for it.
