.. meta::
   :description: rocALUTION design and philosophy
   :keywords: rocALUTION, ROCm, library, API, design, philosophy

.. _design-philosophy:

**********************************
rocALUTION design and philosophy
**********************************

rocALUTION is implemented in C++ and HIP and is designed to separate algorithmic logic from hardware-specific details.
This allows you to write portable sparse linear algebra code without binding it to a specific execution device at compile time.

Design overview
-----------------

rocALUTION objects are intentionally decoupled from the underlying hardware:

- When you declare a matrix, vector, or solver, the object is initially allocated on the host (CPU).
- You can move any object to a selected accelerator using a simple function call.
- The execution model is based on run-time type information (RTTI), allowing you to select the execution location and method at run time.
- This approach differs from template-based libraries, which require hardware decisions to be made at compile time.

This design allows you to use the same source code across different hardware configurations without modification.

Hardware abstraction philosophy
--------------------------------

The core philosophy of rocALUTION is to abstract hardware-specific functions from the code that describes your algorithm.
This abstraction addresses two common challenges:

1. Large simulation codes based on sparse computations are difficult to port and maintain across rapidly evolving hardware architectures.
2. Modern high-performance accelerators can significantly reduce execution time for critical computational kernels.

By separating these concerns, rocALUTION enables you to focus on algorithm design while still benefiting from hardware acceleration.

Parallelism and accelerator support
-------------------------------------

The hardware abstraction layer is a central element of the rocALUTION design:

- It is built to explore fine-grained parallelism suitable for multi-core and many-core devices.
- This approach differs from many parallel sparse libraries that rely primarily on domain decomposition techniques.
- As a result, the design of iterative solvers and preconditioners in rocALUTION is fundamentally different.

Another key aspect of the design is native accelerator support:

- Memory allocation, data transfers, and hardware-specific operations are handled internally by the library.
- You do not need to manage device-specific details explicitly in your application code.

Optional accelerator usage
---------------------------

rocALUTION does not require you to use accelerator hardware:

- You can compile and run the same source code on systems with or without accelerators.
- Offloading algorithms and solvers to an accelerator is optional and does not affect code portability.

Fallback mechanisms
---------------------

Not all routines can be executed efficiently on many-core accelerator devices.
To ensure full functionality, rocALUTION includes internal fallback mechanisms:

- The library checks at run time whether a specific routine is implemented on the selected accelerator.
- If the routine is not available, the associated object is moved back to the host.
- The computation is then performed on the CPU automatically.

This behavior ensures that your application runs correctly on any supported accelerator, regardless of feature availability.
