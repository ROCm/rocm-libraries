.. meta::
   :description: rocALUTION API usage notes
   :keywords: rocALUTION, ROCm, library, API, notes

.. _remarks:

*******************************
rocALUTION library usage notes
*******************************

The following are general notes regarding how to use the rocALUTION library.

Performance
===========
* Solvers can be built on the accelerator. In many cases, this is faster compared to building on the host.
* Small-sized problems tend to perform better on the host (CPU), due to the good caching system, while large-sized problems typically perform better on the accelerator devices.
* Avoid accessing vectors using [] operators. Use techniques based on :cpp:func:`rocalution::LocalVector::SetDataPtr` and :cpp:func:`rocalution::LocalVector::LeaveDataPtr` instead.
* By default, the OpenMP backend picks the maximum number of threads available. However, if your CPU supports SMT, it will allow to run two times more threads than number of cores. This, in many cases, leads to lower performance. You may observe a performance increase by setting the number of threads (see :cpp:func:`rocalution::set_omp_threads_rocalution`) equal to the number of physical cores.
* If you need to solve a system with multiple right-hand-sides, avoid constructing the solver/preconditioner every time.
* If you are solving similar linear systems, you might want to consider to use the same preconditioner to avoid long building phases.
* In most of the cases, the classical CSR matrix format performs very similar to all other formats on the CPU. On accelerators with many-cores (such as GPUs), formats such as DIA and ELL typically perform better. However, for general sparse matrices one could use HYB format to avoid large memory overhead. The multi-colored preconditioners can be performed in ELL for most of the matrices.
* Not all matrix conversions are performed on the device, the platform will give you a warning if the object need to be moved.
* If you are deploying the rocALUTION library into another software framework try to design your integration functions to avoid :cpp:func:`rocalution::init_rocalution` and :cpp:func:`rocalution::stop_rocalution` every time you call a solver in the library.
* Be sure to compile the library with the correct optimization level (-O3).
* Check if your solver is really performed on the accelerator by printing the matrix information (:cpp:func:`rocalution::BaseRocalution::Info`) just before calling the :cpp:func:`rocalution::Solver::Solve` function.
* Check the configuration of the library for your hardware with :cpp:func:`rocalution::info_rocalution`.
* Mixed-Precision defect correction technique is recommended for accelerators (e.g. GPUs) with partial or no double precision support. The stopping criteria for the inner solver has to be tuned well for good performance.

Accelerators
============
* Avoid PCI-Express communication whenever possible (such as copying data from/to the accelerator). Also check the internal structure of the functions.
* Pinned memory allocation (page-locked) can be used for all host memory allocations when using the HIP backend. This provides faster transfers over the PCI-Express and allows asynchronous data movement. By default, this option is disabled. To enable the pinned memory allocation uncomment `#define ROCALUTION_HIP_PINNED_MEMORY` in file `src/utils/allocate_free.hpp`.
* Asynchronous transfers are available for the HIP backend.

Correctness
===========
* If you are assembling or modifying your matrix, you can check it in octave/MATLAB by just writing it into a matrix-market file and read it via `mmread()` function. You can also input a MATLAB/octave matrix in such a way.
* Be sure, to set the correct relative and absolute tolerance values for your problem.
* Check the computation of the relative stopping criteria, if it is based on :math:`|b-Ax^k|_2/|b-Ax^0|_2` or :math:`|b-Ax^k|_2/|b|_2`.
* Solving very ill-conditioned problems by iterative methods without a proper preconditioning technique might produce wrong results. The solver could stop by showing a low relative tolerance based on the residual but this might be wrong.
* Building the Krylov subspace for many ill-conditioned problems could be a tricky task. To ensure orthogonality in the subspace you might want to perform double orthogonalization (i.e. re-orthogonalization) to avoid rounding errors.
* If you read/write matrices/vectors from files, check the ASCII format of the values (e.g. 34.3434 or 3.43434E + 01).
