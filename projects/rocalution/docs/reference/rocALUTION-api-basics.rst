.. meta::
   :description: rocALUTION API basics
   :keywords: rocALUTION, ROCm, HIP SDK, library, API, basics

.. _basics:

***********************
rocALUTION API Basics
***********************

This document covers basic information about the rocALUTION APIs and their uses.

Operators and vectors
=====================

The main objects in rocALUTION are linear operators and vectors. All objects can be moved to an accelerator at run-time. Linear operators are defined as local or global matrices (i.e. on a single-node or distributed/multi-node) and local stencils (i.e. matrix-free linear operations).

The only template parameter of the operators and vectors is the data type (``ValueType``).
The operator data type could be float, double, complex float, or complex double, while the vector data type can be int, float, double, complex float or complex double (int is used mainly for the permutation vectors).
In the current version, cross ``ValueType`` object operations are not supported. The following figure gives an overview of supported operators and vectors.
For more details, refer to the :ref:`design-philosophy`.

.. _operators:
.. figure:: ../data/operators.png
  :alt: operator and vector classes
  :align: center

  Operator and vector classes.

Each object contains a local copy of the hardware descriptor created by the :cpp:func:`rocalution::init_rocalution` function. This allows you to modify it according to your needs and to obtain two or more objects with different hardware specifications (e.g. different amounts of OpenMP threads, HIP block sizes, etc.).

Local operators and vectors
---------------------------

The local operators and vectors correspond to the local matrices and stencils, and local vectors. The term "local" implies the fact that they stay on a single system. A system can contain several CPUs via UMA or NUMA memory system, as well as an accelerator.

.. doxygenclass:: rocalution::LocalMatrix
.. doxygenclass:: rocalution::LocalStencil
.. doxygenclass:: rocalution::LocalVector

Global operators and vectors
----------------------------

Global operators and vectors correspond to the global matrix and global vectors. The term "global" implies the fact that they stay on a single or multiple nodes in a network. For this type of computation, the communication is based on MPI.

.. doxygenclass:: rocalution::GlobalMatrix
.. doxygenclass:: rocalution::GlobalVector

Backend descriptor and user control
===================================

Naturally, not all routines and algorithms can be performed efficiently on many-core systems (i.e. on accelerators).
To provide full functionality, the library has internal mechanisms to check if a particular routine is implemented on the accelerator.
If not, the object is moved to the host and the routine is computed there.
This ensures that the application runs (maybe not in the most efficient way) with any accelerator regardless of the availability of the required functionality for it.

Initialization of rocALUTION
----------------------------

The body of a rocALUTION code should simply contain the header file and the namespace of the library.
The program must contain an initialization call to :cpp:func:`init_rocalution <rocalution::init_rocalution>` that checks and allocates the hardware and a finalizing call to :cpp:func:`stop_rocalution <rocalution::stop_rocalution>` that releases the allocated hardware.

.. doxygenfunction:: rocalution::init_rocalution
.. doxygenfunction:: rocalution::stop_rocalution

Thread-core mapping
-------------------

The number of threads used by rocALUTION can be modified by the function :cpp:func:`set_omp_threads_rocalution <rocalution::set_omp_threads_rocalution>` or by the global OpenMP environment variable (for Unix-like OS this is `OMP_NUM_THREADS`).
During the initialization phase, the library provides affinity thread-core mapping:

- If the number of cores (including SMT cores) is greater than or equal to twice the number of threads, then all the threads can occupy every second core ID (e.g. 0,2,4,...).
  This is to avoid having two threads working on the same physical core, when SMT is enabled.
- If the number of threads is less than or equal to the number of cores (including SMT), and the previous clause is false, then the threads can occupy every core ID (e.g. 0,1,2,3,...).
- If none of the above criteria is matched, then the default thread-core mapping is used (typically set by the operating system).

.. note:: The thread-core mapping is available for Unix-like operating systems only.
.. note:: The user can disable the thread affinity with :cpp:func:`set_omp_affinity_rocalution <rocalution::set_omp_affinity_rocalution>`, before initializing the library.

OpenMP threshold size
---------------------

When working on a small problem, OpenMP host backend might be slightly slower than using no OpenMP.
This is mainly attributed to the small amount of work, which every thread should perform and the large overhead of forking/joining threads.
This can be avoided by the OpenMP threshold size parameter in rocALUTION.
The default threshold is set to 10.000, which means that all matrices under (and equal to) this size use only one thread (irrespective of the number of OpenMP threads set in the system).
To modify the threshold, use :cpp:func:`set_omp_threshold_rocalution <rocalution::set_omp_threshold_rocalution>`.

Accelerator selection
---------------------

To select the accelerator device id to be used for the computation, use :cpp:func:`set_device_rocalution <rocalution::set_device_rocalution>`.

Disable the accelerator
-----------------------

To disable the accelerator without having to re-compile the library, use :cpp:func:`disable_accelerator_rocalution <rocalution::disable_accelerator_rocalution>`.

Backend information
-------------------

To print the detailed information about the current backend / accelerator in use as well as the available accelerators, use :cpp:func:`info_rocalution <rocalution::info_rocalution>`.

MPI and multi-accelerators
--------------------------

When initializing the library with MPI, you need to pass the rank of the MPI process as well as the number of accelerators available on each node.
Basically, this way you can specify the mapping of MPI process and accelerators - the allocated accelerator is ``rank % num_dev_per_node``.
Thus, you can run two MPI processes on systems with two accelerators by specifying the number of devices to 2, as illustrated in the example code below.

.. code-block:: cpp

  #include <rocalution.hpp>
  #include <mpi.h>

  using namespace rocalution;

  int main(int argc, char* argv[])
  {
      MPI_Init(&argc, &argv);
      MPI_Comm comm = MPI_COMM_WORLD;

      int num_processes;
      int rank;

      MPI_Comm_size(comm, &num_processes);
      MPI_Comm_rank(comm, &rank);

      int nacc_per_node = 2;

      init_rocalution(rank, nacc_per_node);

      // ... do some work

      stop_rocalution();

      return 0;
  }

.. _rocalution_obj_tracking:

Automatic object tracking
=========================

rocALUTION supports automatic object tracking.
After the initialization of the library, all objects created by the user application can be tracked.
Once :cpp:func:`stop_rocalution <rocalution::stop_rocalution>` is called, all memory from tracked objects gets deallocated.
This avoids memory leaks when the objects are allocated but not freed.
The user can enable or disable the tracking by editing ``src/utils/def.hpp``.
By default, automatic object tracking is disabled.

.. _rocalution_verbose:

Verbose output
==============

rocALUTION provides different levels of output messages.
The ``VERBOSE_LEVEL`` can be modified in ``src/utils/def.hpp`` before the compilation of the library.
By setting a higher level, you can obtain more detailed information about the internal calls and data transfers to and from the accelerators.
By default, the ``VERBOSE_LEVEL`` is set to 2.

.. _rocalution_logging:

Verbose output and MPI
======================

To prevent all MPI processes from printing information to ``stdout``, the default configuration allows only ``RANK 0`` to output information.
You can change the ``RANK`` or allow all processes to print by setting ``LOG_MPI_RANK`` to 1 in ``src/utils/def.hpp``.
If file logging is enabled, all ranks write into the corresponding log files.

.. _rocalution_debug:

Debug output
============

Debug output prints almost every detail in the program, including object constructor/destructor, address of the object, memory allocation, data transfers, all function calls for matrices, vectors, solvers, and preconditioners.
The flag ``DEBUG_MODE`` can be set in ``src/utils/def.hpp``.
When enabled, additional ``assert()s`` are checked during the computation.
This might significantly reduce the performance of some operations.

File logging
============

To enable rocALUTION trace file logging, set the environment variable ``ROCALUTION_LAYER`` to 1.
rocALUTION then logs each rocALUTION function call including object constructor/destructor, address of the object, memory allocation, data transfers, all function calls for matrices, vectors, solvers, and preconditioners.
The log file is placed in the working directory.
The log file naming convention is ``rocalution-rank-<rank>-<time_since_epoch_in_msec>.log``.
By default, the environment variable ``ROCALUTION_LAYER`` is unset and logging is disabled.

.. note:: Performance might degrade when logging is enabled.

Versions
========

For checking the rocALUTION version in an application, use pre-defined macros:

.. code-block:: cpp

  #define __ROCALUTION_VER_MAJOR  // version major
  #define __ROCALUTION_VER_MINOR  // version minor
  #define __ROCALUTION_VER_PATCH  // version patch
  #define __ROCALUTION_VER_TWEAK  // commit id (sha-1)

  #define __ROCALUTION_VER_PRE    // version pre-release (alpha or beta)

  #define __ROCALUTION_VER        // version

The final ``__ROCALUTION_VER`` holds the version number as ``10000 * major + 100 * minor + patch``, as defined in ``src/base/version.hpp.in``.
