.. meta::
  :description: rocFFT documentation and API reference library
  :keywords: rocFFT, ROCm, API, documentation

.. _runtime_compilation:

********************************************************************
Runtime Compilation Design Document for rocFFT
********************************************************************

Summary
=======

This document describes runtime compilation (RTC) as it is used in
rocFFT.  Runtime compilation helps reduce binary size and build times
for the library, and can allow for optimizations that are not
practical versus ahead-of-time compiled kernels.

Problem
=======

Stockham FFT kernels make up the vast majority of the rocFFT library.
Kernels handling specific problem sizes are chosen as part of the
rocFFT build process, and are compiled for all the variants that
might be required at runtime.

The count of variants for each problem size has a number of stacking
multipliers applied to it.  Any given problem size needs variants for:

* Each supported GPU architecture;

* Six interleaved/planar and in-place/out-of-place variants;

* Forward and inverse transforms;

* At least two precisions (single/double);

* Unit- and non-unit- strides;

* With and without callback support.

Runtime compilation has advantages over pre-compiling all the
above variants for all problem sizes.  These include:

* Handling of new problem sizes does not require rebuilding the
  library.

* Build times are faster.

* The library binary is smaller.  This in turn means:
  installation is faster; and difficulties arising from limited
  memory addressing in shared objects (the default memory model for
  shared objects built with ``-fPIC`` only allows for 2 GiB binaries,
  resulting in `build breaks`_) are reduced.

.. _build breaks: https://www.ibm.com/support/pages/intel-compiler-error-relocation-truncated-fit-rx8664pc32

Solution
========

HIP provides runtime compilation facilities, in the hiprtc.h header.

The code generator is embedded into the library itself.  During FFT
planning, we can run the code generator to produce suitable FFT
source code for the specific problem being solved.  Then, we can
runtime-compile that source into GPU kernels that we launch at FFT
execution time.

Empirical testing of runtime compilation on our FFT kernels shows
that compilation times for single kernels range between 0.5 and 2
seconds on modern hardware, with more complicated kernels taking more
time.  Kernel execution time is identical to the ahead-of-time
compiled version.

Implementation
==============

Embedding and running the generator
-----------------------------------

A generator implemented in C++ can be built into the library like any
other C++ code that makes up the library.

During plan building, we can execute the generator code to
produce a string of source code for the required problem sizes.

The generator needs sufficient input to have it produce exactly the
variant that is required, e.g. length-336, inverse, out-of-place,
interleaved-to-planar, double precision, etc.

Compilation
-----------

Compilation should be done during plan building, and the generated
kernels can be attached directly to the ``TreeNode`` for that step of
the FFT.

If the kernels are available on the ``TreeNode``, we will have less
overhead at execution time, since we don't need to do any work to
find the right kernel to run.

Caching kernels at runtime
--------------------------

If a process needs to create multiple plans that would compile the
same FFT kernel variant, it's nice to reuse kernels we've already
compiled.  Reusing an already-compiled kernel would save compilation
time on subsequent runs.

Compiled kernels may be persisted to disk for maximal reuse.
However, rocFFT may be used in distributed systems where the
file system is shared among multiple compute nodes, and having
multiple nodes all contend for the same shared file is problematic
for performance.

By default, kernels are only cached in memory, to prevent this
contention.  The cache location may still be overridden at runtime
using mechanisms described below.

Cache keys
^^^^^^^^^^

The cache keys need to be chosen carefully to ensure that an obsolete
kernel is not reused when a new version really **should** be
recompiled.

The kernel function name shall be the main key field in the cache.
The function name shall encode all the parameters by which kernel
functions could differ, including:

* scheme (e.g. whether this is a standard Stockham kernel, or a
  variant that does different read/write tiling)

* length (typically 1D length, may be 2D or more for single kernels
  that handle higher dimension transforms)

* placement (in-place or out-of-place)

* direction (forward or backward)

* input and output formats (planar, interleaved, real)

* precision (single or double)

* stride type (unit stride or non-unit stride)

* large 1D twiddle type and base (for kernels that need to integrate a
  twiddle multiplication step)

* callback type (run callbacks or don't run callbacks)

Encoding all of these parameters into the kernel name is necessary
anyway, so that logs and profilers will tell users and developers
exactly which kernel is running, even if it's been runtime-compiled.

Using just the kernel name as the main key is also helpful because
the caching code needn't be aware of all the possible parameters that
kernels could differ by.  New parameters can be added at anytime, and
as long as the kernel names are updated accordingly, the cache will
just work.

The cache will also need to store other key fields to ensure that a
kernel is compiled if any of these changes:

* GPU architecture

* HIP runtime version

* Kernel generator version

Practically, these key field choices will ensure that users are
always running the latest kernels that rocFFT provides and which are
appropriate for the hardware present.

User control of cache
^^^^^^^^^^^^^^^^^^^^^

Distributed workflows will want additional control over the cache.
For example, a workload that distributes FFT computation over many MPI nodes will want to ensure that the kernels are built
once centrally rather than by each node.

MPI nodes might also have no access to disk (either shared with other
nodes or local to each node).

rocFFT needs to expose APIs to:

* Serialize the current cache to a library-allocated buffer

* Free the library-allocated serialization buffer

* Deserialize a buffer into a cache (which might need to be in-memory
  for diskless nodes)

The example MPI computation described above would be able to build
plans on the rank 0 node to populate the cache once.  Then, it can
use these new APIs along with MPI APIs to distribute the cache to
each work node.

Backing store implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The cache may be written to disk, and if so it must be robust in the
face of concurrent access, crashes during library operation, and so
on.

We really would like the cache to have ACID properties of database
systems.

The easiest way to achieve this is to use SQLite to manage the
storage.  It's easily embeddable in our library (or is readily
available as its own library), and provides all the properties
we'd want for the storage backend.

It also provides APIs to serialize a database, as required for the
distributed workflows described above.

Pre-built kernels
^^^^^^^^^^^^^^^^^

Even if rocFFT is prepared to runtime-compile any FFT kernel, we can
still pre-compile kernels by populating a cache at library build time
and shipping the cache with the library.

Cache location
^^^^^^^^^^^^^^

The main challenge here is installing this pre-built cache in a place
that the library will be able to find.

The easiest solution here, as employed by `other math libraries` is
to look for this the cache file relative to the shared library itself.

.. _other math libraries: https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/d8e00e169ccc7ca21211705643e85545e98e455a/library/src/tensile_host.cpp#L521

Environment variables can override the locations of caches used by
rocFFT.  During normal operation, we would expect one read-only cache
shipped with the library and one modifiable cache updated as the user
runs transforms that use new kernels.

We support two environment variables for these two locations:

* ROCFFT_RTC_SYS_CACHE_PATH - the pre-built read-only system-level cache.
* ROCFFT_RTC_CACHE_PATH - the read-write user-level cache.

Note that if the library is linked statically, we will not be able to
find any files relative to the library.  The
ROCFFT_RTC_SYS_CACHE_PATH environment variable will then be required
for rocFFT to find the system-level cache, but rocFFT will still
update the user-level cache and have correct behavior without a
system-level cache.

Populating the cache
^^^^^^^^^^^^^^^^^^^^

Populating this shipped cache is done via a helper executable that is
built and run during the rocFFT build.  A separate helper executable
(which is not itself shipped with rocFFT) is necessary so that it can
share rocFFT's generator and RTC code, without requiring rocFFT to
expose extra symbols just for this task.

This helper should work at the kernel level, e.g. build Stockham
kernels for all desired combinations of:

* supported architectures (gfx908, gfx90a, gfx1030, etc.)
* precisions
* problem sizes
* array formats
* etc.

The criteria for which kernels to pre-build can be arbitrary.  Less
common choices will be runtime-compiled, and runtime compilation is
still a fallback in case a pre-built kernel is not available for
whatever reason.

An inferior option would be for the helper to work at the plan level
(i.e. use rocFFT to build a set of plans and save the resulting RTC
kernels).  However, creating plans involves doing a lot of other
unnecessary work, like generating twiddle tables and deciding on
buffer assignment.

Impact on tests
^^^^^^^^^^^^^^^

Accuracy tests are maximally affected in terms of runtime by this
change, since they run a huge number of problem sizes in the context
of a single process.  That means the costs of generating and
compiling a large variety of kernel variants will be the most painful
here, once more problem sizes are handled by the new generator.

An increase in test runtime is an unfortunate side effect of runtime
compilation.  This cost is made more acceptable because the compile
time of the library has already been reduced prior to running the tests.

A possible solution here might be to do a parallel traversal of the
test cases, building rocFFT plans for each of them (but not actually
executing plans).  This would runtime-compile the whole suite's
kernels in parallel, which would save a lot of time.

Interaction with callbacks
^^^^^^^^^^^^^^^^^^^^^^^^^^

Callback-enabled FFTs require a different kernel variant to be
generated, but the decision of whether to actually run with a
callback is made by the user after the plan is constructed.

To solve this, we generate both a callback and non-callback variant
where necessary during plan creation.

Parallel compilation
^^^^^^^^^^^^^^^^^^^^

Because of the potential need for callback-enabled kernels, most
plans will be generated faster if kernels can be compiled in
parallel.  Unfortunately, hipRTC has process-wide locks in it that
prevent useful multithreading of compilation.

Instead, we can spawn a helper process for subsequent compilations if
a compilation is already in-progress in the original process.  This
helper would need to be shipped with the library, in a location
that's knowable by the library.  If we fail to find or spawn that
helper, compilation must fall back to compiling in-process.

Code organization
=================

The whole of rocFFT runtime compilation can be broken down into
separate subsystems:

1. Generating source to be compiled, further subdivided into
   generators for each type of kernel (Stockham, transpose,
   Bluestein, etc).  Input specifications of the desired kernel
   include problem size, precision, result placement, and so on.

   Files to implement this are named:

   * rtc_stockham_gen.cpp
   * rtc_transpose_gen.cpp
   * etc.

2. Compiling source code into object code, which can be further subdivided:

   a. Compiling code in the current process
   b. Compiling code in a sub-process

   The files to implement these are named:

   rtc_compile.cpp
   rtc_subprocess.cpp

3. Reading/writing the cache of compiled object code.

   The file to implement this is named:

   rtc_cache.cpp

4. Compiling and launching the correct kernel for a TreeNode in an
   FFT plan.  This subsystem would need to derive the correct input
   specifications for the generator, given the data in the TreeNode.
   It would also need to derive the correct launch arguments to pass
   to the kernel.

   Files to implement this are named:

   * rtc_stockham_kernel.cpp
   * rtc_transpose_kernel.cpp
   * etc.

   These files are named rtc_*_kernel.cpp because they implement
   subclasses of the generic RTCKernel type.

In this list, 1 and 2 are independent.  2b depends on 2a.  3 depends
on 1 and 2.  4 depends on 3.  2a requires the hipRTC library, 3
requires the SQLite library, and 4 requires the full HIP runtime
library (amdhip64).

Build-time processes that populate a cache to ship with the library
depend on 3.  The helper process to support parallel compilation
depends on 2a.

It's important to avoid using the full HIP runtime at build time -
Windows build environments in particular may not have the sufficient
libraries or infrastructure to successfully load the full runtime,
but they are able to load hipRTC.

Future work
===========

Moving away from chosen problem sizes
-------------------------------------

Once the infrastructure is in place, we could consider enabling
runtime compilation for all FFT sizes, not just those that are chosen
ahead of time.  The generator is already able to auto-factorize
arbitrary sizes, though we haven't yet tested the limits of this
ability.

Copyright and disclaimer
========================

The information contained herein is for informational purposes only,
and is subject to change without notice. While every precaution has
been taken in the preparation of this document, it may contain
technical inaccuracies, omissions and typographical errors, and AMD is
under no obligation to update or otherwise correct this information.
Advanced Micro Devices, Inc. makes no representations or warranties
with respect to the accuracy or completeness of the contents of this
document, and assumes no liability of any kind, including the implied
warranties of non-infringement, merchantability or fitness for
particular purposes, with respect to the operation or use of AMD
hardware, software or other products described herein.  No license,
including implied or arising by estoppel, to any intellectual property
rights is granted by this document.  Terms and limitations applicable
to the purchase or use of AMD’s products are as set forth in a signed
agreement between the parties or in AMD's Standard Terms and
Conditions of Sale.

AMD is a trademark of Advanced Micro Devices, Inc. Other product names
used in this publication are for identification purposes only and may
be trademarks of their respective companies.

Copyright (c) 2022 - 2024 Advanced Micro Devices, Inc. All rights
reserved.

