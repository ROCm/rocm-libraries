.. meta::
   :description: hipBLASLt experimental API reference
   :keywords: hipBLASLt, ROCm, library, API, reference, experimental

.. _experimental-api:

********************************
hipBLASLt experimental API
********************************

hipBLASLt ships some features through an experimental API before they become part of the
stable :doc:`hipBLASLt API <./api-reference>` or :doc:`hipBLASLtExt API <./ext-reference>`.
The experimental API lets early adopters evaluate a feature and give feedback while its interface
is still being refined.

.. warning::

   The experimental API is not covered by the hipBLASLt backward compatibility guarantees. Any
   experimental function, type, or enumerator can change or be removed in any ROCm release,
   including patch releases. Code that uses it must keep a working path without it.

Enabling the experimental API
=============================

The experimental API is declared in ``hipblaslt/hipblaslt-experimental.h``, which is installed
with every hipBLASLt build but is not included by ``hipblaslt/hipblaslt.h``. The header declares
nothing unless the consumer defines ``ROCM_ENABLE_EXPERIMENTAL_APIS`` to a nonzero value before
including it. The same switch enables the experimental APIs of every ROCm library; there is no
per-library switch.

.. code-block:: c

   #define ROCM_ENABLE_EXPERIMENTAL_APIS 1
   #include <hipblaslt/hipblaslt-experimental.h>

CMake consumers can link ``roc::hipblaslt_experimental`` instead of ``roc::hipblaslt``. It is
``roc::hipblaslt`` plus the define, and it is exported with the hipBLASLt package but not listed in
``hipblaslt_LIBRARIES``:

.. code-block:: cmake

   find_package(hipblaslt CONFIG REQUIRED)
   target_link_libraries(my_target PRIVATE roc::hipblaslt_experimental)

Any other build system can set the define, for example
``target_compile_definitions(my_target PRIVATE ROCM_ENABLE_EXPERIMENTAL_APIS=1)``.

Defining it to ``0`` turns the experimental API off explicitly. Including the header without the
switch is not an error, so code that checks for features (below) builds either way.

Checking for a feature
======================

Each experimental feature defines a revision macro, ``HIPBLASLT_EXPERIMENTAL_<FEATURE>``, set to
the feature's current revision. The revision starts at 1 and increases only when the feature
changes in a way that can break existing callers. Additions don't change it.

Because the shared library version does not track experimental changes, also check at runtime
that the loaded library implements the revision you compiled against.
``hipblasLtExperimentalGetFeatureRevision`` returns the loaded library's revision of a feature, or
0 if it doesn't have the feature. Code that uses a feature should check both and fall back when
either doesn't match:

.. code-block:: cpp

   #if defined(HIPBLASLT_EXPERIMENTAL_PARTITIONED_GEMM) && HIPBLASLT_EXPERIMENTAL_PARTITIONED_GEMM == 1
       if(hipblasLtExperimentalGetFeatureRevision("PARTITIONED_GEMM") == 1)
       {
           /* Use the experimental feature. */
       }
       else
   #endif
       {
           /* Stable path. */
       }

Code that cannot work without a feature can report an error instead of falling back. Framework
code, which is built both with and without the switch, must keep the fallback.

Experimental enumerators
========================

Experimental values of stable enum types, such as new epilogues or matmul descriptor attributes,
are declared in their feature's header with ``HIPBLASLT_EXPERIMENTAL_ENUM_BIT`` set. They are
passed to the existing stable functions, for example ``hipblasLtMatmulDescSetAttribute``. When an
experimental enumerator graduates, it is assigned a new value in the stable enum.

Experimental enumerator values are never renumbered or reused, even after the enumerator is
removed. ``HIPBLASLT_EXPERIMENTAL_ENUM_HIGH_WATER`` records the highest value ever assigned and
only increases. A removed value is rejected with ``HIPBLAS_STATUS_INVALID_VALUE``. The mark is
shared by every experimental enum, C++ scoped enums included, so a new enum's first value is
above it: one library-wide number is what lets the check compare only two versions of the
headers and still never reuse a value.

Lifecycle
=========

* A breaking change never changes a declaration in place. A function whose signature or
  documented behavior changes gets a new name (for example with a ``_v2`` suffix) and the old name
  is removed, so a program built against the old revision fails to load or falls back instead of
  calling the new code with old arguments. The same applies to structs, unions and typedefs: any
  change to one, including adding a member, gives it a new name, and the functions that use it
  get new names too. Each breaking change increments the feature's revision
  and is listed under an ``Experimental`` heading in the CHANGELOG.
* C++ features follow the same rule, applied to what decides which symbol a caller binds to and
  what memory it assumes:

  * A function or method is identified by its scope, name and parameter types. Adding a method
    or an overload is an addition. Removing one is a breaking change. A different parameter list
    is a different overload, so changing one is a removal plus an addition. The return type,
    default arguments and qualifiers (``const``, ``noexcept``, ``= delete``) never change under
    the same parameter list. Renaming a parameter is not a change.
  * A class's layout is its bases, its data members and its virtual functions. Any change to it
    gives the class a new name. A class that keeps its state behind a private pointer to an
    implementation (as ``PartitionedGemm`` does) can then gain methods freely. Private methods
    are not part of the API.
  * Templates are not allowed in experimental headers, because they can't be compared without
    instantiating them.
* An experimental feature ships in at most four ROCm releases, counted from the release that
  introduced it (patch releases don't count). Revision increments don't reset this. By then it
  either graduates to the stable API or is removed. A feature can be extended by one release,
  once, with an approver recorded in the registry.
* At most three experimental features exist at a time.
* When a feature graduates, the stable declaration is added under a new name and the
  experimental feature, including its revision macro, is removed in the same release. After that,
  ``hipblasLtExperimentalGetFeatureRevision`` returns 0 for it.

Adding or changing an experimental feature
==========================================

These rules are checked automatically, so they cannot be forgotten.

* Each feature has its own header, ``library/include/hipblaslt/hipblaslt-experimental-<feature>.h``,
  which ``hipblaslt-experimental.h`` includes and which defines the feature's revision macro.
  Enumerators must have explicit integer values.
* Each feature is registered in ``library/experimental/features.cmake`` with an owner, a tracking
  issue, the ROCm release that introduced it, and the last ROCm release that may ship it. Releases
  are counted on the ordered list in ``cmake/rocm_feature_releases.cmake``, and the release being
  built is ``HIPBLASLT_ROCM_RELEASE``. CMake configure warns in the last release and fails after
  it, so the change that starts the next release cannot merge until the feature graduates, is
  removed, or is extended. Configure also fails if a feature header is not registered, is
  missing its revision macro, or if there are too many features.
* Each feature has contract tests in ``clients/tests/src/experimental/contract_<feature>*_gtest.cpp``.
  They define the feature's promised behavior. Adding a contract test is free; changing or
  deleting one is a breaking change. Headers they share, such as test helpers, are named
  ``contract_<feature>*.hpp`` in the same directory and are part of the contract too. Every CI
  lane checks part of the contract: a feature that is limited to some architectures has a
  contract test that it refuses on the others, so lanes without the supported hardware don't
  just skip.
* ``tools/scripts/check_experimental_api.py`` runs on every pull request that touches the
  experimental API. It compares the declarations and contract tests with the target branch and
  fails if:

  * a function, C++ method, struct, union, typedef, C++ class layout or enumerator was changed in
    place, an enumerator value was reused, or
    ``HIPBLASLT_EXPERIMENTAL_ENUM_HIGH_WATER`` went down or is too low;
  * a declaration, contract test or contract header was removed or changed without incrementing
    that feature's revision;
  * a header declares a template;
  * a revision went backwards, or a new feature doesn't start at revision 1 or higher.

  It warns when a revision was incremented with no breaking change. Run it locally with:

  .. code-block:: shell

     python3 tools/scripts/check_experimental_api.py --base origin/develop

  The checker cannot see implementation changes. If a change alters a feature's behavior, a
  contract test fails, and updating that test triggers the check.

Experimental API reference
==========================

.. doxygendefine:: HIPBLASLT_EXPERIMENTAL_ENUM_BIT

.. doxygendefine:: HIPBLASLT_EXPERIMENTAL_ENUM_HIGH_WATER

hipblasLtExperimentalGetFeatureRevision()
-----------------------------------------
.. doxygenfunction:: hipblasLtExperimentalGetFeatureRevision

Experimental feature PARTITIONED_GEMM
-------------------------------------

``PartitionedGemm`` is supported on gfx1250 devices only. On any other architecture
``setProblem`` returns ``HIPBLAS_STATUS_NOT_SUPPORTED``.

``hipblaslt_experimental::PartitionedGemm`` (C++ only) runs one GEMM as slices on the devices of a
partitioned GPU, for example the two partitions of a DPX card, where each partition is a separate
HIP device. The caller passes one hipBLASLt handle per partition (created with that partition's
device current), and one stream, workspace, and event per partition when it initializes and runs.
``SplitDim`` picks the dimension to split (M, N, or the batch). The first ``N - 1`` partitions take
``size / N`` rounded up to a 256-byte boundary of every matrix they cut when that changes the
balance by at most 1%, and the last takes the rest; otherwise they take ``size / N`` and a hint is
logged (``HIPBLASLT_LOG_MASK=4``). A slice that starts off a 256-byte boundary runs much slower, so
pad the leading dimensions (and batch strides) to multiples of 256 bytes, which is enough for N and
Batch splits; for an M split also make M a multiple of ``parts * 256 / element size``. hipBLASLt
does not allocate or copy any matrix, and does not enable peer access: every partition's slices
must be accessible from its device (local memory, or peer or VMM access that the caller has set
up), otherwise the run faults like any GEMM given an inaccessible pointer. Run in DPX mode for
speed; SPX (every partition on one device) gives correct results but is only meant for checking
them without a DPX GPU. ``setProblem`` queries where the memory is, so call it when a pointer or
the shape changes, not on every launch. The caller passes one pointer per partition for each
matrix: the same pointer shares a matrix between the partitions, different pointers give each
partition its own copy. Where the memory lives (``hipMalloc`` on one device, or a VMM range with
pages on several devices) is up to the caller: the library gives each partition the slice whose
pages are on that partition's device, and ``getSlice`` reports it. See
``clients/samples/31_hipblaslt_gemm_partitioned_ext`` for an example, including the revision
checks, and ``hipblaslt-bench --partitions`` to benchmark it. This current implementation supports
plain GEMM; the device user arguments overloads are declared and return
``HIPBLAS_STATUS_NOT_SUPPORTED``.

.. doxygendefine:: HIPBLASLT_EXPERIMENTAL_PARTITIONED_GEMM

.. doxygenclass:: hipblaslt_experimental::PartitionedGemm
    :members:
    :protected-members:
    :private-members:

.. doxygenenum:: hipblaslt_experimental::SplitDim
