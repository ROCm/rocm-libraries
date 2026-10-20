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
   experimental function, type, or enumerator can change or be removed in any release, including
   minor and patch releases. Do not build production integrations on the experimental API.

Enabling the experimental API
=============================

The experimental API is declared in ``hipblaslt/hipblaslt-experimental.h``, which is installed
with every hipBLASLt build but is not included by ``hipblaslt/hipblaslt.h``. To use it, define
``HIPBLASLT_ENABLE_EXPERIMENTAL_API`` to the experimental API version you have reviewed before
including the header:

.. code-block:: c

   #define HIPBLASLT_ENABLE_EXPERIMENTAL_API 1
   #include <hipblaslt/hipblaslt-experimental.h>

Alternatively, set the define from your build system, for example with CMake:

.. code-block:: cmake

   target_compile_definitions(my_target PRIVATE HIPBLASLT_ENABLE_EXPERIMENTAL_API=1)

The header fails to compile if ``HIPBLASLT_ENABLE_EXPERIMENTAL_API`` is not defined, or if its
value does not match ``HIPBLASLT_EXPERIMENTAL_API_VERSION``. The version is incremented whenever
the experimental API changes in a way that can break existing callers, so upgrading hipBLASLt
intentionally breaks the build of code that uses the experimental API until the changes listed in
the CHANGELOG are reviewed and the define is updated. Additions do not increment the version.

Because the shared library version does not track experimental changes, also check at runtime
that the loaded library implements the version you compiled against:

.. code-block:: c

   if(hipblasLtExperimentalGetApiVersion() != HIPBLASLT_EXPERIMENTAL_API_VERSION)
   {
       /* Fall back to the stable API. */
   }

Experimental enumerators
========================

Experimental values of stable enum types, such as new epilogues or matmul descriptor attributes,
are declared in ``hipblaslt-experimental.h`` with ``HIPBLASLT_EXPERIMENTAL_ENUM_BIT`` set. They are
passed to the existing stable functions, for example ``hipblasLtMatmulDescSetAttribute``. When an
experimental enumerator graduates, it is assigned a new value in the stable enum.

Lifecycle
=========

* ``HIPBLASLT_EXPERIMENTAL_API_VERSION`` is incremented by any breaking change: removing or
  renaming a declaration, changing a signature or enumerator value, changing documented
  behavior, or graduating a feature. Each increment is listed under an ``Experimental`` heading
  in the CHANGELOG. Adding a feature, function, or enumerator does not increment the version.
* An experimental feature ships in at most four hipBLASLt releases (counted in minor versions;
  patch releases don't count). By then it either graduates to the stable API or is removed. A
  feature can be extended by one release, once, with an approver recorded in the registry.
* At most three experimental features exist at a time.
* When a feature graduates, the stable declaration is added under a new name and the
  experimental declaration is removed in the same release, which increments the version.

Adding or changing an experimental feature
==========================================

These rules are checked automatically, so they cannot be forgotten.

* Each feature has its own header, ``library/include/hipblaslt/hipblaslt-experimental-<feature>.h``,
  which ``hipblaslt-experimental.h`` includes. Enumerators must have explicit values.
* Each feature is registered in ``library/experimental/features.cmake`` with an owner, a tracking
  issue, the release that introduced it, and the last release that may ship it. CMake configure
  warns in the last release and fails after it, so the version bump that starts the next release
  cannot merge until the feature graduates, is removed, or is extended. Configure also fails if
  a feature header is not registered or if there are too many features.
* Each feature has contract tests in ``clients/tests/src/experimental/contract_<feature>_gtest.cpp``.
  They define the feature's promised behavior. Adding a contract test is free; changing or
  deleting one is a breaking change.
* ``tools/scripts/check_experimental_api.py`` runs on every pull request that touches the
  experimental API. It compares the declarations and contract tests with the target branch and
  fails if anything was removed or changed without incrementing
  ``HIPBLASLT_EXPERIMENTAL_API_VERSION``. Run it locally with:

  .. code-block:: shell

     python3 tools/scripts/check_experimental_api.py --base origin/develop

  The checker cannot see implementation changes. If a change alters a feature's behavior, a
  contract test fails, and updating that test triggers the check.

Experimental API reference
==========================

.. doxygendefine:: HIPBLASLT_EXPERIMENTAL_API_VERSION

.. doxygendefine:: HIPBLASLT_EXPERIMENTAL_ENUM_BIT

hipblasLtExperimentalGetApiVersion()
------------------------------------
.. doxygenfunction:: hipblasLtExperimentalGetApiVersion

Experimental feature DEMO
-------------------------
.. doxygenfunction:: hipblasLtExperimentalDemoAlignSize

.. doxygenenum:: hipblasLtExperimentalDemoRounding_t
