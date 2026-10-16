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
the experimental API changes, so upgrading hipBLASLt intentionally breaks the build of code that
uses the experimental API until the changes listed in the CHANGELOG are reviewed and the define is
updated.

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

* Every addition, change, or removal in the experimental API increments
  ``HIPBLASLT_EXPERIMENTAL_API_VERSION`` and is listed under an ``Experimental`` heading in the
  CHANGELOG.
* An experimental feature either graduates to the stable API or is removed within three releases
  of being introduced.
* When a feature graduates, the stable declaration is added under a new name. The experimental
  name remains available for one more release, then it is removed.

Experimental API reference
==========================

.. doxygendefine:: HIPBLASLT_EXPERIMENTAL_API_VERSION

.. doxygendefine:: HIPBLASLT_EXPERIMENTAL_ENUM_BIT

hipblasLtExperimentalGetApiVersion()
------------------------------------
.. doxygenfunction:: hipblasLtExperimentalGetApiVersion
