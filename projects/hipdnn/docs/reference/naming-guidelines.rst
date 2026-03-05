.. meta::
  :description: Follow these best practices for coding style and naming guidelines when using hipDNN.
  :keywords: hipDNN, ROCm, coding, naming

.. _guidelines:

******************************************
hipDNN coding style and naming guidelines
******************************************

Follow these best practices for coding style and naming guidelines when using hipDNN.

Naming summary
==============

.. list-table::
   :widths: 3 3 5
   :header-rows: 1

   * - Kind
     - Format
     - Example
   * - Class / Struct
     - PascalCase
     - ``TensorDescriptor``
   * - Interface
     - I + PascalCase
     - ``ITensorView``
   * - Function
     - ``camelCase``
     - ``buildGraph()``
   * - Variable (local / parameter)
     - ``camelCase``
     - ``workspaceSize``
   * - Private Member variable
     - ``_camelCase``
     - ``_cachedPlan``
   * - Static variable
     - ``s_camelCase``
     - ``s_engineCount``
   * - Global variable
     - ``g_camelCase`` 
     - ``g_globalState``
   * - Constant / Macro 
     - UPPER_CASE
     - ``MAX_WORKSPACE_BYTES``
   * - Enum Type
     - PascalCase
     - ``EngineMode``
   * - Enum Value 
     - ``UPPER_SNAKE``
     - ``ENGINE_MODE_DEFAULT``

File and class naming
=====================

- Prefer the filename to match the primary class it contains: ``ExecutionPlan.hpp``, ``ExecutionPlan.cpp``.
- Utility collections (no single dominant class) may use a name that makes sense to describe the contents (for example, ``Error.hpp``).
- Interfaces: prefix ``I``, for example, ``IAllocator.hpp``.
- Keep one major class per file when practical.
- Avoid excessively long filenames; lean on directory structure for grouping.

Functions
=========

Use descriptive action-oriented verbs: ``createPlan``, ``finalizeConfig``, ``launchKernels``.

Unused function arguments
--------------------------

Prefer use of ``[[maybe_unused]]`` rather than commenting-out argument names or using ``std::ignore``.
Exception is for arguments that *shall not* be used, such as the case in a legacy version of a method that has an argument that is no longer relevant and shouldn't be used.  
In this case, comment-out the argument name and leave a comment indicating the reason.

Variables
=========

Favor clarity over abbreviation: prefer ``intermediateSize`` to ``intSz``.

Members
=======

- Private / protected data members: prefix single underscore ``_`` then camelCase (``_opGraph``).
- Static data members: ``s_camelCase``.
- Exposed constants inside a class: ``static constexpr`` UPPER_CASE.
- Plain structs whose intent is a passive aggregate (all or mostly public data). Do *not* prefix member names with ``_``; just use camelCase.
  
  - Rationale: underscores communicate encapsulation; POD-style structs are transparent.

Example:

.. code:: cpp

  struct TensorExtent {
      int n;
      int c;
      int h;
      int w;
  };


If later you add invariants or non-trivial behavior, consider converting to a class and applying the underscore rule to newly private members.

Globals
=======

Avoid unless absolutely required; prefix ``g_`` to make visibility explicit.

Interfaces
==========

- Naming: ``IInterfaceName``.
- Keep pure abstract; avoid data members.

Enums
=====

- Enum type name: PascalCase (for example, ``EngineMode``, ``ConvolutionMode``).
- Enumerator names: UPPER_SNAKE (``ENGINE_MODE_DEFAULT``, ``ENGINE_MODE_DETERMINISTIC``).
- When mirroring external APIs, keep exact enumerator spellings.

Constants
=========

- UPPER_CASE with optional single underscores: ``DEFAULT_ALIGNMENT``, ``MAX_TENSOR_RANK``.
- Prefer ``constexpr`` over macros when possible.

Namespaces
==========

- lower_snake_case with single underscores
- Nested namespaces should be defined on the same line, for example,
  
  .. code::

    ```
    namespace hipdnn_data_sdk::test_utilities::pointwise
    {
      ...
    } // namespace hipdnn_data_sdk::test_utilities::pointwise

- Do not use redundant namespace qualifiers (for example, do not use ``hipdnn_data_sdk::`` qualifier when inside the ``hipdnn_data_sdk`` namespace).
- Most code should fit generally within a few namespaces:
  
  - ``hipdnn_<component>``: (for example, ``hipdnn_frontend``) Contains all basic code required for the component.
  - ``utilities``: Contains code that can aid and assist in using component code.
  - ``test_utilities``: Contains code that can aid and assist in testing component code.

Test naming guidelines
======================

GoogleTest reserves underscores in test suite and test names for future expansion. Current repository names with underscores risk future incompatibility; we proactively constrain test suite naming.

Rules below apply ONLY to the TestSuite name (first parameter of ``TEST`` / ``TEST_F`` / ``TEST_P``). 
The TestCase (second parameter) can be descriptive but should still avoid the reserved keywords where noted.  
When writing parameterized tests the prefix(parameter name is ``InstantiationName``) in the ``INSTANTIATE_TEST_SUITE_P`` macro can be left blank or used for another purpose that make sense for that test.

Keywords
--------

- **Integration**: Only for integration tests, always first if present
- **Test**: Mainly for unit tests, always first if test is not an Integration test.
- **Gpu**: Optional after Integration or Test but before suite name if the test needs Gpu support.
- **Datatypes**: Bfp16, Fp16, Fp32. Always last if present.

Unit Tests
----------

In most cases unit style tests should be named so they directly mirror the class under test.  
If the class is named ``MyClass``, then the test suite should be named ``TestMyClass``. 
In general these kinds of tests should try to avoid using anything that requires Gpu support. 
This is not always possible, in the cases where Gpu support is required, the test suite should be named ``TestGpuMyClass``.

Naming examples
---------------

.. code:: cpp

  TestBackendLogger
  TestHandle
  TestGpuHandle
  TestBatchnormBwdPlan
  TestBatchnormBwdPlanFp32
  TestBatchnormBwdPlanFp16


File naming
-----------

The test file name should mirror the primary test suite it contains. For example, if the main test in a suite is ``TestMyClass``, the file should be named ``TestMyClass.cpp``.  
That same file may also contain ``TestGpuMyClass`` but it is not the primary test suite so the file name does not need to reflect it.

Integration tests
------------------

See `Testing Strategy on GitHub <https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipdnn/docs/testing/TestingStrategy.md>`_ for more information on integration tests. 
Integration tests should be named to reflect the feature or component under test.

Naming examples
---------------

.. code:: cpp

  IntegrationGpuBatchnormBackwardNchwFp32
  IntegrationGpuBatchnormBackwardNchwBfp16
  IntegrationGpuBatchnormBackwardNchwFp16
  IntegrationGraphFusion


File naming
-----------

For integration tests, the main test suite might be named ``IntegrationGpuFeatureX`` but have several child suites like ``IntegrationGpuFeatureXFp32`` and ``IntegrationGpuFeatureXBfp16``. 
The parent suite name is the primary suite, so the file name should be ``IntegrationGpuFeatureX`.cpp``.

Test case naming
----------------

May be richly descriptive ``HandlesLargeStride``, ``RejectsMismatchedLayouts`` or very simple ``Correctness``, ``Accuracy``. Avoid duplicating suite-level keywords (``Integration``, ``Gpu``, datatype tokens) redundantly inside the test case name. 
The test case name is the preferred place to list the shape/layout variant being tested (for example, ``Nchw``, ``Nhwc``).  
In general there should not be duplication across the suite name and test case name.  
Otherwise the naming of the test case is entirely up to the developer.

Rationale
---------

- Ordering enforces quick visual parsing (environment → scope → subject → specialization).
- Avoid underscores to remain future-proof with gtest evolution.
- Suffix datatype to emphasize functional context before precision variant.
- Consistent pattern simplifies filtering (for example, ``--gtest_filter=*Gpu*Fp32``).

Examples
========

Class and file
--------------

File: ``ExecutionPlan.hpp``

.. code:: cpp

  class ExecutionPlan {
  public:
      static constexpr int MAX_STEPS = 8;

      explicit ExecutionPlan(int initialSteps);
      void buildGraph();
      int stepCount() const;

  private:
      int _stepCount;
      bool _isFinalized;
  };


Interface
---------

.. code:: cpp

  class IAllocator {
  public:
      virtual ~IAllocator() = default;
      virtual void* allocate(size_t bytes) = 0;
      virtual void deallocate(void* ptr) = 0;
  };


Constant and enum
-----------------

.. code:: cpp

  enum EngineMode {
      ENGINE_MODE_DEFAULT = 0,
      ENGINE_MODE_DETERMINISTIC = 1
  };

  constexpr size_t MAX_WORKSPACE_BYTES = 1ull << 32;


Test (gtest)
------------

.. code:: cpp

  TEST(IntegrationGpuGraphFusionFp32, FusesThreeSequentialOps) {
      // ...
  }
  TEST(IntegrationGpuGraphFusionBfp16, FusesThreeSequentialOps) {
      // ...
  }

  TEST(TestExecutionPlan, BuildsGraphCorrectly) {
      // ...
  }


Decision checklist
==================

When adding new code, verify:

- Names follow the table in Section 1.
- File name matches main class (or is a justified utility collection).
- Test suite names follow ordering & allowed tokens.
- Layout tokens appear before datatype tokens when both used.
- No stray underscores in test suite names.

Deviation process
=================

If an external API or standard library interop forces divergence (for example,, fixed enum value names), document the exception with a brief comment near the declaration.

Automated tooling
=================

The repository includes automated tooling to enforce coding standards and maintain consistency across the codebase.

Clang-Tidy rules
----------------

The project uses clang-tidy to automatically enforce many of the coding style guidelines defined in this document. The configuration can be found in ``.clang-tidy`` at the repository root.

The CI pipeline automatically runs clang-tidy on all pull requests to ensure compliance before merging.

Test naming enforcement tool
----------------------------

A dedicated test naming enforcement tool is available to automatically validate that all test names follow the conventions outlined in Section 11.

The tool is located at ``cmake/scripts/test_name_validator.py`` and is integrated into the build system.

Use this command to run the validation manually:

.. code:: bash

  ninja validate_test_names


.. note::

  This validation also runs automatically as part of the ``ninja check`` target.

This tool:

- Parses all test executables to extract test names.
- Validates ordering of keywords (Integration, Gpu, Feature, Layout, Datatype).
- Checks for prohibited underscores in test suite names.
- Generates reports on non-compliant test names.
- Is integrated with CI to block merges with invalid test names.


Adhering to these rules and utilizing the automated tooling maintains readability, consistency, and tooling friendliness across the codebase.