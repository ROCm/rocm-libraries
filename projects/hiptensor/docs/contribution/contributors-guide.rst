.. meta::
   :description: A high-performance HIP library for tensor primitives
   :keywords: hipTensor, ROCm, library, API, tool, contribution

.. _contributors-guide:

============================
Contributing to hipTensor
============================

This document provides the coding guidelines for contributing to the hipTensor APIs.

License agreement
=================

1. The code I am contributing is mine and I have the right to license
   it.

2. By submitting a pull request for this project, I am granting you a
   license to distribute said code under the MIT License for the
   project.

Pull-request guidelines
=======================

Our code contribution guidelines closely follows the model of `GitHub
pull-requests <https://help.github.com/articles/using-pull-requests/>`__.
The hipTensor repository uses a workflow with a ``release`` branch for releases and a
``develop`` branch for integrating new code. Follow the guidelines below when creating a pull request:

-  Target the ``develop`` branch for integration.
-  Ensure that the code builds successfully.
-  Do not break existing test cases.
-  Be informed that a new functionality is only merged with new unit tests.
-  Ensure that new unit tests integrate within the existing GoogleTest framework.
-  Design the tests with thorough code coverage.
-  Include benchmark tests and ensure performance approaches the compute-bound or memory-bound limits.


Style guide
============

This project follows the `CPP Core
guidelines <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md>`__,
with a few modifications and additions listed below. Follow these guidelines when creating pull requests.

Interface
---------

-  Use C++17 for the library code.
-  Avoid camel case.

.. note::

   The above rules specifically apply to publicly visible APIs, but are also encouraged, though not required, for internal code.

Philosophy
----------

-  Use ISO Standard C++14 to ensure compatibility across Windows, Linux, and macOS platforms.
-  Prefer compile-time check to run-time check.

Implementation
--------------

-  Use the ``.cpp`` suffix for code files and ``.hpp`` suffix for interface files, if your project doesn't already follow another
   convention.
-  Ensure that each ``.cpp`` file includes the ``.hpp`` file(s) that defines its interface.
-  Avoid placing global ``using`` directives in header files.
-  Use ``#include`` guards for all ``.hpp`` files.
-  Avoid using unnamed (anonymous) ``namespace``\ s in header files.
-  Use ``std::array`` or ``std::vector`` instead of C arrays.
-  Minimize the exposure of class members.
-  Keep functions short and simple.
-  To return multiple output values, prefer using a ``std::tuple``.
-  Manage resources automatically using RAII, including ``std::unique_ptr`` and ``std::shared_ptr``.
-  Use ``auto`` to avoid redundant repetition of type names.
-  Always initialize an object.
-  Use the ``{}`` initializer syntax wherever possible.
-  Expect your code to run as part of a multi-threaded program.
-  Avoid using global variables.

Format
------

C++ code is formatted using ``clang-format``. To run ``clang-format``,
use the version in the ``/opt/rocm/llvm/bin`` directory. Avoid using your
system's built-in ``clang-format``, as it might be outdated and produce inconsistent results.

To format a file, use:

.. code-block:: bash

   /opt/rocm/llvm/bin/clang-format -style=file -i <path-to-source-file>

To format all files, run the following script in the hipTensor directory:

.. code-block:: bash

   #!/bin/bash
   git ls-files -z *.cc *.cpp *.h *.hpp *.cl *.h.in *.hpp.in *.cpp.in | xargs -0 /opt/rocm/llvm/bin/clang-format -style=file -i

To install githooks to format the code per-commit, use:

.. code-block:: bash

   ./.githooks/install

Example
-------

1. Create and track a hipTensor fork.
2. Clone your fork:

   .. code-block:: bash

      git clone -b develop https://github.com/<your_fork>/hipTensor.git .
      .githooks/install
      git checkout -b <new_branch>
      ...
      git add <new_work>
      git commit -m "What was changed"
      git push origin <new_branch>
      ...

3. Create a pull request to ``ROCm/hipTensor``, targeting the ``develop`` branch.
4. Respond to code reviews.
5. Await CI and approval feedback.
6. Once approved, await for dev team to merge!

.. note::
    
   Install the githooks via ``.githooks/install``, as there are triggers for clang formatting in commits.
