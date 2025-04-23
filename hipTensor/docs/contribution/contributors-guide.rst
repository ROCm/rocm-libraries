.. meta::
   :description: A high-performance HIP library for tensor primitives
   :keywords: hipTensor, ROCm, library, API, tool, contribution

.. _contributors-guide:

============================
Contributing to hipTensor
============================

This document provides the coding guidelines to be followed while contributing to the hipTensor APIs.

License agreement
=================

1. The code I am contributing is mine and I have the right to license
   it.

2. By submitting a pull request for this project I am granting you a
   license to distribute said code under the MIT License for the
   project.

Pull-request guidelines
=======================

Our code contribution guidelines closely follows the model of `GitHub
pull-requests <https://help.github.com/articles/using-pull-requests/>`__.
The hipTensor repository follows a workflow which dictates a /master branch where releases are cut, and a
/develop branch which serves as an integration branch for new code. Follow the guidelines below while creating a pull request:

-  Target the **develop** branch for integration
-  Ensure that the code builds successfully
-  Do not break existing test cases
-  Be informed that a new functionality is only merged with new unit tests
-  Ensure that new unit tests integrate within the existing GoogleTest framework
-  Design the tests with good code coverage
-  Ensure that the code contains benchmark tests and performance approaches
   the compute bound limit or memory bound limit

Style guide
============

This project follows the `CPP Core
guidelines <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md>`__,
with few modifications and additions as given below. We encourage you to follow the below-mentioned guidelines while creating the pull requests.

Interface
---------

-  Use C++17 for the library code
-  Avoid camel case
-  The above-given rules apply specifically to publicly visible APIs, but is also
   encouraged (not mandated) for internal code

Philosophy
----------

-  Write in ISO Standard C++14 (especially to support Windows, Linux and
   macOS platforms )
-  Prefer compile-time check to run-time check

Implementation
--------------

-  Use a ``.cpp`` suffix for code files and an ``.hpp`` suffix for interface files if your project doesn't already follow another
   convention
-  Ensure that a ``.cpp`` file mandatorily includes the ``.hpp`` file(s) that defines its interface
-  Don't put a global ``using`` -directive in a header file
-  Use ``#include`` guards for all ``.hpp`` files
-  Don't use an unnamed (anonymous) ``namespace`` in a header
-  Prefer using ``std::array`` or ``std::vector`` instead of a C array
-  Minimize the exposure of class members
-  Keep functions short and simple
-  To return multiple 'out' values, prefer returning a ``std::tuple``
-  Manage resources automatically using RAII ( includes ``std::unique_ptr`` and ``std::shared_ptr``)
-  Use ``auto`` to avoid redundant repetition of type names
-  Always initialize an object
-  Prefer the ``{}`` initializer syntax
-  Expect your code to run as part of a multi-threaded program
-  Avoid global variables

Format
------

C++ code is formatted using ``clang-format``. To run clang-format,
use the version in the ``/opt/rocm/llvm/bin`` directory. Don't use your
system's built-in ``clang-format``, which could be an older version leading to different results.

To format a file, use:

.. code-block:: bash

   /opt/rocm/llvm/bin/clang-format -style=file -i <path-to-source-file>

To format all files, run the following script in the hipTensor directory:

.. code-block:: bash

   #!/bin/bash
   git ls-files -z *.cc *.cpp *.h *.hpp *.cl *.h.in *.hpp.in *.cpp.in | xargs -0 /opt/rocm/llvm/bin/clang-format -style=file -i

Also, to install githooks to format the code per-commit, use:

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

3. Create a pull request to ROCmSoftwarePlatform/hipTensor targeting develop branch.
4. Make sure to respond to code reviews.
5. Await CI and approval feedback.
6. Once approved, await for dev team to merge!

.. note::
    Please don't forget to install the githooks via ``.githooks/install`` as there are triggers for clang formatting in commits.
