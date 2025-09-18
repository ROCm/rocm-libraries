.. meta::
  :description: rocSPARSE contribution guidelines
  :keywords: rocSPARSE, ROCm, API, documentation, contributing

.. _contributing-to:

*************************
Contributing to rocSPARSE
*************************

AMD welcomes and values community contributions to rocSPARSE, including bug reports, bug fixes, documentation enhancements,
performance notes, and other improvements. Follow these details to ensure your contributions will be successfully accepted.

The rocSPARSE code contribution guidelines closely follow the `GitHub pull-requests <https://help.github.com/articles/using-pull-requests/>`_ model.
This repository uses the `git-flow <http://nvie.com/posts/a-successful-git-branching-model/>`_ workflow,
which includes a ``release`` branch that is used to cut releases and a ``develop`` branch which serves as an integration branch for new code.

Issue discussion
================

Use the GitHub Issues tab to notify the rocSPARSE team of issues.

*  Use your best judgement when creating an issue. If the issue is already listed, upvote it and
   comment or post to provide additional details, such as how you reproduced this issue.
*  If you're not sure whether your issue is the same, error on the side of caution and file your issue.
   You can add a comment to include the issue number (and link) for the similar issue. If
   your issue is determined to be the same as the existing issue, the rocSPARSE team will close the duplicate.
*  If your issue doesn't already exist, use the issue template to file a new issue.

   *  When filing an issue, provide as much information as possible, including script output, so the rocSPARSE team
      can collect information about your configuration. This helps reduce the time required to reproduce your issue.
   *  Check your issue regularly, to see whether additional information is required to successfully reproduce the issue.

*  You can also open an issue to ask the maintainers questions about whether a proposed change
   meets the acceptance criteria or to discuss an idea pertaining to the library.

Acceptance criteria
===================

rocSPARSE is a library that contains basic linear algebra subroutines for sparse matrices and vectors. It's written in HIP for GPU devices
and is designed to be used with C and C++ code. The functionality of rocSPARSE is organized into the following categories:

* :ref:`rocsparse_auxiliary_functions_`: Helper functions that are required for subsequent library calls.
* :ref:`rocsparse_level1_functions_`: Operations between a vector in sparse format and a vector in dense format.
* :ref:`rocsparse_level2_functions_`: Operations between a matrix in sparse format and a vector in dense format.
* :ref:`rocsparse_level3_functions_`: Operations between a matrix in sparse format and multiple vectors in dense format.
* :ref:`rocsparse_extra_functions_`: Operations that manipulate sparse matrices.
* :ref:`rocsparse_precond_functions_`: Manipulations on a matrix in sparse format to obtain a preconditioner.
* :ref:`rocsparse_conversion_functions_`: Operations on a matrix in sparse format to obtain a different matrix in sparse format.
* :ref:`rocsparse_reordering_functions_`: Operations on a matrix in sparse format to obtain a reordering.
* :ref:`rocsparse_utility_functions_`: Routines for checking sparse matrices for valid data.
* :ref:`rocsparse_generic_functions_`: Generic routines that allow using different index and data types.

The rocSPARSE team is interested in contributions that:

* Fix bugs, improve documentation, enhance testing, and reduce complexity.
* Improve the performance of existing routines.
* Add missing functionality found in one of the categories above.
* Add additional sparse matrix formats or allow an existing format to be used with an existing routine.

Some of the routines in rocSPARSE let users choose between multiple different algorithms.
This is useful for obtaining the best performance for different use cases
(because some algorithms might perform better than others depending on the sparse matrix) or for satisfying
important user requirements (such as run-to-run reproducibility).
The following is a non-exhaustive list of reasons for including alternate algorithms:

* They might perform better when a sparse matrix has roughly the same number of non-zeros per row.
* They might perform better when a sparse matrix has a large variation in the number of non-zeros per row.
* They might perform better if they are allowed to use a large amount of device memory.
* They might perform better or worse if the intent is to perform the computation only once or many times.
* They might allow for reproducibility between runs, for example, by not using atomic operations.
* They might not require any additional memory allocation or analysis phase.
* They might handle different ranges in the sparse matrix size, for instance, the number of rows or number of non-zeros.

The opportunity exists for contributors to add different algorithms that are optimized for important user requirements
and performance considerations. Contributors are encouraged to leverage the GitHub "Issues" tab to discuss possible additions.

Exceptions
----------

rocSPARSE places a heavy emphasis on high performance. Because of this, contributions that add new routines (or that modify existing routines)
must take the perspective of offering high performance on the hardware they run on. In rocSPARSE, this evaluation
is typically done using approximations of GFLOPS/s or GB/s, comparing the results to what the device is estimated to achieve.
Comparison to other sparse math libraries is also useful.

Additionally, new routines must offer enough value to enough users to be included,
because compile times, binary sizes, and general library complexity are important considerations.
The rocSPARSE team reserves the right to decide whether a proposed routine is too niche or specialized to be worth including.

Code structure
==============

This section describes the structure of the rocSPARSE library in the GitHub repository.
A more detailed description of the directory structure can be found in the :doc:`rocSPARSE design notes <../conceptual/rocsparse-design>`.

*  ``library/include/``: Contains the ``rocsparse.h`` header (which itself includes headers defining the rocSPARSE public API). This directory also contains the headers for all the rocSPARSE public types.
*  ``library/src/``: Contains the implementations of all the rocSPARSE routines. It includes subdirectories for each category of routines, for instance, ``level1``, ``level2``, and ``level3``. These directories contain both the C++ and HIP kernel code.
*  ``clients/``: Contains the testing and benchmarking code and samples demonstrating rocSPARSE usage.
*  ``docs/``: Contains the documentation files.
*  ``scripts/``: Contains potentially useful Python and shell scripts for downloading test matrices (see ``scripts/performance/matrices/``) as well as plotting tools. See the :doc:`rocSPARSE design notes <../conceptual/rocsparse-design>` for more details.

Coding style
============

In general, follow the style of the surrounding code. C and C++ code is formatted using ``clang-format``.
Use the clang-format version installed with ROCm (found in the ``/opt/rocm/llvm/bin`` directory).
Do not use the built-in ``clang-format`` for your system. This is a different version that might result in incorrect results.

To format a file, use:

.. code-block:: shell

   /opt/rocm/hcc/bin/clang-format -style=file -i <path-to-source-file>

To format all files, run the following script in rocSPARSE directory:

.. code-block:: shell

   #!/bin/bash
   git ls-files -z *.cc *.cpp *.h *.hpp *.cl *.h.in *.hpp.in *.cpp.in | xargs -0 /opt/rocm/hcc/bin/clang-format  -style=file -i

You can also install githooks to format the code on a per-commit basis:

.. code-block:: shell

   ./.githooks/install

Pull request guidelines
=======================

When you create a pull request (PR), target the default branch. This is the ``develop`` branch,
which also serves as the integration branch.

Deliverables
------------

When raising a PR in rocSPARSE, here are some important steps to follow:

*  For each new file in the repository, include the licensing header and adjust the date to the current year:

   .. code-block:: cpp
      :caption: rocsparse_file_header

      /* ************************************************************************
      * Copyright (C) 20xx Advanced Micro Devices, Inc. All rights Reserved.
      *
      * Permission is hereby granted, free of charge, to any person obtaining a copy
      * of this software and associated documentation files (the "Software"), to deal
      * in the Software without restriction, including without limitation the rights
      * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
      * copies of the Software, and to permit persons to whom the Software is
      * furnished to do so, subject to the following conditions:
      *
      * The above copyright notice and this permission notice shall be included in
      * all copies or substantial portions of the Software.
      *
      * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
      * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
      * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
      * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
      * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
      * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
      * THE SOFTWARE.
      *
      * ************************************************************************ */

   When a file is only modified, the pre-commit script should automatically update the date.

*  When adding a new routine, ensure you also add appropriate testing code. These new unit tests should integrate within the existing `GoogleTest framework <https://github.com/google/googletest/blob/main/docs/primer.md>`_.
   This typically involves adding the following files to the ``clients/testing/`` directory:

   *  ``testing_<routine_name>.cpp``
   *  ``test_<routine_name>.cpp``
   *  ``test_<routine_name>.yaml``

   Review the existing tests for guidance.

*  When modifying an existing routine, add appropriate test coverage to the ``test_<routine_name>.yaml`` file in the  ``clients/tests/`` directory.

*  Tests must have good code coverage.

*  At a minimum, rocSPARSE supports the following data and compute formats.
   When adding a new routine that uses data or compute values, you must support these four types at a minimum.

   * ``float``
   * ``double``
   * ``rocsparse_float_complex``
   * ``rocsparse_double_complex``

*  Ensure the code builds successfully. This includes ensuring that the code can compile, that it's properly formatted, and that all tests pass.

*  Do not break existing test cases.

Process
-------

When a PR is raised for the ``develop`` branch in the rocSPARSE repository, the CI process is automatically triggered.
The CI does the following:

*  Verifies that the PR passes static analysis (for example, it ensures the clang formatting rules have been followed).
*  Verifies the documentation can be properly built.
*  Ensures the PR compiles on different operating system and GPU device architecture combinations.
*  Ensures all tests pass on different operating system  and GPU device architecture combinations.

In your PR, feel free to ask questions about any CI failures you might encounter. The reviewers are listed in the ``CODEOWNERS`` file.
