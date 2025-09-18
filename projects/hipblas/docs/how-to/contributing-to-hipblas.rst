.. meta::
  :description: How to contribute to hipBLAS
  :keywords: hipBLAS, rocBLAS, BLAS, ROCm, API, contribution

.. _contribute:

********************************************************************
Contributing to hipBLAS
********************************************************************

This topic explains how to contribute to the hipBLAS code base, including coding
and pull request (PR) guidelines and information about static code analysis.

Pull request guidelines
=======================

The hipBLAS code contribution guidelines closely follow the `GitHub
pull requests <https://help.github.com/articles/using-pull-requests/>`__ model.
The hipBLAS repository follows a workflow that dictates a ``release`` branch, from which releases are cut, and a
``develop`` branch, which serves as an integration branch for new code. Pull requests should
adhere to these guidelines:

*  Target the **develop** branch for integration.
*  Ensure code builds successfully.
*  Do not break existing test cases.
*  New unit tests should integrate with the existing GoogleTest framework.
*  Tests must have good code coverage.

Coding guidelines:
==================

*  Don't use unnamed namespaces inside of header files.
*  Use either ``template`` or ``inline`` (or both) for functions defined outside of classes in header files.
*  Do not declare namespace-scope (not ``class``-scope) functions ``static`` inside of header files
   unless:

   *  There is a very good reason.
   *  The function does not have any non-``const`` ``static`` local variables.
   *  It is acceptable that each compilation unit will have its own independent definition of the function and
      its ``static`` local variables.

   .. note::

      ``static`` ``class`` member functions defined in header files are okay.

*  Use ``static`` for ``constexpr`` ``template`` variables until C++17. After C++17,
   ``constexpr`` variables become ``inline`` variables and can be defined in multiple compilation units.
   It is okay if the ``constexpr`` variables remain ``static`` in C++17, but it means there might
   be some redundancy between compilation units.

Code format
-----------

C and C++ code is formatted using ``clang-format``. To run ``clang-format``,
use the version in the ``/opt/rocm/llvm/bin`` directory. Don't use your
system's built-in ``clang-format``, because it might be an older version that
could generate different results.

To format a file, use:

::

    /opt/rocm/llvm/bin/clang-format -style=file -i <path-to-source-file>

To format all files, run the following script in hipBLAS directory:

::

    #!/bin/bash
    git ls-files -z *.cc *.cpp *.h *.hpp *.cl *.h.in *.hpp.in *.cpp.in | xargs -0 /opt/rocm/llvm/bin/clang-format -style=file -i

The githooks application can also be installed to format the code on a per-commit basis:

::

    ./.githooks/install

Static code analysis
=====================

``cppcheck`` is an open-source static analysis tool. hipBLAS uses this tool to perform static code analysis.

Use the following command to run ``cppcheck`` locally and generate the report for all files.

.. code:: bash

   $ cd hipBLAS
   $ cppcheck --enable=all --inconclusive --library=googletest --inline-suppr -i./build --suppressions-list=./CppCheckSuppressions.txt --template="{file}:{line}: {severity}: {id} :{message}" . 2> cppcheck_report.txt

For more information on the ``cppcheck`` command line options, see the `cppcheck <http://cppcheck.net/>`_ manual.
