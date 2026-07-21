.. meta::
  :description: Tensile documentation and API reference
  :keywords: Tensile, GEMM, Tensor, tensor, ROCm, API, Documentation

.. _cli-reference:

**********************
Tensile CLI reference
**********************

The Tensile project provides several command line tools. Here is the standard syntax for the CLI tools:

.. table:: Usage syntax for Tensile's command-line documentation

   ================================= ==================================
   Notation                          Description
   ================================= ==================================
   ``<Text inside angle brackets>``  Required argument.
   ``[Text inside square brackets]`` Optional arguments.
   ``{Text inside braces}``          Set of arguments. One is required.
   ================================= ==================================

To see the usage of ``TensileCreateLibrary`` tool, use ``help``:

.. code-block:: shell

   Tensile/bin/TensileCreateLibrary --help

.. warning::
   Consider undocumented command-line options experimental or deprecated.
