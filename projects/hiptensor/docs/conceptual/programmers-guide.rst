.. meta::
   :description: A high-performance HIP library for tensor primitives
   :keywords: hipTensor, cuTensor, ROCm, library, API, tool

.. _programmers-guide:

===================
Programming guide
===================

This document provides insight into the library source code organization, design implementation, development guidance, and testing and benchmarking details.

--------------------------------
Infrastructure
--------------------------------

- Doxygen and Sphinx are used to generate the project's documentation.
- Jenkins is used to automate Continuous Integration (CI) testing, with configurations stored in the ``.jenkins`` folder.
- hipTensor is hosted and maintained by AMD on `GitHub  <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hiptensor>`_.

  .. note::

    The hipTensor repository for ROCm 7.1.1 and earlier is located at `<https://github.com/ROCm/hipTensor>`_.

- The hipTensor project is organized and configured using ``CMake``, with ``CMakeLists.txt`` files in the root of each directory.
- ``clang-format`` is used to format C++ code. ``.githooks/install`` ensures that a clang-format pass will run on each committed file.
- ``GTest`` is used to implement test suite organization and execution.
- ``CTest`` is used to consolidate and invoke multiple test targets. The ``<hipTensor_install_dir>/CTestTestfile.cmake`` file lists the testing targets executed when ``ctest`` is invoked.
- The preferred compiler for hipTensor is ``CC=<path_to_rocm>/bin/amdclang and CXX=<path_to_rocm>/bin/amdclang++``. ``hipcc`` is also supported, however may be deprecated in future ROCm releases.

--------------------------------
Design concepts
--------------------------------

hipTensor is a library developed with the ``C++20`` language standard. It uses meta-programming techniques to optimize code at compile time and generate efficient GPU kernels.
hipTensor employs Composable Kernel as a functional backend and is written in different layers.

The outer API layer serves as a functional interface for users to define tensor data abstractions and manipulations.
The second layer, the hipTensor solution interface, connects the API objects with the intended functionality.
The solution layer handles translating input problem parameters into solution candidates, selecting candidates, managing resources, and logging.
Solution candidates provide interface abstractions into functional backends, such as Composable Kernel objects, which can be invoked with results returned through the API.
The Composable Kernel library is used as a header library, where all kernel instances are customized by hipTensor and statically bundled by the hipTensor functional backend layer. This allows future backends to remain isolated in their own modules, as they are now.
The hipTensor solution layer is divided into functional components, such as contraction, permutation, and reduction.
Each component contains a registry of backend instances as potential solution candidates for given input parameters.
These instances are selected based on API-provided hints and populated with the appropriate arguments for invocation by the API.

hipTensor tests and samples are consumers of the hipTensor library and demonstrate the usages of the API in different contexts, such as tensor contractions, permutations and reductions.

--------------------------------
Nomenclature
--------------------------------

Tensor contraction
^^^^^^^^^^^^^^^^^^^

In general, a tensor contraction is a multiply-accumulate problem over elements between two or more multi-dimensional tensors.
hipTensor uses Einstein notation, where repeated indices are summed and each index appears at most twice in each mathematical term.
In the process of accumulating over summation dimensions, they are effectively collapsed or contracted.

hipTensor supports the following contraction forms:

* **Binary bilinear contraction**: :math:`D = \alpha \, A \, B + \beta \, C`,
  where :math:`A`, :math:`B`, :math:`C` are input tensors and :math:`D` is the
  output tensor. Created with ``hiptensorCreateContraction`` and executed with
  ``hiptensorContract``.

* **Binary scale contraction**: :math:`D = \alpha \, A \, B`. The scale form is
  selected by passing a null tensor descriptor (and a null pointer at execution
  time) for :math:`C`; the bilinear API is reused with the bias term omitted.

* **Trinary bilinear contraction**: :math:`E = \alpha \, A \, B \, C + \beta \, D`,
  where :math:`A`, :math:`B`, :math:`C` are input tensors, :math:`D` is the
  bias input, and :math:`E` is the output. Created with
  ``hiptensorCreateContractionTrinary`` and executed with
  ``hiptensorContractTrinary``.

* **Trinary scale contraction**: :math:`E = \alpha \, A \, B \, C`. The scale
  form is selected by passing a null tensor descriptor (and a null pointer at
  execution time) for :math:`D`; the trinary API is reused with the bias term
  omitted.

All four contraction forms support per-input-tensor unary element-wise
operators (for example, ``HIPTENSOR_OP_RELU``, ``HIPTENSOR_OP_EXP``,
``HIPTENSOR_OP_COS``) applied before the contraction, in addition to the
default ``HIPTENSOR_OP_IDENTITY``.

Tensor permutation
^^^^^^^^^^^^^^^^^^^

Tensor permutation reorders stride indices, changing the data dimensional locality relationships.

Tensor reduction
^^^^^^^^^^^^^^^^^^^

Tensor reduction transforms a higher-dimensional tensor into a lower-dimensional one by performing some operations on the original tensor.

Tensor rank
^^^^^^^^^^^

Tensor rank refers to the data dimensionality, such as the number of modes. In Einstein notation, repeated modes indicate the dimensions being contracted during tensor contractions.

In contractions, modes can be categorized as M, N, and K, defined as follows:

* Tensor A modes [M0, ..., Mn, K0, ..., Kn]
* Tensor B modes [N0, ..., Nn, K0, ..., Kn]
* Tensor C/D modes [M0, ..., Mn, N0, ..., Nn]

Repeated indices K0, ..., Kn are indices that are contracted. Contractions currently support up to M6N6K6, allowing up to 6 dimensions for each M, N and K.
A tensor contraction with A = [M0, ..., M5, K0, ..., K5], B = [N0, ..., N5, K0, ..., K5], and C/D [M0, ..., M5, N0, ..., N5] is considered rank 12.

Tensor mode
^^^^^^^^^^^

Tensor modes let users specify the order or labels of the input strides that define the dimensional data relationship.
They describe the data's memory layout and spatial relationships.

--------------------------------
Library source code organization
--------------------------------

The hipTensor code is split into four major parts:

- The ``library`` directory contains the library source code.
- The ``samples`` directory contains real-world use-cases of the hipTensor API.
- The ``test`` directory contains validation tests for hipTensor API.
- Infrastructure.

``library`` directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``library`` directory contains the following include and source files:

- ``library/include/hiptensor/``: C++ include files for the hipTensor API. These files also contain Doxygen comments that document the API.
- ``library/include/hiptensor/internal``: Include files for utility code and tensor utility generation.
- ``library/src/``: Source files for logging, device management, and performance functions.
- ``library/src/contraction/``: Source files for core initialization and management of contraction module.
- ``library/src/contraction/device``: Source files for composable kernel backend bilinear and scale instances.
- ``library/src/elementwise/``: Source files for core initialization and management of permutation module.
- ``library/src/elementwise/device``: Source files for composable kernel backend permute instances.
- ``library/src/reduction/``: Source files for core initialization and management of reduction module.
- ``library/src/reduction/device``: Source files for composable kernel backend reduction instances.
- ``library/src/include``: Infrastructure support for backend and logging management.

``samples`` directory
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``samples`` directory contains the sample codes for the following demonstrations:

- ``01_contraction/simple_bilinear_contraction``: Abstract base code for bilinear contractions.
- ``01_contraction/simple_scale_contraction``: Abstract base code for scale contractions.
- ``01_contraction/simple_bilinear_contraction_*``: Specialized bilinear contraction demonstrations per data type.
- ``01_contraction/simple_scale_contraction_*``: Specialized bilinear contraction demonstrations per data type.
- ``01_contraction/simple_trinary_bilinear_contraction_*``: Specialized trinary bilinear contraction demonstrations per data type.
- ``01_contraction/simple_trinary_scale_contraction_*``: Specialized trinary scale contraction demonstrations per data type.
- ``01_contraction/simple_contraction_plan_cache``: Simple contraction with plan cache demonstration.
- ``01_contraction/simple_contraction_c``: Simple contraction demonstration in C.
- ``02_elementwise/elementwise_permute``: Simple permutation demonstration.
- ``02_elementwise/elementwise_binary``: Simple element-wise binary operation demonstration.
- ``02_elementwise/elementwise_trinary``: Simple element-wise trinary operation demonstration.
- ``02_elementwise/elementwise_binary_c``: Simple element-wise binary operation demonstration in C.
- ``03_reduction/reduction``: Simple reduction demonstration.
- ``03_reduction/reduction_c``: Simple reduction demonstration in C.

``test`` directory
^^^^^^^^^^^^^^^^^^^^^^^

The ``test`` directory contains the test codes for testing the following functionalities:

- ``00_unit/logger_test``: Tests logger API functions of hipTensor.
- ``00_unit/yaml_test``: Tests the YAML serialization and de-serialization for testing parameters.
- ``01_contraction/bilinear_contraction_test_*``: Testing harness for the bilinear contractions.
- ``01_contraction/scale_contraction_test_*``: Testing harness for the scale contractions.
- ``01_contraction/complex_bilinear_contraction_test_*``: Testing harness for the bilinear contractions with complex data types.
- ``01_contraction/complex_scale_contraction_test_*``: Testing harness for the scale contractions with complex data types.
- ``01_contraction/trinary_contraction_test``: Testing harness for the trinary contractions.
- ``01_contraction/trinary_bilinear_contraction_test``: Testing harness for the trinary bilinear contractions.
- ``01_contraction/trinary_scale_contraction_test``: Testing harness for the trinary scale contractions.
- ``01_contraction/bilinear_contraction_with_unary_ops_test``: Testing harness for binary bilinear contractions with non-identity unary operators.
- ``01_contraction/scale_contraction_with_unary_ops_test``: Testing harness for binary scale contractions with non-identity unary operators.
- ``01_contraction/contraction_resource``: Shared resource infrastructure for testing contractions.
- ``01_contraction/trinary_contraction_resource``: Shared resource infrastructure for testing trinary contractions.
- ``01_contraction/configs``: YAML files with actual contraction testing parameters.
- ``02_elementwise/elementwise_*``: Testing infrastructure for element-wise operation tests.
- ``02_elementwise/rank?_elementwise_permute_*``: Testing harnesses for permutation of a particular rank.
- ``02_elementwise/rank?_elementwise_binary_op*``: Testing harnesses for elementwise binary operation of a particular rank.
- ``02_elementwise/rank?_elementwise_trinary_op*``: Testing harnesses for elementwise trinary operation of a particular rank.
- ``02_elementwise/configs``: YAML files with actual permutation testing parameters.
- ``03_conduction/conduction*``: Testing infrastructure for conduction tests.
- ``03_conduction/rank*``: Testing harnesses for conduction of a particular rank.
- ``03_conduction/configs``: YAML files with actual conduction testing parameters.

``performance`` directory
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``scripts/performance`` directory contains the benchmarking scripts:

- ``BenchmarkContraction.sh``: Benchmarking script for contraction.
- ``BenchmarkPermutation.sh``: Benchmarking script for permutation.
- ``BenchmarkReduction.sh``: Benchmarking script for reduction.

``emulation test`` script
^^^^^^^^^^^^^^^^^^^^^^^^^

The emulation test script ``rtest.py`` is located in the project's root directory.

Contributing
^^^^^^^^^^^^

To contribute to the project, see :ref:`contributors-guide`.
