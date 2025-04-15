.. meta::
   :description: A high-performance HIP library for tensor primitives
   :keywords: hipTensor, cuTensor, ROCm, library, API, tool

.. _programmers-guide:

===================
Programming guide
===================

This document provides insight into the library source code organization, design implementation details, helpful information for new development, and testing and benchmarking details.

--------------------------------
Infrastructure
--------------------------------

- Doxygen and Sphinx are used to generate the project's documentation.
- Jenkins is used to automate Continuous Integration (CI) testing (``.jenkins`` folder has configurations).
- hipTensor is hosted and maintained by AMD on `Github  <https://github.com/ROCm/hipTensor>`_.
- The hipTensor project is organized and configured via ``CMake`` and the collection of ``CMakeLists.txt`` in the base of each directory.
- ``clang-format`` is used to format C++ code. ``.githooks/install`` ensures that a clang-format pass will run on each committed file.
- ``GTest`` is used to implement test suite organization and execution.
- ``CTest`` is used to consolidate and invoke multiple test targets. In the ``<hipTensor_install_dir>/CTestTestfile.cmake`` file, testing targets are listed that will be run when ``ctest`` is invoked.
- The preferred compiler for hipTensor is ``CC=<path_to_rocm>/bin/amdclang and CXX=<path_to_rocm>/bin/amdclang++``. ``hipcc`` is also supported, however may be deprecated in future ROCm releases.

--------------------------------
General Design Concepts
--------------------------------

hipTensor is developed with the ``C++17`` language standard. The library takes advantage of several meta-programming techniques that help to statically
optimize code at compile time and generate more efficient GPU kernels. hipTensor employs Composable Kernel as a functional backend, therefore the library is written in different layers.

The outer API layer serves as a functional interface for the user to define tensor data abstractions and manipulations. The second layer is the hipTensor solution interface that bridges the communication gap
between the API objects and the desired functionality. The solution layer encompasses the translation of the input problem parameters into solution candidates, candidate selection, resource management and logging.
Solution candidates provide interface abstractions into functional backends such as Composable Kernel objects which may be invoked and whose results are passed back up through the API. The Composable Kernel library
is consumed as a header library where all kernel instances are customized by hipTensor and statically bundled which is managed by the hipTensor functional backend layer. This way if additional backends were
to be considered in the future, the backends could be isolated into their own modules as they are now. The hipTensor solution layer is also split up into functional components, such as contraction, permutation and reduction. Each component contains a registry of backend instances which are held as potential solution candidates
to a given set of input parameters. These instances go through selection processing as directed with hints from the API, and are populated with appropriate arguments and readied for invocation by the API.

hipTensor tests and samples are consumers of the hipTensor library and demonstrate the usages of the API in different contexts, such as tensor contractions, permutations and reductions.

--------------------------------
Nomenclature
--------------------------------

Tensor contraction
^^^^^^^^^^^^^^^^^^^

In general, a tensor contraction is a multiply-accumulate problem over elements between two multi-dimensional tensors. hipTensor will use the Einstein notation for the contraction notation. Repeated indices are summed over, and each index may appear a maximum of twice in each mathematical term.
In the process of accumulating over summation dimensions, they are effectively collapsed, or contracted.

Tensor permutation
^^^^^^^^^^^^^^^^^^^

Tensor permutation is essentially the re-ordering of the stride indices such that the data dimensional locality relationships are changed.

Tensor reduction
^^^^^^^^^^^^^^^^^^^

Tensor reduction refers to obtaining a lower-dimensional tensor from a higher-dimensional one by performing some operation on the original tensor.

Tensor rank
^^^^^^^^^^^

In terms of tensor rank, this is considered the dimensionality of the data. For example this would be the number of modes. We consider Einstein notation in contractions such as repeated modes are
the dimensions which are contracted.

In contractions, we may differentiate modes in terms of M's, N's and K's, in which:

* Tensor A modes [M0, ..., Mn, K0, ..., Kn]
* Tensor B modes [N0, ..., Nn, K0, ..., Kn]
* Tensor C/D modes [M0, ..., Mn, N0, ..., Nn]

and repeated indices K0, ..., Kn are indices that are contracted. Contractions currently support up to M6N6K6 which means we may have up to 6 dimensions for each M, N and K.
A tensor contraction with A = [M0, ..., M5, K0, ..., K5] B = [N0, ..., N5, K0, ..., K5] and C/D [M0, ..., M5, N0, ..., N5] would be considered a rank 12 contraction.

Tensor mode
^^^^^^^^^^^

Tensor modes are a way for the user to easily specify the ordering or labeling of the input strides that define the dimensional data relationship. This is used to describe how
the data is laid out in memory and how they are related to each other spatially.

--------------------------------
Library source code organization
--------------------------------

The hipTensor code is split into four major parts:

- The ``library`` directory contains the library source code.
- The ``samples`` directory contains real-world use-cases of the hipTensor API.
- The ``test`` directory contains validation tests for hipTensor API.
- Infrastructure

``library`` directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``library`` directory contains the following include and source files:

- ``library/include/hiptensor/``: C++ include files for the hipTensor API. These files also contain Doxygen comments that document the API.
- ``library/include/hiptensor/internal``: Include files for utility code and generate tensor utility.
- ``library/src/``: Source files for Logger, device, and performance functions.
- ``library/src/contraction/``: Source files for core initialization and management of contraction module.
- ``library/src/contraction/device``: Source files for composable kernel backend bilinear and scale instances.
- ``library/src/permutation/``: Source files for core initialization and management of permutation module.
- ``library/src/permutation/device``: Source files for composable kernel backend permute instances.
- ``library/src/reduction/``: Source files for core initialization and management of reduction module.
- ``library/src/reduction/device``: Source files for composable kernel backend reduction instances.
- ``library/src/include``: Infrastructure support for backend and logging management.

``samples`` directory
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``samples`` directory contains the sample codes for the following simple demonstrations:

- ``01_contraction/simple_bilinear_contraction``: Abstract base test for bilinear contractions.
- ``01_contraction/simple_scale_contraction``: Abstract base test for scale contractions.
- ``01_contraction/simple_bilinear_contraction_*``: Specialized bilinear contraction tests per data type.
- ``01_contraction/simple_scale_contraction_*``: Specialized bilinear contraction tests per data type.
- ``02_permutation/permutation``: Simple permutation demonstration.
- ``03_reduction/reduction``: Simple reduction demonstration.

``test`` directory
^^^^^^^^^^^^^^^^^^^^^^^

The ``test`` directory contains the test codes for testing the following functionalities:

- ``00_unit/logger_test``: Tests logger API functions of hipTensor.
- ``00_unit/yaml_test``: Tests the YAML serialization / de-serialization for testing parameters.
- ``01_contraction/contraction_test``: Testing harness for the bilinear and scale contractions.
- ``01_contraction/complex_*_contraction``: Testing harness for the bilinear and scale contractions with complex data types.
- ``01_contraction/contraction_resource``: Shared resource infrastructure for testing contractions.
- ``01_contraction/configs``: YAML files with actual contraction testing parameters.
- ``02_permutation/permutation*``: Testing infrastructure for permutation tests.
- ``02_permutation/rank*``: Testing harnesses for permutation of a particular rank.
- ``02_permutation/configs``: YAML files with actual permutation testing parameters.
- ``03_conduction/conduction*``: Testing infrastructure for conduction tests.
- ``03_conduction/rank*``: Testing harnesses for conduction of a particular rank.
- ``03_conduction/configs``: YAML files with actual conduction testing parameters.

``performance`` directory
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``scripts/performance`` directory contains the benchmarking scripts:

- ``BenchmarkContraction.sh``: Benchmarking script for contraction
- ``BenchmarkPermutation.sh``: Benchmarking script for permutation
- ``BenchmarkReduction.sh``: Benchmarking script for reduction

``emulation test`` script
^^^^^^^^^^^^^^^^^^^^^^^^^

The emulation test script ``rtest.py`` is located within the project's root directory.

Contributing
^^^^^^^^^^^^

For those wishing to contribute to the project, please see :ref:`contributors-guide`.
