.. meta::
  :description: Tensile is a tool for creating a benchmark-driven backend library for GEMM
  :keywords: Tensile developers guide, Tensile contributors guide, Tensile programmers guide, GEMM, Tensor
.. highlight:: none

.. _programmers-guide:

********************************************************************
Programmer's guide
********************************************************************

This topic provides necessary information for programmers interested in contributing to the Tensile source code.

.. _development-environment:

=======================
Development environment
=======================

ROCm is the base requirement for contributing to Tensile. See if ROCm is supported on your platform by verifying the `supported operating systems <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems>`_ list.
Then, follow the steps given in the :ref:`installation` guide.

-------------------------
Developing in Docker
-------------------------

ROCm development images are available on `Docker Hub <https://hub.docker.com/search?q=rocm%2Fdev>`_ for a variety of OS/ROCm versions. See `Docker images in the ROCm ecosystem <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html#docker-images-in-the-rocm-ecosystem>`_ for more details.

==================
Project structure
==================

Here is the project directory structure to help you find the project files available for contribution.

.. code-block::

   Tensile/
   ├── Tensile/              Source code, tests, and utilities for the Tensile project
   │   └── Tests/                Kernels and application tests
   ├── HostLibraryTests/     Tests for host-side code running the Tensile library
   ├── docker/               A collection of useful Dockerfiles
   ├── docs/                 Documentation source files
   ├── requirements.txt      Python dependencies for running Tensile applications
   ├── pytest.ini            Configuration settings for pytest
   ├── tox.ini               Configuration settings for the Tox environment management tool
   └── setup.py              Package build and installation script

=======
Testing
=======

Tensile uses `pytest <https://docs.pytest.org/>`_ to manage library or kernel tests. The Tensile project utilizes `pytest markers <https://docs.pytest.org/en/stable/how-to/mark.html>`_ to filter the tests to be run. Important markers include ``pre_checkin``, ``extended``, ``integration``, and ``unit``. Refer to `pytest.ini <https://github.com/ROCm/Tensile/blob/develop/pytest.ini>`_ for all supported markers.

You can run a test via the ``tox ci`` environment by passing the desired test marker using ``-m <MARKER>``:

.. code-block::

   tox run -e ci -- -m {pre_checkin|extended|integration|unit}

Note that ``--`` is used to pass options to the underlying pytest command.

.. note::

   By default, the ``tox run`` command runs pre-checkin tests, when no markers are specified via ``-m``.

-------------------------------
Unit tests and coverage reports
-------------------------------

All unit tests are available in ``Tensile/Tests/unit/``. A convenience command is included to add coverage reporting:

.. code-block::

   tox run -e unittest

   # OR for 32 processes

   tox run -e unittest -- -n 32

By default, coverage results are dumped to the terminal. To generate reports in other formats such as HTML, use:

.. code-block::

   tox run -e unittest -- --cov-report=html

Files and directories excluded from coverage reporting are itemized in `.coveragerc <https://github.com/ROCm/Tensile/blob/develop/.coveragerc>`_.

Although, we encourage to run unit tests using ``tox`` for consistency, you can also run the tests directly using ``pytest`` for quicker feedback. For example, To run a single test named ``test_foo``, use:

.. code-block::

   pytest unit/test_TensileCreateLibrary.py -k "test_foo" --capture=no -v

------------------
Host library tests
------------------

Host library tests ensure that the generated libraries remain operational when called from the client code such as other libraries or applications.
These tests are built on `gtest <https://github.com/google/googletest>`_. To run them, download the submodule first. Then, from Tensile project's root, run:

.. code-block::

   git submodule update --init

Next, you can configure, build, and run the host library tests using any of the following:

- ``tox``:

  .. code-block::

   tox run -e hostlibtest

  .. note::

   Note that the ``tox`` command wraps `invoke <https://www.pyinvoke.org/index.html>`_, a tool to manage CLI-invokable tasks. Since tox is fundamentally a Python environment manager and test runner, any reusable shell commands that fall outside its purview are managed by invoke (which are again encapsulated by tox sometimes). See `tasks.py <https://github.com/ROCm/Tensile/blob/develop/tasks.py>`_ for details.

- ``invoke``:

  .. code-block::

   invoke hostlibtest --configure --build --run

  Running the preceding command generates an executable ``TensileTests``, which can be further used to run the tests.

- Manually: To build and run the tests manually, see the commands in `tasks.py <https://github.com/ROCm/Tensile/blob/develop/tasks.py>`_.
  For advanced usage like filtering or repeating test cases, see the `gtest documentation <https://github.com/google/googletest/blob/main/docs/advanced.md>`_.

===============
Static analysis
===============

------
Python
------

To run all static analysis, use the top-level ``tox`` label ``static``:

.. code-block::

   tox run -m static

.. note::
   The preceding command might reformat your code, so make sure to commit your changes after running the command.

**Linting** is evaluated using `flake8 <https://flake8.pycqa.org/en/latest/>`_ and **formatting** is conducted using `black <https://black.readthedocs.io/en/stable/>`_ and `isort <https://pycqa.github.io/isort/>`_. To run a check in isolation, either refer to `tox.ini <https://github.com/ROCm/Tensile/blob/develop/tox.ini>`_ or use one the following commands:

.. code-block::

   tox run -e lint
   tox run -e format     # add `-- --check` to check formatting without applying changes
   tox run -e isort      # add `-- --check` to check imports without applying changes


.. tip::

   To ensure consistent formatting, we recommend you to set up the editor to **format on save** using the same formatter settings as in `tox.ini <https://github.com/ROCm/Tensile/blob/develop/tox.ini>`_. Either way, ensuring to commit changes after running static analysis reduces wait times caused by simple CI failures.

---
C++
---

**Formatting** is conducted using `clang-format <https://clang.llvm.org/docs/ClangFormatStyleOptions.html>`_.
The following command formats all given files, however, we recommend you to setup the editor to *format on save*.

.. code-block::

   clang-format -i style=file <files>

Styling rules are configured in `.clang-format <https://github.com/ROCm/Tensile/blob/develop/.clang-format>`_.

=========
Profiling
=========

------
Python
------

To enable profiling, use the ``@profile`` decorator, which must be imported from the ``Tensile.Utilities.Profile`` module. Under the hood, the decorator wraps the function in a `cProfile <https://docs.python.org/3/library/profile.html#module-cProfile>`_ context and generates a ``.prof`` file inside the ``profiling-results-<date>`` directory.

.. note::
   Nested profiling is NOT supported due to the existing limitation with the profiling decorator. This implies that if `func1` calls `func2` in a loop, and both are marked for profiling, the resulting ``.prof`` file for `func1` will display incorrect results.

============
External CI
============

`Azure Pipelines <https://dev.azure.com/ROCm-CI/ROCm-CI/_build?definitionId=256>`_ are run for every pull request and commit targeting the develop and mainline branches.
The pipeline packages up the wheel file and runs pre_checkin tests on a gfx942 system.
`Click <https://dev.azure.com/ROCm-CI/ROCm-CI/_build?definitionId=256>`_ on the job corresponding to the pull request or commit to view execution logs.

========================
Building documentation
========================

To build the documentation locally, use:

.. code-block::

   tox run -e docs

After the documentation is built, the HTML files are generated in ``docs/_build/html``.

=====================
Versioning practices
=====================

Tensile follows semantic versioning practices such as **major.minor.patch**. See `server.org <https://semver.org/>`_ for details.
