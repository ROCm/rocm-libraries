.. meta::
  :description: Introduction to the rocSOLVER documentation and API reference library
  :keywords: rocSOLVER, ROCm, API, documentation

.. _rocsolver:

********************************************************************
rocSOLVER documentation
********************************************************************

rocSOLVER implements `LAPACK routines <https://www.netlib.org/lapack/index.html>`_
on top of the :doc:`AMD ROCm platform <rocm:index>`. rocSOLVER is implemented in the
:doc:`HIP programming language <hip:index>` and optimized for AMD GPUs.

The rocSOLVER public repository is located at `<https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocsolver>`_.

The rocSOLVER repository for ROCm 7.0.2 and earlier is located at `<https://github.com/ROCm/rocSOLVER>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Installation guide <./installation/installlinux>`

  .. grid-item-card:: How to

    * :doc:`Use rocSOLVER <./howto/using>`
    * :doc:`Apply the memory model <./howto/memory>`
    * :doc:`Use multi-level logging <./howto/logging>`
    * :doc:`Run rocSOLVER clients <./howto/clients>`
    * :doc:`Contribute to rocSOLVER <./howto/contribute>`

  .. grid-item-card:: Examples

    * `Client samples <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocsolver/clients/samples>`_

  .. grid-item-card:: API reference

    * :doc:`rocSOLVER API introduction <./reference/intro>`
    * :doc:`rocSOLVER types <./reference/types>`
    * :doc:`rocSOLVER precision support <./reference/precision>`
    * :doc:`rocSOLVER environment variables <./reference/env_variables>`
    * :doc:`LAPACK auxiliary functions <./reference/auxiliary>`
    * :doc:`LAPACK functions <./reference/lapack>`
    * :doc:`LAPACK-like functions <./reference/lapacklike>`
    * :doc:`Refactorization and direct solvers <./reference/refact>`
    * :doc:`Library and logging functions <./reference/helpers>`
    * :doc:`rocSOLVER performance tuning <./reference/tuning>`
    * :doc:`Deprecated components <./reference/deprecated>`

To contribute to the documentation, see `Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
