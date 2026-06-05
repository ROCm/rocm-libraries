.. meta::
  :description: hipDNN (Deep Neural Network) is a graph-based library providing improved performance for deep learning workloads with AMD GPUs.
  :keywords: hipDNN, ROCm, documentation

********************
hipDNN documentation
********************

hipDNN (Deep Neural Network) is a graph-based library that improves performance for deep learning workloads on AMD GPUs.

hipDNN uses operation graphs as an intermediate representation for describing computations, allowing different backend engines to optimize and execute them efficiently.

hipDNN provides an interface that follows established deep learning conventions and a plugin-based architecture, which allows advanced users to extend hipDNN without modifying the core library.

.. tip::

  hipDNN provides instructions to :ref:`migrate-cudnn`.

The public repository for hipDNN is located at `https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn>`_.

.. note::

  hipDNN is in beta. Running production workloads is not recommended.

  hipDNN is only supported on ROCm 7.12 or later.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`hipDNN installation <./install/installation>`
    * :doc:`Add hipDNN to your project <./install/add-project>`

  .. grid-item-card:: Conceptual

    * :doc:`High-level architecture <conceptual/architecture>`
    * :doc:`Engine configuration knobs <conceptual/knobs>`
    * :doc:`Backend architecture <conceptual/backend-architecture>`

  .. grid-item-card:: How to

    * :doc:`Build and execute operation graphs <how-to/build-execute-hipdnn>`
    * :doc:`Get and set engine knob configurations <how-to/get-set-engine-knob>`
    * :doc:`Migrate a cuDNN project to hipDNN <how-to/migrate-cudnn>`
    * :doc:`Develop plugins <how-to/develop-plugins>`

  .. grid-item-card:: Reference

    * :doc:`Operation support <reference/plugins>`
    * :doc:`Environment variables <reference/environment-variables>`
    * :doc:`Glossary <reference/glossary>`
    * :doc:`API reference <reference/api-reference>`

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
