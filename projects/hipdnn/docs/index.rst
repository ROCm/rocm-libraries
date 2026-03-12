.. meta::
  :description: hipDNN (Deep Neural Network) is a graph-based library providing improved performance for deep learning workloads with AMD GPUs.
  :keywords: hipDNN, ROCm, documentation,

********************
hipDNN documentation
********************

hipDNN (Deep Neural Network) is a graph-based library providing improved performance for deep learning workloads with AMD GPUs.

hipDNN uses operation graphs as an intermediate representation to describe computations, allowing different backend engines to optimize and execute these graphs efficiently.

hipDNN provides an interace that follows established deep learning conventions and has a plugin-based architecture which allows advanced users to extend hipDNN without modifying the core library.

The public repository for hipDNN is located at `https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn>`_.

.. note::

  hipDNN is in beta-stage development and consequently not recommended for running production workloads at this time.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`hipDNN prerequisites <./install/hipdnn-prerequisites>`
    * :doc:`Quick start <./install/quickstart>`

  .. grid-item-card:: Conceptual

    * :doc:`High-level architecture <conceptual/architecture>`
    * :doc:`Engine configuration knobs <conceptual/knobs>`
    * :doc:`Backend architecture <conceptual/backend-architecture>`

  .. grid-item-card:: How to

    * :doc:`Build and execute operation graphs <how-to/build-execute-hipdnn>`
    * :doc:`Get/set engine knob configurations <how-to/get-set-engine-knob>`
    * :doc:`Develop plugins <how-to/develop-plugins>`
    * :doc:`Migrate a cudNN project to hipDNN <how-to/migrate-cudnn>`

  .. grid-item-card:: Reference

    * :doc:`Operation support <reference/plugins>`
    * :doc:`Environment variables <reference/environment-variables>`
    * :doc:`Glossary <reference/glossary>`


To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
