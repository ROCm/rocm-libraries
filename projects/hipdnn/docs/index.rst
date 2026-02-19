.. meta::
  :description: hipDNN (Deep Neural Network) is a graph-based deep learning library that enables multi-operation fusion for improved performance on AMD GPUs. 
  :keywords: hipDNN, ROCm, documentation,

********************
hipDNN documentation
********************

hipDNN (Deep Neural Network) is a graph-based deep learning library that enables multi-operation fusion for improved performance on AMD GPUs.
Each plugin implements specific operations with support for different datatypes, layouts, and features.
hipDNN allows developers to run deep learning workloads on AMD GPUS by providing an interface modeled after NVIDIA's cuDNN frontend API.

The component public repository is located at `https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`hipDNN prerequisites <./install/hipdnn-prerequisites>`
    * :doc:`hipDNN installation (Linux) <./install/hipdnn-install>`
    * :doc:`hipDNN installation (Windows) <./install/hipdnn-install-windows>`

  .. grid-item-card:: Conceptual

    * :doc:`High-level architecture <conceptual/architecture>`
    * :doc:`MIOpen Provider plugin architecture <conceptual/miopen-plugin>`
    * :doc:`Backend architecture <conceptual/backend-architecture>`
  
  .. grid-item-card:: How to

    * :doc:`Migrate a cudNN project to hipDNN <how-to/migrate-cudnn>`
    * :doc:`Build and execute operation graphs <how-to/build-execute-hipdnn>`
    * :doc:`Extend hipDNN functionality <how-to/extend-hipdnn>`
    * :doc:`Develop plugins <how-to/develop-plugins>`
    * :doc:`Get/set engine knob configurations <how-to/get-set-engine-knob>`

  .. grid-item-card:: Reference

    * :doc:`Environment variables <reference/environment-variables>`
    * :doc:`Coding style and naming guidelines <reference/naming-guidelines>`
    * :doc:`Glossary <reference/glossary>`


To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
