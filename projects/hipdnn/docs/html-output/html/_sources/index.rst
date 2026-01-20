.. meta::
  :description: hipDNN is a plugin-based deep learning library that provides graph-based operation support through various backend plugins
  :keywords: hipDNN, ROCm, documentation,

********************
hipDNN documentation
********************

hipDNN (Deep Neural Network) is a plugin-based deep learning library that provides graph-based operation support through various backend plugins. 
Each plugin implements specific operations with support for different datatypes, layouts, and features.
hipDNN allows developers to run deep learning workloads on ROCm and AMD GPUs while maintaining compatibility with NVIDIA's cuDNN API.

The component public repository is located at `https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn <https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipdnn>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`hipDNN prerequisites <./install/hipdnn-prerequisites>`
    * :doc:`hipDNN installation (Linux) <./install/hipdnn-install>`
    * :doc:`ipDNN installation (Windows) <./install/hipdnn-install-windows>`

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Conceptual

    * :doc:`hipDNN high-level architecture <conceptual/architecture>`
    * :doc:`MIOpen Provider plugin architecture <conceptual/miopen-plugin>`
  
  .. grid-item-card:: How to

    * :doc:`Build and execute operation graphs in hipDNN <how-to/build-execute-hipdnn>`

  .. grid-item-card:: Reference

    * :doc:`hipDNN environment variables <reference/environment-variables>`
    * :doc:`hipDNN coding style and naming guidelines <reference/naming-guidelines>`
    * :doc:`Glossary <reference/glossary>`

  .. grid-item-card:: Advanced

    * :doc:`hipDNN backend architecture <conceptual/backend-architecture>`
    * :doc:`Extend hipDNN functionality <how-to/extend-hipdnn>`
    * :doc:`Develop plugins for hipDNN <how-to/develop-plugins>`

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
