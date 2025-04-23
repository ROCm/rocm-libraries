.. meta::
  :description: rocAL documentation and API reference library
  :keywords: rocAL, ROCm, API, documentation

.. _rocal:

********************************************************************
rocAL documentation
********************************************************************

The ROCm Augmentation Library (rocAL) lets you improve the throughput and performance of your deep learning applications. It's designed to efficiently decode and process image and video pipelines from a variety of storage formats on AMD GPUs and CPUs. Its C++ and Python APIs let you program customizable pipelines for different datasets and models. rocAL is optimized for loading and pre-processing data for deep learning applications, with support for multiple data formats and augmentations. 

The rocAL public repository is located at `https://github.com/ROCm/rocAL <https://github.com/ROCm/rocAL>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Installation

      * :doc:`rocAL prerequisites <./install/rocAL-prerequisites>`
      * :doc:`Installing rocAL with the package installer <./install/rocAL-package-install>`
      * :doc:`Building and installing rocAL from source <./install/rocAL-build-and-install>`

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Conceptual

    * :ref:`overview`
    * :ref:`architecture`

  .. grid-item-card:: How to

    * :doc:`Use the rocAL C++ API <./how-to/using-with-cpp>`
    * :doc:`Use the rocAL Python API <./how-to/using-with-python>`
    * :doc:`Use ML framework integration <./how-to/framework>`

  .. grid-item-card:: Examples
    
    * `rocAL image processing examples <https://github.com/ROCm/rocAL/tree/master/docs/examples/image_processing>`_ 
    * `rocAL PyTorch examples <https://github.com/ROCm/rocAL/tree/master/docs/examples/pytorch>`_ 
    * `rocAL TensorFlow examples <https://github.com/ROCm/rocAL/tree/master/docs/examples/tf>`_
    * `rocAL Jupyter notebooks <https://github.com/ROCm/rocAL/tree/master/docs/examples/notebooks>`_ 

  .. grid-item-card:: Reference

    * :doc:`rocAL RNNT dataloading <./reference/rocAL-and-RNNT>`  
    * `rocAL API Modules <https://rocm.docs.amd.com/projects/rocAL/en/latest/doxygen/html/modules.html>`_
    * `rocAL API <https://rocm.docs.amd.com/projects/rocAL/en/latest/doxygen/html/group__group__rocal.html>`_
    * `rocAL Datatypes <https://rocm.docs.amd.com/projects/rocAL/en/latest/doxygen/html/group__group__rocal__types.html>`_
    * `rocAL Augmentation API <https://rocm.docs.amd.com/projects/rocAL/en/latest/doxygen/html/group__group__rocal__augmentations.html>`_
    * `rocAL Data Loaders API <https://rocm.docs.amd.com/projects/rocAL/en/latest/doxygen/html/group__group__rocal__data__loaders.html>`_
    * `rocAL Data Transfer API <https://rocm.docs.amd.com/projects/rocAL/en/latest/doxygen/html/group__group__rocal__data__transfer.html>`_
    * `rocAL Info API <https://rocm.docs.amd.com/projects/rocAL/en/latest/doxygen/html/group__group__rocal__info.html>`_
    * `rocAL Metadata API <https://rocm.docs.amd.com/projects/rocAL/en/latest/doxygen/html/group__group__rocal__meta__data.html>`_
    * `rocAL Parameter API <https://rocm.docs.amd.com/projects/rocAL/en/latest/doxygen/html/group__group__rocal__parameters.html>`_
    * `rocAL Header Files <https://rocm.docs.amd.com/projects/rocAL/en/latest/doxygen/html/files.html>`_

To contribute to the documentation refer to `Contributing to ROCm Docs <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.

