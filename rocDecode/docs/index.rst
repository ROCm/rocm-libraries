.. meta::
  :description: rocDecode documentation and API reference library
  :keywords: rocDecode, ROCm, API, documentation, video, decode, decoding, acceleration

********************************************************************
rocDecode documentation
********************************************************************

rocDecode provides APIs, utilities, and samples that you can use to easily access the video decoding
features of your media engines (VCNs). It also allows interoperability with other compute engines on
the GPU using Video Acceleration API (VA-API)/HIP. To learn more, see :doc:`what-is-rocDecode`

The rocDecode public repository is located at `https://github.com/ROCm/rocDecode <https://github.com/ROCm/rocDecode>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Installing rocDecode with the package installer <./install/rocDecode-package-install>`
    * :doc:`Building and installing rocDecode from source code <./install/rocDecode-build-and-install>`
    * `rocDecode Docker containers <https://github.com/ROCm/rocDecode/tree/develop/docker>`_

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Conceptual

    * :doc:`Video decoding pipeline <./conceptual/video-decoding-pipeline>`
    * :doc:`rocDecode surface memory locations <./conceptual/rocDecode-memory-types>`

  .. grid-item-card:: How to

    * :doc:`Understand the rocDecode videodecode.cpp sample <./how-to/using-rocDecode-videodecode-sample>`
    * :doc:`Use the rocDecode RocVideoDecoder <./how-to/using-rocDecode-video-decoder>`
    * :doc:`Use the rocDecode FFmpeg demultiplexer <./how-to/using-rocDecode-ffmpeg>`
    * :doc:`Use the rocDecode bitstream reader APIs <./how-to/using-rocDecode-bitstream>`    
    * :doc:`Use the rocDecode core APIs <./how-to/using-rocdecode>`


  .. grid-item-card:: Samples

    * :doc:`rocDecode samples <./tutorials/rocDecode-samples>`

  .. grid-item-card:: Reference

    * :doc:`rocDecode codec support and hardware capabilities <./reference/rocDecode-formats-and-architectures>`
    * :doc:`rocDecode tested configurations <./reference/rocDecode-tested-configurations>`
    * :doc:`API library <../doxygen/html/files>`
    * :doc:`Functions <../doxygen/html/globals>`
    * :doc:`Data structures <../doxygen/html/annotated>`
  
To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
