.. meta::
  :description: rocPyDecode documentation
  :keywords: rocPyDecode, rocDecode, ROCm, documentation, video, decode, decoding, acceleration

********************************************************************
rocPyDecode documentation
********************************************************************

rocPyDecode is a Python binding for the `rocDecode <https://rocm.docs.amd.com/projects/rocDecode/en/latest/>`_ APIs. It connects Python and C/C++ libraries, enabling function calling and data passing between the two languages.

rocPyDecode decodes compressed video streams while keeping the resulting decoded frames in video memory, avoiding unnecessary data copies. rocPyDecode lets you use the FFMpeg demultiplexer (demuxer) to seek and demultiplex (demux) packetized media files, and integrates with machine learning frameworks such as PyTorch to facilitate machine learning on decoded surfaces.
 
The rocPyDecode python library uses the rocDecode C++ SDK library to decode video streams based on the number of available media engines (VCNs) on the GPU. AMD GPUs contain one or more VCNs that can be used for accelerated, hardware-based video decoding. Hardware decoders offload decoding tasks from the CPU to the GPU, reducing power consumption and boosting decoding throughput.

The rocPyDecode public repository is located at `https://github.com/ROCm/rocPyDecode <https://github.com/ROCm/rocPyDecode>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`rocPyDecode prerequisites <./install/rocPyDecode-prerequisites>`
    * :doc:`rocPyDecode installation overview <./install/rocPyDecode-install>`
    * :doc:`rocPyDecode CMake installation <./install/rocPyDecode-cmake-install>`
    * :doc:`rocPyDecode pip installation <./install/rocPyDecode-pip-install>`

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Conceptual

    * :doc:`rocPyDecode surface data memory locations <./conceptual/rocPyDecode-mem-types>`
  
  .. grid-item-card:: How to

    * :doc:`Use rocPyDecode <how-to/using-rocPydecode>`

  .. grid-item-card:: Tutorials

    * `rocPyDecode samples <https://github.com/ROCm/rocPyDecode/tree/develop/samples>`_   
  

  .. grid-item-card:: Reference

    * :doc:`rocDecode supported codecs and hardware <reference/rocPyDecode-codecs-and-hardware>`
    * :doc:`rocPyDecode API reference<reference/rocPyDecode>`
      
      * :doc:`rocPyDecode structures <./reference/structures>`
      * :doc:`rocPyDecode decoder class <./reference/decoderClass>`
      * :doc:`rocPyDecode demuxer CLass <./reference/demuxerClass>`


To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
