.. meta::
  :description: rocPyDecode supported codex and hardware capabilities
  :keywords: install, rocPyDecode, AMD, ROCm, GPU, codec
  
********************************************************************
rocPyDecode supported codecs and hardware capabilities
********************************************************************

rocPyDecode supports the 8-bit and 10-bit H.265 (HEVC) codec, and the 8-bit H.264 (AVC) codec.


The following table shows the codec support and capabilities of the VCN for each supported GPU
architecture.

.. csv-table::
  :header: "GPU Architecture", "VCN Generation", "Number of VCNs", "Max width, Max height - H.265", "Max width, Max height - H.264"

  "gfx908", "VCN 2.5.0", "2", "4096, 2176", "4096, 2160"
  "gfx90a", "VCN 2.6.0", "2", "4096, 2176", "4096, 2160"
  "gfx942", "VCN 3.0", "3", "7680, 4320", "4096, 2176"
  "gfx1030, gfx1031, gfx1032", "VCN 3.x", "2", "7680, 4320", "4096, 2176"
  "gfx1100, gfx1102", "VCN 4.0", "2", "7680, 4320", "4096, 2176"
  "gfx1101", "VCN 4.0", "1", "7680, 4320", "4096, 2176"
