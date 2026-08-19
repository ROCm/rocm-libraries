.. meta::
  :description: ROCm Performance Primitives (RPP) supported functionalities
  :keywords: RPP, ROCm, Performance Primitives, documentation, support, functionalities, audio, image

********************************************************************
ROCm Performance Primitives supported functionalities and variants
********************************************************************

The following tables show the CPU and GPU support for ROCm Performance Primitives (RPP) functionalities and variants.

CPU support is also referred to as HOST support, and GPU support is provided through the HIP backend.

The functionalities are grouped to match the RPP API headers described in the :doc:`API reference <./rpp-api-reference>`.

Color augmentations
-----------------------------------------------------------------------------------------------

.. csv-table::
  :widths: 7, 3, 3
  :header: "Type", "CPU", "GPU"

  "brightness", "✅", "✅"
  "gamma correction", "✅", "✅"
  "blend", "✅", "✅"
  "hue", "✅", "✅"
  "saturation", "✅", "✅"
  "color twist", "✅", "✅"
  "color jitter", "✅", "❌"
  "color cast", "✅", "✅"
  "exposure", "✅", "✅"
  "contrast", "✅", "✅"
  "lut", "✅", "✅"
  "color temperature", "✅", "✅"
  "histogram equalize", "✅", "✅"

Effects augmentations
-----------------------------------------------------------------------------------------------

.. csv-table::
  :widths: 7, 3, 3
  :header: "Type", "CPU", "GPU"

  "gridmask", "✅", "✅"
  "spatter", "✅", "✅"
  "salt and pepper noise", "✅", "✅"
  "shot noise", "✅", "✅"
  "gaussian noise", "✅", "✅"
  "non-linear blend", "✅", "✅"
  "water", "✅", "✅"
  "ricap", "✅", "✅"
  "vignette", "✅", "✅"
  "jitter", "✅", "✅"
  "erase", "✅", "✅"
  "random erase", "✅", "✅"
  "glitch", "✅", "✅"
  "rain", "✅", "✅"
  "pixelate", "✅", "✅"
  "fog", "✅", "✅"
  "posterize", "✅", "✅"
  "solarize", "✅", "✅"
  "snow", "✅", "✅"
  "channel dropout", "✅", "✅"
  "cutout dropout", "✅", "✅"
  "grid dropout", "✅", "✅"
  "coarse dropout", "✅", "✅"

Geometric augmentations
-----------------------------------------------------------------------------------------------

.. csv-table::
  :widths: 7, 3, 3
  :header: "Type", "CPU", "GPU"

  "crop", "✅", "✅"
  "crop mirror normalize", "✅", "✅"
  "crop and patch", "✅", "✅"
  "flip", "✅", "✅"
  "resize", "✅", "✅"
  "resize mirror normalize", "✅", "✅"
  "resize crop mirror", "✅", "✅"
  "rotate", "✅", "✅"
  "warp affine", "✅", "✅"
  "warp perspective", "✅", "✅"
  "lens correction", "✅", "✅"
  "fisheye", "✅", "✅"
  "phase", "✅", "✅"
  "slice", "✅", "✅"
  "remap", "✅", "✅"
  "transpose", "✅", "✅"
  "concat", "✅", "✅"
  "jpeg compression distortion", "✅", "✅"

Morphological operations
-----------------------------------------------------------------------------------------------

.. csv-table::
  :widths: 7, 3, 3
  :header: "Type", "CPU", "GPU"

  "erode", "✅", "✅"
  "dilate", "✅", "✅"

Filter augmentations
-----------------------------------------------------------------------------------------------

.. csv-table::
  :widths: 7, 3, 3
  :header: "Type", "CPU", "GPU"

  "box filter", "✅", "✅"
  "median filter", "✅", "✅"
  "gaussian filter", "✅", "✅"
  "sobel filter", "✅", "✅"
  "emboss", "✅", "✅"

Arithmetic operations
-----------------------------------------------------------------------------------------------

.. csv-table::
  :widths: 7, 3, 3
  :header: "Type", "CPU", "GPU"

  "add scalar", "✅", "✅"
  "subtract scalar", "✅", "✅"
  "multiply scalar", "✅", "✅"
  "fused multiply add scalar", "✅", "✅"
  "magnitude", "✅", "✅"
  "log", "✅", "✅"
  "log1p", "✅", "✅"
  "tensor add", "✅", "✅"
  "tensor subtract", "✅", "✅"
  "tensor multiply", "✅", "✅"
  "tensor divide", "✅", "✅"

Statistical operations
-----------------------------------------------------------------------------------------------

.. csv-table::
  :widths: 7, 3, 3
  :header: "Type", "CPU", "GPU"

  "tensor sum", "✅", "✅"
  "tensor min", "✅", "✅"
  "tensor max", "✅", "✅"
  "tensor mean", "✅", "✅"
  "tensor stddev", "✅", "✅"
  "normalize", "✅", "✅"
  "threshold", "✅", "✅"

Bitwise operations
-----------------------------------------------------------------------------------------------

.. csv-table::
  :widths: 7, 3, 3
  :header: "Type", "CPU", "GPU"

  "bitwise ``AND``", "✅", "✅"
  "bitwise ``OR``", "✅", "✅"
  "bitwise ``XOR``", "✅", "✅"
  "bitwise ``NOT``", "✅", "✅"
  "tensor ``AND`` tensor", "✅", "✅"
  "tensor ``OR`` tensor", "✅", "✅"
  "tensor ``XOR`` tensor", "✅", "✅"

Data exchange operations
-----------------------------------------------------------------------------------------------

.. csv-table::
  :widths: 7, 3, 3
  :header: "Type", "CPU", "GPU"

  "copy", "✅", "✅"
  "channel permute", "✅", "✅"
  "color to greyscale", "✅", "✅"
  "YUV to RGB", "❌", "✅"
  "YUV to RGB (cubic vertical upsampling)", "❌", "✅"
  "YUV to RGB (linear vertical upsampling)", "❌", "✅"

Audio augmentations
-----------------------------------------------------------------------------------------------

.. csv-table::
  :widths: 7, 3, 3
  :header: "Type", "CPU", "GPU"

  "non-silent region detection", "✅", "✅"
  "to decibels", "✅", "✅"
  "pre-emphasis filter", "✅", "✅"
  "down mixing", "✅", "✅"
  "spectrogram", "✅", "✅"
  "mel filter bank", "✅", "✅"
  "resample", "✅", "✅"
  "audio tensor add tensor", "✅", "✅"
  "audio tensor multiply scalar", "✅", "✅"

3D image (voxel) augmentations
-----------------------------------------------------------------------------------------------

.. csv-table::
  :widths: 7, 3, 3
  :header: "Type", "CPU", "GPU"

  "flip (voxel)", "✅", "✅"
  "gaussian noise (voxel)", "✅", "✅"
  "add scalar", "✅", "✅"
  "subtract scalar", "✅", "✅"
  "multiply scalar", "✅", "✅"
  "fused multiply add scalar", "✅", "✅"
  "slice", "✅", "✅"
  "normalize", "✅", "✅"
