.. meta::
  :description: ROCm Performance Primitives (RPP) supported functionalities
  :keywords: RPP, ROCm, Performance Primitives, documentation, support, functionalities, audio, image

********************************************************************
ROCm Performance Primitives supported functionalities and variants
********************************************************************

The following tables show the CPU and GPU support for ROCm Performance Primitives (RPP) functionalities and variants. 

CPU support is also referred to as HOST support.


Image augmentations
-----------------------------------------------------------------------------------------------

.. csv-table::
  :widths: 7, 3, 3
  :header: "Type", "CPU", "GPU"

  "brightness", "✅", "✅"
  "gamma correction", "✅", "✅"
  "blend", "✅", "✅"
  "blur", "✅", "✅"
  "contrast", "✅", "✅"
  "pixelate", "✅", "✅"
  "jitter", "✅", "✅"
  "snow", "✅", "✅"
  "noise", "✅", "✅"
  "random shadow", "✅", "✅"
  "fog", "✅", "✅"
  "rain", "✅", "✅"
  "random crop letterbox", "✅", "✅"
  "exposure", "✅", "✅"
  "histogram balance", "✅", "❌"

Statistical functions
-----------------------------------------------------------------------------------------------

.. csv-table::
  :widths: 7, 3, 3
  :header: "Type", "CPU", "GPU"

  "thresholding", "✅", "✅"
  "min", "✅", "✅"
  "max", "✅", "✅"
  "min max loc", "✅", "❌"
  "integral", "✅", "❌"
  "histogram_equalization", "✅", "❌"
  "mean stddev", "✅", "❌"

Geometry transforms
-----------------------------------------------------------------------------------------------

.. csv-table::
  :widths: 7, 3, 3
  :header: "Type", "CPU", "GPU"

  "flip", "✅", "✅"
  "resize", "✅", "✅"
  "resize crop", "✅", "✅"
  "rotate", "✅", "✅"
  "warp affine", "✅", "✅"
  "fisheye", "✅", "✅"
  "lens correction", "✅", "✅"
  "scale", "✅", "✅"
  "warp perspective", "✅", "✅"

Advanced augmentations
-----------------------------------------------------------------------------------------------

.. csv-table::
  :widths: 7, 3, 3
  :header: "Type", "CPU", "GPU"

  "water", "✅", "✅"
  "non-linear blend", "✅", "✅"
  "color cast", "✅", "✅"
  "erase", "✅", "✅"
  "crop and patch", "✅", "✅"
  "lut", "✅", "✅"
  "glitch", "✅", "✅"

Fused functions
-----------------------------------------------------------------------------------------------

.. csv-table::
  :widths: 7, 3, 3
  :header: "Type", "CPU", "GPU"

  "color twist", "✅", "✅"
  "crop", "✅", "✅"
  "crop mirror normalize", "✅", "✅"
  "resize crop mirror", "✅", "✅"

Morphological transforms
-----------------------------------------------------------------------------------------------

.. csv-table::
  :widths: 7, 3, 3
  :header: "Type", "CPU", "GPU"

  "erode", "✅", "✅"
  "dilate", "✅", "✅"

Color model conversions
-----------------------------------------------------------------------------------------------

.. csv-table::
  :widths: 7, 3, 3
  :header: "Type", "CPU", "GPU"

  "hue", "✅", "✅"
  "saturation", "✅", "✅"
  "color convert", "✅", "✅"
  "color temperature", "✅", "✅"
  "vignette", "✅", "✅"
  "channel extract", "✅", "❌"
  "channel combine", "✅", "❌"
  "lookup table", "✅", "✅"
  "tensor table lookup", "✅", "❌"

Filter operations
-----------------------------------------------------------------------------------------------

.. csv-table::
  :widths: 7, 3, 3
  :header: "Type", "CPU", "GPU"

  "box filter", "✅", "✅"
  "sobel filter", "✅", "✅"
  "median filter", "✅", "✅"
  "custom convolution", "✅", "❌"
  "non-max suppression", "✅", "✅"
  "gaussian filter", "✅", "✅"
  "non-linear filter", "✅", "✅"

Arithmetic operations
-----------------------------------------------------------------------------------------------

.. csv-table::
  :widths: 7, 3, 3
  :header: "Type", "CPU", "GPU"

  "absolute difference", "✅", "✅"
  "accumulate weighted", "✅", "✅"
  "accumulate", "✅", "✅"
  "add", "✅", "✅"
  "subtract", "✅", "✅"
  "magnitude", "✅", "✅"
  "multiply", "✅", "✅"
  "phase", "✅", "✅"
  "tensor add", "✅", "✅"
  "tensor subtract", "✅", "✅"
  "tensor multiply", "✅", "✅"
  "accumulate squared", "✅", "✅"

Bitwise operations
-----------------------------------------------------------------------------------------------

.. csv-table::
  :widths: 7, 3, 3
  :header: "Type", "CPU", "GPU"

  "bitwise ``AND``", "✅", "✅"
  "bitwise ``NOT``", "✅", "✅"
  "exclusive ``OR``", "✅", "✅"
  "inclusive ``OR``", "✅", "✅"

Computer vision
-----------------------------------------------------------------------------------------------

.. csv-table::
  :widths: 7, 3, 3
  :header: "Type", "CPU", "GPU"

  "local binary pattern", "✅", "✅"
  "data object copy", "✅", "✅"
  "gaussian image pyramid", "✅", "✅"
  "laplacian image pyramid", "✅", "✅"
  "canny edge detector", "✅", "✅"
  "harris corner detector", "✅", "✅"
  "tensor convert bit depth", "✅", "❌"
  "fast corner detector", "✅", "❌"
  "reconstruction laplacian image pyramid", "✅", "❌"
  "control flow", "✅", "✅"
  "hough lines", "✅", "❌"
  "hog", "✅", "❌"
  "remap", "✅", "❌"
  "tensor matrix multiply", "✅", "✅"
  "tensor transpose", "✅", "✅"

Audio
-----------------------------------------------------------------------------------------------

.. csv-table::
  :widths: 7, 3, 3
  :header: "Type", "CPU", "GPU"

  "non Silent Region Detection", "✅", "✅"
  "to Decibels", "✅", "✅"  
  "downmixing", "✅", "✅"
  "pre-emphasis Filter", "✅", "✅"
  "resample", "✅", "✅"
  "mel Filter Bank", "✅", "✅"
  "spectrogram", "✅", "✅"
