.. meta::
  :description: MIOpen library precision support
  :keywords: MIOpen, ROCm, API, documentation, datatypes, data type support, precision

.. _miopen-data-type-support:

********************************************************************
MIOpen precision support
********************************************************************

This topic lists the data type support for the MIOpen library on AMD GPUs.

This page lists the data types supported by the library itself and does not
indicate hardware support. A type listed here is only usable if the GPU
architecture also supports it; otherwise it is unsupported. For data type support
across the other ROCm libraries and by GPU architecture, see the
:doc:`Data types and precision support page <rocm:reference/precision-support>`.

.. _miopen-input-output-type-support:

Supported data types overview
=============================

The following table summarizes the input and output data types supported by
MIOpen. For the full ``miopenDataType_t`` enumeration and per-type support
notes, see the section that follows.

.. list-table::
    :header-rows: 1

    *
      - Icon
      - Definition
    *
      - ✅
      - Fully supported as both an input and output type.
    *
      - ⚠️
      - Partially supported as an input or output type.

Data types not listed in the table below are not supported.

.. datatemplate:yaml:: /data/reference/precision-support.yaml

    .. list-table::
        :header-rows: 1
        :widths: 70, 30

        *
            - Data type
            - Support
    {% for data_type in data.data_types %}
        *
            - {{ data_type.type }}
            - {{ data_type.support }}
    {% endfor %}

Datatype enumeration
====================

MIOpen contains several datatypes at different levels of support. The enumerated datatypes are:

.. code:: cpp

    typedef enum {
        miopenHalf     = 0,
        miopenFloat    = 1,
        miopenInt32    = 2,
        miopenInt8     = 3,
        /* Value 4 is reserved. */
        miopenBFloat16 = 5,
        miopenDouble   = 6,
        miopenFloat8   = 7,
        miopenBFloat8  = 8
    } miopenDataType_t;

Of these types only ``miopenFloat`` and ``miopenHalf`` are fully supported across all layers in MIOpen.
Refer to the individual :doc:`Modules <../doxygen/html/modules>` in the API library for specific
datatype support and limitations.

Type descriptions:

* ``miopenHalf``: 16-bit floating point
* ``miopenFloat``: 32-bit floating point
* ``miopenInt32``: 32-bit integer, used primarily for ``int8`` convolution outputs
* ``miopenInt8``: 8-bit integer; supported by ``int8`` convolution forward path, tensor set, tensor copy, tensor cast, tensor transform, tensor transpose, and im2col
* ``miopenBFloat16``: brain float fp-16 (8-bit exponent, 7-bit fraction); supported by convolutions, tensor set, and tensor copy
* ``miopenDouble``: 64-bit floating point; supported by reduction, layerNorm, and batchNorm
* ``miopenFloat8``: 8-bit floating point (layout 1.4.3, exponent bias 7); supported by convolutions
* ``miopenBFloat8``: 8-bit floating point (layout 1.5.2, exponent bias 15); supported by convolutions

.. note::

   Convolution APIs currently only support uniform input/output datatypes and tensor layouts.

In addition to these standard datatypes, pooling also contains its own indexing datatypes:

.. code:: cpp

    typedef enum {
        miopenIndexUint8  = 0,
        miopenIndexUint16 = 1,
        miopenIndexUint32 = 2,
        miopenIndexUint64 = 3,
    } miopenIndexType_t;

.. _miopen-fp32-compute-precision:

FP32 compute precision (TF32)
=============================

``miopenFloat`` convolutions can be allowed to compute with reduced internal precision. This is a
compute mode, not a storage type: tensors stay ``FP32``.

* **Requesting it:** set ``MIOPEN_CONVOLUTION_ATTRIB_MATH_TYPE`` to ``miopenMathDefault`` with
  ``miopenSetConvolutionAttribute``. A new descriptor starts with the global default from the
  ``MIOPEN_TF32_OVERRIDE`` environment variable (``1``: ``miopenMathDefault``; ``0`` or unset:
  ``miopenMathPedantic``). After the descriptor is created, only the attribute changes it, and
  ``miopenGetConvolutionAttribute`` returns the value in effect.
* **It is a precision floor:** ``miopenMathDefault`` allows ``TF32`` (10-bit mantissa) or higher
  precision. Solvers that support it compute in ``TF32`` or with a three-pass ``bfloat16`` split
  (about 16 mantissa bits); all other solvers compute in full ``FP32``, which always satisfies the
  request.
* **Where it applies:** solvers that support it include the grouped Composable Kernel convolution
  solvers on gfx942 (native ``xf32``) and gfx950 (``bfloat16`` split), and ``ConvHipConv`` on
  gfx950 and gfx1250.
* **No report of what ran:** like cuDNN and cuBLAS, MIOpen treats ``miopenMathDefault`` as
  permission. It does not report whether a particular call used ``TF32``.
* **Results:** expect differences from ``miopenMathPedantic`` within ``TF32`` rounding error.
  Find and performance database records are kept separately for the two math types.
