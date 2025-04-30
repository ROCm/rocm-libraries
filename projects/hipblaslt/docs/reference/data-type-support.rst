.. meta::
   :description: hipBLASLt library data type support
   :keywords: hipBLASLt, ROCm, data type support

.. _data-type-support:

******************************************
Data type support
******************************************

This topic lists the supported data types for the hipBLASLt GEMM operation, 
which is performed by :ref:`hipblasltmatmul`. 

The ``hipDataType`` enumeration defines data precision types and is primarily 
used when the data reference itself does not include type information, such as
in ``void*`` pointers. This enumeration is mainly utilized in BLAS libraries.

The hipBLASLt input and output types are listed in the following table.

.. list-table::
    :header-rows: 1

    *
      - hipDataType
      - hipBLASLt type
      - Description

    * 
      - ``HIP_R_8I``
      - ``hipblasLtInt8``
      - 8-bit real signed integer.

    * 
      - ``HIP_R_32I``
      - ``hipblasLtInt32``
      - 32-bit real signed integer.

    * 
      - ``HIP_R_8F_E4M3_FNUZ``
      - ``hipblaslt_f8_fnuz``
      - 8-bit real float8 precision floating-point

    * 
      - ``HIP_R_8F_E5M2_FNUZ``
      - ``hipblaslt_bf8_fnuz``
      - 8-bit real bfloat8 precision floating-point

    * 
      - ``HIP_R_16F``
      - ``hipblasLtHalf``
      - 16-bit real half precision floating-point

    * 
      - ``HIP_R_16BF``
      - ``hipblasLtBfloat16``
      - 16-bit real bfloat16 precision floating-point

    * 
      - ``HIP_R_32F``
      - ``hipblasLtFloat``
      - 32-bit real single precision floating-point

The hipBLASLt compute modes are listed in the following table.

.. list-table::
    :header-rows: 1

    *
      - hipDataType
      - Description

    * 
      - ``HIPBLAS_COMPUTE_32I``
      - 32-bit integer compute mode.

    * 
      - ``HIPBLAS_COMPUTE_16F``
      - 16-bit half precision floating-point compute mode.

    * 
      - ``HIPBLAS_COMPUTE_32F``
      - 32-bit singple precision floating-point compute mode.

    * 
      - ``HIPBLAS_COMPUTE_64F``
      - 64-bit double precision floating-point compute mode.

    * 
      - ``HIPBLAS_COMPUTE_32F_FAST_16F``
      - Enables the library to utilize Tensor Cores with 32-bit float computation for matrices with 16-bit half precision input and output.

    * 
      - ``HIPBLAS_COMPUTE_32F_FAST_16BF``
      - Enables the library to utilize Tensor Cores with 32-bit float computation for matrices with 16-bit bfloat16 precision input and output.

    * 
      - ``HIPBLAS_COMPUTE_32F_FAST_TF32``
      - Enables the library to utilize Tensor Cores with TF32 computation for matrices with 32-bit input and output.


hipBLASLt GEMM operation equation:

.. math::

 D = Activation(alpha \cdot op(A) \cdot op(B) + beta \cdot op(C) + bias)

Where :math:`op( )` refers to in-place operations, such as transpose and
non-transpose, and :math:`alpha` and :math:`beta` are scalars.

.. note:: 
  
  The ``hipblaslt_f8_fnuz`` and ``hipblaslt_bf8_fnuz`` data types are only
  supported on the gfx94x platform.


.. list-table:: Supported data types
  :header-rows: 1
  :name: supported-data-types

  *
    - A data type
    - B data type
    - C data type
    - D (Output) data type
    - Compute(Scale) data type

  *
    - ``float``
    - ``float``
    - ``float``
    - ``float``
    - ``float``

  *
    - ``half``
    - ``half``
    - ``half``
    - ``half``
    - ``float``

  *
    - ``half``
    - ``half``
    - ``half``
    - ``float``
    - ``float``


  *
    - ``bfloat16``
    - ``bfloat16``
    - ``bfloat16``
    - ``bfloat16``
    - ``float``

  *
    - ``float8``
    - ``float8``
    - ``float``
    - ``float``
    - ``float``

  *
    - ``float8``
    - ``float8``
    - ``half``
    - ``half``
    - ``float``

  *
    - ``float8``
    - ``float8``
    - ``bfloat16``
    - ``bfloat16``
    - ``float``

  *
    - ``float8``
    - ``float8``
    - ``float8``
    - ``float8``
    - ``float``

  *
    - ``float8``
    - ``float8``
    - ``bfloat8``
    - ``bfloat8``
    - ``float``

  *
    - ``bfloat8``
    - ``bfloat8``
    - ``float``
    - ``float``
    - ``float``

  *
    - ``bfloat8``
    - ``bfloat8``
    - ``half``
    - ``half``
    - ``float``

  *
    - ``bfloat8``
    - ``bfloat8``
    - ``bfloat16``
    - ``bfloat16``
    - ``float``

  *
    - ``bfloat8``
    - ``bfloat8``
    - ``float8``
    - ``float8``
    - ``float``

  *
    - ``bfloat8``
    - ``bfloat8``
    - ``bfloat8``
    - ``bfloat8``
    - ``float``

  *
    - ``int8_t``
    - ``int8_t``
    - ``int8_t``
    - ``int8_t``
    - ``int32_t``

The :cpp:func:`hipblasLtMatrixTransform` :cpp:func:`hipblasLtMatmul` functions
data type support is listed separately on the :ref:`hipBLASLt API reference page <api-reference>`. 

For more information about data type support for the other ROCm libraries, see 
:doc:`Data types and precision support page <rocm:reference/precision-support>`. 
