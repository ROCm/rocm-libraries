.. meta::
   :description: hipBLASLt API reference
   :keywords: hipBLASLt, ROCm, library, API, reference

.. _api-reference:

***********************
hipBLASLt API reference
***********************

hipblasLtCreate()
------------------------------------------
.. doxygenfunction:: hipblasLtCreate

hipblasLtDestroy()
------------------------------------------
.. doxygenfunction:: hipblasLtDestroy

hipblasLtMatrixLayoutCreate()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixLayoutCreate

hipblasLtMatrixLayoutDestroy()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixLayoutDestroy

hipblasLtMatrixLayoutSetAttribute()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixLayoutSetAttribute

hipblasLtMatrixLayoutGetAttribute()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixLayoutGetAttribute

hipblasLtMatmulDescCreate()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulDescCreate

hipblasLtMatmulDescDestroy()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulDescDestroy

hipblasLtMatmulDescSetAttribute()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulDescSetAttribute

hipblasLtMatmulDescGetAttribute()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulDescGetAttribute

hipblasLtMatmulPreferenceCreate()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulPreferenceCreate

hipblasLtMatmulPreferenceDestroy()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulPreferenceDestroy

hipblasLtMatmulPreferenceSetAttribute()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulPreferenceSetAttribute

hipblasLtMatmulPreferenceGetAttribute()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulPreferenceGetAttribute

.. _hipblasltmatmulalgogetheuristic:

hipblasLtMatmulAlgoGetHeuristic()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmulAlgoGetHeuristic

.. _hipblasltmatmul:

hipblasLtMatmul()
------------------------------------------
.. doxygenfunction:: hipblasLtMatmul

Supported data types
------------------------------------------

``hipblasLtMatmul`` supports the following computeType, scaleType, Bias type, Atype/Btype, and Ctype/Dtype:

============================= =================== =============== ===============
computeType                   scaleType/Bias type Atype/Btype     Ctype/Dtype
============================= =================== =============== ===============
HIPBLAS_COMPUTE_32F           HIP_R_32F           HIP_R_32F       HIP_R_32F
HIPBLAS_COMPUTE_32F_FAST_TF32 HIP_R_32F           HIP_R_32F       HIP_R_32F
HIPBLAS_COMPUTE_32F           HIP_R_32F           HIP_R_16F       HIP_R_16F
HIPBLAS_COMPUTE_32F           HIP_R_32F           HIP_R_16F       HIP_R_32F
HIPBLAS_COMPUTE_32F           HIP_R_32F           HIP_R_16BF      HIP_R_16BF
============================= =================== =============== ===============

For ``FP8`` type Matmul, hipBLASLt supports the type combinations shown in the following table:

* This table uses simpler abbrieviations:

  *  **FP16** means **HIP_R_16F**
  *  **BF16** means **HIP_R_16BF**
  *  **FP32** means **HIP_R_32F**
  *  **FP8** means **HIP_R_8F_E4M3**
  *  **BF8** means **HIP_R_8F_E5M2**
  *  **FP8_FNUZ** means **HIP_R_8F_E4M3_FNUZ** and
  *  **BF8_FNUZ** means **HIP_R_8F_E5M2_FNUZ**

*  The table applies to all transpose types (NN/NT/TT/TN).
*  **Default bias type** indicates the type when the bias type is not explicitly specified.

+-------+-------+-------+-------+-------------+----------+----------+------------+-----------+
| Atype | Btype | Ctype | Dtype | computeType | scaleA,B | scaleC,D | Bias type  | Default   |
|       |       |       |       |             |          |          |            | bias type |
+=======+=======+=======+=======+=============+==========+==========+======+=====+===========+
| FP8   | FP8   | FP16  | FP16  | FP32        | Yes      | No       | FP32, FP16 | FP16      |
|       |       +-------+-------+             +          +          +------------+-----------+
|       |       | BF16  | BF16  |             |          |          | FP32, BF16 | BF16      |
|       |       +-------+-------+             +          +          +------------+-----------+
|       |       | FP32  | FP32  |             |          |          | FP32, BF16 | BF16      |
|       |       +-------+-------+             +          +----------+------------+-----------+
|       |       | FP8   | FP8   |             |          | Yes      | FP32, FP16 | FP16      |
|       |       +-------+-------+             +          +          +------------+-----------+
|       |       | BF8   | BF8   |             |          |          | FP32, FP16 | FP16      |
|       +-------+-------+-------+             +          +----------+------------+-----------+
|       | BF8   | FP16  | FP16  |             |          | No       | FP32, FP16 | FP16      |
|       |       +-------+-------+             +          +          +------------+-----------+
|       |       | BF16  | BF16  |             |          |          | FP32, BF16 | BF16      |
|       |       +-------+-------+             +          +          +------------+-----------+
|       |       | FP32  | FP32  |             |          |          | FP32, BF16 | BF16      |
|       |       +-------+-------+             +          +----------+------------+-----------+
|       |       | FP8   | FP8   |             |          | Yes      | FP32, FP16 | FP16      |
|       |       +-------+-------+             +          +          +------------+-----------+
|       |       | BF8   | BF8   |             |          |          | FP32, FP16 | FP16      |
+-------+-------+-------+-------+             +          +----------+------------+-----------+
| BF8   | FP8   | FP16  | FP16  |             |          | No       | FP32, FP16 | FP16      |
|       |       +-------+-------+             +          +          +------------+-----------+
|       |       | BF16  | BF16  |             |          |          | FP32, BF16 | BF16      |
|       |       +-------+-------+             +          +          +------------+-----------+
|       |       | FP32  | FP32  |             |          |          | FP32, BF16 | BF16      |
|       |       +-------+-------+             +          +----------+------------+-----------+
|       |       | FP8   | FP8   |             |          | Yes      | FP32, FP16 | FP16      |
|       |       +-------+-------+             +          +          +------------+-----------+
|       |       | BF8   | BF8   |             |          |          | FP32, FP16 | FP16      |
|       +-------+-------+-------+             +          +----------+------------+-----------+
|       | BF8   | FP16  | FP16  |             |          | No       | FP32, FP16 | FP16      |
|       |       +-------+-------+             +          +          +------------+-----------+
|       |       | BF16  | BF16  |             |          |          | FP32, BF16 | BF16      |
|       |       +-------+-------+             +          +          +------------+-----------+
|       |       | FP32  | FP32  |             |          |          | FP32, BF16 | BF16      |
|       |       +-------+-------+             +          +----------+------------+-----------+
|       |       | FP8   | FP8   |             |          | Yes      | FP32, FP16 | FP16      |
|       |       +-------+-------+             +          +          +------------+-----------+
|       |       | BF8   | BF8   |             |          |          | FP32, FP16 | FP16      |
+-------+-------+-------+-------+-------------+----------+----------+------------+-----------+

To use special data ordering for ``HIPBLASLT_ORDER_COL16_4R8`` and ``HIPBLASLT_ORDER_COL16_4R16`` in ``hipblasLtMatmul`` for the gfx94x architecture, choose one of these valid combinations of transposes and orders of input and output matrices:

+-------+-------+-------+-------+------+-------+-----------------------------+-----------------------------+---------------------+---------------------+
| Atype | Btype | CType | DType |  opA |  opB  |  orderA                     |  orderB                     |  orderC             |   orderD            |
+=======+=======+=======+=======+======+=======+=============================+=============================+=====================+=====================+
|  FP8  | FP8   |  FP16 | FP16  |  T   |   N   |  HIPBLASLT_ORDER_COL16_4R16 |  HIPBLASLT_ORDER_COL        | HIPBLASLT_ORDER_COL | HIPBLASLT_ORDER_COL |
+-------+-------+-------+-------+------+-------+-----------------------------+-----------------------------+---------------------+---------------------+
|  FP16 | FP16  |  FP16 | FP16  |  T   |   N   |  HIPBLASLT_ORDER_COL16_4R8  |  HIPBLASLT_ORDER_COL        | HIPBLASLT_ORDER_COL | HIPBLASLT_ORDER_COL |
+-------+-------+-------+-------+------+-------+-----------------------------+-----------------------------+---------------------+---------------------+
|  BF16 | BF16  |  BF16 | BF16  |  T   |   N   |  HIPBLASLT_ORDER_COL16_4R8  |  HIPBLASLT_ORDER_COL        | HIPBLASLT_ORDER_COL | HIPBLASLT_ORDER_COL |
+-------+-------+-------+-------+------+-------+-----------------------------+-----------------------------+---------------------+---------------------+

hipblasLtMatrixTransformDescCreate()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixTransformDescCreate

hipblasLtMatrixTransformDescDestroy()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixTransformDescDestroy

hipblasLtMatrixTransformDescSetAttribute()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixTransformDescSetAttribute

hipblasLtMatrixTransformDescGetAttribute()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixTransformDescGetAttribute

hipblasLtMatrixTransform()
------------------------------------------
.. doxygenfunction:: hipblasLtMatrixTransform

``hipblasLtMatrixTransform`` supports the following Atype/Btype/Ctype and scaleType:

======================= ===================
Atype/Btype/Ctype       scaleType
======================= ===================
HIP_R_32F               HIP_R_32F
HIP_R_16F               HIP_R_32F/HIP_R_16F
HIP_R_16BF              HIP_R_32F
HIP_R_8I                HIP_R_32F
HIP_R_32I               HIP_R_32F
======================= ===================
