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

An algorithm is valid only for the problem it was queried for. ``hipblasLtMatmul`` and
``hipblaslt_ext::Gemm::initialize`` refuse an algorithm for a hand-written kernel with
``HIPBLAS_STATUS_INVALID_VALUE`` when the problem no longer satisfies that kernel's problem
predicates, such as its sizes, strides, or ``alpha`` value.

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

* This table uses simpler abbreviations:

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
|  FP8  | FP8   |  BF16 | BF16  |  T   |   N   |  HIPBLASLT_ORDER_COL16_4R16 |  HIPBLASLT_ORDER_COL        | HIPBLASLT_ORDER_COL | HIPBLASLT_ORDER_COL |
+-------+-------+-------+-------+------+-------+-----------------------------+-----------------------------+---------------------+---------------------+
|  FP16 | FP16  |  FP32 | FP32  |  T   |   N   |  HIPBLASLT_ORDER_COL16_4R8  |  HIPBLASLT_ORDER_COL        | HIPBLASLT_ORDER_COL | HIPBLASLT_ORDER_COL |
+-------+-------+-------+-------+------+-------+-----------------------------+-----------------------------+---------------------+---------------------+
|  FP16 | FP16  |  FP16 | FP16  |  T   |   N   |  HIPBLASLT_ORDER_COL16_4R8  |  HIPBLASLT_ORDER_COL        | HIPBLASLT_ORDER_COL | HIPBLASLT_ORDER_COL |
+-------+-------+-------+-------+------+-------+-----------------------------+-----------------------------+---------------------+---------------------+
|  BF16 | BF16  |  BF16 | BF16  |  T   |   N   |  HIPBLASLT_ORDER_COL16_4R8  |  HIPBLASLT_ORDER_COL        | HIPBLASLT_ORDER_COL | HIPBLASLT_ORDER_COL |
+-------+-------+-------+-------+------+-------+-----------------------------+-----------------------------+---------------------+---------------------+
|  FP8  | FP8   |  FP16 | FP16  |  T   |   N   |  HIPBLASLT_ORDER_COL        |  HIPBLASLT_ORDER_COL16_4R16 | HIPBLASLT_ORDER_COL | HIPBLASLT_ORDER_COL |
+-------+-------+-------+-------+------+-------+-----------------------------+-----------------------------+---------------------+---------------------+
|  FP8  | FP8   |  BF16 | BF16  |  T   |   N   |  HIPBLASLT_ORDER_COL        |  HIPBLASLT_ORDER_COL16_4R16 | HIPBLASLT_ORDER_COL | HIPBLASLT_ORDER_COL |
+-------+-------+-------+-------+------+-------+-----------------------------+-----------------------------+---------------------+---------------------+
|  FP16 | FP16  |  FP32 | FP32  |  T   |   N   |  HIPBLASLT_ORDER_COL        |  HIPBLASLT_ORDER_COL16_4R8  | HIPBLASLT_ORDER_COL | HIPBLASLT_ORDER_COL |
+-------+-------+-------+-------+------+-------+-----------------------------+-----------------------------+---------------------+---------------------+
|  FP16 | FP16  |  FP16 | FP16  |  T   |   N   |  HIPBLASLT_ORDER_COL        |  HIPBLASLT_ORDER_COL16_4R8  | HIPBLASLT_ORDER_COL | HIPBLASLT_ORDER_COL |
+-------+-------+-------+-------+------+-------+-----------------------------+-----------------------------+---------------------+---------------------+
|  BF16 | BF16  |  BF16 | BF16  |  T   |   N   |  HIPBLASLT_ORDER_COL        |  HIPBLASLT_ORDER_COL16_4R8  | HIPBLASLT_ORDER_COL | HIPBLASLT_ORDER_COL |
+-------+-------+-------+-------+------+-------+-----------------------------+-----------------------------+---------------------+---------------------+

There are restrictions on the supported problem sizes for the ``HIP_R_4F_E2M1``,
``HIP_R_8F_E4M3``, and ``HIP_R_8F_E5M2`` data types.
When ``HIPBLASLT_MATMUL_DESC_A_SCALE_MODE`` and ``HIPBLASLT_MATMUL_DESC_B_SCALE_MODE`` are both set
to ``HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0``, the following restrictions apply:

*  Atype and Btype can be any combination of: ``HIP_R_8F_E4M3``, ``HIP_R_8F_E5M2``, and ``HIP_R_4F_E2M1``.
*  Ctype must be the same as Dtype.
*  Dtype can be any of the following: ``HIP_R_32F``, ``HIP_R_16F``, or ``HIP_R_16BF``.
*  ``m % 16`` must be equal to ``0``.
*  ``n % 16`` must be equal to ``0``.
*  ``K % 128`` must be equal to ``0``.
*  ``B`` must be equal to ``1``.
*  ``opA`` must be equal to ``T``.
*  ``opB`` must be equal to ``N``.
*  Epilogues are not supported.
*  The scaling data pointed to by ``HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER`` must be stored in the same order as ``A``.
*  The scaling data pointed to by ``HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER`` must be stored in the same order as ``B``.

When ``HIPBLASLT_MATMUL_DESC_A_SCALE_MODE`` and ``HIPBLASLT_MATMUL_DESC_B_SCALE_MODE`` are both set
to ``HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT``, the following restrictions apply:

*  Atype and Btype must each be ``HIP_R_6F_E2M3`` or ``HIP_R_6F_E3M2``.
*  The order of ``A`` must be ``HIPBLASLT_ORDER_COL256_R128_EXT``, and the order of ``B`` must be
   ``HIPBLASLT_ORDER_COL256_R128_EXT`` or ``HIPBLASLT_ORDER_COL``. The caller stores ``A``, and ``B`` when it
   uses that order, tiled in that order, and both scales shuffled in tiles of 128 rows by 4 K blocks. The
   descriptions of both enum values in ``hipblaslt.h`` give the byte offset of every element and scale.
*  ``opA`` must be equal to ``T``, and ``opB`` must be equal to ``N``.
*  ``m`` must be a multiple of 256 and ``K`` a multiple of 128. ``n`` must be a multiple of 256 with ``B`` in
   ``HIPBLASLT_ORDER_COL256_R128_EXT``, and a multiple of 128 with ``B`` in ``HIPBLASLT_ORDER_COL``.
*  ``lda`` and ``ldb`` must be equal to ``K``.

The solutions for this scale mode are rocKE GEMM kernels, which add these restrictions:

*  Atype and Btype must be the same type.
*  Ctype and Dtype must be ``HIP_R_16BF``, and the compute type must be ``HIPBLAS_COMPUTE_32F``.
*  ``ldd`` must be equal to ``m``.
*  ``B`` must be equal to ``1``.
*  Epilogues are not supported.
*  ``alpha`` must be equal to ``1``, and ``beta`` must be equal to ``0``.
*  Solutions are registered for gfx950 PCI chip IDs ``0x75a0`` and ``0x75a3``, at ``m = n = 4096`` with
   ``K = 65536`` or ``K = 131584``. With ``B`` in ``HIPBLASLT_ORDER_COL``, they exist only for
   ``HIP_R_6F_E2M3`` at ``K = 65536``.
*  ``hipblasLtMatmulAlgoGetHeuristic`` does not return these solutions, because it does not assume
   ``beta = 0``. Query them through the extension API (``hipblaslt_ext::Gemm::setProblem`` with ``beta = 0``,
   then ``algoGetHeuristic``), and run the returned algorithm with ``hipblaslt_ext::Gemm::run`` or
   ``hipblasLtMatmul``.

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
