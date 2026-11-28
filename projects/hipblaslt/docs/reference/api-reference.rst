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
predicates, such as its sizes, strides, or ``alpha`` value, or uses
``HIPBLASLT_BATCH_MODE_POINTER_ARRAY``, which hand-written kernels do not support.

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

When ``HIPBLASLT_MATMUL_DESC_A_SCALE_MODE`` or ``HIPBLASLT_MATMUL_DESC_B_SCALE_MODE`` is set to
``HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT`` or
``HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_PAGE24_EXT``, or the order of ``A`` or ``B`` is one of the
``HIPBLASLT_ORDER_COL256_R128`` orders, the following restrictions apply:

*  Atype and Btype must each be ``HIP_R_6F_E2M3`` or ``HIP_R_6F_E3M2``.
*  The orders and scale modes must be one of these combinations:

   .. list-table::
      :header-rows: 1

      * - Order of ``A``
        - Order of ``B``
        - Scale mode of ``A``
        - Scale mode of ``B``
      * - ``HIPBLASLT_ORDER_COL256_R128_EXT``
        - ``HIPBLASLT_ORDER_COL256_R128_EXT`` or ``HIPBLASLT_ORDER_COL``
        - ``..._128_4_EXT``
        - ``..._128_4_EXT``
      * - ``HIPBLASLT_ORDER_COL256_R128_P48_EXT``
        - ``HIPBLASLT_ORDER_COL256_R128_P48_EXT``
        - ``..._128_4_EXT``
        - ``..._128_4_EXT``
      * - ``HIPBLASLT_ORDER_COL256_R128_P16_EXT``
        - ``HIPBLASLT_ORDER_COL``
        - ``..._128_4_EXT``
        - ``..._128_4_EXT``
      * - ``HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_EXT``
        - ``HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_SPLIT_EXT``
        - ``..._128_4_PAGE24_EXT``
        - ``..._128_4_EXT``

   The caller stores each operand that uses one of these orders tiled in that order, and the scales shuffled
   in tiles of 128 rows by 4 K blocks. The descriptions of the enum values in ``hipblaslt.h`` give the byte
   offset of every element and scale.
*  ``opA`` must be equal to ``T``, and ``opB`` must be equal to ``N``.
*  ``m`` must be a multiple of 256 and ``K`` a multiple of 128. ``n`` must be a multiple of 256 with ``B`` in
   one of the ``HIPBLASLT_ORDER_COL256_R128`` orders, and a multiple of 128 with ``B`` in
   ``HIPBLASLT_ORDER_COL``.
*  ``lda`` and ``ldb`` must be equal to ``K``.
*  With ``A`` in ``HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_EXT``, ``m`` and ``n`` must be equal to ``4096``,
   and ``K`` must be at most ``98304``.

Each order selects its own solutions: a solution for one ``HIPBLASLT_ORDER_COL256_R128`` order is never
returned for another. Solutions are registered for ``HIPBLASLT_ORDER_COL256_R128_EXT``, for
``HIPBLASLT_ORDER_COL256_R128_P48_EXT`` on ``A`` and ``B``, for ``HIPBLASLT_ORDER_COL256_R128_P16_EXT`` on
``A`` with ``B`` in ``HIPBLASLT_ORDER_COL``, and for ``HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_EXT`` on ``A``
with ``B`` in ``HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_SPLIT_EXT``. The registered solutions are rocKE GEMM
kernels, which add these restrictions:

*  Atype and Btype must be the same type.
*  Ctype and Dtype must be ``HIP_R_16BF``, and the compute type must be ``HIPBLAS_COMPUTE_32F``.
*  ``ldd`` must be equal to ``m``.
*  The batch count must be equal to ``1``.
*  Epilogues are not supported.
*  ``alpha`` must be equal to ``1``, and ``beta`` must be equal to ``0``.
*  Solutions are registered for gfx950 PCI chip IDs ``0x75a0`` and ``0x75a3``, at ``m = n = 4096``:

   *  With ``HIPBLASLT_ORDER_COL256_R128_EXT`` on ``A`` and ``B``, for ``HIP_R_6F_E2M3`` and
      ``HIP_R_6F_E3M2`` at ``K = 65536`` or ``K = 131584``.
   *  With ``HIPBLASLT_ORDER_COL256_R128_EXT`` on ``A`` and ``B`` in ``HIPBLASLT_ORDER_COL``, for
      ``HIP_R_6F_E2M3`` at ``K = 65536`` only.
   *  With ``HIPBLASLT_ORDER_COL256_R128_P48_EXT`` on ``A`` and ``B``,
      ``HIPBLASLT_ORDER_COL256_R128_P16_EXT`` on ``A`` and ``B`` in ``HIPBLASLT_ORDER_COL``, or
      ``HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_EXT`` on ``A`` and
      ``HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_SPLIT_EXT`` on ``B``, for ``HIP_R_6F_E2M3`` and
      ``HIP_R_6F_E3M2`` at ``K = 65536`` only.
*  The ``PAGE24`` solutions are selected on amdgpu 6.19 and 7.1 only. On those drivers each launch passes the
   kernel bit 0 of the device's ``xgmi_physical_id``, which decides which work-groups read which half of the
   placed blocks. The bit affects speed only.
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

It also packs one MXFP6 operand of 4096 rows (M or N) with ``K = 65536`` on gfx950. A pack reads A
and writes C, which must not overlap. ``HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA`` must stay
``HIPBLAS_OP_N`` and B must be NULL; alpha, beta, and scaleType are not used.

* Data: Adesc is ``HIP_R_6F_E2M3`` or ``HIP_R_6F_E3M2`` in ``HIPBLASLT_ORDER_COL``, with 65536 rows,
  4096 columns, and ``ld = 65536``. Cdesc is the same in ``HIPBLASLT_ORDER_COL256_R128_P16_EXT``.
* Scales: ``HIPBLASLT_MATRIX_TRANSFORM_DESC_C_SCALE_MODE_EXT`` is
  ``HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT``. Adesc and Cdesc are ``HIP_R_8F_UE8M0`` in
  ``HIPBLASLT_ORDER_COL``, with 2048 rows, 4096 columns, and ``ld = 2048``.

It also places such an operand for the near-die GEMM, writing only the bytes the destination layout places:

* A data: Adesc is ``HIP_R_6F_E2M3`` or ``HIP_R_6F_E3M2`` in ``HIPBLASLT_ORDER_COL256_R128_P48_EXT``, with
  65536 rows, 4096 columns, and ``ld = 65536``. Cdesc is the same in
  ``HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_EXT``, and C is ``1026*2097152`` bytes.
* B data: as for A, with Cdesc in ``HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_SPLIT_EXT``, and C is
  ``514*2097152`` bytes.
* A scales: ``HIPBLASLT_MATRIX_TRANSFORM_DESC_C_SCALE_MODE_EXT`` is
  ``HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_PAGE24_EXT``. A holds the scales in the
  ``HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT`` layout. Adesc and Cdesc are as for the scale
  pack, and C is ``46*2097152`` bytes.

The transform reads ``/sys/module/amdgpu/version``. On a driver other than amdgpu 6.19 and 7.1 a placement returns
``HIPBLAS_STATUS_NOT_SUPPORTED`` and writes nothing. The GEMM reads each 8 KiB of the placed data and scales from the
die it is on only when C starts at a 512 MiB-aligned offset of a 2 MiB-aligned, physically contiguous allocation
(``hipExtMallocWithFlags`` with ``hipDeviceMallocContiguous``). That affects speed only: in any other buffer the GEMM
reads the same bytes and gives the same results, more slowly.

There is no pack into ``HIPBLASLT_ORDER_COL256_R128_EXT`` or ``HIPBLASLT_ORDER_COL256_R128_P48_EXT``.
Those layouts, any other source layout, and any other shape return ``HIPBLAS_STATUS_INVALID_VALUE``. Other
devices return ``HIPBLAS_STATUS_ARCH_MISMATCH``.
