.. meta::
   :description: C++ library for accelerating mixed precision matrix multiply-accumulate operations
    leveraging specialized GPU matrix cores on AMD's latest discrete GPUs
   :keywords: rocWMMA, ROCm, library, API, tool

.. _api-reference-guide:

====================
API reference guide
====================

This document provides information about rocWMMA functions, data types, and other programming constructs.

Synchronous API
---------------

In general, rocWMMA API functions ( ``load_matrix_sync``, ``store_matrix_sync``, ``mma_sync`` ) are assumed to be synchronous when
used in the context of global memory.

When using these functions in the context of shared memory (e.g. LDS memory), additional explicit workgroup synchronization (``synchronize_workgroup``)
may be required due to the nature of this memory usage.


Supported GPU architectures
----------------------------

List of supported CDNA architectures (wave64):

* gfx908
* gfx90a
* gfx942
* gfx950

.. note::
    gfx9 = gfx908, gfx90a, gfx942, gfx950


List of supported RDNA architectures (wave32):

* gfx1100
* gfx1101
* gfx1102
* gfx1200
* gfx1201

.. note::
    gfx11 = gfx1100, gfx1101, gfx1102
    gfx12 = gfx1200, gfx1201


Supported data types
--------------------

rocWMMA mixed precision multiply-accumulate operations support the following data type combinations.

Data Types **<Ti / To / Tc>** = <Input type / Output Type / Compute Type>, where:

* Input Type = Matrix A / B
* Output Type = Matrix C / D
* Compute Type = Math / accumulation type

* i8 = 8-bit precision integer
* f8 = 8-bit precision floating point
* bf8 = 8-bit precision brain floating point
* f16 = half-precision floating point
* bf16 = half-precision brain floating point
* f32 = single-precision floating point
* i32 = 32-bit precision integer
* xf32 = single-precision tensor floating point
* f64 = double-precision floating point

.. note::
    f16 represents equivalent support for both _Float16 and __half types.

    Current f8 support is NANOO (optimized) format.

.. tabularcolumns::
   |C|C|C|C|C|

+------------------------------+------------+-----------+---------------+----------------------------+--------------------+
|Ti / To / Tc                  |BlockM      |BlockN     |BlockK Range*  |       CDNA Support         |    RDNA Support    |
|                              |            |           |(Powers of 2)  |                            |                    |
+==============================+============+===========+===============+============================+====================+
|                              |16          |16         | 32+           |                            |       gfx12        |
|     bf8 / f32 / f32          +------------+-----------+---------------+       gfx940, gfx950       +--------------------+
|                              |32          |32         | 16+           |                            |        \-          |
+------------------------------+------------+-----------+---------------+----------------------------+--------------------+
|                              |16          |16         | 32+           |                            |       gfx12        |
|     f8 / f32 / f32           +------------+-----------+---------------+       gfx940, gfx950       +--------------------+
|                              |32          |32         | 16+           |                            |        \-          |
+------------------------------+------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 16            |      gfx908, gfx90a        |   gfx11, gfx12     |
|                              |     16     |    16     +---------------+----------------------------+--------------------+
|                              |            |           | 32            |      gfx940, gfx950        |        \-          |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |            |           | 64+           |           gfx950           |        \-          |
|     i8 / i32 / i32           +------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 8             |      gfx908, gfx90a        |        \-          |
|                              |     32     |    32     +---------------+----------------------------+--------------------+
|                              |            |           | 16            |      gfx940, gfx950        |        \-          |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |            |           | 32+           |           gfx950           |        \-          |
+------------------------------+------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 16            |      gfx908, gfx90a        |   gfx11, gfx12     |
|                              |     16     |    16     +---------------+----------------------------+--------------------+
|                              |            |           | 32            |      gfx940, gfx950        |        \-          |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |            |           | 64+           |           gfx950           |        \-          |
|     i8 / i8 / i32            +------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 8             |      gfx908, gfx90a        |        \-          |
|                              |     32     |    32     +---------------+----------------------------+--------------------+
|                              |            |           | 16            |      gfx940, gfx950        |        \-          |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |            |           | 32+           |           gfx950           |        \-          |
+------------------------------+------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 16            |           gfx9             |   gfx11, gfx12     |
|     f16 / f32 / f32          |     16     |    16     +---------------+----------------------------+--------------------+
|                              |            |           | 32+           |          gfx950            |        \-          |
|                              +------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 8             |           gfx9             |        \-          |
|                              |     32     |    32     +---------------+----------------------------+--------------------+
|                              |            |           | 16+           |           gfx950           |        \-          |
+------------------------------+------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 16            |           gfx9             |   gfx11, gfx12     |
|     f16 / f16 / f32          |     16     |    16     +---------------+----------------------------+--------------------+
|                              |            |           | 32+           |          gfx950            |        \-          |
|                              +------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 8             |           gfx9             |        \-          |
|                              |     32     |    32     +---------------+----------------------------+--------------------+
|                              |            |           | 16+           |           gfx950           |        \-          |
+------------------------------+------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 16            |           gfx9             |   gfx11, gfx12     |
|     f16 / f16 / f16**        |     16     |    16     +---------------+----------------------------+--------------------+
|                              |            |           | 32+           |          gfx950            |        \-          |
|                              +------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 8             |           gfx9             |        \-          |
|                              |     32     |    32     +---------------+----------------------------+--------------------+
|                              |            |           | 16+           |           gfx950           |        \-          |
+------------------------------+------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 8             |          gfx908            |        \-          |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |     16     |    16     | 16            |   gfx90a, gfx942, gfx950   |       gfx11        |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |            |           | 32+           |           gfx950           |        \-          |
|     bf16 / f32 / f32         +------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 4+            |          gfx908            |        \-          |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |     32     |    32     | 8             |   gfx90a, gfx942, gfx950   |        \-          |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |            |           | 16+           |          gfx950            |        \-          |
+------------------------------+------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 8             |          gfx908            |        \-          |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |     16     |    16     | 16            |   gfx90a, gfx942, gfx950   |       gfx11        |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |            |           | 32+           |           gfx950           |        \-          |
|     bf16 / bf16 / f32        +------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 4+            |          gfx908            |        \-          |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |     32     |    32     | 8             |   gfx90a, gfx942, gfx950   |        \-          |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |            |           | 16+           |          gfx950            |        \-          |
+------------------------------+------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 8             |          gfx908            |        \-          |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |     16     |    16     | 16            |   gfx90a, gfx942, gfx950   |       gfx11        |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |            |           | 32+           |           gfx950           |        \-          |
|     bf16 / bf16 / bf16**     +------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 4+            |          gfx908            |        \-          |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |     32     |    32     | 8             |   gfx90a, gfx942, gfx950   |        \-          |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |            |           | 16+           |          gfx950            |        \-          |
+------------------------------+------------+-----------+---------------+----------------------------+--------------------+
|                              |16          |16         | 4+            |           gfx9             |        \-          |
|     f32 / f32 / f32          +------------+-----------+---------------+----------------------------+--------------------+
|                              |32          |32         | 2+            |           gfx9             |        \-          |
+------------------------------+------------+-----------+---------------+----------------------------+--------------------+
|                              |16          |16         | 8+            |                            |                    |
|     xf32 / xf32 / xf32       +------------+-----------+---------------+          gfx942            |        \-          |
|                              |32          |32         | 4+            |                            |                    |
+------------------------------+------------+-----------+---------------+----------------------------+--------------------+
|      f64 / f64 / f64         |16          |16         | 4+            |   gfx90a, gfx942, gfx950   |        \-          |
+------------------------------+------------+-----------+---------------+----------------------------+--------------------+

.. note::
    \* = BlockK range lists the minimum possible value. Other values in the range are powers of 2 larger than the minimum. Practical BlockK values are usually 32 and smaller.

    \*\* = CDNA architectures matrix unit accumulation is natively 32-bit precision and is converted to the desired type.


Supported matrix layouts
------------------------

(N = col major, T = row major)

.. tabularcolumns::
   |C|C|C|C|

+---------+--------+---------+--------+
|LayoutA  |LayoutB |Layout C |LayoutD |
+=========+========+=========+========+
|N        |N       |N        |N       |
+---------+--------+---------+--------+
|N        |N       |T        |T       |
+---------+--------+---------+--------+
|N        |T       |N        |N       |
+---------+--------+---------+--------+
|N        |T       |T        |T       |
+---------+--------+---------+--------+
|T        |N       |N        |N       |
+---------+--------+---------+--------+
|T        |N       |T        |T       |
+---------+--------+---------+--------+
|T        |T       |N        |N       |
+---------+--------+---------+--------+
|T        |T       |T        |T       |
+---------+--------+---------+--------+

Supported thread block sizes
----------------------------

rocWMMA generally supports and tests up to 4 wavefronts per threadblock. The X dimension is expected to be a multiple of the wave size and will be scaled as such.

.. tabularcolumns::
   |C|C|

+------------+------------+
|TBlock_X    |TBlock_Y    |
+============+============+
|WaveSize    |1           |
+------------+------------+
|WaveSize    |2           |
+------------+------------+
|WaveSize    |4           |
+------------+------------+
|WaveSize*2  |1           |
+------------+------------+
|WaveSize*2  |2           |
+------------+------------+
|WaveSize*4  |1           |
+------------+------------+

.. note::
    WaveSize (RDNA) = 32

    WaveSize (CDNA) = 64


Using rocWMMA API
-----------------

This section describes how to use the rocWMMA library API.

rocWMMA datatypes
-----------------

matrix_a
^^^^^^^^

.. doxygenstruct:: rocwmma::matrix_a


matrix_b
^^^^^^^^

.. doxygenstruct:: rocwmma::matrix_b


accumulator
^^^^^^^^^^^

.. doxygenstruct:: rocwmma::accumulator


row_major
^^^^^^^^^

.. doxygenstruct:: rocwmma::row_major


col_major
^^^^^^^^^

.. doxygenstruct:: rocwmma::col_major


fragment
^^^^^^^^

.. doxygenclass:: rocwmma::fragment
   :members:


rocWMMA enumeration
-------------------

layout_t
^^^^^^^^

.. doxygenenum:: rocwmma::layout_t


rocWMMA API functions
----------------------

.. doxygenfunction:: rocwmma::fill_fragment

.. doxygenfunction:: rocwmma::load_matrix_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>& frag, const DataT* data, uint32_t ldm)

.. doxygenfunction:: rocwmma::load_matrix_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT>& frag, const DataT* data, uint32_t ldm, layout_t layout)

.. doxygenfunction:: rocwmma::store_matrix_sync(DataT* data, fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT> const& frag, uint32_t ldm)

.. doxygenfunction:: rocwmma::store_matrix_sync(DataT* data, fragment<MatrixT, BlockM, BlockN, BlockK, DataT> const& frag, uint32_t ldm, layout_t layout)

.. doxygenfunction:: rocwmma::mma_sync

.. doxygenfunction:: rocwmma::synchronize_workgroup

rocWMMA cooperative API functions
---------------------------------

.. doxygenfunction:: rocwmma::load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>& frag, const DataT* data, uint32_t ldm, uint32_t waveIndex, uint32_t waveCount)

.. doxygenfunction:: rocwmma::load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>& frag, const DataT* data, uint32_t ldm)

.. doxygenfunction:: rocwmma::load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>& frag, const DataT* data, uint32_t ldm, uint32_t waveIndex)

.. doxygenfunction:: rocwmma::store_matrix_coop_sync(DataT* data, fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT> const& frag, uint32_t ldm, uint32_t waveIndex, uint32_t waveCount)

.. doxygenfunction:: rocwmma::store_matrix_coop_sync(DataT* data, fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT> const& frag, uint32_t ldm)

.. doxygenfunction:: rocwmma::store_matrix_coop_sync(DataT* data, fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT> const& frag, uint32_t ldm, uint32_t waveIndex)

rocWMMA transforms API functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: rocwmma::applyTranspose(FragT &&frag)

.. doxygenfunction:: rocwmma::applyDataLayout(FragT &&frag)

Sample programs
----------------

See a sample code for calling rocWMMA functions ``load_matrix_sync``, ``store_matrix_sync``, ``fill_fragment``, and ``mma_sync`` `here <https://github.com/ROCm/rocWMMA/blob/develop/samples/simple_hgemm.cpp>`_.
For more such sample programs, refer to the `Samples directory <https://github.com/ROCm/rocWMMA/tree/develop/samples>`_.

Emulation tests
---------------

The emulation test is a smaller test suite specifically designed for emulators. It comprises a selection of test cases from the full ROCWMM test set, allowing for significantly faster execution on emulated platforms. Despite its concise nature, the emulation test supports ``smoke``, ``regression``, and ``extended`` modes.

For example, run a smoke test.

.. code-block:: bash

   rtest.py --install_dir <build_dir> --emulation smoke
