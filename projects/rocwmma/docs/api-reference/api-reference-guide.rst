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

rocWMMA API functions such as ``load_matrix_sync``, ``store_matrix_sync``, and ``mma_sync`` are synchronous when
used with global memory. However, when you use these functions with shared memory, for example, LDS memory,
explicit workgroup synchronization (``synchronize_workgroup``) might be required.


Supported GPU architectures
----------------------------

Supported CDNA architectures:

* gfx908 (wave64)
* gfx90a (wave64)
* gfx942 (wave64)
* gfx950 (wave64)
* gfx1250 (wave32)

.. note::
    gfx9 refers to gfx908, gfx90a, gfx942, and gfx950.


Supported RDNA architectures (wave32):

* gfx1100
* gfx1101
* gfx1102
* gfx1200
* gfx1201

.. note::
    gfx11 refers to gfx1100, gfx1101, and gfx1102.


.. _rocwmma-supported-data-types:

Supported data types
--------------------

rocWMMA mixed precision multiply-accumulate operations support the following data type combinations.

Data Types **<Ti / To / Tc>** = <Input type / Output Type / Compute Type>, where:

* Input Type = Matrix A / B
* Output Type = Matrix C / D
* Compute Type = Math / Accumulation type

Supported data types:

* i8: 8-bit precision integer
* f8: 8-bit precision floating point
* bf8: 8-bit precision brain floating point
* f16: half-precision floating point
* bf16: half-precision brain floating point
* f32: single-precision floating point
* i32: 32-bit precision integer
* xf32: single-precision tensor floating point
* f64: double-precision floating point

.. note::
    f16 includes support for both _Float16 and __half types.

    f8 NANOO (optimized) format is only supported on gfx942, otherwise f8 OCP is assumed on targets that support f8 datatypes.

    f8(bf8) represents any one of mixed  Matrix A / B input type: f8 / f8, f8 / bf8, bf8 / f8, bf8 / bf8.

.. tabularcolumns::
   |C|C|C|C|C|

+------------------------------+------------+-----------+---------------+----------------------------+--------------------+
|Ti / To / Tc                  |BlockM      |BlockN     |BlockK Range*  |       CDNA Support         |    RDNA Support    |
|                              |            |           |(Powers of 2)  |                            |                    |
+==============================+============+===========+===============+============================+====================+
|                              |16          |16         | 32+           |      gfx942, gfx950,       |  gfx1200, gfx1201  |
|     bf8 / f32 / f32          +------------+-----------+---------------+      gfx1250               +--------------------+
|                              |32          |32         | 16+           |                            |        \-          |
+------------------------------+------------+-----------+---------------+----------------------------+--------------------+
|                              |16          |16         | 32+           |      gfx942, gfx950,       |  gfx1200, gfx1201  |
|     f8 / f32 / f32           +------------+-----------+---------------+      gfx1250               +--------------------+
|                              |32          |32         | 16+           |                            |        \-          |
+------------------------------+------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 16            |      gfx908, gfx90a,       |  gfx11, gfx1200,   |
|                              |            |           |               |      gfx1250               |  gfx1201           |
|                              |     16     |    16     +---------------+----------------------------+--------------------+
|                              |            |           | 32            |      gfx942, gfx950,       |        \-          |
|                              |            |           |               |      gfx1250               |                    |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |            |           | 64+           |    gfx950, gfx1250         |        \-          |
|     i8 / i32 / i32           +------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 8             |      gfx908, gfx90a        |        \-          |
|                              |     32     |    32     +---------------+----------------------------+--------------------+
|                              |            |           | 16            |      gfx942, gfx950        |        \-          |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |            |           | 32+           |           gfx950           |        \-          |
+------------------------------+------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 16            |    gfx908, gfx90a,         |   gfx11, gfx1200,  |
|                              |            |           |               |    gfx1250                 |   gfx1201          |
|                              |     16     |    16     +---------------+----------------------------+--------------------+
|                              |            |           | 32            |    gfx942, gfx950,         |        \-          |
|                              |            |           |               |    gfx1250                 |                    |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |            |           | 64+           |    gfx950, gfx1250         |        \-          |
|     i8 / i8 / i32            +------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 8             |      gfx908, gfx90a        |        \-          |
|                              |     32     |    32     +---------------+----------------------------+--------------------+
|                              |            |           | 16            |      gfx942, gfx950        |        \-          |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |            |           | 32+           |           gfx950           |        \-          |
+------------------------------+------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 16            |          gfx1250           |        \-          |
|     f8(bf8) / f16 / f16      |     16     |    16     +---------------+----------------------------+--------------------+
|                              |            |           | 32            |          gfx1250           |        \-          |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |            |           | 64            |          gfx1250           |        \-          |
+------------------------------+------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 16            |          gfx1250           |  gfx1200, gfx1201  |
|     f8(bf8) / f32 / f32      |     16     |    16     +---------------+----------------------------+--------------------+
|                              |            |           | 32            |          gfx1250           |        \-          |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |            |           | 64            |          gfx1250           |        \-          |
+------------------------------+------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 16            |           gfx9             |  gfx11, gfx1200,   |
|                              |            |           |               |                            |  gfx1201           |
|     f16 / f32 / f32          |     16     |    16     +---------------+----------------------------+--------------------+
|                              |            |           | 32+           |          gfx950            |        \-          |
|                              +------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 8             |           gfx9             |        \-          |
|                              |     32     |    32     +---------------+----------------------------+--------------------+
|                              |            |           | 16+           |           gfx950           |        \-          |
+------------------------------+------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 16            |     gfx9, gfx1250          |  gfx11, gfx1200,   |
|                              |            |           |               |                            |  gfx1201           |
|     f16 / f16 / f32          |     16     |    16     +---------------+----------------------------+--------------------+
|                              |            |           | 32+           |    gfx950, gfx1250         |        \-          |
|                              +------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 8             |           gfx9             |        \-          |
|                              |     32     |    32     +---------------+----------------------------+--------------------+
|                              |            |           | 16+           |           gfx950           |        \-          |
+------------------------------+------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 16            |     gfx9, gfx1250          |  gfx11, gfx1200,   |
|                              |            |           |               |                            |  gfx1201           |
|     f16 / f16 / f16**        |     16     |    16     +---------------+----------------------------+--------------------+
|                              |            |           | 32+           |    gfx950, gfx1250         |        \-          |
|                              +------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 8             |           gfx9             |        \-          |
|                              |     32     |    32     +---------------+----------------------------+--------------------+
|                              |            |           | 16+           |           gfx950           |        \-          |
+------------------------------+------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 8             |          gfx908            |        \-          |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |     16     |    16     | 16            |   gfx90a, gfx942, gfx950   |  gfx11, gfx1200,   |
|                              |            |           |               |                            |  gfx1201           |
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
|                              |     16     |    16     | 16            |   gfx90a, gfx942, gfx950,  |  gfx11, gfx1200,   |
|                              |            |           |               |   gfx1250                  |  gfx1201           |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |            |           | 32+           |    gfx950, gfx1250         |        \-          |
|     bf16 / bf16 / f32        +------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 4+            |          gfx908            |        \-          |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |     32     |    32     | 8             |   gfx90a, gfx942, gfx950   |        \-          |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |            |           | 16+           |          gfx950            |        \-          |
+------------------------------+------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 8             |          gfx908            |        \-          |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |     16     |    16     | 16            |   gfx90a, gfx942, gfx950,  |  gfx11, gfx1200,   |
|                              |            |           |               |   gfx1250                  |  gfx1201           |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |            |           | 32+           |    gfx950, gfx1250         |        \-          |
|     bf16 / bf16 / bf16**     +------------+-----------+---------------+----------------------------+--------------------+
|                              |            |           | 4+            |           gfx908           |        \-          |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |     32     |    32     | 8             |   gfx90a, gfx942, gfx950   |        \-          |
|                              |            |           +---------------+----------------------------+--------------------+
|                              |            |           | 16+           |          gfx950            |        \-          |
+------------------------------+------------+-----------+---------------+----------------------------+--------------------+
|                              |16          |16         | 4+            |     gfx9, gfx1250          |        \-          |
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
    BlockM/N values are minimum recommended values. Below these values padding is used which may impact performance. Above this value powers of 2 are acceptable.

    \* BlockK range specifies the minimum recommended value. Below this value padding is used which may impact performance. Above this value powers of 2 are acceptable. In practice, BlockK values are typically 32 or less.

    \*\* On CDNA architectures, matrix unit accumulation is performed in natively 32-bit precision and then converted to the target data type.

.. note::
    rocWMMA supports partial fragment sizes where ``FragMNK`` may be smaller than the ``BlockMNK`` sizes listed in the table above. These fragments are internally padded to nearest supported ``BlockMNK`` sizes.


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

rocWMMA supports up to four wavefronts per thread block. The X dimension should be a multiple of the wave size and is scaled accordingly.

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


default_schedule
^^^^^^^^^^^^^^^^

.. doxygentypedef:: rocwmma::fragment_scheduler::default_schedule


coop_row_major_2d
^^^^^^^^^^^^^^^^^

.. doxygentypedef:: rocwmma::fragment_scheduler::coop_row_major_2d


coop_col_major_2d
^^^^^^^^^^^^^^^^^

.. doxygentypedef:: rocwmma::fragment_scheduler::coop_col_major_2d


coop_row_slice_2d
^^^^^^^^^^^^^^^^^

.. doxygentypedef:: rocwmma::fragment_scheduler::coop_row_slice_2d


coop_col_slice_2d
^^^^^^^^^^^^^^^^^

.. doxygentypedef:: rocwmma::fragment_scheduler::coop_col_slice_2d


single
^^^^^^^^^

.. doxygentypedef:: rocwmma::fragment_scheduler::single


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

.. doxygenfunction:: rocwmma::load_matrix_sync(FragT &frag, const DataT* data, uint32_t ldm)

.. doxygenfunction:: rocwmma::load_matrix_sync(FragT &frag, const DataT* data, uint32_t ldm, layout_t layout)

.. doxygenfunction:: rocwmma::store_matrix_sync(DataT* data, FragT const& frag, uint32_t ldm)

.. doxygenfunction:: rocwmma::store_matrix_sync(DataT* data, FragT const& frag, uint32_t ldm, layout_t layout)

.. doxygenfunction:: rocwmma::mma_sync

.. doxygenfunction:: rocwmma::synchronize_workgroup


rocWMMA transforms API functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: rocwmma::apply_transpose(FragT &&frag)

.. doxygenfunction:: rocwmma::apply_data_layout(FragT &&frag)

.. doxygenfunction:: rocwmma::apply_fragment(FragT &&frag)

.. doxygenfunction:: rocwmma::to_register_file(FragT &&frag)

.. doxygenfunction:: rocwmma::from_register_file(FragT &&frag)

Sample programs
----------------

A sample demonstrating the use of rocWMMA functions ``load_matrix_sync``, ``store_matrix_sync``, ``fill_fragment``, and ``mma_sync`` is available `here <https://github.com/ROCm/rocm-libraries/blob/develop/projects/rocwmma/samples/simple_hgemm.cpp>`_.
For more sample programs, refer to the `samples directory <https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocwmma/samples>`_.

Emulation tests
---------------

The emulation test is a smaller test suite designed for emulators. It includes a subset of ROCWMMA test cases for faster execution on emulated platforms. It supports ``smoke``, ``regression``, and ``extended`` modes.

For example, to run a smoke test:

.. code-block:: bash

   rtest.py --install_dir <build_dir> --emulation smoke
