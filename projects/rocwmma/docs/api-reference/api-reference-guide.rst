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


Supported data types
--------------------

rocWMMA supports a range of input, output, and compute data types, including
mixed-precision multiply-accumulate combinations. For the full list of supported
data types and their per-architecture availability, see the
:doc:`rocWMMA data type support <./data-type-support>` page.

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
