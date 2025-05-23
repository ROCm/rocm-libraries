.. meta::
  :description: rocPRIM documentation and API reference library
  :keywords: rocPRIM, ROCm, API, documentation

.. _intrinsics:

********************************************************************
 Intrinsics
********************************************************************

Hardware Architecture
=====================

.. doxygenfunction:: rocprim::arch::wavefront::size()
.. doxygenfunction:: rocprim::arch::wavefront::min_size()
.. doxygenfunction:: rocprim::arch::wavefront::max_size()

.. doxygenenum:: rocprim::arch::wavefront::target
.. doxygenfunction:: rocprim::arch::wavefront::target()
.. doxygenfunction:: rocprim::arch::wavefront::size_from_target()

Bitwise
========

.. doxygenfunction:: rocprim::get_bit(int x, int i)
.. doxygenfunction:: rocprim::bit_count(unsigned int x)
.. doxygenfunction:: rocprim::bit_count(unsigned long long x)
.. doxygenfunction:: rocprim::ctz(unsigned int x)
.. doxygenfunction:: rocprim::ctz(unsigned long long x)

Warp size
===========

.. doxygenfunction:: rocprim::host_warp_size(const int device_id, unsigned int& warp_size)
.. doxygenfunction:: rocprim::host_warp_size(const hipStream_t stream, unsigned int& warp_size)

Lane and Warp ID
=================

.. doxygengroup:: intrinsicsmodule_warp_id
   :content-only:

Flat ID
==========

.. doxygengroup:: intrinsicsmodule_flat_id
   :content-only:

Flat Size
===========

.. doxygenfunction:: rocprim::flat_block_size()
.. doxygenfunction:: rocprim::flat_tile_size()

Synchronization
=================

.. doxygenfunction:: rocprim::syncthreads()
.. doxygenfunction:: rocprim::wave_barrier()

Active threads
==================

.. doxygenfunction:: rocprim::ballot (int predicate)
.. doxygenfunction:: rocprim::group_elect(lane_mask_type mask)
.. doxygenfunction:: rocprim::masked_bit_count (lane_mask_type x, unsigned int add=0)
.. doxygenfunction:: rocprim::match_any(unsigned int label, bool valid = true)
.. doxygenfunction:: rocprim::match_any(unsigned int label, unsigned int label_bits, bool valid = true)
