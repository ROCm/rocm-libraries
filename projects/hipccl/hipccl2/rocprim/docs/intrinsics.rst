Intrinsics
==========

Bitwise
-------

.. doxygenfunction:: rocprim::get_bit(int x, int i)
.. doxygenfunction:: rocprim::bit_count(unsigned int x)
.. doxygenfunction:: rocprim::bit_count(unsigned long long x)

Warp size
---------

.. doxygenfunction:: rocprim::warp_size()
.. doxygenfunction:: rocprim::host_warp_size()
.. doxygenfunction:: rocprim::device_warp_size()

Lane and Warp ID
----------------

.. doxygengroup:: intrinsicsmodule_warp_id
   :content-only:

Flat ID
-------

.. doxygengroup:: intrinsicsmodule_flat_id
   :content-only:

Flat Size
---------

.. doxygenfunction:: rocprim::flat_block_size()
.. doxygenfunction:: rocprim::flat_tile_size()

Synchronization
---------------

.. doxygenfunction:: rocprim::syncthreads()
.. doxygenfunction:: rocprim::wave_barrier()

Active threads
--------------


.. doxygenfunction:: rocprim::ballot (int predicate)
.. doxygenfunction:: rocprim::masked_bit_count (lane_mask_type x, unsigned int add=0)
