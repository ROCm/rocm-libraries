.. meta::
   :description: C++ library for accelerating mixed precision matrix multiply-accumulate operations
    leveraging specialized GPU matrix cores on AMD's latest discrete GPUs
   :keywords: rocWMMA, ROCm, library, API, tool

.. _migration-guide:

===============================
Migration guide for rocWMMA 2.0 
===============================

This document outlines the key API changes and new features introduced in rocWMMA 2.0, with examples to help you migrate from earlier versions.

Starting with version 2.0, rocWMMA introduces significant changes to its API, including:

- Removal of the cooperative API
- Transforms API no longer requires wave count template parameters
- rocWMMA fragments now have a fragment scheduler template argument
- rocWMMA fragments now support partial fragment sizes

-----------------------
Cooperative API changes
-----------------------

Previous releases began deprecating cooperative API functions such as those defined in ``rocwmma/rocwmma_coop.hpp``:

.. code-block:: c++

   template <uint32_t WaveCount, typename MatrixT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT, typename DataLayoutT>
   ROCWMMA_DEVICE void load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>& frag,
                                             const DataT*                                                   data,
                                             uint32_t                                                       ldm,
                                             uint32_t                                                       waveIndex);

   template <uint32_t WaveCount, typename MatrixT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT, typename DataLayoutT>
   ROCWMMA_DEVICE void store_matrix_coop_sync(DataT*                                                               data,
                                             fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT> const& frag,
                                             uint32_t                                                             ldm,
                                             uint32_t                                                             waveIndex);

These functions previously required ``WaveCount`` as a template parameter and passed ``waveIndex`` as an argument to the API calls. This information was used to distribute data responsibilities across participating waves, aiming to balance and optimize data transactions within a thread block. Cooperation between wavefronts in a thread block requires the use of a separate cooperative API, along with propagation of wave count and wave index values.

Example of deprecated cooperative API:

.. code-block:: c++

   // Global read (macro tile)
   using GRBuffA = fragment<matrix_a, MACRO_TILE_X, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutA>;

   // Local warp coordinate relative to current thread block (wg).
   constexpr auto warpDims        = make_coord2d(WARPS_X, WARPS_Y);
   auto           localWarpCoord  = make_coord2d(threadIdx.x / WARP_SIZE, threadIdx.y);

   // WorkItems will be split up by minimum IOCount to perform either global read or local write.
   // These are inputs to cooperative functions.
   constexpr auto warpCount = get<0>(warpDims) * get<1>(warpDims);

   // Scheduling warp order is analogous to row major priority.
   // E.g. Wg = (128, 2) = 2x2 warps
   // (0, 0)   (0, 1)   Share Schedule: w0 = (0, 0), w1 = (0, 1),
   // (1, 0)   (1, 1)                   w2 = (1, 0), w3 = (1, 1), count = 4
   const auto warpIndex = get<0>(localWarpCoord) * get<1>(warpDims) + get<1>(localWarpCoord);

   // Transfer data from global memory to local memory
   GRBuffA grBuffA;
   load_matrix_coop_sync<warpCount>(grBuffA, gAddrA, lda, warpIndex);
   store_matrix_coop_sync<warpCount>(ldsAddr, applyDataLayout<DataLayoutLds, warpCount>(applyTranspose(grBuffA)), ldsld, warpIndex);

Calculating the warp count and warp index requires extra boilerplate code. It is important to supply the same warp count and warp index values to matching pairs of load, store, and transform APIs. Providing mismatched values to APIs that depend on matching warp count and index poses a risk of incorrect behavior. Embedding the warp count and index into the fragment object helps mitigate the risk.

As a result, fragments in rocWMMA 2.0 are augmented with an additional fragment scheduler template parameter. Fragment schedulers are classes that represent thread block scheduling models. These models provide static values for both the wave count and wave order (wave index). Fragment schedulers are classified as either non-cooperative (the default, where waves act independently) or cooperative (where waves collaborate within a thread block). Their names reflect their ordering scheme.

Example:

.. code-block:: c++

   namespace fragment_scheduler
   {
      //! @struct default
      //! @brief The default fragment scheduler; each wave operates independently.
      using default_schedule = IOScheduler::Default;

      //! @struct coop_row_major_2d
      //! @brief  A cooperative scheduling strategy where each wave in the 2d thread block
      //! will contribute to the fragment operation in row_major grid order.
      //! All waves are scheduled in row_major order.
      //! E.g. (TBlockX, TBlockY) => 2x2 waves
      //! w0 = (0, 0),  w1 = (0, 1),
      //! w2 = (1, 0),  w3 = (1, 1)
      //! @tparam TBlockX the size of the thread-block in the X dimension
      //! @tparam TBlockY the size of the thread-block in the Y dimension
      template <uint32_t TBlockX, uint32_t TBlockY>
      using coop_row_major_2d = IOScheduler::RowMajor2d<TBlockX, TBlockY>;

      ...
   }

Here is the simplified usage with new cooperative fragment changes:

.. code-block:: c++

   // Global read (macro tile)
   // Distribute segments of macro tile data between waves of the thread block in
   // row major order.
   using CoopScheduler = fragment_scheduler::coop_row_major_2d<TBLOCK_X, TBLOCK_Y>;
   using GRBuffA = fragment<matrix_a, MACRO_TILE_X, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutA, CoopScheduler>;

   // Transfer data from global memory to local memory
   GRBuffA grBuffA;
   load_matrix_sync(grBuffA, gAddrA, lda);
   store_matrix_sync(ldsAddr, apply_data_layout<DataLayoutLds>(apply_transpose(grBuffA)), ldsld);

To summarize, the ``CoopScheduler`` template parameter allows you to express the required cooperative behavior with the fragment class declaration. Boilerplate code for calculating wave count and wave indices is wrapped into the ``CoopScheduler`` class. You can use fragments with the standard rocWMMA API without the need to externally propagate matching wave counts or wave indices, making rocWMMA more compact and expressive than previous versions.

.. note::

   Cooperative fragment schedulers require template parameters for ``TBLOCK_X`` and ``TBLOCK_Y`` dimensions. This design enables various optimizations by allowing the schedulers to provide a static wave count at compile time. As a result, rocWMMA no longer supports run-time wave count calculations in favor of better performance.

------------------------
Partial fragment support
------------------------

In previous rocWMMA versions, fragment sizes were required to be a multiple of the minimum block sizes, as described in the :doc:`programmers-guide`. This was a function of the MMA implementation of hardware acceleration. Thus, rocWMMA serves as a direct hardware enablement to employ block-wise decomposition of matrix-multiply problems. In absence of perfect block-wise decompositions, there is a need to accommodate odd-sized blocks or partials. To increase the utility of rocWMMA to more applications, rocWMMA was extended to include support for partial tile sizes, allowing fragment dimensions (FragMNK) to differ from the minimum block-wise dimensions required for MMA (BlockMNK). rocWMMA now pads FragMNK dimensions to meet the minimal BlockMNK dimensions, ensuring compatibility with MMA hardware.

.. code-block:: c++

   // Fragment types, assuming ROCWMMA_MNK are minimum block sizes.
   // These fragments will not use any padding.
   using FragA = fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutA>;
   using FragB = fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutB>;
   using Accum = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, AccumT>;

   FragA fragA;
   FragB fragB;
   Accum accum;
   fill_fragment(accum, 0);

   load_matrix_sync(fragA, gAddrA, lda);
   load_matrix_sync(fragB, gAddrB, ldb);
   mma_sync(accum, fragA, fragB, accum);

   store_matrix_sync(gResC, accum, ldc, layout_t::mem_row_major);

   // Now also supported

   // Fragment types, which are partial fragments.
   // These fragments will use padding to minimum block sizes internally.
   // Note: The dimensions (2, 3, 1) are smaller than BlockMNK, creating partial fragments
   using FragA = fragment<matrix_a, 2, 3, 1, InputT, DataLayoutA>;
   using FragB = fragment<matrix_b, 2, 3, 1, InputT, DataLayoutB>;
   using Accum = fragment<accumulator, 2, 3, 1, AccumT>;

   FragA fragA;
   FragB fragB;
   Accum accum;
   fill_fragment(accum, 0);

   load_matrix_sync(fragA, gAddrA, lda);
   load_matrix_sync(fragB, gAddrB, ldb);
   mma_sync(accum, fragA, fragB, accum);

   store_matrix_sync(gResC, accum, ldc, layout_t::mem_row_major);


In summary, partial tiles are padded to the minimum MMA block dimensions to accommodate a wider range of fragment sizes. However, this added flexibility comes at a cost: extra registers used for padding might increase kernel register pressure for small tiles and incur extra overhead for checking boundary conditions. Padded fragments are logically restricted to writing in FragMNK dimensions and zeroing boundary conditions.

.. note::

   Padded fragment internals always use padded-sized resources instead of fragment-sized resources.
   However, fragment element-wise accesses, such as uniform FMA, should continue to use ``fragment.num_elements``, assuming that any padded elements will be zero.

   Example:

   .. code-block:: c++

      // Fused multiply-add still valid for partials as padded elements are 0
      for(int i = 0; i < frag.num_elements; i++)
      {
         frag.x[i] = frag.x[i] * (alpha + 1);
      }
