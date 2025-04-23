/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef ROCWMMA_COOP_API_HPP
#define ROCWMMA_COOP_API_HPP

#include "rocwmma.hpp"

//! rocWMMA cooperative API complements the rocWMMA API with support for cooperative operations.
//! Operations may be split into smaller unit work items that are assigned to wavefronts
//! in a collaborative pool using a round-robin fashion.
//!
//! From each wavefront perspective, the data responsibility determines the cohesiveness
//! of the data local to the wavefront's fragment. Cooperative operations are designed to be used
//! in conjunction with each other to cooperatively move a larger full block of data,
//! as is common to do in GEMM algorithms. This helps to optimize potential for data re-use.
//!
//! \n
//! **load_matrix_coop_sync / store_matrix_coop_sync**
//!
//! Loads data from memory according to Matrix Layout.
//! Matrix A layout loads / stores matrix columns in the K direction
//! (Matrix A = M x K, fragA = BlockM x BlockK)
//! Matrix B layout loads / stores matrix rows in the K direction
//! (Matrix B = K x N, fragB = BlockK x BlockN)
//! Matrix C layout loads / stores matrix rows in vector width of 4
//! (Matrix C = M x N, fragAcc = BlockM x BlockN)
//!
//! Fragments are stored in packed registers in optimal load / store patterns.
//! In-register elements have no guaranteed order, which have been optimized for loading / storing efficiency.

namespace rocwmma
{

    //! Loads the fragment from memory address cooperatively across wavefronts.
    //! Each cooperating wavefront is responsible in loading a portion of the final fragment.
    //! This function may be paired with store_matrix_coop_sync to move a single fragment collaboratively between memory locations.
    //! @note Individual wavefronts only load a smaller portion of the full data that they are responsible for.
    //!
    //! The full load is split into work items (splitCount).
    //! Work items are assigned in round robin fashion to waves in the range of [0, waveCount).
    //! The current wave index determines the order of the current wave in the collaboration pool.
    //! Work items are consumed in order by waves [0, waveCount) until
    //! there are no more work items and the operation is completed.
    //!
    //! @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
    //! @param data Data pointer to global/local memory
    //! @param ldm Leading dimension size
    //! @param waveIndex Index assignment of current wave in collaboration
    //! @param waveCount Number of waves assigned for collaboration
    //! @param splitCount Number of work items to split the operation
    //! @tparam MatrixT fragment context
    //! @tparam BlockM/N/K block dimensions
    //! @tparam DataT data type
    //! @tparam DataLayoutT in-memory layout as col_major or row_major
    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE ROCWMMA_DEPRECATED inline void
        load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>& frag,
                              const DataT*                                                   data,
                              uint32_t                                                       ldm,
                              uint32_t waveIndex,
                              uint32_t waveCount,
                              uint32_t splitCount);

    //! Loads the fragment from memory address cooperatively across wavefronts.
    //! Each cooperating wavefront is responsible in loading a portion of the final fragment.
    //! This function may be paired with store_matrix_coop_sync to move a single fragment collaboratively between memory locations.
    //! @note Individual wavefronts only load a smaller portion of the full data that they are responsible for.
    //!
    //! The full load is split into work items (default = waveCount).
    //! Work items are assigned in round robin fashion to waves in the range of [0, waveCount).
    //! The current wave index determines the order of the current wave in the collaboration pool.
    //! Work items are consumed in order by waves [0, waveCount) until there are no more
    //! work items and the operation is completed.
    //!
    //! @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
    //! @param data Data pointer to global/local memory
    //! @param ldm Leading dimension size
    //! @param waveIndex Index assignment of current wave in collaboration
    //! @param waveCount Number of waves assigned for collaboration
    //! @tparam MatrixT fragment context
    //! @tparam BlockM/N/K block dimensions
    //! @tparam DataT data type
    //! @tparam DataLayoutT in-memory layout as col_major or row_major
    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE ROCWMMA_DEPRECATED inline void
        load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>& frag,
                              const DataT*                                                   data,
                              uint32_t                                                       ldm,
                              uint32_t waveIndex,
                              uint32_t waveCount);

    //! Loads the fragment from memory address cooperatively across wavefronts.
    //! Each cooperating wavefront is responsible in loading a portion of the final fragment.
    //! This function may be paired with store_matrix_coop_sync to move a single fragment collaboratively between memory locations.
    //! @note Individual wavefronts only load a smaller portion of the full data that they are responsible for.
    //!
    //! The full load is split into work items (current waveCount).
    //! Work items are assigned in round robin fashion to waves in the range of [0, waveCount).
    //! The current wave index determines the order of the current wave in the collaboration pool.
    //! Work items are consumed in order by waves [0, waveCount) until
    //! there are no more work items and the operation is completed.
    //!
    //! @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
    //! @param data Data pointer to global/local memory
    //! @param ldm Leading dimension size
    //! @tparam MatrixT fragment context
    //! @tparam BlockM/N/K block dimensions
    //! @tparam DataT data type
    //! @tparam DataLayoutT in-memory layout as col_major or row_major
    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE ROCWMMA_DEPRECATED inline void
        load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>& frag,
                              const DataT*                                                   data,
                              uint32_t                                                       ldm);

    // @cond
    //! Loads the fragment from memory address cooperatively across wavefronts.
    //! Each cooperating wavefront is responsible in loading a portion of the final fragment.
    //! This function may be paired with store_matrix_coop_sync to move a single fragment collaboratively between memory locations.
    //! @note Individual wavefronts only load a smaller portion of the full data that they are responsible for.
    //!
    //! This flavor of cooperative load includes WaveCount and SplitCount as template parameters that may be used
    //! to optimize during compile time, and is preferred over these arguments as runtime function arguments.
    //!
    //! The full load is split into work items (SplitCount).
    //! Work items are assigned in round robin fashion to waves in the range of [0, waveCount).
    //! The current wave index determines the order of the current wave in the collaboration pool.
    //! Work items are consumed in order by waves [0, waveCount) until
    //! there are no more work items and the operation is completed.
    //!
    //! @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
    //! @param data Data pointer to global/local memory
    //! @param ldm Leading dimension size
    //! @param waveIndex Index assignment of current wave in collaboration
    //! @tparam uint32_t WaveCount
    //! @tparam uint32_t SplitCount
    //! @tparam MatrixT fragment context
    //! @tparam BlockM/N/K block dimensions
    //! @tparam DataT data type
    //! @tparam DataLayoutT in-memory layout as col_major or row_major
    template <uint32_t WaveCount,
              uint32_t SplitCount,
              typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE ROCWMMA_DEPRECATED inline void
        load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>& frag,
                              const DataT*                                                   data,
                              uint32_t                                                       ldm,
                              uint32_t waveIndex);
    // @endcond

    //! Loads the fragment from memory address cooperatively across wavefronts.
    //! Each cooperating wavefront is responsible in loading a portion of the final fragment.
    //! This function may be paired with store_matrix_coop_sync to move a single fragment collaboratively between memory locations.
    //! @note Individual wavefronts only load a smaller portion of the full data that they are responsible for.
    //!
    //! This flavor of cooperative load includes WaveCount as a template parameter that may be used
    //! to optimize during compile time, and is preferred over providing this value as runtime function argument.
    //!
    //! The full load is split into work items (WaveCount). Work items are assigned
    //! in round robin fashion to waves in the range of [0, waveCount). The current
    //! wave index determines the order of the current wave in the collaboration pool.
    //! Work items are consumed in order by waves [0, waveCount) until there are no more
    //! work items and the operation is completed.
    //!
    //! @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
    //! @param data Data pointer to global/local memory
    //! @param ldm Leading dimension size
    //! @param waveIndex Index assignment of current wave in collaboration
    //! @tparam uint32_t WaveCount
    //! @tparam MatrixT fragment context
    //! @tparam BlockM/N/K block dimensions
    //! @tparam DataT data type
    //! @tparam DataLayoutT in-memory layout as col_major or row_major
    template <uint32_t WaveCount,
              typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE ROCWMMA_DEPRECATED inline void
        load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>& frag,
                              const DataT*                                                   data,
                              uint32_t                                                       ldm,
                              uint32_t waveIndex);

    //! Cooperative Store Matrix - Stores the entire fragment to data address cooperatively across waves.
    //! Each cooperative wave is responsible in storing a portion of the final fragment.
    //! @note The full fragment data is not required to be cohesive for individual waves as they
    //! only store a piece of the data. This function may be paired with load_matrix_coop_sync to move a single fragment
    //! collaboratively between memory locations.
    //!
    //! The full store is split into work items (splitCount). Work items are assigned
    //! in round robin fashion to waves in the range of [0, waveCount). The current
    //! wave index determines the order of the current wave in the collaboration pool.
    //! Work items are consumed in order by waves [0, waveCount) until there are no more
    //! work items and the operation is completed.
    //!
    //! @param data Data pointer to global/local memory
    //! @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
    //! @param ldm Leading dimension size
    //! @param waveIndex Index assignment of current wave in collaboration
    //! @param waveCount Number of waves assigned for collaboration
    //! @param splitCount Number of work items to split the operation
    //! @tparam MatrixT fragment context
    //! @tparam BlockM/N/K block dimensions
    //! @tparam DataT data type
    //! @tparam DataLayoutT in-memory layout as col_major or row_major
    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE ROCWMMA_DEPRECATED inline void store_matrix_coop_sync(
        DataT*                                                               data,
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT> const& frag,
        uint32_t                                                             ldm,
        uint32_t                                                             waveIndex,
        uint32_t                                                             waveCount,
        uint32_t                                                             splitCount);

    //! Cooperative Store Matrix - Stores the entire fragment to data address cooperatively across waves.
    //! Each cooperative wave is responsible in storing a portion of the final fragment.
    //! @note The full fragment data is not required to be cohesive for individual waves as they
    //! only store a piece of the data. This function may be paired with load_matrix_coop_sync to move a single fragment
    //! collaboratively between memory locations.
    //!
    //! The full store is split into work items (default = waveCount). Work items are assigned
    //! in round robin fashion to waves in the range of [0, waveCount). The current
    //! wave index determines the order of the current wave in the collaboration pool.
    //! Work items are consumed in order by waves [0, waveCount) until there are no more
    //! work items and the operation is completed.
    //!
    //! @param data Data pointer to global/local memory
    //! @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
    //! @param ldm Leading dimension size
    //! @param waveIndex Index assignment of current wave in collaboration
    //! @param waveCount Number of waves assigned for collaboration
    //! @tparam MatrixT fragment context
    //! @tparam BlockM/N/K block dimensions
    //! @tparam DataT data type
    //! @tparam DataLayoutT in-memory layout as col_major or row_major
    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE ROCWMMA_DEPRECATED inline void store_matrix_coop_sync(
        DataT*                                                               data,
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT> const& frag,
        uint32_t                                                             ldm,
        uint32_t                                                             waveIndex,
        uint32_t                                                             waveCount);

    //! Cooperative Store Matrix - Stores the entire fragment to data address cooperatively across waves.
    //! Each cooperative wave is responsible in storing a portion of the final fragment.
    //! @note The full fragment data is not required to be cohesive for individual waves as they
    //! only store a piece of the data. This function may be paired with load_matrix_coop_sync to move a single fragment
    //! collaboratively between memory locations.
    //!
    //! The full store is split into work items (current waveCount).
    //! Work items are assigned in round robin fashion to waves in the range of [0, waveCount).
    //! The current wave index determines the order of the current wave in the collaboration pool.
    //! Work items are consumed in order by waves [0, waveCount) until
    //! there are no more work items and the operation is completed.
    //!
    //! @param data Data pointer to global/local memory
    //! @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
    //! @param ldm Leading dimension size
    //! @tparam MatrixT fragment context
    //! @tparam BlockM/N/K block dimensions
    //! @tparam DataT data type
    //! @tparam DataLayoutT in-memory layout as col_major or row_major
    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE ROCWMMA_DEPRECATED inline void store_matrix_coop_sync(
        DataT*                                                               data,
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT> const& frag,
        uint32_t                                                             ldm);

    // @cond
    //! Cooperative Store Matrix - Stores the entire fragment to data address cooperatively across waves.
    //! Each cooperative wave is responsible in storing a portion of the final fragment.
    //! @note The full fragment data is not required to be cohesive for individual waves as they
    //! only store a piece of the data. This function may be paired with load_matrix_coop_sync to move a single fragment
    //! collaboratively between memory locations.
    //!
    //! This flavor of cooperative store includes WaveCount and SplitCount as a template parameter that may be used
    //! to optimize during compile time, and is preferred over providing this value as runtime function argument.
    //!
    //! The full store is split into work items (SplitCount). Work items are assigned
    //! in round robin fashion to waves in the range of [0, waveCount). The current
    //! wave index determines the order of the current wave in the collaboration pool.
    //! Work items are consumed in order by waves [0, waveCount) until there are no more
    //! work items and the operation is completed.
    //!
    //! @param data Data pointer to global/local memory
    //! @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
    //! @param ldm Leading dimension size
    //! @param waveIndex Index assignment of current wave in collaboration
    //! @tparam WaveCount Number of waves participating
    //! @tparam SplitCount Number of work items
    //! @tparam MatrixT fragment context
    //! @tparam BlockM/N/K block dimensions
    //! @tparam DataT data type
    //! @tparam DataLayoutT in-memory layout as col_major or row_major
    template <uint32_t WaveCount,
              uint32_t SplitCount,
              typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE ROCWMMA_DEPRECATED inline void store_matrix_coop_sync(
        DataT*                                                               data,
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT> const& frag,
        uint32_t                                                             ldm,
        uint32_t                                                             waveIndex);
    // @endcond

    //! Cooperative Store Matrix - Stores the entire fragment to data address cooperatively across waves.
    //! Each cooperative wave is responsible in storing a portion of the final fragment.
    //! @note The full fragment data is not required to be cohesive for individual waves as they
    //! only store a piece of the data. This function may be paired with load_matrix_coop_sync to move a single fragment
    //! collaboratively between memory locations.
    //!
    //! This flavor of cooperative store includes WaveCount as a template parameter that may be used
    //! to optimize during compile time, and is preferred over providing this value as runtime function argument.
    //!
    //! The full store is split into work items (WaveCount). Work items are assigned
    //! in round robin fashion to waves in the range of [0, waveCount). The current
    //! wave index determines the order of the current wave in the collaboration pool.
    //! Work items are consumed in order by waves [0, waveCount) until there are no more
    //! work items and the operation is completed.
    //!
    //! @param data Data pointer to global/local memory
    //! @param frag Fragment of type MatrixT with its associated block sizes, data type and layout
    //! @param ldm Leading dimension size
    //! @param waveIndex Index assignment of current wave in collaboration
    //! @tparam WaveCount Number of waves participating
    //! @tparam MatrixT fragment context
    //! @tparam BlockM/N/K block dimensions
    //! @tparam DataT data type
    //! @tparam DataLayoutT in-memory layout as col_major or row_major
    template <uint32_t WaveCount,
              typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE ROCWMMA_DEPRECATED inline void store_matrix_coop_sync(
        DataT*                                                               data,
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT> const& frag,
        uint32_t                                                             ldm,
        uint32_t                                                             waveIndex);

} // namespace rocwmma

#include "rocwmma_coop_impl.hpp"

#endif // ROCWMMA_COOP_API_HPP
