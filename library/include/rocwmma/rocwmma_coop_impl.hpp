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
#ifndef ROCWMMA_COOP_API_IMPL_HPP
#define ROCWMMA_COOP_API_IMPL_HPP

#include "internal/coop_io_config.hpp"
#include "internal/coop_load.hpp"
#include "internal/coop_store.hpp"

#include "rocwmma_coop.hpp"

namespace rocwmma
{
    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE inline void
        load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>& frag,
                              const DataT*                                                   data,
                              uint32_t                                                       ldm,
                              uint32_t waveIndex,
                              uint32_t waveCount,
                              uint32_t splitCount)
    {
        // splitCount unused
        load_matrix_coop_sync(frag, data, ldm, waveIndex, waveCount);
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE inline void
        load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>& frag,
                              const DataT*                                                   data,
                              uint32_t                                                       ldm,
                              uint32_t waveIndex,
                              uint32_t waveCount)
    {

        using FragT    = decay_t<decltype(frag)>;
        using IOConfig = GetCoopIOConfig_t<FragT, 1u>;
        using Loader   = typename IOConfig::Loader;
        using PostLoad = typename IOConfig::PostLoadXForm;

        // Sanity checks
        static_assert(!is_same<DataLayoutT, void>::value,
                      "Must provide layout information. Either statically assign data layout in "
                      "fragment declaration or use the run-time function overload.");

        // TODO: Fix
        // static_assert(
        //     is_same<typename FragT::Traits::AccessT, typename Loader::Traits::OutputT>::value,
        //     "Fragment access and coop load output types do not match");

        // Load and implicit pack
        // Note: the frag will only be partially filled with useful data.
        // Layout and thread locality is not guaranteed.
        Loader::exec(frag.mAccess, data, ldm, waveIndex, waveCount);

        // Post-load transformation
        frag.mAccess = PostLoad::exec(frag.mAccess);
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE inline void
        load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>& frag,
                              const DataT*                                                   data,
                              uint32_t                                                       ldm)
    {
        using FragT       = decay_t<decltype(frag)>;
        using MappingUtil = GetMappingUtil_t<FragT>;

        // Default: all waves participate in 'row major' order
        auto waveCoord = MappingUtil::waveCoord();
        auto wgDim     = MappingUtil::workgroupDim();

        auto waveIndex = get<0>(waveCoord) * get<1>(wgDim) + get<1>(waveCoord);
        auto waveCount = get<0>(wgDim) * get<1>(wgDim);
        load_matrix_coop_sync(forward<FragT&>(frag), data, ldm, waveIndex, waveCount);
    }

    // @cond
    template <uint32_t WaveCount,
              uint32_t SplitCount,
              typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE inline void
        load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>& frag,
                              const DataT*                                                   data,
                              uint32_t                                                       ldm,
                              uint32_t waveIndex)
    {
        // SplitCount is unused
        load_matrix_coop_sync<WaveCount>(frag, data, ldm, waveIndex);
    }
    // @endcond

    template <uint32_t WaveCount,
              typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE void
        load_matrix_coop_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>& frag,
                              const DataT*                                                   data,
                              uint32_t                                                       ldm,
                              uint32_t waveIndex)
    {
        using FragT    = decay_t<decltype(frag)>;
        using IOConfig = GetCoopIOConfig_t<FragT, WaveCount>;
        using Loader   = typename IOConfig::Loader;
        using PostLoad = typename IOConfig::PostLoadXForm;

        // Sanity checks
        static_assert(!is_same<DataLayoutT, void>::value,
                      "Must provide layout information. Either statically assign data layout in "
                      "fragment declaration or use the run-time function overload.");

        // TODO: Fix
        // static_assert(
        //     is_same<typename FragT::Traits::AccessT, typename Loader::Traits::OutputT>::value,
        //     "Fragment access and coop load output types do not match");

        // Load and implicit pack
        // Note: the frag will only be partially filled with useful data.
        // Layout and thread locality is not guaranteed.
        Loader::exec(frag.mAccess, data, ldm, waveIndex);

        // Post-load transformation
        frag.mAccess = PostLoad::exec(frag.mAccess);
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE inline void store_matrix_coop_sync(
        DataT*                                                               data,
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT> const& frag,
        uint32_t                                                             ldm,
        uint32_t                                                             waveIndex,
        uint32_t                                                             waveCount,
        uint32_t                                                             splitCount)
    {
        // splitCount unused
        store_matrix_coop_sync(data, frag, ldm, waveIndex, waveCount);
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE inline void store_matrix_coop_sync(
        DataT*                                                               data,
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT> const& frag,
        uint32_t                                                             ldm,
        uint32_t                                                             waveIndex,
        uint32_t                                                             waveCount)
    {
        using FragT    = decay_t<decltype(frag)>;
        using IOConfig = GetCoopIOConfig_t<FragT, 1u>;
        using PreStore = typename IOConfig::PreStoreXForm;
        using Storer   = typename IOConfig::Storer;

        // Sanity checks
        static_assert(!is_same<DataLayoutT, void>::value,
                      "Must provide data layout. Either statically assign data layout in "
                      "fragment declaration or use the run-time function overload.");

        // TODO: Fix
        // static_assert(
        //     is_same<typename FragT::Traits::AccessT, typename Storer::Traits::InputT>::value,
        //     "Fragment access and coop store input types do not match");

        // Implicit unpack and store
        // Note: the frag is only be partially filled with useful data.
        // Layout and thread locality is not guaranteed.
        Storer::exec(data, PreStore::exec(frag.mAccess), ldm, waveIndex, waveCount);
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE inline void store_matrix_coop_sync(
        DataT*                                                               data,
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT> const& frag,
        uint32_t                                                             ldm)
    {
        using FragT       = decay_t<decltype(frag)>;
        using MappingUtil = GetMappingUtil_t<FragT>;

        // Default: all waves participate in 'row major' order
        auto waveCoord = MappingUtil::waveCoord();
        auto wgDim     = MappingUtil::workgroupDim();

        auto waveIndex = get<0>(waveCoord) * get<1>(wgDim) + get<1>(waveCoord);
        auto waveCount = get<0>(wgDim) * get<1>(wgDim);
        store_matrix_coop_sync(data, frag, ldm, waveIndex, waveCount);
    }

    // @cond
    template <uint32_t WaveCount,
              uint32_t SplitCount,
              typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE inline void store_matrix_coop_sync(
        DataT*                                                               data,
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT> const& frag,
        uint32_t                                                             ldm,
        uint32_t                                                             waveIndex)
    {
        // Implicit unpack and store
        // Note: the frag is only be partially filled with useful data.
        // Layout and thread locality is not guaranteed.
        store_matrix_coop_sync<WaveCount>(data, frag, ldm, waveIndex);
    }
    // @endcond

    template <uint32_t WaveCount,
              typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE void store_matrix_coop_sync(
        DataT*                                                               data,
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT> const& frag,
        uint32_t                                                             ldm,
        uint32_t                                                             waveIndex)
    {
        using FragT    = decay_t<decltype(frag)>;
        using IOConfig = GetCoopIOConfig_t<FragT, WaveCount>;
        using PreStore = typename IOConfig::PreStoreXForm;
        using Storer   = typename IOConfig::Storer;

        // Sanity checks
        static_assert(!is_same<DataLayoutT, void>::value,
                      "Must provide data layout. Either statically assign data layout in "
                      "fragment declaration or use the run-time function overload.");

        // static_assert(
        //     is_same<typename FragT::Traits::AccessT, typename Storer::Traits::InputT>::value,
        //     "Fragment access and coop stor input types do not match");

        // Implicit unpack and store
        // Note: the frag is only be partially filled with useful data.
        // Layout and thread locality is not guaranteed.
        Storer::exec(data, PreStore::exec(frag.mAccess), ldm, waveIndex);
    }

} // namespace rocwmma

#endif // ROCWMMA_COOP_API_IMPL_HPP
