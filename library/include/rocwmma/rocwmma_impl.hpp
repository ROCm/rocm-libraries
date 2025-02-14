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
#ifndef ROCWMMA_API_IMPL_HPP
#define ROCWMMA_API_IMPL_HPP

#include "rocwmma.hpp"

#include "internal/accessors.hpp"
#include "internal/blend.hpp"
#include "internal/broadcast.hpp"
#include "internal/constants.hpp"
#include "internal/convert.hpp"
#include "internal/dpp.hpp"
#include "internal/flow_control.hpp"
#include "internal/io_config.hpp"
#include "internal/io_layout.hpp"
#include "internal/io_shape.hpp"
#include "internal/io_traits.hpp"
#include "internal/layout/layout.hpp"
#include "internal/mapping_util.hpp"
#include "internal/mfma.hpp"
#include "internal/mma.hpp"
#include "internal/mma_config.hpp"
#include "internal/opaque_load.hpp"
#include "internal/opaque_store.hpp"
#include "internal/pack_util.hpp"
#include "internal/permute.hpp"
#include "internal/swizzle.hpp"
#include "internal/transforms.hpp"
#include "internal/types.hpp"
#include "internal/utils.hpp"
#include "internal/vector.hpp"
#include "internal/vector_iterator.hpp"
#include "internal/vector_util.hpp"
#include "internal/wmma.hpp"

namespace rocwmma
{
    // fragment implementations
    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>::fragment(
        const fragment& other)
        : mStorage(other.mStorage)
    {
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>&
                   fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>::operator=(
            const fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>& other)
    {
        mStorage = other.mStorage;
        return *this;
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE inline DataT&
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>::operator[](uint32_t index)
    {
        return mAccess.data[index];
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE inline auto
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>::operator*() ->
        typename Traits::StorageT&
    {
        return mStorage;
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE inline DataT const&
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>::operator[](
            uint32_t index) const
    {
        return mAccess.data[index];
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE inline auto
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>::operator*() const ->
        typename Traits::StorageT const&
    {
        return mStorage;
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE constexpr inline uint32_t
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>::height()
    {
        return GetIOShape_t<decltype(fragment())>::BlockHeight;
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE constexpr inline uint32_t
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>::width()
    {
        return GetIOShape_t<decltype(fragment())>::BlockWidth;
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE constexpr inline uint32_t
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>::blockDim()
    {
        return GetIOShape_t<decltype(fragment())>::BlockDim;
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE constexpr inline uint32_t
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>::kDim()
    {
        return GetIOShape_t<decltype(fragment())>::KDim;
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE constexpr inline uint32_t
        fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>::size()
    {
        return num_elements;
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE void
        fill_fragment(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>& frag,
                      DataT                                                          value)
    {
        using FragT       = decay_t<decltype(frag)>;
        using Broadcaster = typename GetIOConfig_t<FragT>::Broadcaster;

        // Sanity check
        static_assert(is_same<typename Broadcaster::Traits::BroadcastT,
                              typename FragT::Traits::AccessT>::value,
                      "Broadcast input and fragment access types do not match");

        Broadcaster::exec(frag.mAccess, value);
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE void
        load_matrix_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT>& frag,
                         const DataT*                                                   data,
                         uint32_t                                                       ldm)
    {
        using FragT    = decay_t<decltype(frag)>;
        using IOConfig = GetIOConfig_t<FragT>;
        using Loader   = typename IOConfig::Loader;
        using PostLoad = typename IOConfig::PostLoadXForm;

        // Sanity checks
        static_assert(!is_same<DataLayoutT, void>::value,
                      "Must provide layout information. Either statically assign data layout in "
                      "fragment declaration or use the run-time function overload.");

        static_assert(is_same<typename FragT::Traits::AccessT, typename Loader::BufferT>::value,
                      "Fragment access and load buffer types do not match");

        // Load then implicit pack
        Loader::exec(frag.mAccess, data, ldm);

        // Post-load transformation
        frag.mAccess = PostLoad::exec(frag.mAccess);
    }

    template <typename MatrixT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
    ROCWMMA_DEVICE void load_matrix_sync(fragment<MatrixT, BlockM, BlockN, BlockK, DataT>& frag,
                                         const DataT*                                      data,
                                         uint32_t                                          ldm,
                                         layout_t                                          layout)
    {
        using FragRowMajor = fragment<MatrixT, BlockM, BlockN, BlockK, DataT, row_major>;
        using FragColMajor = fragment<MatrixT, BlockM, BlockN, BlockK, DataT, col_major>;

        // Dispatch on layout type
        if(layout == layout_t::mem_row_major)
        {
            load_matrix_sync(reinterpret_cast<FragRowMajor&>(frag), data, ldm);
        }
        else
        {
            load_matrix_sync(reinterpret_cast<FragColMajor&>(frag), data, ldm);
        }
    }

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    ROCWMMA_DEVICE void
        store_matrix_sync(DataT*                                                               data,
                          fragment<MatrixT, BlockM, BlockN, BlockK, DataT, DataLayoutT> const& frag,
                          uint32_t                                                             ldm)
    {
        using FragT    = decay_t<decltype(frag)>;
        using IOConfig = GetIOConfig_t<FragT>;
        using PreStore = typename IOConfig::PreStoreXForm;
        using Storer   = typename IOConfig::Storer;

        // Sanity check
        static_assert(!is_same<DataLayoutT, void>::value,
                      "Must provide data layout. Either statically assign data layout in "
                      "fragment declaration or use the run-time function overload.");

        static_assert(is_same<typename FragT::Traits::AccessT, typename Storer::BufferT>::value,
                      "Fragment access and store input types do not match");

        // Implicit unpack and then store
        Storer::exec(data, PreStore::exec(frag.mAccess), ldm);
    }

    template <typename MatrixT, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
    ROCWMMA_DEVICE void
        store_matrix_sync(DataT*                                                  data,
                          fragment<MatrixT, BlockM, BlockN, BlockK, DataT> const& frag,
                          uint32_t                                                ldm,
                          layout_t                                                layout)
    {
        using FragRowMajor = fragment<MatrixT, BlockM, BlockN, BlockK, DataT, row_major>;
        using FragColMajor = fragment<MatrixT, BlockM, BlockN, BlockK, DataT, col_major>;

        // Dispatch on layout type
        if(layout == layout_t::mem_row_major)
        {
            store_matrix_sync(data, reinterpret_cast<FragRowMajor const&>(frag), ldm);
        }
        else
        {
            store_matrix_sync(data, reinterpret_cast<FragColMajor const&>(frag), ldm);
        }
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputTA,
              typename InputTB,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD>
    ROCWMMA_DEVICE void
        mma_sync(fragment<accumulator, BlockM, BlockN, BlockK, ComputeT, LayoutD>&       d,
                 fragment<matrix_a, BlockM, BlockN, BlockK, InputTA, LayoutA> const&     a,
                 fragment<matrix_b, BlockM, BlockN, BlockK, InputTB, LayoutB> const&     b,
                 fragment<accumulator, BlockM, BlockN, BlockK, ComputeT, LayoutC> const& c)
    {
        using MmaConfig = MmaConfig<BlockM,
                                    BlockN,
                                    BlockK,
                                    InputTA,
                                    InputTB,
                                    ComputeT,
                                    LayoutA,
                                    LayoutB,
                                    LayoutC,
                                    LayoutD>;

        // Transforms
        using XA = typename MmaConfig::PreMmaXFormA;
        using XB = typename MmaConfig::PreMmaXFormB;
        using XC = typename MmaConfig::PreMmaXFormC;
        using XD = typename MmaConfig::PostMmaXFormD;

        // PackUtil
        using PackA = typename MmaConfig::PackA;
        using PackB = typename MmaConfig::PackB;
        using PackC = typename MmaConfig::PackC;
        using PackD = typename MmaConfig::PackD;

        using Mma = typename MmaConfig::Mma;

        // 1. Perform input pre-ops on A, B, Acc (unpacked mAccess)
        // 2. Mma (packed)
        // 3. Perform acc post-op on Acc
        // 4. Pack back to register
        d.mAccess = XD::exec(PackD::unpack(Mma::exec(PackA::pack(XA::exec(a.mAccess)),
                                                     PackB::pack(XB::exec(b.mAccess)),
                                                     PackC::pack(XC::exec(c.mAccess)))));
    }

    ROCWMMA_DEVICE void synchronize_workgroup()
    {
        __syncthreads();
    }

} // namespace rocwmma

#endif // ROCWMMA_API_IMPL_HPP
