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
#ifndef ROCWMMA_IO_CONFIG_HPP
#define ROCWMMA_IO_CONFIG_HPP

#include "./layout/register_layout_transforms.hpp"
#include "broadcast.hpp"
#include "coop_load.hpp"
#include "coop_store.hpp"
#include "io_shape.hpp"
#include "opaque_load.hpp"
#include "opaque_store.hpp"
#include "pack_util.hpp"
#include "types.hpp"

namespace rocwmma
{
    /**
     * \defgroup Rocwmma_ioconf ROCWMMA IOConfig
     * @brief ROCWMMA fragment input and output configurations
     * @{
     */

    /*! \struct IOConfig
  *  \brief Definition of fragment input / output configurations
 *         in specific matrix context.
 *
 * @tparam Matrix fragment context
 * @tparam BlockM/N/K block dimensions
 * @tparam DataT data type
 * @tparam DataLayoutT in-memory layout as col_major or row_major
 * @param IOShape dimensional properties of the fragment
 * @param IOLayout 1d and 2d layouts of the fragment
 * @param IOTraits meta-properties for input and output of the fragment
 * @param PackUtil utility for packing / unpacking fragment data
 * @param Broadcaster utility for assigning a single value to entire fragment
 * @param MappingUtil global mapping utility for current fragment
 * @param Loader Issues load instructions for raw fragment data
 * @param Storer Issues store instructions for raw fragment data
 */

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT>
    struct IOConfig
    {
        using IOShape = IOShape<MatrixT, BlockM, BlockN, BlockK>;
        using IOLayout
            = IOLayout<MatrixT, IOShape::BlockDim, IOShape::KDim, DataT, DataLayoutT, 1u>;
        using IOTraits = IOTraits<IOShape::BlockDim, IOShape::KDim, DataT, IOLayout::VW>;

        using PackUtil    = PackUtil<DataT>;
        using Broadcaster = Broadcast<DataT, IOTraits::UnpackedSize>;

        using MappingUtil
            = MappingUtil<IOShape::BlockHeight, IOShape::BlockWidth, DataT, DataLayoutT>;

        using Loader = OpaqueLoad<typename IOLayout::DataLayout, typename IOLayout::MatrixLayout>;

        using PostLoadXForm = register_layout_transform<typename IOLayout::StorageLayout,
                                                        typename IOLayout::FragmentLayout>;

        using PreStoreXForm = register_layout_transform<typename IOLayout::FragmentLayout,
                                                        typename IOLayout::StorageLayout>;

        using Storer = OpaqueStore<typename IOLayout::DataLayout, typename IOLayout::MatrixLayout>;
    };

    /************************************************
 * Matrix C/D (accumulator) with undetermined DataLayout
 *
 * Fewer specific indications for matrix data geometry I/O, however
 * general IOTraits, Pack/Unpack, Broadcast still available.
 *
 * */
    template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT>
    struct IOConfig<accumulator, BlockM, BlockN, BlockK, DataT, void>
    {
        using IOShape  = IOShape<accumulator, BlockM, BlockN, BlockK>;
        using IOLayout = IOLayout<accumulator, IOShape::BlockDim, IOShape::KDim, DataT, void, 1u>;
        using IOTraits = IOTraits<IOShape::BlockDim, IOShape::KDim, DataT>;
        using PackUtil = PackUtil<DataT>;
        using Broadcaster = Broadcast<DataT, IOTraits::UnpackedSize>;

        using PreMmaXForm = register_layout_transform<typename IOLayout::FragmentLayout,
                                                      typename IOLayout::MmaLayout>;

        using PostMmaXForm = register_layout_transform<typename IOLayout::MmaLayout,
                                                       typename IOLayout::FragmentLayout>;
    };
    /** @}*/

} // namespace rocwmma

#endif // ROCWMMA_IO_CONFIG_HPP
