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
#ifndef ROCWMMA_COOP_IO_CONFIG_HPP
#define ROCWMMA_COOP_IO_CONFIG_HPP

#include "coop_load.hpp"
#include "coop_store.hpp"
#include "io_layout.hpp"
#include "io_shape.hpp"
#include "io_traits.hpp"
#include "layout/register_layout_transforms.hpp"
#include "pack_util.hpp"
#include "types.hpp"

namespace rocwmma
{

    /**
     * \defgroup Rocwmma_ioconf ROCWMMA IOConfig
     * @brief ROCWMMA cooperative fragment input and output configurations
     * @{
     */

    /*! \struct CoopIOConfig
 *  \brief Definition of cooperative fragment input / output configurations
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
 * @param MappingUtil global mapping utility for current fragment
 * @param Loader Issues cooperative load instructions for raw fragment data
 * @param Storer Issues cooperative store instructions for raw fragment data
 */

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename DataLayoutT,
              uint32_t WaveCount>
    struct CoopIOConfig
    {
        using IOShape = IOShape<MatrixT, BlockM, BlockN, BlockK>;
        using IOLayout
            = IOLayout<MatrixT, IOShape::BlockDim, IOShape::KDim, DataT, DataLayoutT, WaveCount>;
        using IOTraits = IOTraits<IOShape::BlockDim, IOShape::KDim, DataT, IOLayout::VW>;

        using PackUtil = PackUtil<DataT>;
        using MappingUtil
            = MappingUtil<IOShape::BlockHeight, IOShape::BlockWidth, DataT, DataLayoutT>;

        using Loader = CooperativeLoad<typename IOLayout::DataLayout,
                                       typename IOLayout::MatrixLayout,
                                       WaveCount>;

        using PostLoadXForm = register_layout_transform<typename IOLayout::StorageLayout,
                                                        typename IOLayout::FragmentLayout>;

        using PreMmaXForm = register_layout_transform<typename IOLayout::FragmentLayout,
                                                      typename IOLayout::MmaLayout>;

        using PreStoreXForm = register_layout_transform<typename IOLayout::FragmentLayout,
                                                        typename IOLayout::StorageLayout>;

        using Storer = CooperativeStore<typename IOLayout::DataLayout,
                                        typename IOLayout::MatrixLayout,
                                        WaveCount>;
    };

    /************************************************
 * Matrix C/D (accumulator) with undetermined DataLayout
 *
 * Fewer specific indications for matrix data geometry I/O, however
 * general IOShape, PackUtil, Broadcast still available.
 *
 * */
    template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename DataT, uint32_t WaveCount>
    struct CoopIOConfig<accumulator, BlockM, BlockN, BlockK, DataT, void, WaveCount>
    {
        using IOShape  = IOShape<accumulator, BlockM, BlockN, BlockK>;
        using PackUtil = PackUtil<DataT>;
    };
    /** @}*/

} // namespace rocwmma

#endif // ROCWMMA_COOP_IO_CONFIG_HPP
