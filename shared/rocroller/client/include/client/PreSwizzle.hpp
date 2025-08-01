/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#pragma once

#include <vector>

#include <rocRoller/TensorDescriptor.hpp>

#include <client/GEMMParameters.hpp>

namespace rocRoller::Client
{

    inline std::tuple<TensorDescriptor, TensorDescriptor>
        view_64x4(TensorDescriptor desc, int dimN, int dimK)
    {
        // TensorDescriptor desc;

        // if(desc1.stride(1) > desc1.stride(0))
        // {
        //     desc = TensorDescriptor(desc1.dataType(), {desc1.size(1), desc1.size(0)});
        // }
        // else
        // {
        //     desc = desc1;
        // }

        AssertFatal(desc.dimensions() > 1, ShowValue(desc.dimensions()));
        AssertFatal(dimN < desc.dimensions(), ShowValue(dimN), ShowValue(desc.dimensions()));
        AssertFatal(dimK < desc.dimensions(), ShowValue(dimK), ShowValue(desc.dimensions()));
        AssertFatal(dimN < dimK, ShowValue(dimN), ShowValue(dimN));

        {
            size_t expectedStride = 1;
            for(int dim = 0; dim < desc.dimensions(); dim++)
            {
                AssertFatal(desc.stride(dim) == expectedStride,
                            ShowValue(desc.stride(dim)),
                            ShowValue(expectedStride),
                            ShowValue(desc.sizes()),
                            ShowValue(desc.strides()));
                expectedStride *= desc.size(dim);
            }
        }

        // def pre_shuffle_weight_16x16x128_B4(x_)
        //     x_ = x_.view(B, N//16, 16, K//128, 4, 32)
        //     x_ = x_.permute(0,1,3,4,2,5)
        //     x_ = x_.contiguous()
        //     # TODO: view x_
        //     return x_

        // auto sizes = desc.sizes();
        // AssertFatal(sizes[dimK] % 128 == 0, ShowValue(sizes[dimK]));
        // AssertFatal(sizes[dimN] % 16 == 0, ShowValue(sizes[dimK]));

        // auto afterK = std::next(sizes.begin(), dimK + 1);
        // sizes.insert(afterK, {16, 4});
        // sizes[dimK] /= 64;

        // auto afterN = std::next(sizes.begin(), dimN);
        // sizes.insert(afterN, 4);
        // sizes[dimN] /= 4;

        // std::vector order{dimK + 3, dimK + 1, dimN + 1, dimK + 2, dimN};
        // if(dimN == 1)
        //     order.push_back(0);

        int M = 1;
        int K = 0;

        // std::vector<size_t> sizes = {desc.size(M) / 16, 16, desc.size(K) / 64, 16, 4};
        // std::vector<int> order = {0, 2, 4, 3, 1};
        std::vector<size_t> sizes = {4, 16, desc.size(K) / 64, 4, desc.size(M) / 4};
        std::vector<int>    order = {4, 1, 2, 0, 5};

        TensorDescriptor dst, src;

        {
            AssertFatal(
                order.size() == sizes.size(), ShowValue(order.size()), ShowValue(sizes.size()));
            std::vector<size_t> strides(sizes.size(), 0);
            size_t              stride = 1;

            for(auto idx : std::ranges::reverse_view(order))
            {
                AssertFatal(strides.at(idx) == 0);
                strides[idx] = stride;
                stride *= sizes.at(idx);
            }

            dst = TensorDescriptor(desc.dataType(), sizes, std::move(strides));
        }

        {
            std::vector<size_t> strides(sizes.size(), 0);
            size_t              stride = 1;
            for(int idx = 0; idx < sizes.size(); idx++)
            {
                strides[idx] = stride;
                stride *= sizes[idx];
            }
            std::cerr << ShowValue(sizes) << ShowValue(strides);

            src = TensorDescriptor(desc.dataType(), sizes, std::move(strides));
        }

        return {dst, src};
    }

    //     template <typename T>
    //     inline std::vector<T> shuffleDims(std::vector<T> const&   input,
    //                                       TensorDescriptor const& dst,
    //                                       TensorDescriptor const& src)
    //     {
    //         AssertFatal(dst.dimensions() > 1, ShowValue(dst.dimensions()));
    //         AssertFatal(dst.sizes() == src.sizes(), ShowValue(dst.sizes()), ShowValue(src.sizes()));
    //         AssertFatal(dst.dataType() == src.dataType());
    //         // AssertFatal(TypeInfo<T>::Var == dst.dataType());

    //         auto const& sizes = dst.sizes();
    //         auto        count = CoordCount(sizes.begin(), std::prev(sizes.end()));

    //         std::vector<T> rv(input.size());

    // #pragma omp parallel for
    //         for(size_t coordNum = 0; coordNum < count; coordNum++)
    //         {
    //             std::vector<size_t> coord(dst.dimensions(), 0);
    //             CoordNumbered(coordNum,
    //                           coord.begin(),
    //                           std::prev(coord.end()),
    //                           sizes.begin(),
    //                           std::prev(sizes.end()));

    //             for(coord.back() = 0; coord.back() < sizes.back(); coord.back()++)
    //             {
    //                 auto dstIdx = dst.index(coord);
    //                 auto srcIdx = src.index(coord);

    //                 rv.at(dstIdx) = input.at(srcIdx);
    //             }
    //         }

    //         return rv;
    //     }

    template <typename T>
    inline std::vector<T> preSwizzle(std::vector<T> const&        input,
                                     TensorDescriptor const&      desc,
                                     std::array<size_t, 3> const& tile)
    {
        auto blockSize = 32;

        auto descTmp = desc.withNormalizedDimensions();
        descTmp = TensorDescriptor(desc.dataType(), {descTmp.size(0) / blockSize, descTmp.size(1)});

        AssertFatal(descTmp.totalAllocatedElements() == input.size(),
                    ShowValue(descTmp),
                    ShowValue(input.size()));

        // AssertFatal(problemParams.types.scaleShuffleTile.has_value());
        auto [tileM, tileK, subTileK] = tile;

        size_t instPerTileK  = tileK / subTileK;
        size_t instKPerTileM = tileM / subTileK;

        std::vector<size_t> srcSizes = {subTileK,
                                        instPerTileK,
                                        descTmp.size(0) / (tileK),
                                        instKPerTileM,
                                        subTileK,
                                        descTmp.size(1) / (tileM)};

        TensorDescriptor src(descTmp.dataType(), srcSizes);

        AssertFatal(src.totalAllocatedElements() == descTmp.totalAllocatedElements(),
                    ShowValue(src.totalAllocatedElements()),
                    ShowValue(descTmp.totalAllocatedElements()),
                    ShowValue(src.totalAllocatedElements() / descTmp.totalAllocatedElements()),
                    ShowValue(src),
                    ShowValue(descTmp));

#if 1
        auto dst
            = TensorDescriptor::ShuffledNoPadding(descTmp.dataType(), srcSizes, {4, 1, 2, 3, 0, 5});

        AssertFatal(src.totalAllocatedElements() == dst.totalAllocatedElements());

        {
            auto          tmp  = iota<int>(0, input.size()).template to<std::vector>();
            auto          tmp2 = shuffleDims(tmp, dst, src);
            std::ofstream file("tensor.txt");
            file << writeTensor(tmp2, descTmp);
        }

        return shuffleDims(input, dst, src);
#else
        return input;
#endif
    }

}
