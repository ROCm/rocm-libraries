/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc.
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
#include <cstddef>
#include <iostream>
#include <roc/host_validation/tensor.hpp>
#include <span>
#include <vector>

namespace
{
    template <typename T>
    void printTensor(std::ostream& os, const roc::host_validation::Tensor& tensor)
    {
        std::vector<size_t> indices(tensor.shape().rank(), 0);
        os << '[';

        auto printDimension = [&](auto&& self, size_t dimension) -> void {
            os << '[';
            for(size_t index = 0; index < tensor.shape()[dimension]; ++index)
            {
                indices[dimension] = index;
                if(dimension + 1 == tensor.shape().rank())
                {
                    os << static_cast<float>(tensor.loadAs<T>(std::span<const size_t>(indices)))
                       << ", ";
                }
                else
                {
                    self(self, dimension + 1);
                }
            }
            os << "], ";
            if(dimension + 1 == tensor.shape().rank())
                os << '\n';
        };

        printDimension(printDimension, 0);
        os << "]\n";
    }
}

int main(int argc, char** argv)
{
    constexpr size_t m{18};
    constexpr size_t k{34};
    std::vector<int> weightStorage(m * k);

    for(size_t i = 0; i < m; ++i)
    {
        for(size_t j = 0; j < k; ++j)
        {
            weightStorage[i * k + j] = i * k + j;
        }
    }

    using roc::host_validation::Layout;
    using roc::host_validation::Shape;
    using roc::host_validation::Tensor;
    const Tensor weight
        = Tensor::copyNativeStorage(Layout::contiguousLastDimensionFastest(Shape{m, k}), std::span<const int>(weightStorage));

    std::cout << "Original weight:\n";
    printTensor<int>(std::cout, weight);
    constexpr size_t MiM          = 16;
    constexpr size_t MiK          = 16;
    constexpr size_t MiKv         = 4;
    constexpr size_t PackK        = 2;
    constexpr auto   MultipleM    = MiM;
    constexpr auto   MultipleK    = MiK * PackK;
    const auto       paddedM      = (m / MultipleM + !!(m % MultipleM)) * MultipleM;
    const auto       paddedK      = (k / MultipleK + !!(k % MultipleK)) * MultipleK;
    const Tensor     paddedWeight = weight.copyWithZeroPadding(Shape{paddedM, paddedK});
    std::cout << "Padded weight:\n";
    printTensor<int>(std::cout, paddedWeight);
    const Tensor permuted
        = paddedWeight
              .reshapeSharingStorage(Shape{paddedM / MiM, MiM, paddedK / (MiK * PackK), MiK / MiKv, MiKv * PackK})
              .copyWithPermutedDimensions({0, 2, 3, 1, 4});
    std::cout << "Swizzle weight:\n";
    printTensor<int>(std::cout, permuted);
    return 0;
}
