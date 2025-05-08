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

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "CustomSections.hpp"
#include "SimpleTest.hpp"

#include <common/mxDataGen.hpp>

using namespace rocRoller;
using namespace DGen;
using namespace Catch::Matchers;

namespace mxDataGeneratorTest
{
    class mxDataGeneratorTest : public SimpleTest
    {
    public:
        mxDataGeneratorTest() = default;

        template <typename rrDT>
        void exeDataGeneratorTest(unsigned    dim1,
                                  unsigned    dim2,
                                  const float min          = -1.f,
                                  const float max          = 1.f,
                                  const int   blockScaling = 32)
        {
            using DGenDT = typename rrDT2DGenDT<rrDT>::type;

            auto             dataType = TypeInfo<rrDT>::Var.dataType;
            TensorDescriptor desc(dataType, {dim1, dim2}, "T");

            uint32_t seed = 9861u;

            const auto dgen = getDataGenerator<rrDT>(desc, min, max, seed, blockScaling);

            auto byteData = dgen.getDataBytes();
            auto scale    = dgen.getScaleBytes();
            auto ref      = dgen.getReferenceFloat();

            auto rrVector1 = DGenVector<rrDT>(desc, min, max, seed, true, 32);
            auto rrVector2 = getRandomVector<rrDT>(dgen, true);

            std::vector<uint8_t>& byteData1 = reinterpret_cast<std::vector<uint8_t>&>(rrVector1);
            std::vector<uint8_t>& byteData2 = reinterpret_cast<std::vector<uint8_t>&>(rrVector2);

            for(int i = 0; i < ref.size(); i++)
            {
                int scale_i = i / blockScaling;
                CHECK(toFloatPacked<DGenDT>(&scale[0], &byteData[0], scale_i, i) == ref[i]);
                CHECK(toFloatPacked<DGenDT>(&scale[0], &byteData1[0], scale_i, i) == ref[i]);
                CHECK(toFloatPacked<DGenDT>(&scale[0], &byteData2[0], scale_i, i) == ref[i]);
            }
        }
    };

    TEMPLATE_TEST_CASE(
        "Use mxDataGenerator", "[mxDataGenerator]", FP4, FP6, BF6, FP8, BF8, Half, BFloat16, float)
    {
        mxDataGeneratorTest t;

        SUPPORTED_ARCH_SECTION(arch)
        {
            const int dim1 = 32;
            const int dim2 = 32;

            t.exeDataGeneratorTest<TestType>(dim1, dim2);
        }
    }
}
