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
