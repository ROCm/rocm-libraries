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
        void exeDataGeneratorTest(const int         dim1,
                                  const int         dim2,
                                  const float       min          = -1.f,
                                  const float       max          = 1.f,
                                  const int         blockScaling = 1,
                                  const DataPattern pattern      = DataPattern::Bounded)
        {
            using DGenDT = typename rrDT2DGenDT<rrDT>::type;

            uint32_t seed = 1713573849u;

            auto dgen = getDataGenerator<rrDT>(dim1, dim2, min, max, seed, blockScaling, pattern);

            auto byteData = dgen.getDataBytes();
            auto scale    = dgen.getScaleBytes();
            auto ref      = dgen.getReferenceFloat();

            for(int i = 0; i < ref.size(); i++)
            {
                CHECK(toFloatPacked<DGenDT>(&scale[0], &byteData[0], i, i) == ref[i]);
            }
        }
    };

    TEMPLATE_TEST_CASE("Use mxDataGenerator", "[mxDataGenerator]", FP8, BF8, Half, BFloat16, float)
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
