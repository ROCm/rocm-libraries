#include "GPUContextFixture.hpp"
#include <common/mxDataGen.hpp>

using namespace rocRoller;
using namespace DGen;

namespace rocRollerTest
{
    class DataGenerationTest : public GPUContextFixtureParam<rocRoller::DataType>
    {
    };

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
            EXPECT_EQ(toFloatPacked<DGenDT>(&scale[0], &byteData[0], i, i), ref[i]);
        }
    }

    TEST_P(DataGenerationTest, DataGeneratorValueTest)
    {
        auto      dType = std::get<rocRoller::DataType>(GetParam());
        const int dim1  = 32;
        const int dim2  = 32;

        if(dType == rocRoller::DataType::FP8)
        {
            exeDataGeneratorTest<FP8>(dim1, dim2);
        }
        else if(dType == rocRoller::DataType::BF8)
        {
            exeDataGeneratorTest<BF8>(dim1, dim2);
        }
        else if(dType == rocRoller::DataType::Half)
        {
            exeDataGeneratorTest<Half>(dim1, dim2);
        }
        else if(dType == rocRoller::DataType::BFloat16)
        {
            exeDataGeneratorTest<BFloat16>(dim1, dim2);
        }
        else if(dType == rocRoller::DataType::Float)
        {
            exeDataGeneratorTest<float>(dim1, dim2);
        }
        else
        {
            Throw<FatalError>("Invalid rocRoller data type for data generator test.");
        }
    }

    INSTANTIATE_TEST_SUITE_P(DataGenerationTest,
                             DataGenerationTest,
                             ::testing::Combine(currentGPUISA(),
                                                ::testing::Values(rocRoller::DataType::FP8,
                                                                  rocRoller::DataType::BF8,
                                                                  rocRoller::DataType::Half,
                                                                  rocRoller::DataType::BFloat16,
                                                                  rocRoller::DataType::Float)));
}
