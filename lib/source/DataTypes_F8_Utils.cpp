#include <rocRoller/DataTypes/DataTypes_BF8.hpp>
#include <rocRoller/DataTypes/DataTypes_F8_Utils.hpp>
#include <rocRoller/DataTypes/DataTypes_FP8.hpp>

namespace rocRoller
{
    float fp8_to_float(const FP8 v)
    {
        return DataTypes::cast_from_f8<3, 4, float, true /*negative_zero_nan*/>(v.data);
    }

    FP8 float_to_fp8(const float v)
    {
        FP8 fp8;
        fp8.data = DataTypes::cast_to_f8<3, 4, float, true /*negative_zero_nan*/, true /*clip*/>(
            v, 0 /*stochastic*/, 0 /*rng*/);
        return fp8;
    }

    float bf8_to_float(const BF8 v)
    {
        return DataTypes::cast_from_f8<2, 5, float, true /*negative_zero_nan*/>(v.data);
    }

    BF8 float_to_bf8(const float v)
    {
        BF8 bf8;
        bf8.data = DataTypes::cast_to_f8<2, 5, float, true /*negative_zero_nan*/, true /*clip*/>(
            v, 0 /*stochastic*/, 0 /*rng*/);
        return bf8;
    }
}
