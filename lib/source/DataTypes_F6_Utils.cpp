#include <rocRoller/DataTypes/DataTypes_BF6.hpp>
#include <rocRoller/DataTypes/DataTypes_F6_Utils.hpp>
#include <rocRoller/DataTypes/DataTypes_FP6.hpp>

namespace rocRoller
{
    float fp6_to_float(const FP6 v)
    {
        return cast_from_f6<float>(v.data, DataTypes::FP6_FMT);
    }

    FP6 float_to_fp6(const float v)
    {
        FP6 fp6;
        fp6.data = cast_to_f6<float>(v, DataTypes::FP6_FMT);
        return fp6;
    }

    float bf6_to_float(const BF6 v)
    {
        return cast_from_f6<float>(v.data, DataTypes::BF6_FMT);
    }

    BF6 float_to_bf6(const float v)
    {
        BF6 bf6;
        bf6.data = cast_to_f6<float>(v, DataTypes::BF6_FMT);
        return bf6;
    }

};
