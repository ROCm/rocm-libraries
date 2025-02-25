#include <rocRoller/DataTypes/DataTypes_BF16_Utils.hpp>
#include <rocRoller/DataTypes/DataTypes_BFloat16.hpp>

namespace rocRoller
{
    float bf16_to_float(const BFloat16 v)
    {
        return DataTypes::cast_from_bf16<float>(v.data);
    }

    BFloat16 float_to_bf16(const float v)
    {
        BFloat16 bf16;
        bf16.data = DataTypes::cast_to_bf16<float>(v);
        return bf16;
    }
}
