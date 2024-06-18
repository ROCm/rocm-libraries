#include <rocRoller/DataTypes/DataTypes_BF8.hpp>
#include <rocRoller/DataTypes/DataTypes_F8_Utils.hpp>
#include <rocRoller/DataTypes/DataTypes_FP8.hpp>
#include <rocRoller/Utilities/Settings.hpp>

namespace rocRoller
{
    float fp8_to_float(const FP8 v)
    {
        auto f8Mode = Settings::getInstance()->get(Settings::F8ModeOption);
        switch(f8Mode)
        {
        case rocRoller::F8Mode::NaNoo:
            return DataTypes::
                cast_from_f8<3, 4, float, /*negative_zero_is_nan*/ true, /*has_infinity*/ false>(
                    v.data);
        case rocRoller::F8Mode::OCP:
            return DataTypes::
                cast_from_f8<3, 4, float, /*negative_zero_is_nan*/ false, /*has_infinity*/ false>(
                    v.data);
        default:
            Throw<FatalError>("Unexpected F8Mode.");
        }
    }

    FP8 float_to_fp8(const float v)
    {
        FP8  fp8;
        auto f8Mode = Settings::getInstance()->get(Settings::F8ModeOption);
        switch(f8Mode)
        {
        case rocRoller::F8Mode::NaNoo:
            fp8.data = DataTypes::cast_to_f8<3,
                                             4,
                                             float,
                                             /*is_ocp*/ false,
                                             /*negative_zero_is_nan*/ true,
                                             /*clip*/ true>(v, 0 /*stochastic*/, 0 /*rng*/);
            break;
        case rocRoller::F8Mode::OCP:
            fp8.data = DataTypes::cast_to_f8<3,
                                             4,
                                             float,
                                             /*is_ocp*/ true,
                                             /*negative_zero_is_nan*/ false,
                                             /*clip*/ true>(v, 0 /*stochastic*/, 0 /*rng*/);
            break;
        default:
            Throw<FatalError>("Unexpected F8Mode.");
        }
        return fp8;
    }

    float bf8_to_float(const BF8 v)
    {
        auto f8Mode = Settings::getInstance()->get(Settings::F8ModeOption);
        switch(f8Mode)
        {
        case rocRoller::F8Mode::NaNoo:
            return DataTypes::
                cast_from_f8<2, 5, float, /*negative_zero_is_nan*/ true, /*has_infinity*/ true>(
                    v.data);
        case rocRoller::F8Mode::OCP:
            return DataTypes::
                cast_from_f8<2, 5, float, /*negative_zero_is_nan*/ false, /*has_infinity*/ true>(
                    v.data);
        default:
            Throw<FatalError>("Unexpected F8Mode.");
        }
    }

    BF8 float_to_bf8(const float v)
    {
        BF8  bf8;
        auto f8Mode = Settings::getInstance()->get(Settings::F8ModeOption);
        switch(f8Mode)
        {
        case rocRoller::F8Mode::NaNoo:
            bf8.data = DataTypes::cast_to_f8<2,
                                             5,
                                             float,
                                             /*is_ocp*/ false,
                                             /*negative_zero_is_nan*/ true,
                                             /*clip*/ true>(v, 0 /*stochastic*/, 0 /*rng*/);
            break;
        case rocRoller::F8Mode::OCP:
            bf8.data = DataTypes::cast_to_f8<2,
                                             5,
                                             float,
                                             /*is_ocp*/ true,
                                             /*negative_zero_nan*/ false,
                                             /*clip*/ true>(v, 0 /*stochastic*/, 0 /*rng*/);
            break;
        default:
            Throw<FatalError>("Unexpected F8Mode.");
        }
        return bf8;
    }
}
