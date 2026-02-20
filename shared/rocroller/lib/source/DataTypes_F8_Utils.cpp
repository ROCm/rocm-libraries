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
                                             /*is_bf8*/ false,
                                             /*negative_zero_is_nan*/ true,
                                             /*clip*/ true>(v, 0 /*stochastic*/, 0 /*rng*/);
            break;
        case rocRoller::F8Mode::OCP:
            fp8.data = DataTypes::cast_to_f8<3,
                                             4,
                                             float,
                                             /*is_ocp*/ true,
                                             /*is_bf8*/ false,
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
                cast_from_f8<2, 5, float, /*negative_zero_is_nan*/ true, /*has_infinity*/ false>(
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
                                             /*is_bf8*/ true,
                                             /*negative_zero_is_nan*/ true,
                                             /*clip*/ true>(v, 0 /*stochastic*/, 0 /*rng*/);
            break;
        case rocRoller::F8Mode::OCP:
            bf8.data = DataTypes::cast_to_f8<2,
                                             5,
                                             float,
                                             /*is_ocp*/ true,
                                             /*is_bf8*/ true,
                                             /*negative_zero_nan*/ false,
                                             /*clip*/ true>(v, 0 /*stochastic*/, 0 /*rng*/);
            break;
        default:
            Throw<FatalError>("Unexpected F8Mode.");
        }
        return bf8;
    }
}
