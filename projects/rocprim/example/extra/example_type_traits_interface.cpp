// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#include <rocprim/type_traits.hpp>

#include <iostream>
#include <sstream>
#include <string>

template<class InputType,
         std::enable_if_t<rocprim::traits::get<InputType>().is_floating_point()>* = nullptr>
void some_algo()
{
    constexpr auto input_traits = rocprim::traits::get<InputType>();

    std::stringstream ss;

    ss << "root_type:\t[" << typeid(InputType).name() << "] this type is floating point\n";
    ss << "is_arithmetic:\t" << input_traits.is_arithmetic() << '\n';
    ss << "is_fundamental:\t" << input_traits.is_fundamental() << '\n';
    ss << "is_scalar:\t" << input_traits.is_scalar() << '\n';
    constexpr auto mask = input_traits.float_bit_mask();
    ss << "sign_bit:\t" << mask.sign_bit << '\n';
    ss << "exponent:\t" << mask.exponent << '\n';
    ss << "mantissa:\t" << mask.mantissa << '\n';
    std::cout << ss.str() << "\n";
}

template<class InputType,
         std::enable_if_t<rocprim::traits::get<InputType>().is_integral()>* = nullptr>
void some_algo()
{
    constexpr auto input_traits = rocprim::traits::get<InputType>();

    std::stringstream ss;
    ss << "root_type:\t[" << typeid(InputType).name() << "] this type is integral\n";
    ss << "is_arithmetic:\t" << input_traits.is_arithmetic() << '\n';
    ss << "is_fundamental:\t" << input_traits.is_fundamental() << '\n';
    ss << "is_scalar:\t" << input_traits.is_scalar() << '\n';
    ss << "is_signed:\t" << input_traits.is_signed() << '\n';
    ss << "is_unsigned:\t" << input_traits.is_unsigned() << '\n';

    std::cout << ss.str() << "\n";
}

template<class InputType,
         std::enable_if_t<!rocprim::traits::get<InputType>().is_integral()
                          && !rocprim::traits::get<InputType>().is_floating_point()>* = nullptr>
void some_algo()
{
    constexpr auto    input_traits = rocprim::traits::get<InputType>();
    std::stringstream ss;
    ss << "root_type:\t[" << typeid(InputType).name()
       << "] this type is neither integral nor floating point\n";
    ss << "is_arithmetic:\t" << input_traits.is_arithmetic() << '\n';
    ss << "is_fundamental:\t" << input_traits.is_fundamental() << '\n';
    ss << "is_signed:\t" << input_traits.is_signed() << '\n';
    ss << "is_unsigned:\t" << input_traits.is_unsigned() << '\n';
    ss << "is_scalar:\t" << input_traits.is_scalar() << '\n';

    std::cout << ss.str() << "\n";
}

// Your type definition
struct custom_float_type
{};

// you should add this struct specialization for the type to implement the traits
template<>
struct rocprim::traits::define<custom_float_type>
{
    using is_arithmetic = rocprim::traits::is_arithmetic::values<true>;
    using number_format
        = rocprim::traits::number_format::values<number_format::kind::floating_point_type>;
    using float_bit_mask = rocprim::traits::float_bit_mask::values<uint32_t, 10, 10, 10>;
};

// Your type definition
struct custom_int_type
{};

// you should add this struct specialization for the type to implement the traits
template<>
struct rocprim::traits::define<custom_int_type>
{
    using is_arithmetic = rocprim::traits::is_arithmetic::values<true>;
    using number_format
        = rocprim::traits::number_format::values<number_format::kind::integral_type>;
    using integral_sign
        = rocprim::traits::integral_sign::values<traits::integral_sign::kind::signed_type>;
};

int main()
{
    // C++ types
    some_algo<int>();
    some_algo<float>();
    some_algo<double>();

    // rocprim type
    some_algo<rocprim::bfloat16>();
    some_algo<rocprim::half>();
    some_algo<rocprim::int128_t>();
    some_algo<rocprim::uint128_t>();

    // other types
    some_algo<custom_float_type>();
    some_algo<custom_int_type>();

    return 0;
}
