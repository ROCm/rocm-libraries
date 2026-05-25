/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

#ifndef MIOPEN_GUARD_TEST_SERIALIZE_HPP
#define MIOPEN_GUARD_TEST_SERIALIZE_HPP

#include <miopen/each_args.hpp>
#include <miopen/tensor_holder.hpp>

#include <type_traits>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>

template <class T>
struct is_trivial_serializable : std::is_trivially_copy_constructible<T>
{
};

template <>
struct is_trivial_serializable<half_float::half> : std::true_type
{
};

template <class T>
std::enable_if_t<is_trivial_serializable<T>{}> serialize(std::ostream& os, const T& x)
{
    os.write(reinterpret_cast<const char*>(&x), sizeof(T));
}

template <class T>
auto serialize(std::ostream& os,
               const T& x) -> decltype(x.begin(), x.end(), T(x.begin(), x.end()), void())
{
    std::size_t n = std::distance(x.begin(), x.end());
    serialize(os, n);
    for(auto&& y : x)
        serialize(os, y);
}

template <class... Ts>
std::enable_if_t<not is_trivial_serializable<std::tuple<Ts...>>{}>
serialize(std::ostream& os, const std::tuple<Ts...>& t)
{
    miopen::unpack(
        [&](auto&&... xs) { miopen::each_args([&](auto&& x) { serialize(os, x); }, xs...); }, t);
}

template <class T>
std::enable_if_t<is_trivial_serializable<T>{}> serialize(std::istream& is, T& x)
{
    is.read(reinterpret_cast<char*>(&x), sizeof(T));
}

template <class T>
std::enable_if_t<is_trivial_serializable<T>{}> serialize(std::istream& is, std::vector<T>& x)
{
    std::size_t n;
    serialize(is, n);
    x.resize(n);
    is.read(reinterpret_cast<char*>(x.data()), sizeof(T) * n);
}

template <class T>
auto serialize(std::istream& is,
               T& x) -> decltype(x.begin(), x.end(), x.assign(x.begin(), x.end()), void())
{
    using value_type = std::decay_t<decltype(*x.begin())>;
    std::size_t n;
    serialize(is, n);
    std::vector<value_type> v;
    v.reserve(n);
    for(std::size_t i = 0; i < n; i++)
    {
        value_type y;
        serialize(is, y);
        v.push_back(y);
    }
    x.assign(v.begin(), v.end());
}

template <class... Ts>
std::enable_if_t<not is_trivial_serializable<std::tuple<Ts...>>{}>
serialize(std::istream& is,
          // cppcheck-suppress constParameter
          std::tuple<Ts...>& t)
{
    miopen::unpack(
        [&](auto&&... xs) { miopen::each_args([&](auto&& x) { serialize(is, x); }, xs...); }, t);
}

template <class T>
void serialize(std::istream& is, tensor<T>& x)
{
    std::vector<std::size_t> lens;
    serialize(is, lens);
    std::vector<std::size_t> strides;
    serialize(is, strides);
    x.desc = miopen::TensorDescriptor{miopen_type<T>{}, lens, strides};
    serialize(is, x.data);
}

template <class T>
void serialize(std::ostream& os, const tensor<T>& x)
{
    const auto& lens = x.desc.GetLengths();
    serialize(os, lens);
    const auto& strides = x.desc.GetStrides();
    serialize(os, strides);
    serialize(os, x.data);
}

#endif
