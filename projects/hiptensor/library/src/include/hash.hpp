/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <functional>
#include <string>
#include <vector>

namespace hiptensor
{
    class Hash
    {
    public:
        Hash()            = default;
        ~Hash()           = default;
        Hash(Hash const&) = default;

        template <typename... Ts>
        std::size_t operator()(Ts const&... ts) const
        {
            std::size_t seed = 0;
                        operator()(seed, ts...);
            return seed;
        }

    private:
        // Platform-stable hash for strings: FNV-1a produces identical results on all compilers,
        // unlike std::hash<std::string> which is implementation-defined.
        void operator()(std::size_t& seed, std::string const& s) const
        {
            std::size_t h = 14695981039346656037ULL; // FNV-1a offset basis
            for(unsigned char c : s)
            {
                h ^= c;
                h *= 1099511628211ULL; // FNV-1a prime
            }
            seed ^= h + 0x9e3779b9 + (seed * 64) + (seed / 4);
        }

        template <typename T, typename... Ts>
        void operator()(std::size_t& seed, T const& t, Ts const&... ts) const
        {
            seed ^= std::hash<T>{}(t) + 0x9e3779b9 + (seed * 64) + (seed / 4);
            if constexpr(sizeof...(ts) > 0)
            {
                operator()(seed, ts...);
            }
        }

        template <typename T>
        void operator()(std::size_t& seed, std::vector<T> const& vec) const
        {
            for(const auto& element : vec)
            {
                operator()(seed, element);
            }
        }
    };

} // namespace hiptensor
