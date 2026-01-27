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

#pragma once

#include <cstdint>
#include <iostream>
#include <tuple>
#include <vector>

namespace DGen
{
    typedef uint64_t index_t;

    template <typename CartesianProductVectorType, typename... Ts>
    auto cartesian_product_helper(CartesianProductVectorType& vec, std::tuple<Ts...> t)
    {
        vec.emplace_back(t);
    }

    template <typename CartesianProductVectorType,
              typename... Ts,
              typename Iter,
              typename... TailIters>
    auto cartesian_product_helper(CartesianProductVectorType& vec,
                                  std::tuple<Ts...>           t,
                                  Iter                        b,
                                  Iter                        e,
                                  TailIters... tail_iters)
    {
        for(auto iter = b; iter != e; ++iter)
            cartesian_product_helper(vec, std::tuple_cat(t, std::make_tuple(*iter)), tail_iters...);
    }

    template <typename CartesianProductVectorType, typename... Iters>
    auto cartesian_product(CartesianProductVectorType& vec, Iters... iters)
    {
        cartesian_product_helper(vec, std::make_tuple(), iters...);
    }

    struct dimension_iterator
    {
        const std::vector<index_t> dimensions;
        explicit dimension_iterator(std::vector<index_t> const& dim)
            : dimensions(dim)
        {
        }

        struct iterator
        {
        private:
            const std::vector<index_t> dimensions;
            std::vector<index_t>       current;
            std::vector<index_t>*      curr_ptr;

        public:
            explicit iterator(std::vector<index_t> const& dim)
                : dimensions(dim)
                , current(dim.size(), 0)
                , curr_ptr(&current)
            {
            }

            iterator()
                : dimensions(0)
                , current(0)
                , curr_ptr(nullptr)
            {
            }

            void print()
            {
                for(const auto e : current)
                {
                    std::cout << e << " ";
                }
                std::cout << "\n";
            }

            const std::vector<index_t>& operator*() const
            {
                return *curr_ptr;
            }
            const std::vector<index_t>* operator->()
            {
                return curr_ptr;
            }

            // Prefix increment
            iterator& operator++()
            {
                for(size_t i = 0; i < dimensions.size(); ++i)
                {
                    current[i] += 1;
                    if(current[i] >= dimensions[i])
                    {
                        current[i] = 0;
                    }
                    else
                    {
                        return *this;
                    }
                }

                curr_ptr = nullptr;
                return *this;
            }

            // Postfix increment
            iterator operator++(int)
            {
                iterator tmp = *this;
                ++(*this);
                return tmp;
            }

            friend bool operator==(const iterator& a, const iterator& b)
            {
                if(a.curr_ptr == nullptr || b.curr_ptr == nullptr)
                {
                    return a.curr_ptr == b.curr_ptr;
                }
                return *a.curr_ptr == *b.curr_ptr;
            }
            friend bool operator!=(const iterator& a, const iterator& b)
            {
                return !(a == b);
            }
        };

        iterator begin() const
        {
            return iterator(dimensions);
        }
        iterator end() const
        {
            return iterator();
        }
    };

    inline index_t get_strided_idx(const std::vector<index_t>& indices, const std::vector<index_t>& stride)
    {
        index_t res = 0;
        for(size_t i = 0; i < indices.size(); i++)
        {
            res += indices[i] * stride[i];
        }

        return res;
    }

    inline void split_dynamic(uint64_t  input,
                              uint32_t  exp_size,
                              uint32_t  man_size,
                              uint8_t*  sign,
                              uint32_t* exponent,
                              uint64_t* mantissa)
    {
        if(sign)
            *sign = input >> (exp_size + man_size);
        if(exponent)
            *exponent = (input >> man_size) & ((1UL << exp_size) - 1UL);
        if(mantissa)
            *mantissa = input & ((1ULL << man_size) - 1ULL);
    }

    inline void split_double(double val, uint8_t* sign, uint32_t* exponent, uint64_t* mantissa)
    {
        union
        {
            double   input;
            uint64_t output;
        } bit_rep;

        bit_rep.input = val;

        split_dynamic(bit_rep.output, 11u, 52u, sign, exponent, mantissa);
    }
}
