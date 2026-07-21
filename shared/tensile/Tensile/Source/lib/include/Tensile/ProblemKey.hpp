/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/Debug.hpp>
#include <Tensile/Properties.hpp>

namespace Tensile
{

    namespace ProblemKey
    {
        /**
         * This exists to provide an abstraction around the different syntax of creating
         * a vector of a size given at runtime vs. creating an array with a fixed size.
         */
        template <typename Key>
        struct KeyFactory
        {
        };

        template <typename T>
        struct KeyFactory<std::vector<T>>
        {
            static std::vector<T> MakeKey(size_t size)
            {
                return std::vector<T>(size);
            }
        };

        template <typename T, size_t N>
        struct KeyFactory<std::array<T, N>>
        {
            static std::array<T, N> MakeKey(size_t size)
            {
                return std::array<T, N>();
            }
        };

        template <typename Key, typename Problem, typename Value = size_t>
        Key keyForProblem(Problem const&                                                problem,
                          std::vector<std::shared_ptr<Property<Problem, Value>>> const& properties)
        {
            bool debug = Debug::Instance().printPropertyEvaluation();

            Key myKey = ProblemKey::KeyFactory<Key>::MakeKey(properties.size());

            for(int i = 0; i < properties.size(); i++)
                myKey[i] = (*properties[i])(problem);

            if(debug)
            {
                std::cout << "Object key: ";
                streamJoin(std::cout, myKey, ", ");
                std::cout << std::endl;
            }

            return myKey;
        }
    }
}
