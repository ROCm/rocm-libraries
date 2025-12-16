/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include <istream>
#include <string>

#include <amd_comgr/amd_comgr.h>

namespace rocRoller
{
    namespace Serialization
    {
        /**
         * Note that the functions declared here will assume that you have included the deserialization headers for any
         * type(s) you want to deserialize.
         */

        /**
         * Parses ELF from a filepath into a T.
         */
        template <typename T>
        T fromELFFile(std::string const& filename);

        /**
         * Parses ELF from a file into a T.
         */
        template <typename T>
        T fromELF(std::string const& filename);
    }
}

#include <rocRoller/Serialization/ELF_impl.hpp>
