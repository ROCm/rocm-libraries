/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2026 AMD ROCm(TM) Software
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

#include <rocRoller/DataTypes/DataTypes.hpp>

#include <string>

namespace rocRoller
{
    namespace Parameters
    {
        namespace Solution
        {
            enum class StorePath : int
            {
                VGPRToGlobalMemoryWithBuffer, // Store from VGPR to buffer using buffer_store_X
                VGPRToGlobalMemoryWithGlobal, // Store from VGPR to global using global_store_X
                VGPRToGlobalMemoryViaLDSWithBuffer, // Store to LDS first, then to buffer (former storeLDSD=true)
                VGPRToGlobalMemoryViaLDSWithGlobal, // Store to LDS first, then to global
                Count,
            };

            std::string   toString(StorePath path);
            std::ostream& operator<<(std::ostream& stream, StorePath const& path);
            std::istream& operator>>(std::istream& stream, StorePath& path);

            MemoryType GetMemoryType(StorePath const& path);
            bool       IsLDSStore(StorePath const& path);
        } // namespace Solution
    } // namespace Parameters
} // namespace rocRoller
