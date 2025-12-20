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

#include <common/Utilities.hpp>

#include <catch2/matchers/catch_matchers.hpp>

#include <hip/hip_runtime.h>
#include <string>

namespace rocRollerTest
{
    class HasHipSuccessMatcher : public Catch::Matchers::MatcherBase<hipError_t>
    {
    public:
        bool match(hipError_t const& result) const override
        {
            m_last = result;
            return result == hipSuccess;
        }

        std::string describe() const override
        {
            if(m_last == hipSuccess)
                return "HIP call returns hipSuccess";

            return std::string("HIP call returns hipSuccess (got: ") + hipGetErrorString(m_last)
                   + ")";
        }

    private:
        mutable hipError_t m_last = hipSuccess;
    };

    inline HasHipSuccessMatcher HasHipSuccess()
    {
        return HasHipSuccessMatcher{};
    }
}
