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

#include <common/SourceMatcher.hpp>

#include <catch2/matchers/catch_matchers.hpp>

#include <string>
#include <utility>

namespace rocRollerTest
{
    class MatchesSourceMatcher : public Catch::Matchers::MatcherBase<std::string>
    {
    public:
        explicit MatchesSourceMatcher(std::string ref, bool includeComments)
            : m_ref(std::move(ref))
            , m_includeComments(includeComments)
        {
        }

        bool match(std::string const& arg) const override
        {
            auto normalizedRef = NormalizedSource(m_ref, m_includeComments);
            auto normalizedArg = NormalizedSource(arg, m_includeComments);
            return normalizedRef == normalizedArg;
        }

        std::string describe() const override
        {
            return m_includeComments ? "matches source after normalization (including comments)"
                                     : "matches source after normalization";
        }

    private:
        std::string m_ref;
        bool        m_includeComments = false;
    };

    inline MatchesSourceMatcher MatchesSource(std::string ref)
    {
        return MatchesSourceMatcher(std::move(ref), /*includeComments=*/false);
    }

    inline MatchesSourceMatcher MatchesSourceIncludingComments(std::string ref)
    {
        return MatchesSourceMatcher(std::move(ref), /*includeComments=*/true);
    }
}