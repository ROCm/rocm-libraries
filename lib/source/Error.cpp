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

#include <rocRoller/Utilities/Error.hpp>

#include <rocRoller/Utilities/Settings.hpp>

namespace rocRoller
{
    [[noreturn]] void Crash()
    {
        auto will_be_null = GetNullPointer();
        // cppcheck-suppress [nullPointer, knownConditionTrueFalse]
        if((*will_be_null = 0))
            throw std::runtime_error("Impossible 1");

        throw std::runtime_error("Impossible 2");
    }

    int* GetNullPointer()
    {
        return nullptr;
    }

    bool Error::BreakOnThrow()
    {
        return Settings::getInstance()->get(Settings::BreakOnThrow);
    }

    const char* Error::what() const noexcept
    {
        if(m_annotatedMessage.empty())
            return std::runtime_error::what();

        return m_annotatedMessage.c_str();
    }

    void Error::annotate(std::string const& msg)
    {
        if(m_annotatedMessage.empty())
            m_annotatedMessage = fmt::format("{}\n{}", std::runtime_error::what(), msg);
        else
        {
            m_annotatedMessage += "\n";
            m_annotatedMessage += msg;
        }
    }

}
