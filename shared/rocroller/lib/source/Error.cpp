// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
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
