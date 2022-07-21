#pragma once

#include "Register_fwd.hpp"

namespace rocRoller
{
    class LabelAllocator
    {
    public:
        LabelAllocator(std::string const& prefix);

        Register::ValuePtr label(std::string const& name);

    private:
        std::string m_prefix;
        int         m_count = 0;
    };
}

#include "LabelAllocator_impl.hpp"
