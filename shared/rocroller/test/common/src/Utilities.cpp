// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Test suite utilities.
 */

#include "common/Utilities.hpp"

int countSubstring(const std::string& str, const std::string& sub)
{
    if(sub.length() == 0)
        return 0;
    int count = 0;
    for(size_t offset = str.find(sub); offset != std::string::npos;
        offset        = str.find(sub, offset + sub.length()))
    {
        ++count;
    }
    return count;
}

std::shared_ptr<void> make_shared_device(rocRoller::CommandArgumentValue const& arg)
{
    auto visitor = [](auto const& arg) -> std::shared_ptr<void> {
        using T = std::decay_t<decltype(arg)>;

        auto rv = make_shared_device<int8_t>(sizeof(T));

        auto result = hipMemcpy(rv.get(), &arg, sizeof(T), hipMemcpyHostToDevice);
        if(result != hipSuccess)
        {
            throw std::runtime_error(hipGetErrorString(result));
        }
        return rv;
    };
    return std::visit(visitor, arg);
}
