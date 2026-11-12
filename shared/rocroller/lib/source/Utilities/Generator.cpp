// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <exception>
#include <utility>

#include <rocRoller/Utilities/Generator.hpp>

namespace rocRoller::GeneratorDetail
{
    void RethrowException(std::exception_ptr& exception)
    {
        std::exception_ptr exc = nullptr;
        std::swap(exc, exception);
        std::rethrow_exception(exc);
    }
}
