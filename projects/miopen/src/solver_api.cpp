/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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
#include <miopen/errors.hpp>
#include <miopen/logger.hpp>
#include <miopen/miopen.h>
#include <miopen/solver_id.hpp>

#include <algorithm>
#include <cstring>
#include <string>

extern "C" {
miopenStatus_t miopenGetSolverName(uint64_t solverId, char* nameBuf, size_t nameBufLen)
{
    MIOPEN_LOG_FUNCTION(solverId);
    return miopen::try_([&] {
        if(nameBuf == nullptr || nameBufLen == 0)
            MIOPEN_THROW(miopenStatusBadParm);
        const auto id = miopen::solver::Id{solverId};
        if(!id.IsValid())
            MIOPEN_THROW(miopenStatusBadParm, "Unknown solver id: " + std::to_string(solverId));
        const auto name = id.ToString();
        if(name.size() + 1 > nameBufLen)
            MIOPEN_THROW(miopenStatusBadParm,
                         "Buffer too small: need " + std::to_string(name.size() + 1));
        std::copy_n(name.c_str(), name.size() + 1, nameBuf);
    });
}

miopenStatus_t miopenGetSolverIdByName(const char* solverName, uint64_t* solverId)
{
    MIOPEN_LOG_FUNCTION(solverName);
    return miopen::try_([&] {
        if(solverName == nullptr || solverId == nullptr)
            MIOPEN_THROW(miopenStatusBadParm);
        const auto id = miopen::solver::Id{solverName};
        *solverId     = id.Value();
    });
}
} // extern "C"
