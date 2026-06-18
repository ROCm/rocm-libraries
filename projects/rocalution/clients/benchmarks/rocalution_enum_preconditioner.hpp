/* ************************************************************************
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once
#include "rocalution/rocalution.hpp"
#include "utility.hpp"

#include <cstring>

struct rocalution_enum_preconditioner
{

#define LIST_ROCALUTION_ENUM_PRECONDITIONER \
    ENUM_PRECONDITIONER(none)               \
    ENUM_PRECONDITIONER(chebyshev)          \
    ENUM_PRECONDITIONER(FSAI)               \
    ENUM_PRECONDITIONER(SPAI)               \
    ENUM_PRECONDITIONER(TNS)                \
    ENUM_PRECONDITIONER(Jacobi)             \
    ENUM_PRECONDITIONER(GS)                 \
    ENUM_PRECONDITIONER(SGS)                \
    ENUM_PRECONDITIONER(ILU)                \
    ENUM_PRECONDITIONER(ItILU0)             \
    ENUM_PRECONDITIONER(ILUT)               \
    ENUM_PRECONDITIONER(IC)                 \
    ENUM_PRECONDITIONER(MCGS)               \
    ENUM_PRECONDITIONER(MCSGS)              \
    ENUM_PRECONDITIONER(MCILU)

    //
    //
    //
#define ENUM_PRECONDITIONER(x_) x_,

    typedef enum rocalution_enum_preconditioner__ : int
    {
        LIST_ROCALUTION_ENUM_PRECONDITIONER
    } value_type;

    static constexpr value_type  all[]{LIST_ROCALUTION_ENUM_PRECONDITIONER};
    static constexpr std::size_t size = countof(all);

#undef ENUM_PRECONDITIONER

    //
    //
    //
#define ENUM_PRECONDITIONER(x_) #x_,

    static constexpr const char* names[size]{LIST_ROCALUTION_ENUM_PRECONDITIONER};

#undef ENUM_PRECONDITIONER

    bool is_invalid() const;
    rocalution_enum_preconditioner();
    rocalution_enum_preconditioner& operator()(const char* name_);
    rocalution_enum_preconditioner(const char* name_);

    value_type value{};
};

// constexpr const char * rocalution_enum_preconditioner::names[rocalution_enum_preconditioner::size];
// constexpr rocalution_enum_preconditioner::value_type rocalution_enum_preconditioner::all[rocalution_enum_preconditioner::size];
