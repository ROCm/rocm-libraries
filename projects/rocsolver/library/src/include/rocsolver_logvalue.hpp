/* **************************************************************************
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include <fmt/format.h>
#include <fmt/ostream.h>

#include "fmt_rocblas_types.hpp"
#include "rocsolver_datatype2string.hpp"

/***************************************************************************
 * Wrapper for types passed to logger, so we can more easily adjust the
 * default way of printing built-in types without doing it globally. (e.g.
 * changing the printing of bool to "0" or "1" rather than "true" "false".)
 ***************************************************************************/
template <typename T>
struct rocsolver_logvalue
{
    T value;
};

/***************************************************************************
 * Constructs a rocsolver_logvalue given a value.
 * Used so that T can be inferred from the argument.
 ***************************************************************************/
template <typename T>
rocsolver_logvalue<T> rocsolver_make_logvalue(T value)
{
    return rocsolver_logvalue<T>{value};
}

namespace fmt
{
/* By default, forward log values to the original printer for their type. */
template <typename T>
struct formatter<rocsolver_logvalue<T>> : formatter<T>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<T> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<T>::format(wrapper.value, ctx);
    }
};

/* Specialize bool to print 0 or 1 rather than true or false, to match the
   rocsolver-bench CLI.*/
template <>
struct formatter<rocsolver_logvalue<bool>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<bool> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(wrapper.value ? '1' : '0', ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_operation>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_operation> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocsolver::rocblas2char_operation(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_fill>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_fill> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocsolver::rocblas2char_fill(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_diagonal>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_diagonal> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocsolver::rocblas2char_diagonal(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_side>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_side> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocsolver::rocblas2char_side(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_direct>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_direct> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocsolver::rocblas2char_direct(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_pivot>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_pivot> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocsolver::rocblas2char_pivot(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_storev>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_storev> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocsolver::rocblas2char_storev(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_workmode>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_workmode> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocsolver::rocblas2char_workmode(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_svect>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_svect> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocsolver::rocblas2char_svect(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_srange>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_srange> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocsolver::rocblas2char_srange(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_evect>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_evect> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocsolver::rocblas2char_evect(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_eform>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_eform> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocsolver::rocblas2char_eform(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_erange>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_erange> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocsolver::rocblas2char_erange(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_eorder>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_eorder> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocsolver::rocblas2char_eorder(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_esort>> : formatter<char>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_esort> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<char>::format(rocsolver::rocblas2char_esort(wrapper.value), ctx);
    }
};

template <>
struct formatter<rocsolver_logvalue<rocblas_datatype>> : formatter<string_view>
{
    template <typename FormatCtx>
    auto format(rocsolver_logvalue<rocblas_datatype> wrapper, FormatCtx& ctx) ROCSOLVER_FMT_CONST
    {
        return formatter<string_view>::format(rocsolver::rocblas2string_datatype(wrapper.value), ctx);
    }
};

} // namespace
