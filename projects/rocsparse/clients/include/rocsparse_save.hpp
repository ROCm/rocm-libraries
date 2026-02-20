/*! \file */
/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef ROCSPARSE_SAVE_HPP
#define ROCSPARSE_SAVE_HPP

#include "rocsparse_exporter_ascii.hpp"
#include "rocsparse_exporter_format_t.hpp"
#include "rocsparse_exporter_matrixmarket.hpp"
#include "rocsparse_exporter_rocalution.hpp"
#include "rocsparse_exporter_rocsparseio.hpp"

template <rocsparse_exporter_format_t::value_type EXPORTER_FORMAT>
struct rocsparse_exporter_format_traits_t;

template <>
struct rocsparse_exporter_format_traits_t<rocsparse_exporter_format_t::rocalution>
{
    using exporter_t = rocsparse_exporter_rocalution;
};

template <>
struct rocsparse_exporter_format_traits_t<rocsparse_exporter_format_t::rocsparseio>
{
    using exporter_t = rocsparse_exporter_rocsparseio;
};

template <>
struct rocsparse_exporter_format_traits_t<rocsparse_exporter_format_t::ascii>
{
    using exporter_t = rocsparse_exporter_ascii;
};

template <>
struct rocsparse_exporter_format_traits_t<rocsparse_exporter_format_t::matrixmarket>
{
    using exporter_t = rocsparse_exporter_matrixmarket;
};

template <rocsparse_exporter_format_t::value_type EXPORTER_FORMAT, typename T>
rocsparse_status rocsparse_save_template(const char* basename, const char* suffix, T obj)
{
    using exporter_t = typename rocsparse_exporter_format_traits_t<EXPORTER_FORMAT>::exporter_t;
    char filename[256];
    if(snprintf(filename, (size_t)256, "%s%s", basename, suffix) >= 256)
    {
        std::cerr << "rocsparse_save_template: truncated string. " << std::endl;
        return rocsparse_status_invalid_value;
    }

    exporter_t exporter(filename);
    exporter.write(obj);
    return rocsparse_status_success;
}

template <typename T, typename... P>
rocsparse_status rocsparse_save(const char* basename, const char* suffix, T obj, P... params);

template <rocsparse_exporter_format_t::value_type EXPORTER_FORMAT, typename T, typename... P>
rocsparse_status
    rocsparse_save_template(const char* basename, const char* suffix, T obj, P... params)
{
    rocsparse_status status = rocsparse_save_template<EXPORTER_FORMAT>(basename, suffix, obj);
    if(status != rocsparse_status_success)
    {
        return status;
    }

    //
    // Recall dispatch.
    //
    return rocsparse_save(basename, params...);
}

//
// @brief
//
template <typename T, typename... P>
rocsparse_status rocsparse_save(const char* basename, const char* suffix, T obj, P... params)
{
    rocsparse_exporter_format_t format;
    format(suffix);
    switch(format.value)
    {
    case rocsparse_exporter_format_t::unknown:
    {
        std::cerr << "unrecognized exporter file format in suffix '" << suffix << "'" << std::endl;
        return rocsparse_status_invalid_value;
    }
    case rocsparse_exporter_format_t::rocsparseio:
    {
        return rocsparse_save_template<rocsparse_exporter_format_t::rocsparseio, T, P...>(
            basename, suffix, obj, params...);
    }
    case rocsparse_exporter_format_t::rocalution:
    {
        return rocsparse_save_template<rocsparse_exporter_format_t::rocalution, T, P...>(
            basename, suffix, obj, params...);
    }
    case rocsparse_exporter_format_t::matrixmarket:
    {
        return rocsparse_save_template<rocsparse_exporter_format_t::matrixmarket, T, P...>(
            basename, suffix, obj, params...);
    }
    case rocsparse_exporter_format_t::ascii:
    {
        return rocsparse_save_template<rocsparse_exporter_format_t::ascii, T, P...>(
            basename, suffix, obj, params...);
    }
    }
    return rocsparse_status_invalid_value;
}

#endif
