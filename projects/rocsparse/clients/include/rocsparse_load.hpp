/*! \file */
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
#ifndef ROCSPARSE_LOAD_HPP
#define ROCSPARSE_LOAD_HPP

#include "rocsparse_importer_format_t.hpp"
#include "rocsparse_importer_matrixmarket.hpp"
#include "rocsparse_importer_mlbsr.hpp"
#include "rocsparse_importer_mlcsr.hpp"
#include "rocsparse_importer_rocalution.hpp"
#include "rocsparse_importer_rocsparseio.hpp"

template <rocsparse_importer_format_t::value_type IMPORTER_FORMAT>
struct rocsparse_importer_format_traits_t;

template <>
struct rocsparse_importer_format_traits_t<rocsparse_importer_format_t::rocalution>
{
    using importer_t = rocsparse_importer_rocalution;
};

template <>
struct rocsparse_importer_format_traits_t<rocsparse_importer_format_t::rocsparseio>
{
    using importer_t = rocsparse_importer_rocsparseio;
};

template <>
struct rocsparse_importer_format_traits_t<rocsparse_importer_format_t::matrixmarket>
{
    using importer_t = rocsparse_importer_matrixmarket;
};

template <>
struct rocsparse_importer_format_traits_t<rocsparse_importer_format_t::mlbsr>
{
    using importer_t = rocsparse_importer_mlbsr;
};

template <>
struct rocsparse_importer_format_traits_t<rocsparse_importer_format_t::mlcsr>
{
    using importer_t = rocsparse_importer_mlcsr;
};

template <rocsparse_importer_format_t::value_type IMPORTER_FORMAT, typename T>
rocsparse_status rocsparse_load_template(const char* basename, const char* suffix, T& obj)
{
    using importer_t = typename rocsparse_importer_format_traits_t<IMPORTER_FORMAT>::importer_t;
    char filename[256];
    if(snprintf(filename, (size_t)256, "%s%s", basename, suffix) >= 256)
    {
        std::cerr << "rocsparse_load_template: truncated string. " << std::endl;
        return rocsparse_status_invalid_value;
    }

    importer_t importer(filename);
    importer.import(obj);
    return rocsparse_status_success;
}

template <typename T, typename... P>
rocsparse_status rocsparse_load(const char* basename, const char* suffix, T& obj, P... params);

template <rocsparse_importer_format_t::value_type IMPORTER_FORMAT, typename T, typename... P>
rocsparse_status
    rocsparse_load_template(const char* basename, const char* suffix, T obj, P... params)
{
    rocsparse_status status = rocsparse_load_template<IMPORTER_FORMAT>(basename, suffix, obj);
    if(status != rocsparse_status_success)
    {
        return status;
    }

    //
    // Recall dispatch.
    //
    return rocsparse_load(basename, params...);
}

//
// @brief
//
template <typename T, typename... P>
rocsparse_status rocsparse_load(const char* basename, const char* suffix, T& obj, P... params)
{
    rocsparse_importer_format_t format;
    format(suffix);
    switch(format.value)
    {
    case rocsparse_importer_format_t::unknown:
    {
        std::cerr << "unrecognized importer file format in suffix '" << suffix << "'" << std::endl;
        return rocsparse_status_invalid_value;
    }
    case rocsparse_importer_format_t::rocsparseio:
    {
        return rocsparse_load_template<rocsparse_importer_format_t::rocsparseio, T, P...>(
            basename, suffix, obj, params...);
    }
    case rocsparse_importer_format_t::rocalution:
    {
        return rocsparse_load_template<rocsparse_importer_format_t::rocalution, T, P...>(
            basename, suffix, obj, params...);
    }
    case rocsparse_importer_format_t::matrixmarket:
    {
        return rocsparse_load_template<rocsparse_importer_format_t::matrixmarket, T, P...>(
            basename, suffix, obj, params...);
    }
    case rocsparse_importer_format_t::mlbsr:
    {
        return rocsparse_load_template<rocsparse_importer_format_t::mlbsr, T, P...>(
            basename, suffix, obj, params...);
    }
    case rocsparse_importer_format_t::mlcsr:
    {
        return rocsparse_load_template<rocsparse_importer_format_t::mlcsr, T, P...>(
            basename, suffix, obj, params...);
    }
    }
    return rocsparse_status_invalid_value;
}

#endif
