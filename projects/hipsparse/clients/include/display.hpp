/*! \file */
/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
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

/*! \file
 *  \brief display.hpp provides common testing utilities.
 */

#pragma once
#ifndef DISPLAY_HPP
#define DISPLAY_HPP

#include <fstream>
#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include <hipsparse.h>

static constexpr const char* s_timing_info_perf          = "GFlop/s";
static constexpr const char* s_timing_info_bandwidth     = "GB/s";
static constexpr const char* s_timing_info_time          = "msec";
static constexpr const char* s_analysis_timing_info_time = "analysis msec";

hipsparseStatus_t hipsparse_record_output_legend(const std::string& s);
hipsparseStatus_t hipsparse_record_output(const std::string& s);
hipsparseStatus_t hipsparse_record_timing(double msec, double gflops, double gbs);
bool              display_timing_info_is_stdout_disabled();

inline auto& operator<<(std::ostream& out, const hipComplex& z)
{
    std::stringstream ss;
    ss << '(' << z.x << ',' << z.y << ')';
    return out << ss.str();
}

inline auto& operator<<(std::ostream& out, const hipDoubleComplex& z)
{
    std::stringstream ss;
    ss << '(' << z.x << ',' << z.y << ')';
    return out << ss.str();
}

struct display_key_t
{
    //
    // Enumerate keys.
    //
    typedef enum enum_
    {
        gflops = 0,
        bandwidth,
        time_ms,
        iters,
        function,
        ctype,
        itype,
        jtype,
        size,
        M,
        N,
        K,
        LD,
        Mb,
        MbA,
        MbC,
        Nb,
        NbA,
        NbC,
        Kb,
        nrhs,
        ell_width,
        ell_nnz,
        coo_nnz,
        nnz,
        nnzA,
        nnzB,
        nnzC,
        nnzb,
        nnzbA,
        nnzbC,
        block_dim,
        row_block_dim,
        row_block_dimA,
        row_block_dimC,
        col_block_dim,
        col_block_dimA,
        col_block_dimC,
        batch_count,
        batch_countA,
        batch_countB,
        batch_countC,
        batch_stride,
        alpha,
        beta,
        percentage,
        threshold,
        trans,
        transA,
        transB,
        transC,
        transX,
        direction,
        order,
        format,
        fill_mode,
        diag_type,
        solve_policy,
        action,
        partition,
        algorithm,
        permute
    } key_t;

    static const char* to_str(key_t key_)
    {
        switch(key_)
        {
        case gflops:
        {
            return s_timing_info_perf;
        }
        case bandwidth:
        {
            return s_timing_info_bandwidth;
        }
        case time_ms:
        {
            return s_timing_info_time;
        }
        case iters:
        {
            return "iters";
        }
        case function:
        {
            return "function";
        }
        case ctype:
        {
            return "ctype";
        }
        case itype:
        {
            return "itype";
        }
        case jtype:
        {
            return "jtype";
        }
        case size:
        {
            return "size";
        }
        case M:
        {
            return "M";
        }
        case N:
        {
            return "N";
        }
        case K:
        {
            return "K";
        }
        case LD:
        {
            return "LD";
        }
        case Mb:
        {
            return "Mb";
        }
        case MbA:
        {
            return "MbA";
        }
        case MbC:
        {
            return "MbC";
        }
        case Nb:
        {
            return "Nb";
        }
        case NbA:
        {
            return "NbA";
        }
        case NbC:
        {
            return "NbC";
        }
        case Kb:
        {
            return "Kb";
        }
        case nrhs:
        {
            return "nrhs";
        }
        case ell_width:
        {
            return "ell_width";
        }
        case ell_nnz:
        {
            return "ell_nnz";
        }
        case coo_nnz:
        {
            return "coo_nnz";
        }
        case nnz:
        {
            return "nnz";
        }
        case nnzA:
        {
            return "nnzA";
        }
        case nnzB:
        {
            return "nnzB";
        }
        case nnzC:
        {
            return "nnzC";
        }
        case nnzb:
        {
            return "nnzb";
        }
        case nnzbA:
        {
            return "nnzbA";
        }
        case nnzbC:
        {
            return "nnzbC";
        }
        case block_dim:
        {
            return "block_dim";
        }
        case row_block_dim:
        {
            return "row_block_dim";
        }
        case row_block_dimA:
        {
            return "row_block_dimA";
        }
        case row_block_dimC:
        {
            return "row_block_dimC";
        }
        case col_block_dim:
        {
            return "col_block_dim";
        }
        case col_block_dimA:
        {
            return "col_block_dimA";
        }
        case col_block_dimC:
        {
            return "col_block_dimC";
        }
        case batch_count:
        {
            return "batch_count";
        }
        case batch_countA:
        {
            return "batch_countA";
        }
        case batch_countB:
        {
            return "batch_countB";
        }
        case batch_countC:
        {
            return "batch_countC";
        }
        case batch_stride:
        {
            return "batch_stride";
        }
        case alpha:
        {
            return "alpha";
        }
        case beta:
        {
            return "beta";
        }
        case percentage:
        {
            return "percentage";
        }
        case threshold:
        {
            return "threshold";
        }
        case trans:
        {
            return "trans";
        }
        case transA:
        {
            return "transA";
        }
        case transB:
        {
            return "transB";
        }
        case transC:
        {
            return "transC";
        }
        case transX:
        {
            return "transX";
        }
        case direction:
        {
            return "dir";
        }
        case order:
        {
            return "order";
        }
        case format:
        {
            return "format";
        }
        case fill_mode:
        {
            return "uplo";
        }
        case diag_type:
        {
            return "diag";
        }
        case solve_policy:
        {
            return "solve_policy";
        }
        case action:
        {
            return "action";
        }
        case partition:
        {
            return "partition";
        }
        case algorithm:
        {
            return "algorithm";
        }
        case permute:
        {
            return "permute";
        }
        default:
        {
            return nullptr;
        }
        }
    }
};

template <typename S>
inline const char* display_to_string(S s)
{
    return s;
};
template <>
inline const char* display_to_string(display_key_t::key_t s)
{
    return display_key_t::to_str(s);
};

//
// Template to display timing information.
//
template <typename S, typename T, typename... Ts>
inline void display_timing_info_legend(std::ostream& out, int n, S name, T t)
{
    out << std::setw(n) << display_to_string(name);
}

template <typename S, typename T, typename... Ts>
inline void display_timing_info_legend(std::ostream& out, int n, S name, T t, Ts... ts)
{
    out << std::setw(n) << display_to_string(name);
    display_timing_info_legend(out, n, ts...);
}

template <typename S, typename T, typename... Ts>
inline void display_timing_info_values(std::ostream& out, int n, S name, T t)
{
    out << std::setw(n) << t;
}

template <typename S, typename T, typename... Ts>
inline void display_timing_info_values(std::ostream& out, int n, S name, T t, Ts... ts)
{
    out << std::setw(n) << t;
    display_timing_info_values(out, n, ts...);
}

template <typename S, typename T, typename... Ts>
inline void display_timing_info_legend_noresults(std::ostream& out, int n, S name_, T t)
{
    const char* name = display_to_string(name_);
    if(strcmp(name, s_timing_info_perf) && strcmp(name, s_timing_info_bandwidth)
       && strcmp(name, s_timing_info_time))
    {
        out << " " << name;
    }
}

template <typename S, typename T, typename... Ts>
inline void display_timing_info_legend_noresults(std::ostream& out, int n, S name_, T t, Ts... ts)
{
    const char* name = display_to_string(name_);
    if(strcmp(name, s_timing_info_perf) && strcmp(name, s_timing_info_bandwidth)
       && strcmp(name, s_timing_info_time))
    {
        out << " " << name;
    }
    display_timing_info_legend_noresults(out, n, ts...);
}

template <typename S, typename T, typename... Ts>
inline void display_timing_info_values_noresults(std::ostream& out, int n, S name_, T t)
{
    const char* name = display_to_string(name_);

    if(strcmp(name, s_timing_info_perf) && strcmp(name, s_timing_info_bandwidth)
       && strcmp(name, s_timing_info_time))
    {
        out << " " << t;
    }
}

template <typename S, typename T, typename... Ts>
inline void display_timing_info_values_noresults(std::ostream& out, int n, S name_, T t, Ts... ts)
{
    const char* name = display_to_string(name_);
    if(strcmp(name, s_timing_info_perf) && strcmp(name, s_timing_info_bandwidth)
       && strcmp(name, s_timing_info_time))
    {
        out << " " << t;
    }
    display_timing_info_values_noresults(out, n, ts...);
}

template <typename T>
inline void grab_results(double values[3], display_key_t::key_t key, T t)
{
}

template <>
inline void grab_results<double>(double values[3], display_key_t::key_t key, double t)
{
    const char* name = display_to_string(key);
    if(!strcmp(name, s_timing_info_perf))
    {
        values[1] = t;
    }
    else if(!strcmp(name, s_timing_info_bandwidth))
    {
        values[2] = t;
    }
    else if(!strcmp(name, s_timing_info_time))
    {
        values[0] = t;
    }
}

template <typename T>
inline void grab_results(double values[3], const char* name, T t)
{
}

template <>
inline void grab_results<double>(double values[3], const char* name, double t)
{
    if(!strcmp(name, s_timing_info_perf))
    {
        values[1] = t;
    }
    else if(!strcmp(name, s_timing_info_bandwidth))
    {
        values[2] = t;
    }
    else if(!strcmp(name, s_timing_info_time))
    {
        values[0] = t;
    }
}

template <typename S, typename T, typename... Ts>
inline void display_timing_info_grab_results(double values[3], S name, T t)
{
    grab_results(values, name, t);
}

template <typename S, typename T, typename... Ts>
inline void display_timing_info_grab_results(double values[3], S name, T t, Ts... ts)
{
    grab_results(values, name, t);
    display_timing_info_grab_results(values, ts...);
}

//bool display_timing_info_is_stdout_disabled();

template <typename S, typename T, typename... Ts>
inline void display_timing_info_generate(std::ostream& out, int n, S name, T t, Ts... ts)
{
    double values[3]{};
    display_timing_info_grab_results(values, name, t, ts...);
    hipsparse_record_timing(values[0], values[1], values[2]);
    display_timing_info_values(out, n, name, t, ts...);
}

template <typename S, typename T, typename... Ts>
inline void display_timing_info_generate_params(std::ostream& out, int n, S name, T t, Ts... ts)
{
    double values[3]{};
    display_timing_info_grab_results(values, name, t, ts...);
    hipsparse_record_timing(values[0], values[1], values[2]);
    display_timing_info_values_noresults(out, n, name, t, ts...);
}

template <typename S, typename T, typename... Ts>
inline void display_timing_info_max_size_strings(int mx[1], S name_, T t)
{
    const char* name = display_to_string(name_);
    int         len  = strlen(name);
    mx[0]            = std::max(len, mx[0]);
}

template <typename S, typename T, typename... Ts>
inline void display_timing_info_max_size_strings(int mx[1], S name_, T t, Ts... ts)
{
    const char* name = display_to_string(name_);
    int         len  = strlen(name);
    mx[0]            = std::max(len, mx[0]);
    display_timing_info_max_size_strings(mx, ts...);
}

template <typename S, typename... Ts>
inline void display_timing_info_main(S name, Ts... ts)
{
    //
    // To configure the size of std::setw.
    //
    int n = 0;
    display_timing_info_max_size_strings(&n, name, ts...);

    //
    //
    //
    n += 4;

    //
    // Legend
    //
    {
        std::ostringstream out_legend;
        out_legend.precision(2);
        out_legend.setf(std::ios::fixed);
        out_legend.setf(std::ios::left);
        if(!display_timing_info_is_stdout_disabled())
        {
            display_timing_info_legend(out_legend, n, name, ts...);
            std::cout << out_legend.str() << std::endl;
        }
        else
        {
            // store the string.
            display_timing_info_legend_noresults(out_legend, n, name, ts...);
            hipsparse_record_output_legend(out_legend.str());
        }
    }

    std::ostringstream out;
    out.precision(2);
    out.setf(std::ios::fixed);
    out.setf(std::ios::left);
    if(!display_timing_info_is_stdout_disabled())
    {
        display_timing_info_generate(out, n, name, ts...);
        std::cout << out.str() << std::endl;
    }
    else
    {
        display_timing_info_generate_params(out, n, name, ts...);
        // store the string.
        hipsparse_record_output(out.str());
    }
}

inline void hipsparse_get_matrixname(const char* f, char* name)
{
    int n = 0;
    while(f[n] != '\0')
        ++n;
    int cdir = 0;
    for(int i = 0; i < n; ++i)
    {
        if(f[i] == '/' || f[i] == '\\')
        {
            cdir = i + 1;
        }
    }
    int ddir = cdir;
    for(int i = cdir; i < n; ++i)
    {
        if(f[i] == '.')
        {
            ddir = i;
        }
    }

    if(ddir == cdir)
    {
        ddir = n;
    }

    for(int i = cdir; i < ddir; ++i)
    {
        name[i - cdir] = f[i];
    }
    name[ddir - cdir] = '\0';
}

#define display_timing_info(...)                                                \
    do                                                                          \
    {                                                                           \
        const char* ctypename = hipsparse_datatype2string(argus.compute_type);  \
        const char* itypename = hipsparse_indextype2string(argus.index_type_I); \
        const char* jtypename = hipsparse_indextype2string(argus.index_type_J); \
                                                                                \
        display_timing_info_main(__VA_ARGS__,                                   \
                                 display_key_t::iters,                          \
                                 argus.iters,                                   \
                                 "verified",                                    \
                                 (argus.unit_check ? "yes" : "no"),             \
                                 display_key_t::function,                       \
                                 &argus.function[0],                            \
                                 display_key_t::ctype,                          \
                                 ctypename,                                     \
                                 display_key_t::itype,                          \
                                 itypename,                                     \
                                 display_key_t::jtype,                          \
                                 jtypename);                                    \
    } while(false)

#endif // DISPLAY_HPP
