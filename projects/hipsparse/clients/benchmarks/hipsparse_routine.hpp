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
#pragma once
#include "hipsparse_arguments.hpp"

// clang-format off
#define HIPSPARSE_FOREACH_ROUTINE   \
HIPSPARSE_DO_ROUTINE(axpyi)         \
HIPSPARSE_DO_ROUTINE(doti)          \
HIPSPARSE_DO_ROUTINE(dotci)         \
HIPSPARSE_DO_ROUTINE(gthr)          \
HIPSPARSE_DO_ROUTINE(gthrz)         \
HIPSPARSE_DO_ROUTINE(roti)          \
HIPSPARSE_DO_ROUTINE(sctr)          \
HIPSPARSE_DO_ROUTINE(bsrsv2)        \
HIPSPARSE_DO_ROUTINE(coomv)         \
HIPSPARSE_DO_ROUTINE(csrmv)         \
HIPSPARSE_DO_ROUTINE(csrsv)         \
HIPSPARSE_DO_ROUTINE(gemvi)         \
HIPSPARSE_DO_ROUTINE(hybmv)         \
HIPSPARSE_DO_ROUTINE(bsrmm)         \
HIPSPARSE_DO_ROUTINE(bsrsm2)        \
HIPSPARSE_DO_ROUTINE(coomm)         \
HIPSPARSE_DO_ROUTINE(cscmm)         \
HIPSPARSE_DO_ROUTINE(csrmm)         \
HIPSPARSE_DO_ROUTINE(coosm)         \
HIPSPARSE_DO_ROUTINE(csrsm)         \
HIPSPARSE_DO_ROUTINE(gemmi)         \
HIPSPARSE_DO_ROUTINE(csrgeam)       \
HIPSPARSE_DO_ROUTINE(csrgemm)       \
HIPSPARSE_DO_ROUTINE(bsric02)       \
HIPSPARSE_DO_ROUTINE(bsrilu02)      \
HIPSPARSE_DO_ROUTINE(csric02)       \
HIPSPARSE_DO_ROUTINE(csrilu02)      \
HIPSPARSE_DO_ROUTINE(gtsv2)                   \
HIPSPARSE_DO_ROUTINE(gtsv2_nopivot)          \
HIPSPARSE_DO_ROUTINE(gtsv2_strided_batch)    \
HIPSPARSE_DO_ROUTINE(gtsv_interleaved_batch) \
HIPSPARSE_DO_ROUTINE(gpsv_interleaved_batch) \
HIPSPARSE_DO_ROUTINE(bsr2csr) \
HIPSPARSE_DO_ROUTINE(csr2coo) \
HIPSPARSE_DO_ROUTINE(csr2csc) \
HIPSPARSE_DO_ROUTINE(csr2hyb) \
HIPSPARSE_DO_ROUTINE(csr2bsr) \
HIPSPARSE_DO_ROUTINE(csr2gebsr) \
HIPSPARSE_DO_ROUTINE(csr2csr_compress) \
HIPSPARSE_DO_ROUTINE(coo2csr) \
HIPSPARSE_DO_ROUTINE(hyb2csr) \
HIPSPARSE_DO_ROUTINE(csr2dense) \
HIPSPARSE_DO_ROUTINE(csc2dense) \
HIPSPARSE_DO_ROUTINE(coo2dense) \
HIPSPARSE_DO_ROUTINE(dense2csr) \
HIPSPARSE_DO_ROUTINE(dense2csc) \
HIPSPARSE_DO_ROUTINE(dense2coo) \
HIPSPARSE_DO_ROUTINE(gebsr2csr) \
HIPSPARSE_DO_ROUTINE(gebsr2gebsc) \
HIPSPARSE_DO_ROUTINE(gebsr2gebsr)
// clang-format on

template <std::size_t N, typename T>
static constexpr std::size_t countof(T (&)[N])
{
    return N;
}

struct hipsparse_routine
{
private:
public:
#define HIPSPARSE_DO_ROUTINE(x_) x_,
    typedef enum _ : int
    {
        HIPSPARSE_FOREACH_ROUTINE
    } value_type;
    value_type                  value{};
    static constexpr value_type all_routines[] = {HIPSPARSE_FOREACH_ROUTINE};
#undef HIPSPARSE_DO_ROUTINE

    static constexpr std::size_t num_routines = countof(all_routines);

private:
#define HIPSPARSE_DO_ROUTINE(x_) #x_,
    static constexpr const char* s_routine_names[num_routines]{HIPSPARSE_FOREACH_ROUTINE};
#undef HIPSPARSE_DO_ROUTINE

public:
    hipsparse_routine();
    hipsparse_routine& operator()(const char* function);
    explicit hipsparse_routine(const char* function);
    hipsparseStatus_t
        dispatch(const char precision, const char indextype, const Arguments& arg) const;
    constexpr const char* to_string() const;

private:
    template <hipsparse_routine::value_type FNAME, typename T, typename I, typename J = I>
    static hipsparseStatus_t dispatch_call(const Arguments& arg);

    template <hipsparse_routine::value_type FNAME, typename T>
    static hipsparseStatus_t dispatch_indextype(const char cindextype, const Arguments& arg);

    template <hipsparse_routine::value_type FNAME>
    static hipsparseStatus_t
        dispatch_precision(const char precision, const char indextype, const Arguments& arg);

    static bool is_routine_supported(hipsparse_routine::value_type FNAME);
    static void print_routine_support_info(hipsparse_routine::value_type FNAME);
};
