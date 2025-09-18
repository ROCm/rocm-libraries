/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse-auxiliary.h"
#include "rocsparse-version.h"

#include "rocsparse_adaptive_info.hpp"
#include "rocsparse_blas.hpp"
#include "rocsparse_bsrmv_info.hpp"
#include "rocsparse_color_info.hpp"
#include "rocsparse_csrgemm_info.hpp"
#include "rocsparse_csritsv_info.hpp"
#include "rocsparse_csrmv_info.hpp"
#include "rocsparse_dnmat_descr.hpp"
#include "rocsparse_dnvec_descr.hpp"
#include "rocsparse_hyb_mat.hpp"
#include "rocsparse_lrb_info.hpp"
#include "rocsparse_mat_descr.hpp"
#include "rocsparse_mat_info.hpp"
#include "rocsparse_spgeam_descr.hpp"
#include "rocsparse_spmat_descr.hpp"
#include "rocsparse_spvec_descr.hpp"
#include "rocsparse_trm_info.hpp"

#include <fstream>
#include <hip/hip_runtime_api.h>

/********************************************************************************
 * \brief rocsparse_handle is a structure holding the rocsparse library context.
 * It must be initialized using rocsparse_create_handle()
 * and the returned handle must be passed
 * to all subsequent library function calls.
 * It should be destroyed at the end using rocsparse_destroy_handle().
 *******************************************************************************/
struct _rocsparse_handle
{
    // constructor
    _rocsparse_handle();
    // destructor
    ~_rocsparse_handle();

    // set stream
    rocsparse_status set_stream(hipStream_t user_stream);
    // get stream
    rocsparse_status get_stream(hipStream_t* user_stream) const;
    // set pointer mode
    rocsparse_status set_pointer_mode(rocsparse_pointer_mode user_mode);
    // get pointer mode
    rocsparse_status get_pointer_mode(rocsparse_pointer_mode* user_mode) const;

    // device id
    int device;
    // device properties
    hipDeviceProp_t properties;
    // device wavefront size
    int wavefront_size;
    // shared memory per block opt-in
    int shared_mem_per_block_optin;
    // asic revision
    int asic_rev;
    // stream ; default stream is system stream NULL
    hipStream_t stream = 0;
    // pointer mode ; default mode is host
    rocsparse_pointer_mode pointer_mode = rocsparse_pointer_mode_host;
    // logging mode
    rocsparse_layer_mode layer_mode;
    // device buffer
    size_t buffer_size{};
    void*  buffer{};

    void* alpha{};
    void* beta{};

    // device one
    void* sone{};
    void* done{};

    // blas handle
    rocsparse::blas_handle blas_handle;

    // logging streams
    std::ofstream log_trace_ofs;
    std::ofstream log_bench_ofs;
    std::ofstream log_debug_ofs;
    std::ostream* log_trace_os{};
    std::ostream* log_bench_os{};
    std::ostream* log_debug_os{};
};

namespace rocsparse
{
    //
    // Get architecture name.
    //
    std::string handle_get_arch_name(rocsparse_handle handle);

    struct rocpsarse_arch_names
    {
        static constexpr const char* gfx908 = "gfx908";
    };

    //
    // Get xnack mode.
    //
    std::string handle_get_xnack_mode(rocsparse_handle handle);
}
