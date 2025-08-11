/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_bsrmv_info.hpp"
#include "rocsparse_csrgemm_info.hpp"
#include "rocsparse_csritsv_info.hpp"
#include "rocsparse_csrmv_info.hpp"
#include "rocsparse_sorted_coo2csr_info.hpp"
#include "rocsparse_trm_info.hpp"

/********************************************************************************
 * \brief rocsparse_mat_info is a structure holding the matrix info data that is
 * gathered during the analysis routines. It must be initialized by calling
 * rocsparse_create_mat_info() and the returned info structure must be passed
 * to all subsequent function calls that require additional information. It
 * should be destroyed at the end using rocsparse_destroy_mat_info().
 *******************************************************************************/
struct _rocsparse_mat_info
{
protected:
    rocsparse_csrmv_info              csrmv_info{};
    rocsparse_bsrmv_info              bsrmv_info{};
    rocsparse::sorted_coo2csr_info_t* m_sorted_coo2csr_info{};

public:
    rocsparse::trm_info_t* bsrsv_upper_info{};
    rocsparse::trm_info_t* bsrsv_lower_info{};
    rocsparse::trm_info_t* bsrsvt_upper_info{};
    rocsparse::trm_info_t* bsrsvt_lower_info{};
    rocsparse::trm_info_t* bsric0_info{};
    rocsparse::trm_info_t* bsrilu0_info{};
    rocsparse::trm_info_t* bsrsm_upper_info{};
    rocsparse::trm_info_t* bsrsm_lower_info{};
    rocsparse::trm_info_t* bsrsmt_upper_info{};
    rocsparse::trm_info_t* bsrsmt_lower_info{};
    rocsparse::trm_info_t* csric0_info{};
    rocsparse::trm_info_t* csrilu0_info{};
    rocsparse::trm_info_t* csrsv_upper_info{};
    rocsparse::trm_info_t* csrsv_lower_info{};
    rocsparse::trm_info_t* csrsvt_upper_info{};
    rocsparse::trm_info_t* csrsvt_lower_info{};
    rocsparse::trm_info_t* csrsm_upper_info{};
    rocsparse::trm_info_t* csrsm_lower_info{};
    rocsparse::trm_info_t* csrsmt_upper_info{};
    rocsparse::trm_info_t* csrsmt_lower_info{};
    rocsparse_csrgemm_info csrgemm_info{};
    rocsparse_csritsv_info csritsv_info{};

    // zero pivot for csrsv, csrsm, csrilu0, csric0
    void* zero_pivot{};

    // singular pivot for csric0
    void* singular_pivot{};

    // tolerance used for determining near singularity
    double singular_tol{};

    // numeric boost for ilu0
    int         boost_enable{};
    size_t      boost_tol_size{};
    const void* boost_tol{};
    const void* boost_val{};

    ~_rocsparse_mat_info();

    void                 set_csrmv_info(rocsparse_csrmv_info value);
    rocsparse_csrmv_info get_csrmv_info();

    void                 set_bsrmv_info(rocsparse_bsrmv_info value);
    rocsparse_bsrmv_info get_bsrmv_info();

    void set_sorted_coo2csr_info(rocsparse::sorted_coo2csr_info_t* value);
    rocsparse::sorted_coo2csr_info_t* get_sorted_coo2csr_info();
};

namespace rocsparse
{
    /********************************************************************************
   * \brief check_trm_shared checks if the given trm info structure
   * shares its meta data with another trm info structure.
   *******************************************************************************/
    bool check_trm_shared(const rocsparse_mat_info info, rocsparse::trm_info_t* trm);

}
