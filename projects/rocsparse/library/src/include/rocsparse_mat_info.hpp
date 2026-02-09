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
#include "rocsparse_trm_t.hpp"

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
    rocsparse_csrmv_info csrmv_info{};
    rocsparse_bsrmv_info bsrmv_info{};

    rocsparse::sorted_coo2csr_info_t* m_sorted_coo2csr_info{};
    rocsparse::trm_t                  m_trm;

public:
    void duplicate_trdata(const rocsparse_mat_info src, hipStream_t stream);

    std::shared_ptr<_rocsparse_csrsv_info>   get_shared_csrsv_info();
    std::shared_ptr<_rocsparse_csrsm_info>   get_shared_csrsm_info();
    std::shared_ptr<_rocsparse_csrilu0_info> get_shared_csrilu0_info();
    std::shared_ptr<_rocsparse_csric0_info>  get_shared_csric0_info();
    std::shared_ptr<_rocsparse_bsrsv_info>   get_shared_bsrsv_info();
    std::shared_ptr<_rocsparse_bsrsm_info>   get_shared_bsrsm_info();
    std::shared_ptr<_rocsparse_bsrilu0_info> get_shared_bsrilu0_info();
    std::shared_ptr<_rocsparse_bsric0_info>  get_shared_bsric0_info();
    void                                     clear_csrsv_info();
    void                                     clear_csrsm_info();
    void                                     clear_csrilu0_info();
    void                                     clear_csric0_info();
    void                                     clear_bsrsv_info();
    void                                     clear_bsrsm_info();
    void                                     clear_bsrilu0_info();
    void                                     clear_bsric0_info();

    rocsparse_bsric0_info  get_bsric0_info();
    rocsparse::trm_info_t* get_bsric0_info(rocsparse_operation operation,
                                           rocsparse_fill_mode fill_mode);
    void                   set_bsric0_info(rocsparse_operation    operation,
                                           rocsparse_fill_mode    fill_mode,
                                           rocsparse::trm_info_t* trm_info);

    rocsparse_bsrilu0_info get_bsrilu0_info();
    rocsparse::trm_info_t* get_bsrilu0_info(rocsparse_operation operation,
                                            rocsparse_fill_mode fill_mode);
    void                   set_bsrilu0_info(rocsparse_operation    operation,
                                            rocsparse_fill_mode    fill_mode,
                                            rocsparse::trm_info_t* trm_info);

    rocsparse_csric0_info  get_csric0_info();
    rocsparse::trm_info_t* get_csric0_info(rocsparse_operation operation,
                                           rocsparse_fill_mode fill_mode);
    void                   set_csric0_info(rocsparse_operation    operation,
                                           rocsparse_fill_mode    fill_mode,
                                           rocsparse::trm_info_t* trm_info);

    rocsparse_csrilu0_info get_csrilu0_info();
    rocsparse::trm_info_t* get_csrilu0_info(rocsparse_operation operation,
                                            rocsparse_fill_mode fill_mode);
    void                   set_csrilu0_info(rocsparse_operation    operation,
                                            rocsparse_fill_mode    fill_mode,
                                            rocsparse::trm_info_t* trm_info);

    rocsparse_csrsv_info   get_csrsv_info();
    rocsparse::trm_info_t* get_csrsv_info(rocsparse_operation operation,
                                          rocsparse_fill_mode fill_mode);
    void                   set_csrsv_info(rocsparse_operation    operation,
                                          rocsparse_fill_mode    fill_mode,
                                          rocsparse::trm_info_t* trm_info);

    rocsparse_csrsm_info   get_csrsm_info();
    rocsparse::trm_info_t* get_csrsm_info(rocsparse_operation operation,
                                          rocsparse_fill_mode fill_mode);
    void                   set_csrsm_info(rocsparse_operation    operation,
                                          rocsparse_fill_mode    fill_mode,
                                          rocsparse::trm_info_t* trm_info);

    rocsparse_bsrsv_info   get_bsrsv_info();
    rocsparse::trm_info_t* get_bsrsv_info(rocsparse_operation operation,
                                          rocsparse_fill_mode fill_mode);
    void                   set_bsrsv_info(rocsparse_operation    operation,
                                          rocsparse_fill_mode    fill_mode,
                                          rocsparse::trm_info_t* trm_info);

    rocsparse_bsrsm_info   get_bsrsm_info();
    rocsparse::trm_info_t* get_bsrsm_info(rocsparse_operation operation,
                                          rocsparse_fill_mode fill_mode);
    void                   set_bsrsm_info(rocsparse_operation    operation,
                                          rocsparse_fill_mode    fill_mode,
                                          rocsparse::trm_info_t* trm_info);

    rocsparse_csrgemm_info csrgemm_info{};
    rocsparse_csritsv_info csritsv_info{};

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
