/*! \file */
/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_bsric0_info.hpp"
#include "rocsparse_bsrilu0_info.hpp"
#include "rocsparse_bsrsm_info.hpp"
#include "rocsparse_bsrsv_info.hpp"
#include "rocsparse_csric0_info.hpp"
#include "rocsparse_csrilu0_info.hpp"
#include "rocsparse_csrsm_info.hpp"
#include "rocsparse_csrsv_info.hpp"
#include "rocsparse_trm_data_t.hpp"

#include <memory>
namespace rocsparse
{
    struct trm_t
    {
    protected:
        std::shared_ptr<_rocsparse_csrsv_info>   m_csrsv_info;
        std::shared_ptr<_rocsparse_csrsm_info>   m_csrsm_info;
        std::shared_ptr<_rocsparse_csrilu0_info> m_csrilu0_info;
        std::shared_ptr<_rocsparse_csric0_info>  m_csric0_info;
        std::shared_ptr<_rocsparse_bsrsv_info>   m_bsrsv_info;
        std::shared_ptr<_rocsparse_bsrsm_info>   m_bsrsm_info;
        std::shared_ptr<_rocsparse_bsrilu0_info> m_bsrilu0_info;
        std::shared_ptr<_rocsparse_bsric0_info>  m_bsric0_info;

        rocsparse_status destroy_csrsv_info(hipStream_t);
        rocsparse_status destroy_csrsm_info(hipStream_t);
        rocsparse_status destroy_csrilu0_info(hipStream_t);
        rocsparse_status destroy_csric0_info(hipStream_t);
        rocsparse_status destroy_bsrsv_info(hipStream_t);
        rocsparse_status destroy_bsrsm_info(hipStream_t);
        rocsparse_status destroy_bsrilu0_info(hipStream_t);
        rocsparse_status destroy_bsric0_info(hipStream_t);

        void uncouple(rocsparse::trm_data_t* p);

    public:
        void                   info() const;
        rocsparse_csrsv_info   create_csrsv_info();
        rocsparse_csrsm_info   create_csrsm_info();
        rocsparse_csrilu0_info create_csrilu0_info();
        rocsparse_csric0_info  create_csric0_info();
        rocsparse_bsrsv_info   create_bsrsv_info();
        rocsparse_bsrsm_info   create_bsrsm_info();
        rocsparse_bsrilu0_info create_bsrilu0_info();
        rocsparse_bsric0_info  create_bsric0_info();

        std::shared_ptr<_rocsparse_csrsv_info>   get_shared_csrsv_info();
        std::shared_ptr<_rocsparse_csrsm_info>   get_shared_csrsm_info();
        std::shared_ptr<_rocsparse_csrilu0_info> get_shared_csrilu0_info();
        std::shared_ptr<_rocsparse_csric0_info>  get_shared_csric0_info();
        std::shared_ptr<_rocsparse_bsrsv_info>   get_shared_bsrsv_info();
        std::shared_ptr<_rocsparse_bsrsm_info>   get_shared_bsrsm_info();
        std::shared_ptr<_rocsparse_bsrilu0_info> get_shared_bsrilu0_info();
        std::shared_ptr<_rocsparse_bsric0_info>  get_shared_bsric0_info();

        rocsparse_status clear_csrsv_info(hipStream_t);
        rocsparse_status clear_csrsm_info(hipStream_t);
        rocsparse_status clear_csrilu0_info(hipStream_t);
        rocsparse_status clear_csric0_info(hipStream_t);
        rocsparse_status clear_bsrsv_info(hipStream_t);
        rocsparse_status clear_bsrsm_info(hipStream_t);
        rocsparse_status clear_bsrilu0_info(hipStream_t);
        rocsparse_status clear_bsric0_info(hipStream_t);

    public:
        trm_t()  = default;
        ~trm_t() = default;
        rocsparse_status destroy(hipStream_t stream);
        void             copy(const trm_t& that, hipStream_t stream);
    };

}
