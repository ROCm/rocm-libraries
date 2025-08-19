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

#include "rocsparse_trm_data_t.hpp"
#include <memory>
namespace rocsparse
{

    struct trm_t
    {
        typedef enum item_index_
        {
            from_csrsv,
            from_csrsm,
            from_csrilu0,
            from_csric0,
            from_bsrsv,
            from_bsrsm,
            from_bsrilu0,
            from_bsric0
        } index_t;

        static const char* to_string(index_t index)
        {
            switch(index)
            {
            case rocsparse::trm_t::from_csrsv:
            {
                return "csrsv ";
            }
            case rocsparse::trm_t::from_csrsm:
            {
                return "csrsm ";
            }
            case rocsparse::trm_t::from_bsrsv:
            {
                return "bsrsv ";
            }
            case rocsparse::trm_t::from_bsrsm:
            {
                return "bsrsm ";
            }
            case rocsparse::trm_t::from_csric0:
            {
                return "csric0 ";
            }
            case rocsparse::trm_t::from_bsric0:
            {
                return "bsric0 ";
            }
            case rocsparse::trm_t::from_csrilu0:
            {
                return "csrilu0 ";
            }
            case rocsparse::trm_t::from_bsrilu0:
            {
                return "bsrilu0 ";
            }
            }
            return "unknown";
        }

        static constexpr size_t  nitems = 8;
        static constexpr index_t all[8] = {from_csrsv,
                                           from_csrsm,
                                           from_csrilu0,
                                           from_csric0,
                                           from_bsrsv,
                                           from_bsrsm,
                                           from_bsrilu0,
                                           from_bsric0};

    protected:
        std::shared_ptr<rocsparse::trm_data_t> m_data[trm_t::nitems]{};

    public:
        trm_t() = default;
        ~trm_t();

        rocsparse::trm_data_t*                 first();
        std::shared_ptr<rocsparse::trm_data_t> get_shared(rocsparse::trm_t::index_t index);
        void                                   copy(const trm_t& that);

        void                   destroy(rocsparse::trm_t::index_t index);
        rocsparse::trm_data_t* create(rocsparse::trm_t::index_t index);
        void                   clear(rocsparse::trm_t::index_t index);
    };

}
