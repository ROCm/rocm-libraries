/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_trm_uncouple_slot.hpp"
#include "rocsparse_mat_info.hpp"

namespace
{
    bool holds(const rocsparse::trm_data_t* data, const rocsparse::trm_info_t* trm)
    {
        for(auto operation : {rocsparse_operation_none, rocsparse_operation_transpose})
        {
            for(auto fill_mode : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
            {
                if(data->get(operation, fill_mode) == trm)
                {
                    return true;
                }
            }
        }
        return false;
    }
}

void rocsparse::trm_uncouple_slot(rocsparse_mat_info     info,
                                  rocsparse::trm_data_t* data,
                                  rocsparse_operation    operation,
                                  rocsparse_fill_mode    fill_mode)
{
    const rocsparse::trm_info_t* trm
        = (info != nullptr && data != nullptr) ? data->get(operation, fill_mode) : nullptr;
    if(trm == nullptr)
    {
        return;
    }

    const rocsparse::trm_data_t* others[] = {info->get_csrsv_info(),
                                             info->get_csrsm_info(),
                                             info->get_csrilu0_info(),
                                             info->get_csric0_info(),
                                             info->get_csrildlt0_info()};
    for(const rocsparse::trm_data_t* other : others)
    {
        if(other != data && holds(other, trm))
        {
            data->set(operation, fill_mode, nullptr);
            return;
        }
    }
}
