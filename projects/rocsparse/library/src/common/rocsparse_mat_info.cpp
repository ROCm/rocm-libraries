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

#include "rocsparse_mat_info.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

rocsparse_csrsm_info _rocsparse_mat_info::get_csrsm_info()
{
    return this->m_trm.create(rocsparse::trm_t::from_csrsm);
}

rocsparse_bsrsm_info _rocsparse_mat_info::get_bsrsm_info()
{
    return this->m_trm.create(rocsparse::trm_t::from_bsrsm);
}

rocsparse_bsrsv_info _rocsparse_mat_info::get_bsrsv_info()
{
    return this->m_trm.create(rocsparse::trm_t::from_bsrsv);
}

rocsparse_csrsv_info _rocsparse_mat_info::get_csrsv_info()
{
    return this->m_trm.create(rocsparse::trm_t::from_csrsv);
}

rocsparse_csric0_info _rocsparse_mat_info::get_csric0_info()
{
    return this->m_trm.create(rocsparse::trm_t::from_csric0);
}

rocsparse_csrilu0_info _rocsparse_mat_info::get_csrilu0_info()
{
    return this->m_trm.create(rocsparse::trm_t::from_csrilu0);
}

rocsparse_bsric0_info _rocsparse_mat_info::get_bsric0_info()
{
    return this->m_trm.create(rocsparse::trm_t::from_bsric0);
}

rocsparse_bsrilu0_info _rocsparse_mat_info::get_bsrilu0_info()
{
    return this->m_trm.create(rocsparse::trm_t::from_bsrilu0);
}

rocsparse::trm_info_t* _rocsparse_mat_info::get_csrsm_info(rocsparse_operation operation,
                                                           rocsparse_fill_mode fill_mode)
{
    return this->get_trm_info(rocsparse::trm_t::from_csrsm, operation, fill_mode);
}

rocsparse::trm_info_t* _rocsparse_mat_info::get_bsrsm_info(rocsparse_operation operation,
                                                           rocsparse_fill_mode fill_mode)
{
    return this->get_trm_info(rocsparse::trm_t::from_bsrsm, operation, fill_mode);
}
rocsparse::trm_info_t* _rocsparse_mat_info::get_bsrsv_info(rocsparse_operation operation,
                                                           rocsparse_fill_mode fill_mode)
{
    return this->get_trm_info(rocsparse::trm_t::from_bsrsv, operation, fill_mode);
}

rocsparse::trm_info_t* _rocsparse_mat_info::get_csric0_info(rocsparse_operation operation,
                                                            rocsparse_fill_mode fill_mode)
{
    return this->get_trm_info(rocsparse::trm_t::from_csric0, operation, fill_mode);
}

rocsparse::trm_info_t* _rocsparse_mat_info::get_csrilu0_info(rocsparse_operation operation,
                                                             rocsparse_fill_mode fill_mode)
{
    return this->get_trm_info(rocsparse::trm_t::from_csrilu0, operation, fill_mode);
}

rocsparse::trm_info_t* _rocsparse_mat_info::get_bsrilu0_info(rocsparse_operation operation,
                                                             rocsparse_fill_mode fill_mode)
{
    return this->get_trm_info(rocsparse::trm_t::from_bsrilu0, operation, fill_mode);
}

rocsparse::trm_info_t* _rocsparse_mat_info::get_bsric0_info(rocsparse_operation operation,
                                                            rocsparse_fill_mode fill_mode)
{
    return this->get_trm_info(rocsparse::trm_t::from_bsric0, operation, fill_mode);
}

rocsparse::trm_info_t* _rocsparse_mat_info::get_csrsv_info(rocsparse_operation operation,
                                                           rocsparse_fill_mode fill_mode)
{
    return this->get_trm_info(rocsparse::trm_t::from_csrsv, operation, fill_mode);
}

void _rocsparse_mat_info::set_bsrsm_info(rocsparse_operation    operation,
                                         rocsparse_fill_mode    fill_mode,
                                         rocsparse::trm_info_t* trm)
{
    this->set_trm_info(rocsparse::trm_t::from_bsrsm, operation, fill_mode, trm);
}

void _rocsparse_mat_info::set_bsrsv_info(rocsparse_operation    operation,
                                         rocsparse_fill_mode    fill_mode,
                                         rocsparse::trm_info_t* trm)
{
    this->set_trm_info(rocsparse::trm_t::from_bsrsv, operation, fill_mode, trm);
}

void _rocsparse_mat_info::set_csrsv_info(rocsparse_operation    operation,
                                         rocsparse_fill_mode    fill_mode,
                                         rocsparse::trm_info_t* trm)
{
    this->set_trm_info(rocsparse::trm_t::from_csrsv, operation, fill_mode, trm);
}

void _rocsparse_mat_info::set_bsric0_info(rocsparse_operation    operation,
                                          rocsparse_fill_mode    fill_mode,
                                          rocsparse::trm_info_t* trm)
{
    this->set_trm_info(rocsparse::trm_t::from_bsric0, operation, fill_mode, trm);
}

void _rocsparse_mat_info::set_bsrilu0_info(rocsparse_operation    operation,
                                           rocsparse_fill_mode    fill_mode,
                                           rocsparse::trm_info_t* trm)
{
    this->set_trm_info(rocsparse::trm_t::from_bsrilu0, operation, fill_mode, trm);
}

void _rocsparse_mat_info::set_csric0_info(rocsparse_operation    operation,
                                          rocsparse_fill_mode    fill_mode,
                                          rocsparse::trm_info_t* trm)
{
    this->set_trm_info(rocsparse::trm_t::from_csric0, operation, fill_mode, trm);
}

void _rocsparse_mat_info::set_csrilu0_info(rocsparse_operation    operation,
                                           rocsparse_fill_mode    fill_mode,
                                           rocsparse::trm_info_t* trm)
{
    this->set_trm_info(rocsparse::trm_t::from_csrilu0, operation, fill_mode, trm);
}

void _rocsparse_mat_info::set_csrsm_info(rocsparse_operation    operation,
                                         rocsparse_fill_mode    fill_mode,
                                         rocsparse::trm_info_t* trm)
{
    return this->set_trm_info(rocsparse::trm_t::from_csrsm, operation, fill_mode, trm);
}

rocsparse::trm_info_t* _rocsparse_mat_info::get_csrsv_lower_info()
{
    return this->get_csrsv_info(rocsparse_operation_none, rocsparse_fill_mode_lower);
}

rocsparse::trm_info_t* _rocsparse_mat_info::get_csrsvt_upper_info()
{
    return this->get_csrsv_info(rocsparse_operation_transpose, rocsparse_fill_mode_upper);
}

rocsparse::trm_info_t* _rocsparse_mat_info::get_csrsm_lower_info()
{
    return this->get_csrsm_info(rocsparse_operation_none, rocsparse_fill_mode_lower);
}

rocsparse::trm_info_t* _rocsparse_mat_info::get_csrsmt_upper_info()
{
    return this->get_csrsm_info(rocsparse_operation_transpose, rocsparse_fill_mode_upper);
}

rocsparse::trm_info_t* _rocsparse_mat_info::get_bsrsm_lower_info()
{
    return this->get_bsrsm_info(rocsparse_operation_none, rocsparse_fill_mode_lower);
}

rocsparse::trm_info_t* _rocsparse_mat_info::get_bsrsv_lower_info()
{
    return this->get_bsrsv_info(rocsparse_operation_none, rocsparse_fill_mode_lower);
}

void _rocsparse_mat_info::clear(rocsparse::trm_t::index_t index)
{
    this->m_trm.clear(index);
}

void _rocsparse_mat_info::set_trm_info(rocsparse::trm_t::index_t index,
                                       rocsparse_operation       operation,
                                       rocsparse_fill_mode       fill_mode,
                                       rocsparse::trm_info_t*    that)
{

    this->m_trm.create(index)->set(operation, fill_mode, that);
}

std::shared_ptr<_rocsparse_csrsv_info> _rocsparse_mat_info::get_shared_csrsv_info()
{
    return this->m_trm.get_shared(rocsparse::trm_t::from_csrsv);
}

std::shared_ptr<_rocsparse_csrsm_info> _rocsparse_mat_info::get_shared_csrsm_info()
{
    return this->m_trm.get_shared(rocsparse::trm_t::from_csrsm);
}

rocsparse::trm_info_t* _rocsparse_mat_info::get_trm_info(rocsparse::trm_t::index_t index,
                                                         rocsparse_operation       operation,
                                                         rocsparse_fill_mode       fill_mode)
{
    return this->m_trm.create(index)->get(operation, fill_mode);
}

//
// Duplicate all the trm_info_t.
//
void _rocsparse_mat_info::duplicate_trdata(rocsparse_mat_info src)
{
    this->m_trm.copy(src->m_trm);
}

rocsparse_indextype _rocsparse_mat_info::get_indextype_J()
{

    return this->m_trm.first()->get_indextype_J();
}

rocsparse_csrmv_info _rocsparse_mat_info::get_csrmv_info()
{
    return this->csrmv_info;
}

void _rocsparse_mat_info::set_csrmv_info(rocsparse_csrmv_info value)
{
    this->csrmv_info = value;
}

rocsparse_bsrmv_info _rocsparse_mat_info::get_bsrmv_info()
{
    return this->bsrmv_info;
}

void _rocsparse_mat_info::set_bsrmv_info(rocsparse_bsrmv_info value)
{
    this->bsrmv_info = value;
}

void _rocsparse_mat_info::set_sorted_coo2csr_info(rocsparse::sorted_coo2csr_info_t* value)
{
    this->m_sorted_coo2csr_info = value;
}

rocsparse::sorted_coo2csr_info_t* _rocsparse_mat_info::get_sorted_coo2csr_info()
{
    return this->m_sorted_coo2csr_info;
}

_rocsparse_mat_info::~_rocsparse_mat_info()
{

    // Clear csrgemm info struct
    WARNING_IF_ROCSPARSE_ERROR(rocsparse::destroy_csrgemm_info(this->csrgemm_info));

    // Clear csritsv info struct
    WARNING_IF_ROCSPARSE_ERROR(rocsparse::destroy_csritsv_info(this->csritsv_info));

    // Clear zero pivot
    WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->zero_pivot));

    // Clear singular pivot
    WARNING_IF_HIP_ERROR(rocsparse_hipFree(this->singular_pivot));

    //
    // TRM_INFO data are automatically destroyed.
    //

    if(this->csrmv_info != nullptr)
    {
        delete this->csrmv_info;
    }

    if(this->bsrmv_info != nullptr)
    {
        delete this->bsrmv_info;
    }

    rocsparse::sorted_coo2csr_info_t* sorted_coo2csr_info = this->get_sorted_coo2csr_info();
    if(sorted_coo2csr_info != nullptr)
    {
        hipStream_t default_stream = 0;
        std::ignore                = sorted_coo2csr_info->free_memory(default_stream);

        delete sorted_coo2csr_info;
        this->set_sorted_coo2csr_info(nullptr);
    }
}
