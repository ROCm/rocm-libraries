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

#include "rocsparse_trm_data_t.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

void rocsparse::trm_data_t::copy(const rocsparse::trm_data_t* that, hipStream_t stream)
{
    if(that != nullptr)
    {
        for(int i = 0; i < 4; ++i)
        {
            if(that->m_data[i] != nullptr)
            {
                rocsparse::trm_info_t::copy(&this->m_data[i], that->m_data[i]);
            }
        }

        this->copy_pivot_info_async(that, stream);
        THROW_IF_HIP_ERROR(hipStreamSynchronize(stream));
    }
}

void rocsparse::trm_data_t::uncouple(const rocsparse::trm_data_t* that)
{
    if(that != nullptr && that != this)
    {
        for(int i = 0; i < 4; ++i)
        {
            auto a = this->m_data[i];
            if(a != nullptr)
            {
                for(int j = 0; j < 4; ++j)
                {
                    auto b = that->m_data[j];
                    if(b == a)
                    {
                        this->m_data[i] = nullptr;
                        break;
                    }
                }
            }
        }
    }
}

rocsparse::trm_data_t::~trm_data_t()
{
    // Clear zero pivot
    for(int i = 0; i < 4; ++i)
    {
        auto trm_info = this->m_data[i];
        if(trm_info != nullptr)
        {
            delete trm_info;
            this->m_data[i] = nullptr;
        }
    }
}

rocsparse::trm_info_t* rocsparse::trm_data_t::get(rocsparse_operation operation,
                                                  rocsparse_fill_mode fill_mode)
{
    const int idx = rocsparse::trm_data_t::storage_index(operation, fill_mode);
    return this->m_data[idx];
}

void rocsparse::trm_data_t::set(rocsparse_operation    operation,
                                rocsparse_fill_mode    fill_mode,
                                rocsparse::trm_info_t* trm_info)
{
    const int idx     = rocsparse::trm_data_t::storage_index(operation, fill_mode);
    this->m_data[idx] = trm_info;
}

rocsparse_indextype rocsparse::trm_data_t::get_indextype_J() const
{
    for(int i = 0; i < 4; ++i)
    {
        if(this->m_data[i] != nullptr)
            return this->m_data[i]->get_index_indextype();
    }
    return rocsparse_indextype_i32;
}
