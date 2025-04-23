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

#include "../tests/rocsparse_test_enum.hpp"
#include "rocsparse_reproducibility.hpp"
#include <iostream>
#include <string.h>

constexpr rocsparse_test_enum::value_type rocsparse_test_enum::all_test_enum[];

rocsparse_reproducibility_t::test_data_t* rocsparse_reproducibility_t::test_t::get_initial_data()
{
    return this->m_datas[0];
}

const rocsparse_reproducibility_t::test_data_t*
    rocsparse_reproducibility_t::test_t::get_initial_data() const
{
    return this->m_datas[0];
}

static uint64_t compute_hash_segment(void* t, uint64_t nbytes)
{
    if(nbytes == 0)
    {
        return uint64_t(1);
    }
    else
    {
        uint8_t* it    = (uint8_t*)t;
        uint64_t h_sum = it[0];
        uint64_t h_min = h_sum;
        uint64_t h_max = h_sum;
        for(uint64_t i = 1; i < nbytes; ++i)
        {
            const uint64_t v = it[i];
            h_sum            = h_sum + v;
            h_min            = std::min(v, h_min);
            h_max            = std::max(v, h_max);
        }
        return std::max((h_sum * 19 + h_min * 37 + h_max * 59), uint64_t(1));
    }
}

uint64_t rocsparse_reproducibility_t::test_data_t::compute_hash() const
{
    uint64_t h = 1;
    for(uint32_t i = 0; i < this->m_num_objects; ++i)
    {
        h += compute_hash_segment(this->m_objects[i], this->m_object_numbytes[i]);
    }
    return h;
}

rocsparse_reproducibility_t::test_data_t* rocsparse_reproducibility_t::test_t::get_current_data()
{
    if(this->m_next_call)
    {
        if(this->m_datas[1] == nullptr)
        {
            this->m_datas[1] = new rocsparse_reproducibility_t::test_data_t();
        }
        return this->m_datas[1];
    }
    else
    {
        if(this->m_datas[0] == nullptr)
        {
            this->m_datas[0] = new rocsparse_reproducibility_t::test_data_t();
        }
        return this->m_datas[0];
    }
}

void rocsparse_reproducibility_t::test_t::reset()
{
    this->m_next_call = false;
    if(this->m_datas[0] != nullptr)
    {
        delete this->m_datas[0];
        this->m_datas[0] = nullptr;
    }
    if(this->m_datas[1] != nullptr)
    {
        delete this->m_datas[1];
        this->m_datas[1] = nullptr;
    }
}

void rocsparse_reproducibility_t::test_t::reset_next()
{
    if(this->m_datas[1] != nullptr)
    {
        this->m_datas[1]->reset();
        delete this->m_datas[1];
        this->m_datas[1] = nullptr;
    }
}

void rocsparse_reproducibility_t::test_t::set_next()
{
    this->m_next_call = true;
    // create
    this->m_datas[1] = new rocsparse_reproducibility_t::test_data_t();
}

bool rocsparse_reproducibility_t::test_t::is_next() const
{
    return this->m_next_call;
}
