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

#include "rocsparse_trm_t.hpp"

rocsparse::trm_data_t* rocsparse::trm_t::first()
{
    for(const auto i : rocsparse::trm_t::all)
    {
        if(this->m_data[i].get() != nullptr)
        {
            return this->m_data[i].get();
        }
    }
    return nullptr;
}

void rocsparse::trm_t::copy(const trm_t& that)
{
    for(const auto i : rocsparse::trm_t::all)
    {
        rocsparse::trm_data_t* that_trm_data = that.m_data[i].get();
        if(that_trm_data != nullptr)
        {
            if(this->m_data[i].get() == nullptr)
            {
                this->m_data[i]
                    = std::shared_ptr<rocsparse::trm_data_t>(new rocsparse::trm_data_t());
            }
            this->m_data[i].get()->copy(that_trm_data);
        }
    }
}

rocsparse::trm_t::~trm_t()
{
    for(const auto i : rocsparse::trm_t::all)
    {
        for(const auto j : rocsparse::trm_t::all)
        {
            if((i < j) && (this->m_data[i].get() != nullptr))
            {
                this->m_data[i].get()->uncouple(this->m_data[j].get());
            }
        }
    }

    for(const auto i : rocsparse::trm_t::all)
    {
        if(this->m_data[i].get() != nullptr)
        {
            this->m_data[i].reset();
        }
    }
}

std::shared_ptr<rocsparse::trm_data_t> rocsparse::trm_t::get_shared(rocsparse::trm_t::index_t index)
{
    return this->m_data[index];
}

void rocsparse::trm_t::destroy(rocsparse::trm_t::index_t index)
{
    rocsparse::trm_data_t* trm_data = this->m_data[index].get();
    if(trm_data != nullptr)
    {
        this->m_data[index].reset();
    }
}

void rocsparse::trm_t::clear(rocsparse::trm_t::index_t index)
{
    auto that = this->m_data[index];
    if(that != nullptr)
    {
        for(const auto i : rocsparse::trm_t::all)
        {
            if(i != index)
            {
                if(this->m_data[i].get() != nullptr)
                {
                    that->uncouple(this->m_data[i].get());
                }
            }
        }
    }

    this->destroy(index);
}

rocsparse::trm_data_t* rocsparse::trm_t::create(rocsparse::trm_t::index_t index)
{
    auto that = this->m_data[index].get();
    if(that == nullptr)
    {
        that                = new rocsparse::trm_data_t();
        this->m_data[index] = std::shared_ptr<rocsparse::trm_data_t>(that);
    }
    return that;
}
