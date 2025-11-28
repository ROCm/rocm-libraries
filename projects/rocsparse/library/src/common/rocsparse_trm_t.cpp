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

void rocsparse::trm_t::destroy_csrsv_info()
{
    this->m_csrsv_info.reset();
}
void rocsparse::trm_t::destroy_csrsm_info()
{
    this->m_csrsm_info.reset();
}
void rocsparse::trm_t::destroy_csrilu0_info()
{
    this->m_csrilu0_info.reset();
}
void rocsparse::trm_t::destroy_csric0_info()
{
    this->m_csric0_info.reset();
}
void rocsparse::trm_t::destroy_bsrsv_info()
{
    this->m_bsrsv_info.reset();
}
void rocsparse::trm_t::destroy_bsrsm_info()
{
    this->m_bsrsm_info.reset();
}
void rocsparse::trm_t::destroy_bsrilu0_info()
{
    this->m_bsrilu0_info.reset();
}
void rocsparse::trm_t::destroy_bsric0_info()
{
    this->m_bsric0_info.reset();
}

rocsparse_csrsv_info rocsparse::trm_t::create_csrsv_info()
{
    if(this->m_csrsv_info.get() == nullptr)
    {
        this->m_csrsv_info = std::shared_ptr<_rocsparse_csrsv_info>(new _rocsparse_csrsv_info());
    }
    return this->m_csrsv_info.get();
}

rocsparse_csrsm_info rocsparse::trm_t::create_csrsm_info()
{
    if(this->m_csrsm_info.get() == nullptr)
    {
        this->m_csrsm_info = std::shared_ptr<_rocsparse_csrsm_info>(new _rocsparse_csrsm_info());
    }
    return this->m_csrsm_info.get();
}

rocsparse_csrilu0_info rocsparse::trm_t::create_csrilu0_info()
{
    if(this->m_csrilu0_info.get() == nullptr)
    {
        this->m_csrilu0_info
            = std::shared_ptr<_rocsparse_csrilu0_info>(new _rocsparse_csrilu0_info());
    }
    return this->m_csrilu0_info.get();
}

rocsparse_csric0_info rocsparse::trm_t::create_csric0_info()
{
    if(this->m_csric0_info.get() == nullptr)
    {
        this->m_csric0_info = std::shared_ptr<_rocsparse_csric0_info>(new _rocsparse_csric0_info());
    }
    return this->m_csric0_info.get();
}

rocsparse_bsrsv_info rocsparse::trm_t::create_bsrsv_info()
{
    if(this->m_bsrsv_info.get() == nullptr)
    {
        this->m_bsrsv_info = std::shared_ptr<_rocsparse_bsrsv_info>(new _rocsparse_bsrsv_info());
    }
    return this->m_bsrsv_info.get();
}

rocsparse_bsrsm_info rocsparse::trm_t::create_bsrsm_info()
{
    if(this->m_bsrsm_info.get() == nullptr)
    {
        this->m_bsrsm_info = std::shared_ptr<_rocsparse_bsrsm_info>(new _rocsparse_bsrsm_info());
    }
    return this->m_bsrsm_info.get();
}

rocsparse_bsrilu0_info rocsparse::trm_t::create_bsrilu0_info()
{
    if(this->m_bsrilu0_info.get() == nullptr)
    {
        this->m_bsrilu0_info
            = std::shared_ptr<_rocsparse_bsrilu0_info>(new _rocsparse_bsrilu0_info());
    }
    return this->m_bsrilu0_info.get();
}

rocsparse_bsric0_info rocsparse::trm_t::create_bsric0_info()
{
    if(this->m_bsric0_info.get() == nullptr)
    {
        this->m_bsric0_info = std::shared_ptr<_rocsparse_bsric0_info>(new _rocsparse_bsric0_info());
    }
    return this->m_bsric0_info.get();
}

void rocsparse::trm_t::clear_csrsv_info()
{
    this->uncouple(this->m_csrsv_info.get());
    this->destroy_csrsv_info();
}
void rocsparse::trm_t::clear_csrsm_info()
{
    this->uncouple(this->m_csrsm_info.get());
    this->destroy_csrsm_info();
}
void rocsparse::trm_t::clear_csrilu0_info()
{
    this->uncouple(this->m_csrilu0_info.get());
    this->destroy_csrilu0_info();
}

void rocsparse::trm_t::clear_csric0_info()
{
    this->uncouple(this->m_csric0_info.get());
    this->destroy_csric0_info();
}

void rocsparse::trm_t::clear_bsrsv_info()
{
    this->uncouple(this->m_bsrsv_info.get());
    this->destroy_bsrsv_info();
}
void rocsparse::trm_t::clear_bsrsm_info()
{
    this->uncouple(this->m_bsrsm_info.get());
    this->destroy_bsrsm_info();
}
void rocsparse::trm_t::clear_bsrilu0_info()
{
    this->uncouple(this->m_bsrilu0_info.get());
    this->destroy_bsrilu0_info();
}

void rocsparse::trm_t::clear_bsric0_info()
{
    this->uncouple(this->m_bsric0_info.get());
    this->destroy_bsric0_info();
}

void rocsparse::trm_t::copy(const trm_t& that, hipStream_t stream)
{
    {
        auto p = that.m_csrsv_info.get();
        if(p != nullptr)
        {
            this->create_csrsv_info()->copy(p, stream);
        }
    }
    {
        auto p = that.m_csrsm_info.get();
        if(p != nullptr)
        {
            this->create_csrsm_info()->copy(p, stream);
        }
    }

    {
        auto p = that.m_csrilu0_info.get();
        if(p != nullptr)
        {
            this->create_csrilu0_info()->copy(p, stream);
        }
    }

    {
        auto p = that.m_csric0_info.get();
        if(p != nullptr)
        {
            this->create_csric0_info()->copy(p, stream);
        }
    }

    {
        auto p = that.m_bsrsv_info.get();
        if(p != nullptr)
        {
            this->create_bsrsv_info()->copy(p, stream);
        }
    }
    {
        auto p = that.m_bsrsm_info.get();
        if(p != nullptr)
        {
            this->create_bsrsm_info()->copy(p, stream);
        }
    }

    {
        auto p = that.m_bsrilu0_info.get();
        if(p != nullptr)
        {
            this->create_bsrilu0_info()->copy(p, stream);
        }
    }

    {
        auto p = that.m_bsric0_info.get();
        if(p != nullptr)
        {
            this->create_bsric0_info()->copy(p, stream);
        }
    }
}

rocsparse::trm_t::~trm_t()
{

    this->uncouple(m_csrsv_info.get());
    m_csrsv_info.reset();

    this->uncouple(m_csrsm_info.get());
    m_csrsm_info.reset();

    this->uncouple(m_csrilu0_info.get());
    m_csrilu0_info.reset();

    this->uncouple(m_csric0_info.get());
    m_csric0_info.reset();

    this->uncouple(m_bsrsv_info.get());
    m_bsrsv_info.reset();

    this->uncouple(m_bsrsm_info.get());
    m_bsrsm_info.reset();

    this->uncouple(m_bsrilu0_info.get());
    m_bsrilu0_info.reset();

    this->uncouple(m_bsric0_info.get());
    m_bsric0_info.reset();
}

#define GET_SHARED_INFO(TOKEN)                                                               \
    std::shared_ptr<_rocsparse_##TOKEN##_info> rocsparse::trm_t::get_shared_##TOKEN##_info() \
    {                                                                                        \
        if(this->m_##TOKEN##_info.get() == nullptr)                                          \
            this->m_##TOKEN##_info = std::make_shared<_rocsparse_##TOKEN##_info>();          \
        return this->m_##TOKEN##_info;                                                       \
    }

GET_SHARED_INFO(csrsv);
GET_SHARED_INFO(csrsm);
GET_SHARED_INFO(csrilu0);
GET_SHARED_INFO(csric0);
GET_SHARED_INFO(bsrsv);
GET_SHARED_INFO(bsrsm);
GET_SHARED_INFO(bsrilu0);
GET_SHARED_INFO(bsric0);

void rocsparse::trm_t::uncouple(rocsparse::trm_data_t* p)
{
    if(p != nullptr)
    {
        p->uncouple(this->m_csrsv_info.get());
        p->uncouple(this->m_csrsm_info.get());
        p->uncouple(this->m_csrilu0_info.get());
        p->uncouple(this->m_csric0_info.get());
        p->uncouple(this->m_bsrsv_info.get());
        p->uncouple(this->m_bsrsm_info.get());
        p->uncouple(this->m_bsrilu0_info.get());
        p->uncouple(this->m_bsric0_info.get());
    }
}
