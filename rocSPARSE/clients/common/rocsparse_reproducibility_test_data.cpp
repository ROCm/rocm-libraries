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

#include "rocsparse_reproducibility.hpp"
#include <iostream>
#include <string.h>

bool rocsparse_reproducibility_t::test_data_t::is_same(
    const rocsparse_reproducibility_t::test_data_t* that, std::ostream& err) const
{
    if(this->m_num_objects != that->m_num_objects)
    {
        err << "the number of objects is different, this->m_num_objects =  " << this->m_num_objects
            << ", that->m_num_objects = " << that->m_num_objects;
        return false;
    }

    bool same = true;
    for(uint32_t i = 0; i < this->m_num_objects; ++i)
    {
        if(this->m_object_names[i] != that->m_object_names[i])
        {
            err << "test_data_t::is_same: object names are different, this->m_object_names[" << i
                << "] =  " << this->m_object_names[i] << ", that->m_objects_names[" << i
                << "] = " << that->m_object_names[i];
            same = false;
            break;
        }
    }

    if(false == same)
    {
        return false;
    }

    for(uint32_t i = 0; i < this->m_num_objects; ++i)
    {
        if(this->m_object_numbytes[i] != that->m_object_numbytes[i])
        {
            err << "test_data_t::is_same: object numbytes are different, this->m_object_numbytes["
                << i << "] = " << this->m_object_numbytes[i] << ", that->m_objects_numbytes[" << i
                << "] = " << that->m_object_numbytes[i] << ".";
            same = false;
            break;
        }
    }

    if(false == same)
    {
        return false;
    }

    for(uint32_t i = 0; i < this->m_num_objects; ++i)
    {
        if(0 != memcmp(this->m_objects[i], that->m_objects[i], this->m_object_numbytes[i]))
        {
            err << "test_data_t::is_same: objects '" << this->m_object_names[i]
                << "' are binary different";
            same = false;
            break;
        }
    }

    return same;
}

void rocsparse_reproducibility_t::test_data_t::reset()
{
    for(uint32_t i = 0; i < this->m_num_objects; ++i)
    {
        free(this->m_objects[i]);
        this->m_objects[i]         = nullptr;
        this->m_object_numbytes[i] = 0;
    }
    this->m_num_objects = 0;
}

uint32_t rocsparse_reproducibility_t::test_data_t::get_num_objects() const
{
    return this->m_num_objects;
}

const char* rocsparse_reproducibility_t::test_data_t::get_object_name(uint32_t object_index) const
{
    return this->m_object_names[object_index].c_str();
}

const void* rocsparse_reproducibility_t::test_data_t::get_object(uint32_t object_index) const
{
    return this->m_objects[object_index];
}

size_t rocsparse_reproducibility_t::test_data_t::get_object_numbytes(uint32_t object_index) const
{
    return this->m_object_numbytes[object_index];
}

void* rocsparse_reproducibility_t::test_data_t::add(const std::string& name, size_t numbytes)
{

    if(this->m_num_objects == s_maxsize)
    {
        std::cerr << "max 32 objects " << std::endl;
        exit(1);
    }

    void* p                                      = malloc(numbytes);
    this->m_object_numbytes[this->m_num_objects] = numbytes;
    this->m_objects[this->m_num_objects]         = p;
    this->m_object_names[this->m_num_objects]    = name;
    ++this->m_num_objects;
    return p;
}
