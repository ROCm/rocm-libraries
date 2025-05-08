/* ************************************************************************
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "envariables.h"
#include <iostream>
//
//
//
template <typename T>
static bool rocsparse_getenv(const char* name, T& val);
template <>
bool rocsparse_getenv<bool>(const char* name, bool& val)
{
    val                    = false;
    const char* getenv_str = getenv(name);
    if(getenv_str != nullptr)
    {
        auto getenv_int = atoi(getenv_str);
        if((getenv_int != 0) && (getenv_int != 1))
        {
            std::cerr << "rocsparse error, invalid environment variable " << name
                      << " must be 0 or 1." << std::endl;
            val = false;
            return false;
        }
        else
        {
            val = (getenv_int == 1);
            return true;
        }
    }
    return true;
}

constexpr rocsparse::envariables::bool_var rocsparse::envariables::all[];

rocsparse::envariables& rocsparse::envariables::Instance()
{
    static rocsparse::envariables instance;
    return instance;
}

rocsparse::envariables::envariables()
{
    //
    // Query variables.
    //
    for(auto tag : rocsparse::envariables::all)
    {
        switch(tag)
        {
#define ENVARIABLE(x_)                                                                          \
    case rocsparse::envariables::x_:                                                            \
    {                                                                                           \
        auto success                                                                            \
            = rocsparse_getenv("ROCSPARSE_" #x_, this->m_bool_var[rocsparse::envariables::x_]); \
        if(!success)                                                                            \
        {                                                                                       \
            std::cerr << "rocsparse_getenv failed " << std::endl;                               \
            exit(1);                                                                            \
        }                                                                                       \
        break;                                                                                  \
    }

            ROCSPARSE_FOREACH_ENVARIABLES;

#undef ENVARIABLE
        }
    }

    if(this->m_bool_var[rocsparse::envariables::VERBOSE])
    {
        for(auto tag : rocsparse::envariables::all)
        {
            switch(tag)
            {
#define ENVARIABLE(x_)                                                                        \
    case rocsparse::envariables::x_:                                                          \
    {                                                                                         \
        const bool v = this->m_bool_var[rocsparse::envariables::x_];                          \
        std::cout << ""                                                                       \
                  << "env variable ROCSPARSE_" #x_ << " : " << ((v) ? "enabled" : "disabled") \
                  << std::endl;                                                               \
        break;                                                                                \
    }

                ROCSPARSE_FOREACH_ENVARIABLES;

#undef ENVARIABLE
            }
        }
    }
}
