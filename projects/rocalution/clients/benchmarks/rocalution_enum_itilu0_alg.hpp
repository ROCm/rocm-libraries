/*! \file */
/* ************************************************************************
* Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "utility.hpp"
#include <cstring>

//
// List the enumeration values.
//

// clang-format off
#define ROCALUTION_ENUM_ITILU0_ALG_TRANSFORM_EACH		  \
  ROCALUTION_ENUM_ITILU0_ALG_TRANSFORM(Default)		  \
  ROCALUTION_ENUM_ITILU0_ALG_TRANSFORM(AsyncInPlace)	  \
  ROCALUTION_ENUM_ITILU0_ALG_TRANSFORM(AsyncSplit)	  \
  ROCALUTION_ENUM_ITILU0_ALG_TRANSFORM(SyncSplit)		  \
  ROCALUTION_ENUM_ITILU0_ALG_TRANSFORM(SyncSplitFusion)
// clang-format on

struct rocalution_enum_itilu0_alg
{
public:
#define ROCALUTION_ENUM_ITILU0_ALG_TRANSFORM(x_) x_,
    typedef enum rocalution_enum_itilu0_alg__ : int
    {
        ROCALUTION_ENUM_ITILU0_ALG_TRANSFORM_EACH
    } value_type;
    static constexpr value_type all[] = {ROCALUTION_ENUM_ITILU0_ALG_TRANSFORM_EACH};
#undef ROCALUTION_ENUM_ITILU0_ALG_TRANSFORM
    static constexpr std::size_t size = countof(all);
    value_type                   value{};

private:
#define ROCALUTION_ENUM_ITILU0_ALG_TRANSFORM(x_) #x_,
    static constexpr const char* names[size]{ROCALUTION_ENUM_ITILU0_ALG_TRANSFORM_EACH};
#undef ROCALUTION_ENUM_ITILU0_ALG_TRANSFORM
public:
    operator value_type() const
    {
        return this->value;
    };
    rocalution_enum_itilu0_alg();
    rocalution_enum_itilu0_alg& operator()(const char* function);
    rocalution_enum_itilu0_alg(const char* function);
    const char*               to_string() const;
    bool                      is_invalid() const;
    static inline const char* to_string(rocalution_enum_itilu0_alg::value_type value)
    {
        //
        // switch for checking inconsistency.
        //
        switch(value)
        {
            ///
#define ROCALUTION_ENUM_ITILU0_ALG_TRANSFORM(x_) \
    case x_:                                     \
    {                                            \
        if(strcmp(#x_, names[value]))            \
            return nullptr;                      \
        break;                                   \
    }

            ROCALUTION_ENUM_ITILU0_ALG_TRANSFORM_EACH;

#undef ROCALUTION_ENUM_ITILU0_ALG_TRANSFORM
            ///
        }

        return names[value];
    }
};
