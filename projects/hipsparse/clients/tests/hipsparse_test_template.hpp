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

#include "hipsparse_test_call.hpp"
#include "hipsparse_test_check.hpp"
#include "hipsparse_test_dispatch.hpp"
#include "hipsparse_test_functors.hpp"
#include "hipsparse_test_traits.hpp"

namespace
{
    template <hipsparse_test_enum::value_type ROUTINE>
    struct hipsparse_test_template
    {
    private:
        using call_t     = temp::hipsparse_test_call<ROUTINE>;
        using traits_t   = hipsparse_test_traits<ROUTINE>;
        using functors_t = hipsparse_test_functors<ROUTINE>;
        using dispatch_t = hipsparse_test_dispatch<traits_t::s_dispatch>;

    public:
        template <typename... P>
        struct test_call_proxy
        {
            explicit operator bool()
            {
                return true;
            }
            void operator()(const Arguments& arg)
            {
                const char* name_ROUTINE = hipsparse_test_enum::to_string(ROUTINE);
                if(!strcmp(arg.function, name_ROUTINE))
                {
                    call_t::template testing<P...>(arg);
                }
                else
                {
                    std::string s(name_ROUTINE);
                    s += "_bad_arg";
                    if(!strcmp(arg.function, s.c_str()))
                    {
                        call_t::template testing_bad_arg<P...>(arg);
                    }
                    else
                    {
                        FAIL() << "Internal error: Test called with unknown function: "
                               << arg.function;
                    }
                }
            }
        };

        template <typename PROXY, template <typename...> class PROXY_CALL>
        struct test_proxy : HipSPARSE_Test<PROXY, PROXY_CALL>
        {
            using definition = HipSPARSE_Test<PROXY, PROXY_CALL>;
            static bool type_filter(const Arguments& arg)
            {
                return dispatch_t::template dispatch<definition::template type_filter_functor>(arg);
            }

            static bool function_filter(const Arguments& arg)
            {
                const char* name = hipsparse_test_enum::to_string(ROUTINE);
                std::string s(name);
                s += "_bad_arg";
                return !strcmp(arg.function, name) || !strcmp(arg.function, s.c_str());
            }

            static std::string name_suffix(const Arguments& arg)
            {
                std::ostringstream s;
                switch(traits_t::s_dispatch)
                {
                case hipsparse_test_dispatch_enum::t:
                {
                    s << hipsparse_datatype2string(arg.compute_type);
                    break;
                }

                case hipsparse_test_dispatch_enum::it:
                {
                    s << hipsparse_indextype2string(arg.index_type_I) << '_'
                      << hipsparse_datatype2string(arg.compute_type);
                    break;
                }
                case hipsparse_test_dispatch_enum::ijt:
                {
                    s << hipsparse_indextype2string(arg.index_type_I) << '_'
                      << hipsparse_indextype2string(arg.index_type_J) << '_'
                      << hipsparse_datatype2string(arg.compute_type);
                    break;
                }
                }

                // Check if this is bad_arg
                {
                    const char* name = hipsparse_test_enum::to_string(ROUTINE);
                    std::string s1(name);
                    s1 += "_bad_arg";
                    if(!strcmp(arg.function, s1.c_str()))
                    {
                        s << "_bad_arg";
                    }
                    else
                    {
                        const std::string suffix = functors_t::name_suffix(arg);
                        if(suffix.size() > 0)
                        {
                            s << '_' << suffix;
                        }
                    }
                }
                return HipSPARSE_TestName<PROXY>{} << s.str();
            }
        };
    };

    template <hipsparse_test_enum::value_type ROUTINE>
    struct hipsparse_test_t_template
    {
        using check_t = hipsparse_test_check<ROUTINE>;

        template <typename T, typename = void>
        struct test_call : hipsparse_test_invalid
        {
        };

        template <typename T>
        struct test_call<T, typename std::enable_if<check_t::template is_valid_type<T>()>::type>
            : hipsparse_test_template<ROUTINE>::template test_call_proxy<T>
        {
        };

        struct test : hipsparse_test_template<ROUTINE>::template test_proxy<test, test_call>
        {
        };
    };

    template <hipsparse_test_enum::value_type ROUTINE>
    struct hipsparse_test_it_template
    {
        using check_t = hipsparse_test_check<ROUTINE>;

        template <typename T, typename I = int32_t, typename = void>
        struct test_call : hipsparse_test_invalid
        {
        };

        template <typename I, typename T>
        struct test_call<I,
                         T,
                         typename std::enable_if<check_t::template is_valid_type<I, T>()>::type>
            : hipsparse_test_template<ROUTINE>::template test_call_proxy<I, T>
        {
        };

        struct test : hipsparse_test_template<ROUTINE>::template test_proxy<test, test_call>
        {
        };
    };

    template <hipsparse_test_enum::value_type ROUTINE>
    struct hipsparse_test_ijt_template
    {
        using check_t = hipsparse_test_check<ROUTINE>;

        template <typename T, typename I = int32_t, typename J = int32_t, typename = void>
        struct test_call : hipsparse_test_invalid
        {
        };

        template <typename I, typename J, typename T>
        struct test_call<I,
                         J,
                         T,
                         typename std::enable_if<check_t::template is_valid_type<I, J, T>()>::type>
            : hipsparse_test_template<ROUTINE>::template test_call_proxy<I, J, T>
        {
        };

        struct test : hipsparse_test_template<ROUTINE>::template test_proxy<test, test_call>
        {
        };
    };
}