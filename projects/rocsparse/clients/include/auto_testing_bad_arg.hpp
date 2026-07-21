/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

/*! \file
 *  \brief auto_testing_bad_arg.hpp provides common testing utilities
 */

#pragma once

#include "auto_testing_bad_arg_get_status.hpp"
#include "auto_testing_bad_arg_set_invalid.hpp"
#include "rocsparse_clients_envariables.hpp"
#include "rocsparse_test.hpp"
#include "test_check.hpp"
#include <fstream>
#include <hip/hip_runtime_api.h>
#include <sstream>
#include <vector>

extern "C" {
rocsparse_status rocsparse_argdescr_create(void** argdescr);
rocsparse_status rocsparse_argdescr_free(void* argdescr);
rocsparse_status rocsparse_argdescr_get_msg(const void* argdescr, const char**);
rocsparse_status rocsparse_argdescr_get_status(const void* argdescr, rocsparse_status*);
rocsparse_status rocsparse_argdescr_get_index(const void* argdescr, int*);
rocsparse_status rocsparse_argdescr_get_name(const void* argdescr, const char**);
rocsparse_status rocsparse_argdescr_get_function_line(const void* argdescr, int*);
rocsparse_status rocsparse_argdescr_get_function_name(const void* argdescr, const char**);
}

template <typename... T>
struct auto_testing_bad_arg_t
{
    inline auto_testing_bad_arg_t(){};
    inline auto_testing_bad_arg_t(int current, int ith, rocsparse_status& status){};
};

template <typename T, typename... Rest>
struct auto_testing_bad_arg_t<T, Rest...>
{
    inline auto_testing_bad_arg_t(T first, Rest... rest)
        : first(first)
        , rest(rest...)
    {
    }

    inline auto_testing_bad_arg_t(int ith, rocsparse_status& status, T& first, Rest&... rest)
        : auto_testing_bad_arg_t(0, ith, status, first, rest...)
    {
    }

    inline auto_testing_bad_arg_t(
        int current, int ith, rocsparse_status& status, T& first, Rest&... rest)
        : first(first)
        , rest(current + 1, ith, status, rest...)
    {
        if(current == ith)
        {
            status = auto_testing_bad_arg_get_status<T>(first);
            auto_testing_bad_arg_set_invalid(this->first);
        }
    }

    T                               first;
    auto_testing_bad_arg_t<Rest...> rest;
};

template <typename C, typename T>
inline void auto_testing_bad_arg_copy(const C& data, T& t)
{
    t = data.first;
}

template <typename C, typename T, typename... Ts>
inline void auto_testing_bad_arg_copy(const C& data, T& t, Ts&... ts)
{
    t = data.first;
    auto_testing_bad_arg_copy(data.rest, ts...);
}

template <typename T>
inline void auto_testing_bad_arg_print(T& t)
{
    std::cout << " " << t << "," << std::endl;
}

template <typename T, typename... Ts>
inline void auto_testing_bad_arg_print(T& t, Ts&... ts)
{
    std::cout << " " << t << "," << std::endl;
    auto_testing_bad_arg_print(ts...);
}

static constexpr uint32_t stringargs_count(const char* str, uint32_t pos = 0, uint32_t count = 0)
{
    if(str[pos] == '\0')
    {
        return ((pos == 0) ? 0 : count + 1);
    }
    else
    {
        return ((str[pos] == ',') ? stringargs_count(str, pos + 1, count + 1)
                                  : stringargs_count(str, pos + 1, count));
    }
}

static constexpr uint32_t stringargs_trim(const char* str, uint32_t pos)
{
    if(str[pos] == '\0' || str[pos] == ' ' || str[pos] == '\t' || str[pos] == ',')
    {
        return pos;
    }
    else
    {
        return stringargs_trim(str, pos + 1);
    }
}

constexpr uint32_t
    stringargs_to_lst(char str[], uint32_t pos, const char* strlst[], uint32_t strlst_pos)
{
    if(str[pos] == '\0')
    {
        return pos;
    }
    else
    {
        if(str[pos] == ' ' || str[pos] == '\t' || str[pos] == ',')
        {
            str[pos] = '\0';
            return stringargs_to_lst(str, pos + 1, strlst, strlst_pos);
        }
        else
        {
            strlst[strlst_pos] = &str[pos];
            pos                = stringargs_trim(str, pos);
            return stringargs_to_lst(str, pos, strlst, strlst_pos + 1);
        }
    }
}

#define LIST_ARG_STRINGS(...)                                                      \
    char                      stringargs[] = #__VA_ARGS__;                         \
    static constexpr uint32_t stringargs_c = stringargs_count(#__VA_ARGS__, 0, 0); \
    const char*               stringargs_lst[stringargs_c];                        \
    stringargs_to_lst(stringargs, 0, stringargs_lst, 0)

struct rocsparse_local_argdescr
{
private:
    void* argdescr{};

public:
    rocsparse_local_argdescr()
    {
        if(rocsparse_clients_envariables::get(rocsparse_clients_envariables::TEST_DEBUG_ARGUMENTS))
        {
            rocsparse_argdescr_create(&argdescr);
        }
    }

    ~rocsparse_local_argdescr()
    {
        rocsparse_argdescr_free(argdescr);
    }

    // Allow rocsparse_local_mat_descr to be used anywhere rocsparse_mat_descr is expected
    operator void*&()
    {
        return this->argdescr;
    }
    // clang-format off
  operator void* const &() const
  {
    return this->argdescr;
  }
    // clang-format on
};

template <typename F, typename... Ts>
inline void auto_testing_bad_arg_excluding(F f, int n, const int* idx, const char** names, Ts... ts)
{
    //
    // Tell we are passing here to summarize routines that are not.
    //
    test_check::set_auto_testing_bad_arg();

    //
    // Create argument descriptpr.
    //
    rocsparse_local_argdescr argdescr;

    static constexpr int nargs = sizeof...(ts);
    for(int iarg = 0; iarg < nargs; ++iarg)
    {
        bool exclude = false;
        for(int i = 0; i < n; ++i)
        {
            if(idx[i] == iarg)
            {
                exclude = true;
                break;
            }
        }

        if(!exclude)
        {
            //
            //
            //
            auto_testing_bad_arg_t<Ts...> arguments(ts...);

            //
            //
            //
            rocsparse_status              status = rocsparse_status_success;
            auto_testing_bad_arg_t<Ts...> invalid_data(iarg, status, ts...);

            //
            //
            //
            auto_testing_bad_arg_copy(invalid_data, ts...);

            //
            //
            //
            const rocsparse_status status_from_routine = f(ts...);
            if(rocsparse_clients_envariables::get(
                   rocsparse_clients_envariables::TEST_DEBUG_ARGUMENTS))
            {
                //
                // Get the argument name.
                //
                const char* argname;
                EXPECT_ROCSPARSE_STATUS(rocsparse_argdescr_get_name(argdescr, &argname),
                                        rocsparse_status_success);

                //
                // If names do not fit.
                //
                const int cmp = strcmp(argname, names[iarg]);
                if(cmp)
                {
                    std::cout
                        << "auto testing bad arg failed on " //
                        << iarg //
                        << " 'th argument, '" //
                        << names[iarg] //
                        << "'" //
                        << std::endl //
                        << "   reason: argument names do not match, argument checking returns " //
                        << argname //
                        << std::endl;
#ifdef GOOGLE_TEST
                    EXPECT_EQ(cmp, 0);
#endif
                }

                //
                // Get the argument index.
                //
                int argidx;
                EXPECT_ROCSPARSE_STATUS(rocsparse_argdescr_get_index(argdescr, &argidx),
                                        rocsparse_status_success);
                //
                // If argument indices do not fit.
                //
                if(argidx != iarg)
                {
                    std::cout
                        << "auto testing bad arg failed on " //
                        << iarg //
                        << " 'th argument, '" //
                        << names[iarg] //
                        << "'" //
                        << std::endl //
                        << "   reason: argument indices do not match, argument checking returns " //
                        << argidx //
                        << std::endl;
#ifdef GOOGLE_TEST
                    EXPECT_EQ(argidx, iarg);
#endif
                }
            }

            //
            // if statuses do not fit.
            //
            if(status != status_from_routine)
            {
                std::cout << "auto testing bad arg failed on " //
                          << iarg //
                          << " 'th argument, '" //
                          << ((names != nullptr) ? names[iarg] : "") //
                          << "'" //
                          << std::endl
                          << "   reason: statuses do not match, argument checking returns "
                          << status_from_routine << ", but it should return " << status
                          << std::endl;
                auto_testing_bad_arg_print(ts...);
                EXPECT_ROCSPARSE_STATUS(status_from_routine, status);
            }

            //
            //
            //
            auto_testing_bad_arg_copy(arguments, ts...);
        }
    }
}

template <typename F, typename... Ts>
inline void auto_testing_bad_arg(F f, Ts... ts)
{
    //
    // Tell we are passing here to summarize routines that are not.
    //
    test_check::set_auto_testing_bad_arg();
    static constexpr int nargs = sizeof...(ts);
    for(int iarg = 0; iarg < nargs; ++iarg)
    {
        auto_testing_bad_arg_t<Ts...> arguments(ts...);

        {
            rocsparse_status              status;
            auto_testing_bad_arg_t<Ts...> invalid_data(iarg, status, ts...);
            auto_testing_bad_arg_copy(invalid_data, ts...);

            if(status != f(ts...))
            {
                std::cout << "auto testing bad arg failed on " << iarg << " 'th argument"
                          << std::endl;
                auto_testing_bad_arg_print(ts...);
                EXPECT_ROCSPARSE_STATUS(f(ts...), status);
            }
        }

        auto_testing_bad_arg_copy(arguments, ts...);
    }
}

template <typename F, typename... Ts>
inline void auto_testing_bad_arg(F f, int n, const int* idx, Ts... ts)
{
    //
    // Tell we are passing here to summarize routines that are not.
    //
    test_check::set_auto_testing_bad_arg();
    static constexpr int nargs = sizeof...(ts);
    for(int iarg = 0; iarg < nargs; ++iarg)
    {
        bool exclude = false;
        for(int i = 0; i < n; ++i)
        {
            if(idx[i] == iarg)
            {
                exclude = true;
                break;
            }
        }

        if(!exclude)
        {
            auto_testing_bad_arg_t<Ts...> arguments(ts...);

            {
                rocsparse_status              status = rocsparse_status_success;
                auto_testing_bad_arg_t<Ts...> invalid_data(iarg, status, ts...);
                auto_testing_bad_arg_copy(invalid_data, ts...);

                if(status != f(ts...))
                {
                    std::cout << "auto testing bad arg failed on " << iarg << " 'th argument"
                              << std::endl;
                    auto_testing_bad_arg_print(ts...);
                    EXPECT_ROCSPARSE_STATUS(f(ts...), status);
                }
            }

            auto_testing_bad_arg_copy(arguments, ts...);
        }
    }
}

#define bad_arg_analysis(f, ...)                                                        \
    do                                                                                  \
    {                                                                                   \
        if(rocsparse_clients_envariables::get(                                          \
               rocsparse_clients_envariables::TEST_DEBUG_ARGUMENTS))                    \
        {                                                                               \
            LIST_ARG_STRINGS(__VA_ARGS__);                                              \
            auto_testing_bad_arg_excluding(f, 0, nullptr, stringargs_lst, __VA_ARGS__); \
        }                                                                               \
        else                                                                            \
        {                                                                               \
            auto_testing_bad_arg(f, __VA_ARGS__);                                       \
        }                                                                               \
    } while(false)

#define select_bad_arg_analysis(f, n, idx, ...)                                     \
    do                                                                              \
    {                                                                               \
        if(rocsparse_clients_envariables::get(                                      \
               rocsparse_clients_envariables::TEST_DEBUG_ARGUMENTS))                \
        {                                                                           \
            LIST_ARG_STRINGS(__VA_ARGS__);                                          \
            auto_testing_bad_arg_excluding(f, n, idx, stringargs_lst, __VA_ARGS__); \
        }                                                                           \
        else                                                                        \
        {                                                                           \
            auto_testing_bad_arg(f, n, idx, __VA_ARGS__);                           \
        }                                                                           \
    } while(false)
