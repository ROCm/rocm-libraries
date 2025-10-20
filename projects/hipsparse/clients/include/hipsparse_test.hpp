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
#ifndef HIPSPARSE_TEST_HPP
#define HIPSPARSE_TEST_HPP

#include "hipsparse_arguments.hpp"
#include "hipsparse_test_cleanup.hpp"

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef GOOGLE_TEST
#include <gtest/gtest.h>

// The tests are instantiated by filtering through the HipSPARSE_Data stream
// The filter is by category and by the type_filter() and function_filter()
// functions in the testclass
#define INSTANTIATE_TEST_CATEGORY(testclass, categ0ry)                         \
    INSTANTIATE_TEST_SUITE_P(                                                  \
        categ0ry,                                                              \
        testclass,                                                             \
        testing::ValuesIn(HipSPARSE_TestData::begin([](const Arguments& arg) { \
                              return !strcmp(arg.category, #categ0ry)          \
                                     && testclass::type_filter(arg)            \
                                     && testclass::function_filter(arg);       \
                          }),                                                  \
                          HipSPARSE_TestData::end()),                          \
        testclass::PrintToStringParamName());

#if defined(GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST)
#define HIPSPARSE_ALLOW_UNINSTANTIATED_GTEST(testclass) \
    GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(testclass);
#else
#define HIPSPARSE_ALLOW_UNINSTANTIATED_GTEST(testclass)
#endif

// Instantiate all test categories
#define INSTANTIATE_TEST_CATEGORIES(testclass)        \
    HIPSPARSE_ALLOW_UNINSTANTIATED_GTEST(testclass)   \
    INSTANTIATE_TEST_CATEGORY(testclass, quick)       \
    INSTANTIATE_TEST_CATEGORY(testclass, pre_checkin) \
    INSTANTIATE_TEST_CATEGORY(testclass, nightly)     \
    INSTANTIATE_TEST_CATEGORY(testclass, stress)      \
    INSTANTIATE_TEST_CATEGORY(testclass, known_bug)

/* ============================================================================================ */
/*! \brief  Normalized test name to conform to Google Tests */
// Template parameter is used to generate multiple instantiations
template <typename>
class HipSPARSE_TestName
{
    std::ostringstream str;

    static auto& get_table()
    {
        // Placed inside function to avoid dependency on initialization order
        static std::unordered_map<std::string, size_t>* table = test_cleanup::allocate(&table);
        return *table;
    }

public:
    // Convert stream to normalized Google Test name
    // rvalue reference qualified so that it can only be called once
    // The name should only be generated once before the stream is destroyed
    operator std::string() &&
    {
        // This table is private to each instantation of HipSPARSE_TestName
        auto&       table = get_table();
        std::string name(str.str());

        if(name == "")
            name = "1";

        // Warn about unset letter parameters
        if(name.find('*') != name.npos)
            fputs("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                  "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                  "Warning: Character * found in name."
                  " This means a required letter parameter\n"
                  "(e.g., transA, diag, etc.) has not been set in the YAML file."
                  " Check the YAML file.\n"
                  "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                  "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",
                  stderr);

        // Replace non-alphanumeric characters with letters
        std::replace(name.begin(), name.end(), '-', 'n'); // minus
        std::replace(name.begin(), name.end(), '.', 'p'); // decimal point

        // Complex (A,B) is replaced with ArBi
        name.erase(std::remove(name.begin(), name.end(), '('), name.end());
        std::replace(name.begin(), name.end(), ',', 'r');
        std::replace(name.begin(), name.end(), ')', 'i');

        // If parameters are repeated, append an incrementing suffix
        auto p = table.find(name);
        if(p != table.end())
            name += "_t" + std::to_string(++p->second);
        else
            table[name] = 1;

        return name;
    }

    // Stream output operations
    template <typename U> // Lvalue LHS
    friend HipSPARSE_TestName& operator<<(HipSPARSE_TestName& name, U&& obj)
    {
        name.str << std::forward<U>(obj);
        return name;
    }

    template <typename U> // Rvalue LHS
    friend HipSPARSE_TestName&& operator<<(HipSPARSE_TestName&& name, U&& obj)
    {
        name.str << std::forward<U>(obj);
        return std::move(name);
    }

    HipSPARSE_TestName()                          = default;
    HipSPARSE_TestName(const HipSPARSE_TestName&) = delete;
    HipSPARSE_TestName& operator=(const HipSPARSE_TestName&) = delete;
};

// ----------------------------------------------------------------------------
// HipSPARSE_Test base class. All non-legacy hipSPARSE Google tests derive from it.
// It defines a type_filter_functor() and a PrintToStringParamName class
// which calls name_suffix() in the derived class to form the test name suffix.
// ----------------------------------------------------------------------------
template <typename TEST, template <typename...> class FILTER>
class HipSPARSE_Test : public testing::TestWithParam<Arguments>
{
protected:
    // This template functor returns true if the type arguments are valid.
    // It converts a FILTER specialization to bool to test type matching.
    template <typename... T>
    struct type_filter_functor
    {
        bool operator()(const Arguments&)
        {
            return static_cast<bool>(FILTER<T...>{});
        }
    };

public:
    // Wrapper functor class which calls name_suffix()
    struct PrintToStringParamName
    {
        std::string operator()(const testing::TestParamInfo<Arguments>& info) const
        {
            return TEST::name_suffix(info.param);
        }
    };
};

#endif // GOOGLE_TEST

// ----------------------------------------------------------------------------
// Error case which returns false when converted to bool. A void specialization
// of the FILTER class template above, should be derived from this class, in
// order to indicate that the type combination is invalid.
// ----------------------------------------------------------------------------
struct hipsparse_test_invalid
{
    // Return false to indicate the type combination is invalid, for filtering
    explicit operator bool()
    {
        return false;
    }

    // If this specialization is actually called, print fatal error message
    void operator()(const Arguments&)
    {
        static constexpr char msg[] = "Internal error: Test called with invalid types\n";

#ifdef GOOGLE_TEST
        FAIL() << msg;
#else
        fputs(msg, stderr);
        exit(EXIT_FAILURE);
#endif
    }
};

#endif // HIPSPARSE_TEST_HPP