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

///
/// @brief Definition of a utility struct to grab environment variables
/// for the rocsparse clients.
//
/// The corresponding environment variable is the literal enum string
/// with the prefix ROCSPARSE_CLIENTS_.
/// Example: rocsparse_clients_envariables::VERBOSE will have a one to one correspondance with the environment variable
/// ROCSPARSE_CLIENTS_VERBOSE.
/// Obviously it loads environment variables at run time.
///
struct rocsparse_clients_envariables
{

    ///
    /// @brief Enumerate Boolean environment variables.
    ///
    typedef enum var_bool_ : int32_t
    {
        VERBOSE,
        TEST_DEBUG_ARGUMENTS
    } var_bool;

    static constexpr var_bool s_var_bool_all[] = {VERBOSE, TEST_DEBUG_ARGUMENTS};

    ///
    /// @brief Return value of a Boolean variable.
    ///
    static bool get(var_bool v);

    ///
    /// @brief Set value of a Boolean variable.
    ///
    static void set(var_bool v, bool value);

    ///
    /// @brief Is the Boolean enviromnent variable defined ?
    ///
    static bool is_defined(var_bool v);

    ///
    /// @brief Return the name of a Boolean variable.
    ///
    static const char* get_name(var_bool v);

    ///
    /// @brief Return the description of a Boolean variable.
    ///
    static const char* get_description(var_bool v);

    ///
    /// @brief Enumerate string environment variables.
    ///
    typedef enum var_string_ : int32_t
    {
        MATRICES_DIR,
        TEST_DATA_DIR
    } var_string;

    static constexpr var_string s_var_string_all[2] = {MATRICES_DIR, TEST_DATA_DIR};

    ///
    /// @brief Return value of a string variable.
    ///
    static const char* get(var_string v);

    ///
    /// @brief Set value of a string variable.
    ///
    static void set(var_string v, const char* value);

    ///
    /// @brief Return the name of a string variable.
    ///
    static const char* get_name(var_string v);

    ///
    /// @brief Return the description of a string variable.
    ///
    static const char* get_description(var_string v);

    ///
    /// @brief Is the string enviromnent variable defined ?
    ///
    static bool is_defined(var_string v);
};
