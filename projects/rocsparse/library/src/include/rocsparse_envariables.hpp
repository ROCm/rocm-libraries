/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

namespace rocsparse
{
    template <std::size_t N, typename T>
    inline constexpr std::size_t countof(T (&)[N])
    {
        return N;
    }

    //
    // Definition of a utility class to grab environment variables
    // for the rocsparse library.
    //
    // The corresponding environment variable is the literal enum string
    // with the prefix ROCSPARSE_.
    // Example: envariables::VERBOSE will have a one to one correspondance with the environment variable
    // ROCSPARSE_VERBOSE.
    // Obviously it loads environment variables at run time.
    //
    class envariables
    {
    public:
#define ROCSPARSE_FOREACH_ENVARIABLES   \
    ENVARIABLE(DEBUG)                   \
    ENVARIABLE(DEBUG_ARGUMENTS)         \
    ENVARIABLE(DEBUG_ARGUMENTS_VERBOSE) \
    ENVARIABLE(DEBUG_KERNEL_LAUNCH)     \
    ENVARIABLE(DEBUG_VERBOSE)           \
    ENVARIABLE(VERBOSE)                 \
    ENVARIABLE(MEMSTAT)                 \
    ENVARIABLE(MEMSTAT_FORCE_MANAGED)   \
    ENVARIABLE(DEBUG_FORCE_HOST_ASSERT) \
    ENVARIABLE(MEMSTAT_GUARDS)          \
    ENVARIABLE(ROCTX)                   \
    ENVARIABLE(DEBUG_WARNINGS)

        //
        // Specification of the enum and the array of all values.
        //
#define ENVARIABLE(x_) x_,

        typedef enum bool_var_ : int32_t
        {
            ROCSPARSE_FOREACH_ENVARIABLES
        } bool_var;
        static constexpr bool_var all[] = {ROCSPARSE_FOREACH_ENVARIABLES};

#undef ENVARIABLE

        //
        // Specification of names.
        //
#define ENVARIABLE(x_) "ROCSPARSE_" #x_,
        static constexpr const char* names[] = {ROCSPARSE_FOREACH_ENVARIABLES};
#undef ENVARIABLE

        //
        // Number of values.
        //
        static constexpr size_t size          = countof(all);
        static constexpr size_t bool_var_size = size;

        //
        // \brief Return value of a Boolean variable.
        //
        inline bool get(bool_var v) const
        {
            return this->m_bool_var[v];
        };

        //
        // Return the unique instance.
        //
        static envariables& Instance();

    private:
        envariables();
        ~envariables()                  = default;
        envariables(const envariables&) = delete;

        envariables& operator=(const envariables&) = delete;
        bool         m_bool_var[bool_var_size]{};
    };
}

#define ROCSPARSE_ENVARIABLES rocsparse::envariables::Instance()
