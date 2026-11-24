/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

// Helpers shared by the host-area tests in rocsparse-unit-test-device.
#pragma once

#include "unit_test_utils.hpp"

#include "rocsparse_debug.hpp"

#include <filesystem>
#include <string>
#include <system_error>

namespace rocsparse_ut
{
    // Saves every debug switch and restores each one exactly on destruction.
    // The public enable/disable calls are coupled (disable_debug also clears
    // arguments, verbose and warnings), so the restore uses the raw setters.
    class debug_state_guard
    {
    public:
        debug_state_guard()
            : m_debug(var().get_debug())
            , m_verbose(var().get_debug_verbose())
            , m_warnings(var().get_debug_warnings())
            , m_arguments(var().get_debug_arguments())
            , m_arguments_verbose(var().get_debug_arguments_verbose())
            , m_force_host_assert(var().get_debug_force_host_assert())
            , m_kernel_launch(var().get_debug_kernel_launch())
        {
        }

        ~debug_state_guard()
        {
            var().set_debug(m_debug);
            var().set_debug_verbose(m_verbose);
            var().set_debug_warnings(m_warnings);
            var().set_debug_arguments(m_arguments);
            var().set_debug_arguments_verbose(m_arguments_verbose);
            var().set_debug_force_host_assert(m_force_host_assert);
            var().set_debug_kernel_launch(m_kernel_launch);
        }

        debug_state_guard(const debug_state_guard&) = delete;
        debug_state_guard& operator=(const debug_state_guard&) = delete;

    private:
        static rocsparse::debug_variables_st& var()
        {
            return rocsparse_debug_variables;
        }

        bool m_debug;
        bool m_verbose;
        bool m_warnings;
        bool m_arguments;
        bool m_arguments_verbose;
        bool m_force_host_assert;
        bool m_kernel_launch;
    };

    // Owns a descriptor or handle and destroys it when the test ends, even
    // after a failed ASSERT. `get` is filled in by the rocsparse_create_* call.
    template <typename D, auto Destroy>
    struct Owned
    {
        D get = nullptr;

        Owned()             = default;
        Owned(const Owned&) = delete;
        Owned& operator=(const Owned&) = delete;

        ~Owned()
        {
            if(get != nullptr)
            {
                EXPECT_EQ(Destroy(get), rocsparse_status_success);
            }
        }
    };

    // A temp_file_path that is removed when the test ends.
    struct scoped_temp_file
    {
        const std::string path;

        explicit scoped_temp_file(const char* name)
            : path(temp_file_path(name))
        {
        }
        ~scoped_temp_file()
        {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }

        scoped_temp_file(const scoped_temp_file&) = delete;
        scoped_temp_file& operator=(const scoped_temp_file&) = delete;
    };
}
