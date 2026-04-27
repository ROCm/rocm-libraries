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
#pragma once
#ifdef GOOGLE_TEST
#include "rocsparse_clients_test_memory_debug_synchronicity.hpp"
#include <map>
namespace rocsparse_clients_test
{
    struct memory_debug_t
    {
    private:
        bool        m_enabled{};
        std::string m_filename{};
        memory_debug_t() = default;
        bool m_non_permissive{};

    public:
        static std::map<std::string, memory_debug_synchronicity_info_t> s_map;
        const std::string&                                              get_filename() const;
        memory_debug_synchronicity_info_t& get_memory_debug_synchronicity_info(const char* name);
        bool                               get_non_permissive() const;
        void                               set_non_permissive(bool);
        rocsparse_status check(rocsparse_handle, bool non_permissive, std::ostream&) const;

        void                   report(rocsparse_handle, std::ostream&) const;
        static memory_debug_t& instance();
        bool                   enabled() const;
        void                   enable();
        void                   disable();
        void                   set_sync_report_filename(const char* filename);
    };
}

namespace rocsparse_clients_test
{
    void memory_debug_check_synchronicity(rocsparse_handle handle, const char* name);
}

#endif
