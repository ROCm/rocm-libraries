/* **************************************************************************
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include <stdlib.h>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include "common/misc/client_environment_helpers.hpp"

bool set_environment_variable(const char* name, const char* value)
{
#ifdef _WIN32
    return _putenv_s(name, value) == 0;
#else
    return setenv(name, value, 1) == 0;
#endif
}

bool unset_environment_variable(const char* name)
{
#ifdef _WIN32
    return _putenv_s(name, "") == 0;
#else
    return unsetenv(name) == 0;
#endif
}

scoped_envvar::scoped_envvar(const char* name, const char* value)
    : m_name(name)
{
    if(const char* old_value = std::getenv(name))
    {
        m_old_value.emplace(old_value);
    }
    if(!set_environment_variable(name, value))
        throw environment_error(fmt::format("failed to set {:s}={:s}", name, value));
}

scoped_envvar::~scoped_envvar()
{
    if(m_old_value)
        set_environment_variable(m_name.c_str(), m_old_value->c_str());
    else
        unset_environment_variable(m_name.c_str());
}
