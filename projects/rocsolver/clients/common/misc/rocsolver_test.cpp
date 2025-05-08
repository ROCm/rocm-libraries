/* **************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstdlib>
#include <system_error>

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include "clients_utility.hpp"
#include "common/misc/rocsolver_test.hpp"

fs::path get_sparse_data_dir()
{
    // first check an environment variable
    if(const char* datadir = std::getenv("ROCSOLVER_TEST_DATA"))
        return fs::path{datadir};

    std::vector<std::string> considered;

    // check relative to the running executable
    fs::path exe_path = fs::path(rocsolver_exepath());
    std::vector<fs::path> candidates = {"../share/rocsolver/test", "sparsedata"};
    for(const fs::path& candidate : candidates)
    {
        std::error_code ec;
        fs::path exe_relative = fs::canonical(exe_path / candidate, ec);
        if(!ec)
            return exe_relative;
        considered.push_back(exe_relative.string());
    }

    fmt::print(stderr,
               "Warning: default sparse data directories not found. "
               "Defaulting to current working directory.\nExecutable location: {}\n"
               "Paths considered:\n{}\n",
               exe_path.string(), fmt::join(considered, "\n"));

    return fs::current_path();
}
