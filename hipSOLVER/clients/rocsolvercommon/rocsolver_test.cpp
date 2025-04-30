/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************ */

#include <cstdlib>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include "rocsolver_test.hpp"

fs::path get_sparse_data_dir()
{
    // first check an environment variable
    if(const char* datadir = std::getenv("HIPSOLVER_TEST_DATA"))
        return fs::path{datadir};

    std::vector<std::string> paths_considered;

    // check relative to the running executable
    fs::path              exe_path   = fs::path(hipsolver_exepath());
    std::vector<fs::path> candidates = {"../share/hipsolver/test", "sparsedata"};
    for(const fs::path& candidate : candidates)
    {
        fs::path        exe_relative_path = exe_path / candidate;
        std::error_code err;
        fs::path        canonical_path = fs::canonical(exe_relative_path, err);
        if(!err)
            return canonical_path;
        paths_considered.push_back(exe_relative_path.string());
    }

    std::ostringstream oss;
    oss << "Warning: default sparse data directories not found. "
           "Defaulting to current working directory.\nExecutable location: "
        << exe_path.string() << "\nPaths considered:\n";
    for(const std::string& path : paths_considered)
        oss << path << "\n";
    std::cerr << oss.str();

    return fs::current_path();
}
