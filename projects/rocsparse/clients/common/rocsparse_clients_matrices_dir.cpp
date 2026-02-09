/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_clients_matrices_dir.hpp"
#include "rocsparse_clients_envariables.hpp"
#include "utility.hpp"

#include <cstdio>
#include <filesystem>

// Helper to ensure trailing separator using std::filesystem (portable for Windows/Linux)
static std::string ensure_trailing_separator(const std::string& path_str)
{
    namespace fs = std::filesystem;
    if(path_str.empty())
        return path_str;
    fs::path p(path_str);
    // Use make_preferred() to get platform-appropriate separators
    std::string result = p.make_preferred().string();
    if(!result.empty() && result.back() != fs::path::preferred_separator)
    {
        result += fs::path::preferred_separator;
    }
    return result;
}

//
//
//
struct clients_matrices_dir
{
private:
    std::string m_path{};
    std::string m_default_path{};
    bool        m_is_defined{};
    clients_matrices_dir()
    {
        namespace fs       = std::filesystem;
        this->m_is_defined = rocsparse_clients_envariables::is_defined(
            rocsparse_clients_envariables::MATRICES_DIR);
        if(this->m_is_defined)
        {
            this->m_path = ensure_trailing_separator(
                rocsparse_clients_envariables::get(rocsparse_clients_envariables::MATRICES_DIR));
        }

        // Compute default path by checking possible relative locations
        fs::path exe_path = rocsparse_exepath();

        static constexpr const char* possible_relative_paths[] = {
            // Development build: executable in build_dir/clients/staging, matrices in build_dir/clients/matrices
            "../matrices",
            // TheRock installation: executable in TheRock/bin, matrices in TheRock/clients/matrices
            "../clients/matrices",
        };

        for(const auto& rel_path : possible_relative_paths)
        {
            fs::path test_path = exe_path / rel_path;
            if(fs::exists(test_path))
            {
                this->m_default_path = test_path.string() + "/";
                break;
            }
        }

        // If no default path found and no environment variable set, print warning but don't throw.
        // The error will be handled later when the path is actually used.
        if(this->m_default_path.empty() && !this->m_is_defined)
        {
            std::cerr << "rocsparse: warning: could not find matrices directory. Please set "
                         "ROCSPARSE_CLIENTS_MATRICES_DIR "
                         "environment variable, or use the option --matrices-dir"
                      << std::endl;
            // Fall back to the old default path behavior to avoid breaking existing setups
            this->m_default_path = (exe_path / ".." / "matrices" / "").make_preferred().string();
        }
    }

public:
    static clients_matrices_dir& instance()
    {
        static clients_matrices_dir self;
        return self;
    }
    static const std::string& path(bool use_default)
    {
        const clients_matrices_dir& self = instance();
        if(self.m_is_defined)
        {
            return self.m_path;
        }
        else
        {
            if(use_default)
            {
                return self.m_default_path;
            }
            else
            {
                return self.m_path;
            }
        }
    }

    static const std::string& default_path()
    {
        const clients_matrices_dir& self = instance();
        return self.m_default_path;
    }

    static void set(const std::string& p)
    {
        clients_matrices_dir& self = instance();
        self.m_is_defined          = true;
        self.m_path                = ensure_trailing_separator(p);
    }
};

//
//
//
const char* rocsparse_clients_matrices_dir_get(bool use_default_path)
{
    return clients_matrices_dir::path(use_default_path).c_str();
}

//
//
//
void rocsparse_clients_matrices_dir_set(const char* path)
{
    clients_matrices_dir::set(std::string(path));
}
