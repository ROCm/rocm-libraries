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
        this->m_is_defined = rocsparse_clients_envariables::is_defined(
            rocsparse_clients_envariables::MATRICES_DIR);
        if(this->m_is_defined)
        {
            this->m_path
                = rocsparse_clients_envariables::get(rocsparse_clients_envariables::MATRICES_DIR);
            {
                const size_t size = this->m_path.size();
                if((size > 0) && (this->m_path[size - 1] != '/'))
                    this->m_path += "/";
            }
        }

        this->m_default_path = rocsparse_exepath();
        {
            const size_t size = this->m_default_path.size();
            if((size > 0) && (this->m_default_path[size - 1] != '/'))
                this->m_default_path += "/";
        }
        this->m_default_path += "../matrices/";
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
        self.m_path                = p;
        const size_t size          = p.size();
        if((size > 0) && (p[size - 1] != '/'))
            self.m_path += "/";
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
