/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/py.hpp>
#include <migraphx/dynamic_loader.hpp>
#include <migraphx/process.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static std::vector<fs::path> find_available_python_versions()
{
    std::vector<fs::path> result;
    auto path = dynamic_loader::path(&load_py).parent_path();
    for(const auto& entry : fs::directory_iterator{path})
    {
        auto p = entry.path();
        if(not fs::is_regular_file(p))
            continue;
        if(not contains(p.stem().string(), "migraphx_py_"))
            continue;
        result.push_back(p);
    }
    std::sort(result.begin(), result.end(), std::greater<>{});
    return result;
}

static dynamic_loader load_py_lib()
{
    auto libs = find_available_python_versions();
    for(const auto& lib : libs)
    {
        auto result = dynamic_loader::try_load(lib);
        if(result.has_value())
            return *result;
    }
    MIGRAPHX_THROW("Cant find a viable version of python");
}

static dynamic_loader py_lib()
{
    static dynamic_loader lib = load_py_lib();
    return lib;
}

MIGRAPHX_PY_EXPORT program load_py(const std::string& filename)
{
    static auto f = py_lib().get_function<program(const std::string&)>("migraphx_load_py");
    return f(filename);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
