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
#include <migraphx/config.hpp>
#include <migraphx/program.hpp>
#include <migraphx/dynamic_loader.hpp>
#include <migraphx/file_buffer.hpp>
#include <pybind11/embed.h>

namespace py = pybind11;

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif
// extern "C" is used to disable name mangling, but the function will still be called from C++
extern "C" program migraphx_load_py(const std::string& filename);
#ifdef __clang__
#pragma clang diagnostic pop
#endif

const std::string& python_path()
{
    static const auto path = dynamic_loader::path(&migraphx_load_py).parent_path().string();
    return path;
}

static py::dict run_file(const std::string& file)
{
    py::object scope = py::module_::import("__main__").attr("__dict__");
    std::string buffer;
    buffer.append("import sys\n");
    buffer.append("sys.path.insert(0, '" + python_path() + "')\n");
    buffer.append("import migraphx\n");
    buffer.append(read_string(file));
    py::exec(buffer, scope);
    return scope.cast<py::dict>();
}

extern "C" program migraphx_load_py(const std::string& filename)
{
    py::scoped_interpreter guard{};
    py::dict vars = run_file(filename);
    auto it       = std::find_if(vars.begin(), vars.end(), [](const auto& p) {
        return py::isinstance<migraphx::program>(p.second);
    });
    if(it == vars.end())
        MIGRAPHX_THROW("No program variable found");
    return it->second.cast<migraphx::program>();
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
