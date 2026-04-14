/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
#include <nanobind/nanobind.h>

namespace nb = nanobind;

void init_base(nb::module_ m);
void init_containers(nb::module_ m);
void init_label(nb::module_ m);
void init_enum(nb::module_ m);
void init_inst(nb::module_ m);
void init_code(nb::module_ m);
void init_count(nb::module_ m);
void init_pass(nb::module_ m);
void init_macro(nb::module_ m);
void init_func(nb::module_ m);
void init_register(nb::module_ m);

NB_MODULE(rocisa, m)
{
    m.doc() = "Module rocisa.";
    init_base(m);
    init_containers(m);
    init_label(m);
    init_enum(m);
    init_inst(m);
    init_code(m);
    init_count(m);
    init_pass(m);
    init_macro(m);
    init_func(m);
    init_register(m);
}
