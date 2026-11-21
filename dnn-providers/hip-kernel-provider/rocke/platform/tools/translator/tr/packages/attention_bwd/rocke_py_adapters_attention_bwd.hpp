// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * rocke_py_adapters_attention_bwd.hpp -- the adapters of the attention backward
 * package that need its own headers (the lane-map evaluator of
 * _attention_bwd_frag). The generic adapters are in rocke_py_adapters.hpp.
 *
 * PRIVATE: support for translated sources in this directory; not installed,
 * not API; EXPERIMENTAL. Package data of tools/translator (attention_bwd):
 * tr.regen copies it next to the sources that use it and never over an
 * existing copy.
 */
#ifndef ROCKE_ROCKE_PY_ADAPTERS_ATTENTION_BWD_HPP
#define ROCKE_ROCKE_PY_ADAPTERS_ATTENTION_BWD_HPP

#include <array>
#include <vector>

#include "rocke/error.hpp"
#include "rocke/helper_rocke.instances.common._attention_bwd_frag.h"
#include "rocke/ir.h"
#include "rocke_py.hpp"
#include "rocke_py_adapters.hpp"

namespace py_adapt
{

/* _attention_bwd_frag.eval_layout(layout): lanes[lane][slot] = (row, col). */
inline py::tuple<py::tuple<std::array<int64_t, 2>>> eval_layout(rocke_ir_builder_t* b,
                                                                const rocke_layout_map_t* m)
{
    const int lanes = m->wave_size;
    const int n = m->frag_len;
    std::vector<int> flat((size_t)lanes * (size_t)n * 2);
    if(!rocke_attn_bwd_eval_layout(b, m, flat.data(), (int)flat.size()))
        throw ckc::ValueError("eval_layout failed");
    py::tuple<py::tuple<std::array<int64_t, 2>>> out;
    for(int l = 0; l < lanes; ++l)
    {
        py::tuple<std::array<int64_t, 2>> lane;
        for(int s = 0; s < n; ++s)
        {
            const size_t k = ((size_t)l * (size_t)n + (size_t)s) * 2;
            lane.push_back({(int64_t)flat[k], (int64_t)flat[k + 1]});
        }
        out.push_back(lane);
    }
    return out;
}

/* eval_layout with no builder in scope: on a scratch builder of its own
 * (the lane map's integer ops are interpreted; nothing reaches a kernel). */
inline py::tuple<py::tuple<std::array<int64_t, 2>>> eval_layout_scratch(const rocke_layout_map_t* m)
{
    rocke_ir_builder_t s;
    if(rocke_ir_builder_init(&s, "rocke_eval_layout_scratch") != ROCKE_OK)
        throw ckc::OOMError("scratch builder init failed");
    try
    {
        auto r = eval_layout(&s, m);
        rocke_ir_builder_free(&s);
        return r;
    }
    catch(...)
    {
        rocke_ir_builder_free(&s);
        throw;
    }
}

} // namespace py_adapt

#endif // ROCKE_ROCKE_PY_ADAPTERS_ATTENTION_BWD_HPP
