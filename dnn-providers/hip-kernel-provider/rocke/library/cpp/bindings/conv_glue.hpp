// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * library/cpp/bindings/conv_glue.hpp -- the ConvProblem dict reader shared by
 * the binding translation units whose spec nests a ConvProblem the way
 * kernels/conv/engine_bridge.py's _conv_problem_to_dict writes it
 * (rocke_library_engine_conv.cpp, rocke_library_engine_img2col.cpp).
 */
#ifndef ROCKE_BINDINGS_CONV_GLUE_HPP
#define ROCKE_BINDINGS_CONV_GLUE_HPP

#include <pybind11/pybind11.h>

#include "family_glue.hpp"

extern "C" {
#include "rocke/helper_rocke.instances.common.conv_implicit_gemm.h" /* rocke_conv_problem_t */
}

namespace rocke_conv_bind
{

/* Overwrite the fields of `p` present in the "problem" sub-dict `d`; an absent
 * key keeps the value already in `p`. A key set to None is refused (dict_int
 * casts it to int): Python has no None default for these fields either, and a
 * None there fails its build. */
inline void fill_conv_problem(rocke_conv_problem_t* p, const pybind11::dict& d)
{
    using rocke_bind::dict_bool;
    using rocke_bind::dict_int;

    p->N = dict_int(d, "N", p->N);
    p->Hi = dict_int(d, "Hi", p->Hi);
    p->Wi = dict_int(d, "Wi", p->Wi);
    p->C = dict_int(d, "C", p->C);
    p->K = dict_int(d, "K", p->K);
    p->Y = dict_int(d, "Y", p->Y);
    p->X = dict_int(d, "X", p->X);
    p->sH = dict_int(d, "sH", p->sH);
    p->sW = dict_int(d, "sW", p->sW);
    p->pH = dict_int(d, "pH", p->pH);
    p->pW = dict_int(d, "pW", p->pW);
    p->dH = dict_int(d, "dH", p->dH);
    p->dW = dict_int(d, "dW", p->dW);
    p->groups = dict_int(d, "groups", p->groups);
    p->is_3d = dict_bool(d, "is_3d", p->is_3d);
    p->Di = dict_int(d, "Di", p->Di);
    p->Z = dict_int(d, "Z", p->Z);
    p->sD = dict_int(d, "sD", p->sD);
    p->pD = dict_int(d, "pD", p->pD);
    p->dD = dict_int(d, "dD", p->dD);
}

} // namespace rocke_conv_bind

#endif // ROCKE_BINDINGS_CONV_GLUE_HPP
