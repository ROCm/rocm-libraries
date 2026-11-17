// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * library/cpp/bindings/rocke_library_engine.cpp -- pybind11 module
 * `rocke_library_engine`: the C++ mirrors of the library-owned kernel families
 * (library/cpp), for the ROCKE_BACKEND=cpp / "both" paths of their bridges in
 * library/kernels. The engine's own families stay in `rocke_engine`; this
 * module links the same engine archive plus librocke_library.a and is built
 * beside it, so whatever puts `rocke_engine` on sys.path finds this one too.
 *
 * Entry points keep the `rocke_engine` naming: <family>_lower_llvm /
 * <family>_serialize_ir / <family>_verify, each (spec_dict, arch).
 */
#include <pybind11/pybind11.h>

#include <string>

extern "C" {
#include "rocke/rocke_build_id.h"
}

#include "family_glue.hpp" /* ArchDomainRefusal */

namespace py = pybind11;

/* rocke_library_engine_conv.cpp: implicit-GEMM conv (fwd, wgrad, wgrad
 * reduce), direct grouped / non-grouped conv, deep fused conv+pool. */
void register_conv(py::module_& m);

/* rocke_library_engine_img2col.cpp: a separate TU because img2col's helper
 * header defines its own struct rocke_conv_problem, which clashes with the
 * implicit-GEMM conv headers in one translation unit. */
void register_img2col(py::module_& m);

/* rocke_library_engine_winograd_{fwd,gemm,gen,fused1d}.cpp: the Winograd
 * forward families, one translation unit each (see winograd_glue.hpp). */
void register_winograd_fwd(py::module_& m);
void register_winograd_gemm(py::module_& m);
void register_winograd_gen(py::module_& m);
void register_winograd_fused1d(py::module_& m);

PYBIND11_MODULE(rocke_library_engine, m)
{
    m.doc() = "pybind11 binding for the C++ mirrors of the rocke library kernel "
              "families (convolution). Companion of rocke_engine.";

    /* Same stamps as rocke_engine: the engine archive this module links. */
    m.def(
        "build_id",
        []() { return std::string(rocke_build_id()); },
        "Content hash of the portable-IR native bundle inputs of the linked engine.");
    m.def(
        "engine_version",
        []() { return std::string(rocke_engine_version()); },
        "Human-readable engine version of the linked engine.");

    /* A lowering the arch-domain lane refused, raised by take_lowered /
     * throw_if_arch_domain, reaches Python as this module's ArchDomainError: a
     * NotImplementedError carrying the engine's message, as rocke_engine's
     * does. Module-local, so it never takes over rocke_engine's refusals, and
     * this module's no longer depend on rocke_engine's translator (or arrive
     * as a plain RuntimeError without it). */
    py::register_local_exception<ArchDomainRefusal>(
        m, "ArchDomainError", PyExc_NotImplementedError);

    register_conv(m);
    register_img2col(m);
    register_winograd_fwd(m);
    register_winograd_gemm(m);
    register_winograd_gen(m);
    register_winograd_fused1d(m);
}
