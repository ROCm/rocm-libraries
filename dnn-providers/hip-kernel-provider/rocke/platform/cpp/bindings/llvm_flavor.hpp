// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Python bindings share runtime.comgr's compiler selection via the Python
 * flavor resolver. Never invoke standalone native discovery in this path. */
#ifndef ROCKE_BINDINGS_LLVM_FLAVOR_HPP
#define ROCKE_BINDINGS_LLVM_FLAVOR_HPP

#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>

#include "rocke/lower_llvm.h"

inline rocke_llvm_flavor_t resolve_python_llvm_flavor()
{
    const auto name = pybind11::module_::import("rocke.core.lower_llvm")
                          .attr("_resolve_llvm_flavor")()
                          .cast<std::string>();
    const auto flavor = rocke_llvm_flavor_from_name(name.c_str());
    if(flavor == ROCKE_LLVM_FLAVOR_AUTO)
        throw std::runtime_error("Python resolved an unsupported LLVM flavor: " + name);
    return flavor;
}

#endif // ROCKE_BINDINGS_LLVM_FLAVOR_HPP
