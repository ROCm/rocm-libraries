// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef TENSILELITE_LLVM_WRITING_HPP
#define TENSILELITE_LLVM_WRITING_HPP

#include <string>

#include <Tensile/ContractionSolution_fwd.hpp>

#include <tensilelitehost/export.h>

namespace TensileLite
{
    /// Encodes one solution as a YAML mapping with the traits the YAML loader
    /// reads: the form of an element of a library's `solutions` sequence.
    TENSILELITEHOST_EXPORT std::string LLVMWriteSolutionData(ContractionSolution const& solution);
}

#endif // TENSILELITE_LLVM_WRITING_HPP
