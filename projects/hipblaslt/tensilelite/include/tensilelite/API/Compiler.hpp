// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef ROCM_TENSILELITE_API_COMPILER_HPP
#define ROCM_TENSILELITE_API_COMPILER_HPP

#include <tensilelite/API/Export.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace rocm::tensilelite {

/// Optimization level of LLVM IR code generation, mapping 1:1 onto -O<n>.
enum class OptLevel { o0, o1, o2, o3 };

/// Code object version, mapping 1:1 onto -mcode-object-version=<n>.
enum class CodeObjectVersion { v4 = 4, v5 = 5, v6 = 6 };

/// Options for the LLVM IR code generation step.
struct TENSILELITE_EXPORTED CompileOptions {
  /// Optimization level.
  OptLevel optLevel = OptLevel::o3;
  /// Code object version.
  CodeObjectVersion codeObjectVersion = CodeObjectVersion::v6;
  /// Compiler options appended verbatim after the generated ones; see the
  /// warning on compileToASM.
  std::vector<std::string> extraOptions;
};

/// Compiles LLVM IR to AMDGPU assembly text using COMGR.
///
/// WARNING: extraOptions are forwarded to the underlying compiler
/// unvalidated. An invalid -mllvm option terminates the calling process, and
/// -mcpu overrides the target.
///
/// llvmIR is textual LLVM IR. Precondition: target is a full ISA name, such as
/// amdgcn-amd-amdhsa--gfx950 or amdgcn-amd-amdhsa--gfx950:xnack-. The returned
/// bytes are the assembly text without a trailing NUL.
///
/// Throws std::runtime_error, including the compiler log, if compilation
/// fails.
///
/// COMGR caches results on disk (~/.cache/comgr, see the AMD_COMGR_CACHE*
/// environment variables) and appends the options in
/// AMD_COMGR_DRIVER_OPTIONS_APPEND to every compilation.
TENSILELITE_EXPORTED std::vector<uint8_t>
compileToASM(std::string_view llvmIR, std::string_view target,
             const CompileOptions &options = {});

/// Compiles LLVM IR to a loadable AMDGPU code object (HSACO) using COMGR.
///
/// WARNING: extraOptions are forwarded to the underlying compiler
/// unvalidated. An invalid -mllvm option terminates the calling process, and
/// -mcpu overrides the target.
///
/// Input, target, exception and environment semantics match compileToASM.
/// The options apply to code generation only, never to linking. The returned
/// bytes are an ELF shared object suitable for hipModuleLoadData.
TENSILELITE_EXPORTED std::vector<uint8_t>
compileToHSACO(std::string_view llvmIR, std::string_view target,
               const CompileOptions &options = {});

/// Assembles AMDGPU assembly text and links it into a loadable code object
/// (HSACO) using COMGR.
///
/// target is a full ISA name, as for compileToASM; including feature suffixes,
/// it must match the assembly's .amdgcn_target directive. The code object
/// version comes from the assembly's .amdhsa_code_object_version directive.
/// There is no options parameter because optimization and code object options
/// have no effect on assembling or linking.
///
/// Throws std::runtime_error, including the assembler or linker log, on
/// failure.
TENSILELITE_EXPORTED std::vector<uint8_t>
assembleToHSACO(std::string_view assembly, std::string_view target);

} // namespace rocm::tensilelite

#endif // ROCM_TENSILELITE_API_COMPILER_HPP
