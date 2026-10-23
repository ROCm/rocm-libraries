// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/API/Compiler.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

using namespace rocm::tensilelite;

// Runs compile with the GIL released; compile must only capture C++-owned
// data, such as the std::string copies nanobind makes of str arguments.
template <typename Compile>
static nb::bytes compileWithoutGIL(Compile compile) {
  std::vector<uint8_t> result;
  {
    nb::gil_scoped_release release;
    result = compile();
  }
  return nb::bytes(result.data(), result.size());
}

NB_MODULE(tensilelite_jit, m) {
  m.doc() =
      "COMGR-based compilation of AMDGPU kernels from LLVM IR or assembly.";

  nb::enum_<OptLevel>(
      m, "OptLevel",
      "Optimization level of LLVM IR code generation, mapping 1:1 onto -O<n>.")
      .value("o0", OptLevel::o0)
      .value("o1", OptLevel::o1)
      .value("o2", OptLevel::o2)
      .value("o3", OptLevel::o3);

  nb::enum_<CodeObjectVersion>(
      m, "CodeObjectVersion",
      "Code object version, mapping 1:1 onto -mcode-object-version=<n>.")
      .value("v4", CodeObjectVersion::v4)
      .value("v5", CodeObjectVersion::v5)
      .value("v6", CodeObjectVersion::v6);

  const CompileOptions defaults;
  nb::class_<CompileOptions>(
      m, "CompileOptions",
      "Options for the LLVM IR code generation step.\n\n"
      "extraOptions are appended verbatim "
      "after the generated compiler options; see the warning on compileToASM. "
      "extraOptions is returned as a copy, so assign a new list to change it.")
      .def(
          "__init__",
          [](CompileOptions *self, OptLevel optLevel,
             CodeObjectVersion codeObjectVersion,
             std::vector<std::string> extraOptions) {
            new (self) CompileOptions{optLevel, codeObjectVersion,
                                      std::move(extraOptions)};
          },
          nb::kw_only(), "optLevel"_a = defaults.optLevel,
          "codeObjectVersion"_a = defaults.codeObjectVersion,
          "extraOptions"_a = defaults.extraOptions)
      .def_rw("optLevel", &CompileOptions::optLevel)
      .def_rw("codeObjectVersion", &CompileOptions::codeObjectVersion)
      .def_rw("extraOptions", &CompileOptions::extraOptions);

  // options is taken by value so other threads cannot modify it mid-compile.
  m.def(
      "compileToASM",
      [](const std::string &llvmIR, const std::string &target,
         CompileOptions options) {
        return compileWithoutGIL(
            [&] { return compileToASM(llvmIR, target, options); });
      },
      "llvmIR"_a, "target"_a, "options"_a = CompileOptions(),
      "Compiles textual LLVM IR (str) to AMDGPU assembly text.\n\n"
      "WARNING: options.extraOptions are forwarded to the compiler "
      "unvalidated. An invalid -mllvm option terminates the Python process, "
      "and -mcpu overrides target.\n\n"
      "target must be a full ISA name, such as amdgcn-amd-amdhsa--gfx950. "
      "Returns the assembly as bytes. Raises RuntimeError with the compiler "
      "log if compilation fails. The GIL is released while compiling.");

  m.def(
      "compileToHSACO",
      [](const std::string &llvmIR, const std::string &target,
         CompileOptions options) {
        return compileWithoutGIL(
            [&] { return compileToHSACO(llvmIR, target, options); });
      },
      "llvmIR"_a, "target"_a, "options"_a = CompileOptions(),
      "Compiles textual LLVM IR (str) to a loadable code object (HSACO).\n\n"
      "WARNING: options.extraOptions are forwarded to the compiler "
      "unvalidated. "
      "An invalid -mllvm option terminates the Python process, and -mcpu "
      "overrides target.\n\n"
      "Input, target and error semantics match compileToASM; the options apply "
      "to code generation only. Returns the ELF code object as bytes. The GIL "
      "is released while compiling.");

  m.def(
      "assembleToHSACO",
      [](const std::string &assembly, const std::string &target) {
        return compileWithoutGIL(
            [&] { return assembleToHSACO(assembly, target); });
      },
      "assembly"_a, "target"_a,
      "Assembles AMDGPU assembly text (str) and links it into a loadable "
      "code object (HSACO).\n\n"
      "target must be a full ISA name that, including feature suffixes, "
      "matches the assembly's .amdgcn_target directive; the code object "
      "version comes from its .amdhsa_code_object_version directive. Returns "
      "the ELF code object as bytes. Raises RuntimeError with the assembler or "
      "linker log on failure. The GIL is released while assembling.");
}
