// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/codegen/Kernel.hpp>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace rocm::tensilelite;
using namespace rocm::tensilelite::codegen;

namespace {

// gfx9 reserves VCC, FLAT_SCRATCH and XNACK_MASK after the numbered SGPRs, and
// LLVM counts them in the metadata .sgpr_count of every gfx950 kernel.
constexpr int gfx9ExtraSgprs = 6;
constexpr int maxWorkgroupSizeLimit = 1024;

struct RegisterCounts {
  int nextFreeVgpr;
  int accumOffset;
  int agprCount;
  int nextFreeSgpr;
  int metadataSgprCount;
};

// Kernel and argument names end up unquoted in symbols and YAML.
bool isIdentifier(std::string_view text) {
  if (text.empty())
    return false;
  auto isAlpha = [](char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
  };
  if (!isAlpha(text.front()))
    return false;
  for (char c : text)
    if (!isAlpha(c) && !(c >= '0' && c <= '9'))
      return false;
  return true;
}

int roundUp(int value, int multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

int metadataVersionMinor(CodeObjectVersion version) {
  switch (version) {
  case CodeObjectVersion::v4:
    return 1;
  case CodeObjectVersion::v5:
    return 2;
  case CodeObjectVersion::v6:
    // LLVM emits [1, 2] for v6 too; [1, 3] makes the HIP runtime fault.
    return 2;
  }
  throw std::invalid_argument("unsupported code object version");
}

std::string validatedName(std::string_view name) {
  if (!isIdentifier(name))
    throw std::invalid_argument("invalid kernel name: '" + std::string(name) +
                                "'");
  return std::string(name);
}

int validatedWorkgroupSize(int size) {
  if (size < 1 || size > maxWorkgroupSizeLimit)
    throw std::invalid_argument("workgroup size must be in [1, 1024]");
  return size;
}

CodeObjectVersion validatedVersion(CodeObjectVersion version) {
  // Throws for unsupported versions; the minor version itself is unused here.
  (void)metadataVersionMinor(version);
  return version;
}

RegisterCounts countRegisters(const RegisterAllocator &registers) {
  RegisterCounts counts;
  counts.accumOffset = registers.accumOffset();
  // AGPRs follow the arch VGPRs in the unified register file, as in rocisa's
  // KernelBody::setGprs, and LLVM counts both in .vgpr_count.
  counts.agprCount = registers.highWaterMark(RegisterKind::agpr);
  counts.nextFreeVgpr = counts.accumOffset + counts.agprCount;
  counts.nextFreeSgpr =
      std::max(1, registers.highWaterMark(RegisterKind::sgpr));
  counts.metadataSgprCount = counts.nextFreeSgpr + gfx9ExtraSgprs;
  return counts;
}

std::string emitHeader(const Kernel &kernel) {
  std::string name(kernel.name());
  std::ostringstream out;
  out << ".amdgcn_target \"" << kernel.target().isaName() << "\"\n"
      << ".amdhsa_code_object_version "
      << static_cast<int>(kernel.codeObjectVersion()) << "\n"
      << ".text\n"
      << ".protected " << name << "\n"
      << ".globl " << name << "\n"
      << ".p2align 8\n"
      << ".type " << name << ",@function\n"
      << name << ":\n";
  return out.str();
}

std::string emitSymbolEnd(std::string_view name) {
  std::ostringstream out;
  out << ".L" << name << "_end:\n"
      << ".size " << name << ", .L" << name << "_end - " << name << "\n";
  return out.str();
}

std::string emitDescriptor(const Kernel &kernel, const RegisterCounts &counts) {
  std::ostringstream out;
  out << ".rodata\n"
      << ".p2align 6\n"
      << ".amdhsa_kernel " << kernel.name() << "\n"
      << "  .amdhsa_user_sgpr_kernarg_segment_ptr 1\n"
      << "  .amdhsa_kernarg_size " << kernel.signature().segmentSize() << "\n"
      << "  .amdhsa_accum_offset " << counts.accumOffset << "\n"
      << "  .amdhsa_next_free_vgpr " << counts.nextFreeVgpr << "\n"
      << "  .amdhsa_next_free_sgpr " << counts.nextFreeSgpr << "\n"
      << "  .amdhsa_group_segment_fixed_size 0\n"
      << "  .amdhsa_private_segment_fixed_size 0\n"
      << "  .amdhsa_system_sgpr_workgroup_id_x 1\n"
      << "  .amdhsa_system_vgpr_workitem_id 0\n"
      << "  .amdhsa_float_denorm_mode_32 3\n"
      << "  .amdhsa_float_denorm_mode_16_64 3\n"
      << ".end_amdhsa_kernel\n";
  return out.str();
}

std::string emitArguments(const KernelSignature &signature) {
  if (signature.arguments().empty())
    return {};
  std::ostringstream out;
  out << "    .args:\n";
  for (const KernelArgument &arg : signature.arguments()) {
    bool isPointer = arg.kind == KernelArgumentKind::globalPointer;
    out << "      - .name: " << arg.name << "\n"
        << "        .size: " << arg.size << "\n"
        << "        .offset: " << arg.offset << "\n"
        << "        .value_kind: " << (isPointer ? "global_buffer" : "by_value")
        << "\n";
    if (isPointer)
      out << "        .address_space: global\n";
  }
  return out.str();
}

std::string emitMetadata(const Kernel &kernel, const RegisterCounts &counts) {
  std::ostringstream out;
  out << ".amdgpu_metadata\n"
      << "---\n"
      << "amdhsa.version: [1, "
      << metadataVersionMinor(kernel.codeObjectVersion()) << "]\n"
      << "amdhsa.target: " << kernel.target().isaName() << "\n"
      << "amdhsa.kernels:\n"
      << "  - .name: " << kernel.name() << "\n"
      << "    .symbol: " << kernel.name() << ".kd\n"
      << "    .kernarg_segment_size: " << kernel.signature().segmentSize()
      << "\n"
      << "    .kernarg_segment_align: " << KernelSignature::segmentAlignment
      << "\n"
      << "    .group_segment_fixed_size: 0\n"
      << "    .private_segment_fixed_size: 0\n"
      << "    .max_flat_workgroup_size: " << kernel.workgroupSize() << "\n"
      << "    .wavefront_size: " << kernel.target().wavefrontSize() << "\n"
      << "    .sgpr_count: " << counts.metadataSgprCount << "\n"
      << "    .vgpr_count: " << counts.nextFreeVgpr << "\n"
      << "    .agpr_count: " << counts.agprCount << "\n"
      << "    .sgpr_spill_count: 0\n"
      << "    .vgpr_spill_count: 0\n"
      << emitArguments(kernel.signature()) << "...\n"
      << ".end_amdgpu_metadata\n";
  return out.str();
}

} // namespace

int KernelSignature::add(std::string_view name, KernelArgumentKind kind,
                         int size) {
  if (!isIdentifier(name))
    throw std::invalid_argument("invalid kernel argument name: '" +
                                std::string(name) + "'");
  for (const KernelArgument &arg : args)
    if (arg.name == name)
      throw std::invalid_argument("duplicate kernel argument name: '" +
                                  std::string(name) + "'");
  int end = args.empty() ? 0 : args.back().offset + args.back().size;
  int offset = roundUp(end, size);
  args.push_back({std::string(name), kind, size, offset});
  return offset;
}

int KernelSignature::addPointer(std::string_view name) {
  return add(name, KernelArgumentKind::globalPointer, 8);
}

int KernelSignature::addScalar(std::string_view name, int size) {
  if (size != 4 && size != 8)
    throw std::invalid_argument("scalar kernel argument size must be 4 or 8");
  return add(name, KernelArgumentKind::byValue, size);
}

int KernelSignature::segmentSize() const {
  if (args.empty())
    return 0;
  return roundUp(args.back().offset + args.back().size, segmentAlignment);
}

// Validation runs in the initializers so no partially built kernel exists.
Kernel::Kernel(const Target &target, std::string_view name, int workgroupSize,
               CodeObjectVersion codeObjectVersion)
    : kernelTarget(target), kernelName(validatedName(name)),
      maxWorkgroupSize(validatedWorkgroupSize(workgroupSize)),
      version(validatedVersion(codeObjectVersion)), registers(target),
      bodyModule(kernelName), irBuilder(target, registers, bodyModule),
      kernargPtr(
          registers.reserve(RegisterKind::sgpr, 0, 2, "kernarg_segment_ptr")),
      workgroupIdXReg(
          registers.reserve(RegisterKind::sgpr, 2, 1, "workgroup_id_x")),
      workitemIdReg(
          registers.reserve(RegisterKind::vgpr, 0, 1, "workitem_id")) {}

std::string Kernel::toAssembly() const {
  registers.checkLimits();
  // Printing instructions reads rocisa's per-thread ISA state.
  kernelTarget.selectForCurrentThread();
  RegisterCounts counts = countRegisters(registers);
  return emitHeader(*this) + bodyModule.toString() + emitSymbolEnd(kernelName) +
         emitDescriptor(*this, counts) + emitMetadata(*this, counts);
}
