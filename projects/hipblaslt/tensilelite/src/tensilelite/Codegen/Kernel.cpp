// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/Codegen/Kernel.hpp>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

using namespace rocm::tensilelite;
using namespace rocm::tensilelite::codegen;

// gfx9 reserves VCC, FLAT_SCRATCH and XNACK_MASK after the numbered SGPRs, and
// LLVM counts them in the metadata .sgpr_count of every gfx950 kernel.
static constexpr int32_t gfx9ExtraSgprs = 6;
static constexpr int32_t maxWorkgroupSizeLimit = 1024;

namespace {

struct RegisterCounts {
  int32_t nextFreeVgpr;
  int32_t accumOffset;
  int32_t agprCount;
  int32_t nextFreeSgpr;
  int32_t metadataSgprCount;
};

} // namespace

// Kernel and argument names end up unquoted in symbols and YAML.
static bool isIdentifier(std::string_view text) {
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

static bool isHidden(KernelArgumentKind kind) {
  return kind != KernelArgumentKind::globalPointer &&
         kind != KernelArgumentKind::byValue;
}

static std::string_view valueKindName(KernelArgumentKind kind) {
  switch (kind) {
  case KernelArgumentKind::globalPointer:
    return "global_buffer";
  case KernelArgumentKind::byValue:
    return "by_value";
  case KernelArgumentKind::hiddenBlockCountX:
    return "hidden_block_count_x";
  case KernelArgumentKind::hiddenBlockCountY:
    return "hidden_block_count_y";
  case KernelArgumentKind::hiddenBlockCountZ:
    return "hidden_block_count_z";
  case KernelArgumentKind::hiddenGroupSizeX:
    return "hidden_group_size_x";
  case KernelArgumentKind::hiddenGroupSizeY:
    return "hidden_group_size_y";
  case KernelArgumentKind::hiddenGroupSizeZ:
    return "hidden_group_size_z";
  }
  throw std::invalid_argument("unknown kernel argument kind");
}

static bool isGroupSize(KernelArgumentKind kind) {
  return kind == KernelArgumentKind::hiddenGroupSizeX ||
         kind == KernelArgumentKind::hiddenGroupSizeY ||
         kind == KernelArgumentKind::hiddenGroupSizeZ;
}

static int32_t roundUp(int32_t value, int32_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

static int32_t metadataVersionMinor(CodeObjectVersion version) {
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

static std::string validatedName(std::string_view name) {
  if (!isIdentifier(name))
    throw std::invalid_argument("invalid kernel name: '" + std::string(name) +
                                "'");
  return std::string(name);
}

static int32_t validatedWorkgroupSize(int32_t size) {
  if (size < 1 || size > maxWorkgroupSizeLimit)
    throw std::invalid_argument("workgroup size must be in [1, 1024]");
  return size;
}

static CodeObjectVersion validatedVersion(CodeObjectVersion version) {
  // Throws for unsupported versions; the minor version itself is unused here.
  (void)metadataVersionMinor(version);
  return version;
}

static RegisterCounts countRegisters(const RegisterAllocator &registers) {
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

static std::string emitHeader(const Kernel &kernel) {
  std::string name(kernel.name());
  std::ostringstream out;
  out << ".amdgcn_target \"" << kernel.target().isaName() << "\"\n"
      << ".amdhsa_code_object_version "
      << static_cast<int32_t>(kernel.codeObjectVersion()) << "\n"
      << ".text\n"
      << ".protected " << name << "\n"
      << ".globl " << name << "\n"
      << ".p2align 8\n"
      << ".type " << name << ",@function\n"
      << name << ":\n";
  return out.str();
}

static std::string emitSymbolEnd(std::string_view name) {
  std::ostringstream out;
  out << ".L" << name << "_end:\n"
      << ".size " << name << ", .L" << name << "_end - " << name << "\n";
  return out.str();
}

static std::string emitDescriptor(const Kernel &kernel,
                                  const RegisterCounts &counts) {
  std::ostringstream out;
  out << ".rodata\n"
      << ".p2align 6\n"
      << ".amdhsa_kernel " << kernel.name() << "\n"
      << "  .amdhsa_user_sgpr_kernarg_segment_ptr 1\n"
      << "  .amdhsa_kernarg_size " << kernel.signature().segmentSize() << "\n"
      << "  .amdhsa_accum_offset " << counts.accumOffset << "\n"
      << "  .amdhsa_next_free_vgpr " << counts.nextFreeVgpr << "\n"
      << "  .amdhsa_next_free_sgpr " << counts.nextFreeSgpr << "\n"
      << "  .amdhsa_group_segment_fixed_size " << kernel.ldsSize() << "\n"
      << "  .amdhsa_private_segment_fixed_size 0\n"
      << "  .amdhsa_system_sgpr_workgroup_id_x 1\n";
  if (kernel.hasWorkgroupIdY())
    out << "  .amdhsa_system_sgpr_workgroup_id_y 1\n";
  if (kernel.hasWorkgroupIdZ())
    out << "  .amdhsa_system_sgpr_workgroup_id_z 1\n";
  out << "  .amdhsa_system_vgpr_workitem_id 0\n"
      << "  .amdhsa_float_denorm_mode_32 3\n"
      << "  .amdhsa_float_denorm_mode_16_64 3\n"
      << ".end_amdhsa_kernel\n";
  return out.str();
}

static std::string emitArguments(const KernelSignature &signature) {
  if (signature.arguments().empty())
    return {};
  std::ostringstream out;
  out << "    .args:\n";
  for (const KernelArgument &arg : signature.arguments()) {
    // Hidden arguments have no name, so the size field opens the entry.
    if (isHidden(arg.kind))
      out << "      - .size: " << arg.size << "\n";
    else
      out << "      - .name: " << arg.name << "\n"
          << "        .size: " << arg.size << "\n";
    out << "        .offset: " << arg.offset << "\n"
        << "        .value_kind: " << valueKindName(arg.kind) << "\n";
    if (arg.kind == KernelArgumentKind::globalPointer)
      out << "        .address_space: global\n";
  }
  return out.str();
}

static std::string emitMetadata(const Kernel &kernel,
                                const RegisterCounts &counts) {
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
      << "    .group_segment_fixed_size: " << kernel.ldsSize() << "\n"
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

int32_t KernelSignature::addExplicit(std::string_view name,
                                     KernelArgumentKind kind, int32_t size) {
  if (!args.empty() && isHidden(args.back().kind))
    throw std::invalid_argument(
        "explicit kernel arguments must precede hidden ones");
  if (!isIdentifier(name))
    throw std::invalid_argument("invalid kernel argument name: '" +
                                std::string(name) + "'");
  for (const KernelArgument &arg : args)
    if (arg.name == name)
      throw std::invalid_argument("duplicate kernel argument name: '" +
                                  std::string(name) + "'");
  return append(std::string(name), kind, size);
}

int32_t KernelSignature::append(std::string name, KernelArgumentKind kind,
                                int32_t size) {
  int32_t end = args.empty() ? 0 : args.back().offset + args.back().size;
  int32_t offset = roundUp(end, size);
  args.push_back({std::move(name), kind, size, offset});
  return offset;
}

int32_t KernelSignature::addPointer(std::string_view name) {
  return addExplicit(name, KernelArgumentKind::globalPointer, 8);
}

int32_t KernelSignature::addScalar(std::string_view name, int32_t size) {
  if (size != 4 && size != 8)
    throw std::invalid_argument("scalar kernel argument size must be 4 or 8");
  return addExplicit(name, KernelArgumentKind::byValue, size);
}

int32_t KernelSignature::addHidden(KernelArgumentKind kind, int32_t size) {
  if (!isHidden(kind))
    throw std::invalid_argument("kernel argument kind is not hidden");
  bool sizeAllowed = size == 4 || (size == 2 && isGroupSize(kind));
  if (!sizeAllowed)
    throw std::invalid_argument("invalid hidden kernel argument size");
  for (const KernelArgument &arg : args)
    if (arg.kind == kind)
      throw std::invalid_argument("duplicate hidden kernel argument");
  return append({}, kind, size);
}

int32_t KernelSignature::segmentSize() const {
  if (args.empty())
    return 0;
  return roundUp(args.back().offset + args.back().size, segmentAlignment);
}

// Validation runs in the initializers so no partially built kernel exists.
Kernel::Kernel(const Target &target, std::string_view name,
               int32_t workgroupSize, CodeObjectVersion codeObjectVersion)
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

void Kernel::enableWorkgroupIds(bool y, bool z) {
  if (!y && !z)
    throw std::invalid_argument("no workgroup id requested");
  if (workgroupIdsEnabled)
    throw std::logic_error("workgroup ids already enabled");
  // The hardware packs the enabled ids right after workgroup id x.
  int32_t next = 3;
  if (y)
    workgroupIdYReg.emplace(
        registers.reserve(RegisterKind::sgpr, next++, 1, "workgroup_id_y"));
  if (z)
    workgroupIdZReg.emplace(
        registers.reserve(RegisterKind::sgpr, next, 1, "workgroup_id_z"));
  workgroupIdsEnabled = true;
}

const RegisterRange &Kernel::workgroupIdY() const {
  if (!workgroupIdYReg)
    throw std::logic_error("workgroup id y is not enabled");
  return *workgroupIdYReg;
}

const RegisterRange &Kernel::workgroupIdZ() const {
  if (!workgroupIdZReg)
    throw std::logic_error("workgroup id z is not enabled");
  return *workgroupIdZReg;
}

void Kernel::setLdsSize(int32_t bytes) {
  if (bytes < 0 || bytes > kernelTarget.maxLdsSize())
    throw std::invalid_argument("lds size must be in [0, " +
                                std::to_string(kernelTarget.maxLdsSize()) +
                                "] bytes");
  ldsBytes = bytes;
}

std::shared_ptr<rocisa::SMemLoadInstruction>
Kernel::loadArgument(RegisterRef dst, int32_t offset) {
  if (dst.kind != RegisterKind::sgpr)
    throw std::invalid_argument("kernel argument load needs SGPRs");
  if (dst.count != 1 && dst.count != 2 && dst.count != 4)
    throw std::invalid_argument(
        "kernel argument load must be 1, 2 or 4 dwords");
  if (dst.start % dst.count != 0)
    throw std::invalid_argument(
        "kernel argument load destination is misaligned");
  if (offset < 0 || offset % 4 != 0)
    throw std::invalid_argument("kernel argument offset must be a non-negative "
                                "multiple of 4");
  if (offset + 4 * dst.count > kernelSignature.segmentSize())
    throw std::invalid_argument("kernel argument load is outside the segment");
  std::shared_ptr<rocisa::RegisterContainer> base = kernargPtr.container();
  switch (dst.count) {
  case 1:
    return irBuilder.create<rocisa::SLoadB32>(dst.container(), base, offset);
  case 2:
    return irBuilder.create<rocisa::SLoadB64>(dst.container(), base, offset);
  default:
    return irBuilder.create<rocisa::SLoadB128>(dst.container(), base, offset);
  }
}

std::string Kernel::toAssembly() const {
  registers.checkLimits();
  // Printing instructions reads rocisa's per-thread ISA state.
  kernelTarget.selectForCurrentThread();
  RegisterCounts counts = countRegisters(registers);
  return emitHeader(*this) + bodyModule.toString() + emitSymbolEnd(kernelName) +
         emitDescriptor(*this, counts) + emitMetadata(*this, counts);
}
