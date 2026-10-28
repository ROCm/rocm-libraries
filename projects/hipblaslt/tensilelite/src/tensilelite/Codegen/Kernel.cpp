// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/Codegen/Kernel.hpp>

#include "container.hpp"
#include "instruction/branch.hpp"
#include "instruction/common.hpp"
#include "instruction/mem.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
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
// The kernarg segment pointer is a 64-bit user SGPR pair.
static constexpr int32_t kernargPtrSgprs = 2;

namespace {

struct RegisterCounts {
  int32_t nextFreeVgpr;
  int32_t accumOffset;
  int32_t agprCount;
  int32_t nextFreeSgpr;
  int32_t metadataSgprCount;
};

// Where the hardware initializes the SGPRs the kernel reads at entry.
struct SgprLayout {
  int32_t userSgprCount;
  int32_t workgroupIdX;
  std::optional<int32_t> workgroupIdY;
  std::optional<int32_t> workgroupIdZ;
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

static const KernelAbi &validatedAbi(const KernelAbi &abi) {
  // Throws for unsupported versions; the minor version itself is unused here.
  (void)metadataVersionMinor(abi.codeObjectVersion);
  if (abi.maxPreloadDwords < 0)
    throw std::invalid_argument("max preload dwords must be non-negative");
  return abi;
}

// The window is the longest prefix of whole explicit arguments that fits the
// request and the user SGPRs left after the kernarg pointer. Padding between
// arguments belongs to the window but to no argument.
static int32_t computePreloadDwords(const Target &target, const KernelAbi &abi,
                                    const KernelSignature &signature) {
  if (abi.maxPreloadDwords > 0 && !target.supportsKernargPreload())
    throw std::invalid_argument("kernarg preloading is not supported by " +
                                std::string(target.archName()));
  int32_t limit =
      std::min(abi.maxPreloadDwords, target.maxUserSgprs() - kernargPtrSgprs);
  int32_t dwords = 0;
  for (const KernelArgument &arg : signature.arguments()) {
    int32_t end = (arg.offset + arg.size) / 4;
    if (isHidden(arg.kind) || end > limit)
      break;
    dwords = end;
  }
  return dwords;
}

// All hidden argument kinds are code object v5 additions.
static KernelSignature validatedSignature(KernelSignature signature,
                                          CodeObjectVersion version) {
  if (version != CodeObjectVersion::v4)
    return signature;
  for (const KernelArgument &arg : signature.arguments())
    if (isHidden(arg.kind))
      throw std::invalid_argument(
          "hidden kernel arguments need code object v5 or later");
  return signature;
}

// The user SGPRs come first, then the enabled system SGPRs: workgroup ids x, y
// and z in that order, consuming consecutive registers.
static SgprLayout computeSgprLayout(const KernelOptions &options,
                                    int32_t preloadDwords) {
  // The kernarg preload window goes between the kernarg pointer and the system
  // SGPRs, so it grows the user SGPR count here.
  SgprLayout layout;
  layout.userSgprCount = kernargPtrSgprs + preloadDwords;
  int32_t next = layout.userSgprCount;
  layout.workgroupIdX = next++;
  if (options.workgroupIdY)
    layout.workgroupIdY = next++;
  if (options.workgroupIdZ)
    layout.workgroupIdZ = next++;
  return layout;
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

// Upper bound on the compat prologue: hardware with kernarg preload support
// starts at the entry point plus this many bytes.
static constexpr int32_t preloadEntryOffset = 256;
// Every SMEM instruction is 8 bytes; s_waitcnt and s_branch are 4 each.
static constexpr int32_t smemBytes = 8;
static constexpr int32_t sopBytes = 4;

// Loads the preload window with the largest s_load whose destination is
// suitably aligned (x8 and x4 need 4-aligned SGPRs, x2 an even one).
static int32_t emitWindowLoads(const Kernel &kernel, rocisa::Module &module) {
  std::shared_ptr<rocisa::RegisterContainer> base =
      kernel.kernargSegmentPtr().container();
  int32_t loads = 0;
  int32_t reg = kernargPtrSgprs;
  int32_t left = kernel.preloadDwords();
  while (left > 0) {
    int32_t size = 1;
    for (int32_t candidate : {8, 4, 2})
      if (left >= candidate && reg % std::min(candidate, 4) == 0) {
        size = candidate;
        break;
      }
    std::shared_ptr<rocisa::RegisterContainer> dst =
        RegisterRef{RegisterKind::sgpr, reg, size}.container();
    int32_t offset = 4 * (reg - kernargPtrSgprs);
    switch (size) {
    case 8:
      module.add(std::make_shared<rocisa::SLoadB256>(dst, base, offset));
      break;
    case 4:
      module.add(std::make_shared<rocisa::SLoadB128>(dst, base, offset));
      break;
    case 2:
      module.add(std::make_shared<rocisa::SLoadB64>(dst, base, offset));
      break;
    default:
      module.add(std::make_shared<rocisa::SLoadB32>(dst, base, offset));
      break;
    }
    ++loads;
    reg += size;
    left -= size;
  }
  return loads;
}

// Firmware without preload support starts at the entry point and needs the
// window loaded by the kernel itself; firmware with support starts at the entry
// point plus 256 bytes, where the body begins, with the window preloaded.
static std::string emitPreloadPrologue(const Kernel &kernel) {
  if (kernel.preloadDwords() == 0)
    return {};
  auto label =
      std::make_shared<rocisa::Label>(std::string(kernel.name()) + "_body", "");
  rocisa::Module prologue("preload_prologue");
  int32_t loads = emitWindowLoads(kernel, prologue);
  int32_t bytes = loads * smemBytes + 2 * sopBytes;
  // A longer prologue would silently misalign the body past the entry offset;
  // this is a safety net, as the current limit of 14 dwords needs at most 3
  // loads.
  if (bytes > preloadEntryOffset)
    throw std::logic_error("kernarg preload prologue exceeds 256 bytes");
  prologue.add(std::make_shared<rocisa::SWaitCnt>(-1, -1, -1, 0));
  prologue.add(std::make_shared<rocisa::SBranch>(label->getLabelName()));
  prologue.add(std::make_shared<rocisa::TextBlock>(".p2align 8\n"));
  prologue.add(label);
  return prologue.toString();
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
      << "  .amdhsa_user_sgpr_kernarg_segment_ptr 1\n";
  if (kernel.preloadDwords() > 0)
    out << "  .amdhsa_user_sgpr_count "
        << kernargPtrSgprs + kernel.preloadDwords() << "\n"
        << "  .amdhsa_user_sgpr_kernarg_preload_length "
        << kernel.preloadDwords() << "\n"
        << "  .amdhsa_user_sgpr_kernarg_preload_offset 0\n";
  out << "  .amdhsa_kernarg_size " << kernel.signature().segmentSize() << "\n"
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
               int32_t workgroupSize, KernelSignature signature,
               const KernelOptions &options)
    : kernelTarget(target), kernelName(validatedName(name)),
      maxWorkgroupSize(validatedWorkgroupSize(workgroupSize)),
      version(validatedAbi(options.abi).codeObjectVersion),
      kernelSignature(validatedSignature(std::move(signature), version)),
      preloadDwordCount(
          computePreloadDwords(target, options.abi, kernelSignature)),
      registers(target), bodyModule(kernelName),
      irBuilder(target, registers, bodyModule),
      kernargPtr(registers.reserve(RegisterKind::sgpr, 0, kernargPtrSgprs,
                                   "kernarg_segment_ptr")),
      workgroupIdXReg(registers.reserve(
          RegisterKind::sgpr,
          computeSgprLayout(options, preloadDwordCount).workgroupIdX, 1,
          "workgroup_id_x")),
      workitemIdReg(
          registers.reserve(RegisterKind::vgpr, 0, 1, "workitem_id")) {
  SgprLayout layout = computeSgprLayout(options, preloadDwordCount);
  if (layout.workgroupIdY)
    workgroupIdYReg.emplace(registers.reserve(
        RegisterKind::sgpr, *layout.workgroupIdY, 1, "workgroup_id_y"));
  if (layout.workgroupIdZ)
    workgroupIdZReg.emplace(registers.reserve(
        RegisterKind::sgpr, *layout.workgroupIdZ, 1, "workgroup_id_z"));
  // The hardware initializes s[2 + offset / 4] with each argument inside the
  // window.
  for (const KernelArgument &arg : kernelSignature.arguments()) {
    if (isHidden(arg.kind) || (arg.offset + arg.size) / 4 > preloadDwordCount)
      break;
    preloaded.push_back(
        {arg.offset, arg.size,
         registers.reserve(RegisterKind::sgpr, kernargPtrSgprs + arg.offset / 4,
                           arg.size / 4, "preload_" + arg.name)});
  }
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

const Kernel::PreloadedArgument *Kernel::heldArgumentAt(int32_t offset) const {
  for (const PreloadedArgument &arg : preloaded)
    if (arg.range && offset >= arg.offset && offset < arg.offset + arg.size)
      return &arg;
  return nullptr;
}

Kernel::PreloadedArgument *Kernel::findPreloaded(int32_t offset) {
  bool isArgumentStart = false;
  for (const KernelArgument &arg : kernelSignature.arguments())
    isArgumentStart |= arg.offset == offset;
  if (!isArgumentStart)
    throw std::invalid_argument(
        "kernel argument offset is not the start of an argument");
  for (PreloadedArgument &arg : preloaded)
    if (arg.offset == offset)
      return &arg;
  return nullptr;
}

std::optional<RegisterRange> Kernel::takePreloadedArgument(int32_t offset) {
  PreloadedArgument *arg = findPreloaded(offset);
  if (!arg)
    return std::nullopt;
  return std::exchange(arg->range, std::nullopt);
}

void Kernel::releasePreloadedArgument(int32_t offset) {
  PreloadedArgument *arg = findPreloaded(offset);
  if (arg)
    arg->range.reset();
}

// Copies count SGPRs, as 64-bit moves where both sides are even aligned.
static void emitMoves(IRBuilder &builder, int32_t dstStart, int32_t srcStart,
                      int32_t count) {
  if (dstStart == srcStart)
    return;
  for (int32_t done = 0; done < count;) {
    int32_t dst = dstStart + done;
    int32_t src = srcStart + done;
    bool wide = count - done >= 2 && dst % 2 == 0 && src % 2 == 0;
    int32_t size = wide ? 2 : 1;
    std::shared_ptr<rocisa::RegisterContainer> dstReg =
        RegisterRef{RegisterKind::sgpr, dst, size}.container();
    std::shared_ptr<rocisa::RegisterContainer> srcReg =
        RegisterRef{RegisterKind::sgpr, src, size}.container();
    if (wide)
      builder.create<rocisa::SMovB64>(dstReg, srcReg);
    else
      builder.create<rocisa::SMovB32>(dstReg, srcReg);
    done += size;
  }
}

// Loads count dwords from byte offset with the widest loads whose destination
// is aligned to their size.
static void emitLoads(IRBuilder &builder,
                      const std::shared_ptr<rocisa::RegisterContainer> &base,
                      int32_t dstStart, int32_t offset, int32_t count) {
  for (int32_t done = 0; done < count;) {
    int32_t dst = dstStart + done;
    int32_t size = 1;
    for (int32_t candidate : {4, 2})
      if (count - done >= candidate && dst % candidate == 0) {
        size = candidate;
        break;
      }
    std::shared_ptr<rocisa::RegisterContainer> dstReg =
        RegisterRef{RegisterKind::sgpr, dst, size}.container();
    int32_t byteOffset = offset + 4 * done;
    switch (size) {
    case 1:
      builder.create<rocisa::SLoadB32>(dstReg, base, byteOffset);
      break;
    case 2:
      builder.create<rocisa::SLoadB64>(dstReg, base, byteOffset);
      break;
    default:
      builder.create<rocisa::SLoadB128>(dstReg, base, byteOffset);
      break;
    }
    done += size;
  }
}

void Kernel::loadArgument(RegisterRef dst, int32_t offset) {
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
  // Split into runs of dwords that are held in registers and runs that are not.
  for (int32_t done = 0; done < dst.count;) {
    int32_t byteOffset = offset + 4 * done;
    int32_t chunk = 1;
    if (const PreloadedArgument *held = heldArgumentAt(byteOffset)) {
      int32_t heldDwords = (held->offset + held->size - byteOffset) / 4;
      chunk = std::min(dst.count - done, heldDwords);
      int32_t src = held->range->start() + (byteOffset - held->offset) / 4;
      emitMoves(irBuilder, dst.start + done, src, chunk);
    } else {
      while (done + chunk < dst.count &&
             !heldArgumentAt(byteOffset + 4 * chunk))
        ++chunk;
      emitLoads(irBuilder, base, dst.start + done, byteOffset, chunk);
    }
    done += chunk;
  }
}

std::string Kernel::toAssembly() const {
  registers.checkLimits();
  // Printing instructions reads rocisa's per-thread ISA state.
  kernelTarget.selectForCurrentThread();
  RegisterCounts counts = countRegisters(registers);
  return emitHeader(*this) + emitPreloadPrologue(*this) +
         bodyModule.toString() + emitSymbolEnd(kernelName) +
         emitDescriptor(*this, counts) + emitMetadata(*this, counts);
}
