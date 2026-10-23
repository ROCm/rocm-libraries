// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/codegen/IRBuilder.hpp>

#include "OperandUtils.hpp"

#include "code.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace rocm::tensilelite::codegen;
using detail::registerText;

namespace {

constexpr std::string_view makeRsrcName = "makeRsrc";

void requireDestination(RegisterRef dst) {
  if (dst.kind != RegisterKind::sgpr || dst.count != 4)
    throw std::invalid_argument(std::string(makeRsrcName) +
                                " needs a destination of 4 sgprs, got " +
                                registerText(dst));
  if (dst.start % 4 != 0)
    throw std::invalid_argument(
        std::string(makeRsrcName) + " destination " + registerText(dst) +
        " must start at an sgpr index that is a multiple of 4");
}

void requirePointer(Operand pointer) {
  std::string prefix = std::string(makeRsrcName) + " pointer must be an sgpr ";
  if (pointer.isImmediate())
    throw std::invalid_argument(prefix + "pair, got immediate " +
                                std::to_string(pointer.immediate()));
  RegisterRef reg = pointer.reg();
  if (reg.kind != RegisterKind::sgpr || reg.count != 2)
    throw std::invalid_argument(prefix + "pair, got " + registerText(reg));
}

void requireSize(Operand sizeBytes) {
  // Any int32 is the 32-bit pattern of an unsigned size.
  if (sizeBytes.isImmediate())
    return;
  RegisterRef reg = sizeBytes.reg();
  if (reg.kind != RegisterKind::sgpr || reg.count != 1)
    throw std::invalid_argument(
        std::string(makeRsrcName) +
        " size must be a single sgpr or an immediate, got " +
        registerText(reg));
}

bool isSelfCopy(RegisterRef dstPart, Operand src) {
  return src.isRegister() && src.reg() == dstPart;
}

// Copying the pointer overwrites dst[0:1], so a size held there must be saved
// before that happens.
bool sizeIsClobberedByPointerCopy(RegisterRef dst, Operand pointer,
                                  Operand sizeBytes) {
  RegisterRef pointerDst{RegisterKind::sgpr, dst.start, 2};
  if (isSelfCopy(pointerDst, pointer) || sizeBytes.isImmediate())
    return false;
  int index = sizeBytes.reg().start;
  return index == dst.start || index == dst.start + 1;
}

} // namespace

void IRBuilder::makeRsrc(RegisterRef dst, Operand pointer, Operand sizeBytes) {
  requireDestination(dst);
  requirePointer(pointer);
  requireSize(sizeBytes);

  // The temporary must outlive the size word's mov below.
  std::optional<RegisterRange> sizeCopy;
  if (sizeIsClobberedByPointerCopy(dst, pointer, sizeBytes)) {
    sizeCopy.emplace(allocateSgpr(1));
    movb32(sizeCopy->ref(), sizeBytes);
    sizeBytes = Operand(*sizeCopy);
  }

  // Each source is read by exactly one copy and word 3 has no register source,
  // so this order is safe for any remaining overlap with dst.
  RegisterRef pointerDst{RegisterKind::sgpr, dst.start, 2};
  RegisterRef sizeDst{RegisterKind::sgpr, dst.start + 2, 1};
  RegisterRef upperDst{RegisterKind::sgpr, dst.start + 3, 1};
  if (!isSelfCopy(pointerDst, pointer))
    movb64(pointerDst, pointer);
  if (!isSelfCopy(sizeDst, sizeBytes))
    movb32(sizeDst, sizeBytes);
  int32_t upper = rocisa::SrdUpperValue(target().isaVersion())->getValue();
  movb32(upperDst, upper);
}
