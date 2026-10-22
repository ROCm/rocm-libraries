// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/codegen/IRBuilder.hpp>

#include "Gfx9Legality.hpp"
#include "OperandUtils.hpp"

#include "container.hpp"
#include "instruction/cmp.hpp"
#include "instruction/common.hpp"

#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace rocm::tensilelite::codegen;
using detail::copyToTemporary;
using detail::operandHalf;
using detail::registerText;
using detail::Temporaries;
using detail::toInput;
using detail::toInput64;

namespace {

constexpr std::string_view selectName = "select";

// The sources of one 32-bit word of the result: a is chosen when the condition
// holds, b otherwise.
struct WordSources {
  Operand a;
  Operand b;
};

// The low word, then the high word of a pair; one word for a single register.
using Words = std::vector<WordSources>;

void requireDestination(RegisterRef dst) {
  detail::requireNotAgpr(selectName, dst);
  if (dst.count == 1 || dst.count == 2)
    return;
  throw std::invalid_argument(std::string(selectName) +
                              " needs a destination of 1 or 2 registers, got " +
                              registerText(dst));
}

Operand validateOperand(int width, Operand operand) {
  if (operand.isRegister()) {
    detail::requireNotAgpr(selectName, operand.reg());
    detail::requireRegisterCount(selectName, operand.reg(), width);
    return operand;
  }
  return width == 1 ? detail::canonicalizeImmediate32(selectName, operand)
                    : operand;
}

void requireCondition(RegisterRef cond) {
  if (cond.kind == RegisterKind::sgpr &&
      (cond.count == 1 || gfx9::isLaneMask(cond)))
    return;
  throw std::invalid_argument(
      std::string(selectName) +
      " condition must be an sgpr or an sgpr pair at an even index, got " +
      registerText(cond));
}

void requireScalarSelect(RegisterRef dst, RegisterRef cond, Operand a,
                         Operand b) {
  if (dst.kind != RegisterKind::sgpr)
    return;
  if (cond.count != 1)
    throw std::invalid_argument(
        std::string(selectName) + " sgpr destination " + registerText(dst) +
        " needs a uniform bool condition, got " + registerText(cond));
  for (Operand source : {a, b}) {
    if (!source.isRegister(RegisterKind::vgpr))
      continue;
    throw std::invalid_argument(
        std::string(selectName) + " cannot write sgpr destination " +
        registerText(dst) + " from vgpr operand " + registerText(source.reg()));
  }
}

Words wordSources(int width, Operand a, Operand b) {
  if (width == 1)
    return Words{WordSources{a, b}};
  return Words{WordSources{operandHalf(a, 0), operandHalf(b, 0)},
               WordSources{operandHalf(a, 1), operandHalf(b, 1)}};
}

bool readsRegister(const WordSources &sources, RegisterRef reg) {
  for (Operand source : {sources.a, sources.b})
    if (source.isRegister() && source.reg() == reg)
      return true;
  return false;
}

// Writing a half of dst first clobbers that register, which the step that
// writes the other half must not read.
bool isSafeOrder(RegisterRef dst, const Words &words, bool highFirst) {
  int first = highFirst ? 1 : 0;
  return !readsRegister(words[1 - first], dst.slice(first));
}

// Returns whether the high half is written first. When neither order is safe,
// the high step's copy of dst's low register moves to a temporary of dst's kind
// so that low-first works. Callers emit this before reading the condition into
// SCC, since the copy is a plain move.
bool orderHalves(IRBuilder &builder, RegisterRef dst, Words &words,
                 Temporaries &temps) {
  if (words.size() == 1 || isSafeOrder(dst, words, false))
    return false;
  if (isSafeOrder(dst, words, true))
    return true;
  RegisterRef clobbered = dst.slice(0);
  for (Operand *word : {&words[1].a, &words[1].b})
    if (word->isRegister() && word->reg() == clobbered)
      *word = copyToTemporary(builder, dst.kind, 1, *word, temps);
  assert(isSafeOrder(dst, words, false) && "unsafe half order");
  return false;
}

// Visits the words in write order, passing each with its destination register.
template <typename Emit>
void emitWords(RegisterRef dst, const Words &words, bool highFirst, Emit emit) {
  int count = static_cast<int>(words.size());
  for (int i = 0; i < count; ++i) {
    int index = highFirst ? count - 1 - i : i;
    emit(dst.slice(index), words[index]);
  }
}

// Sets SCC to whether the uniform bool is nonzero.
void testCondition(IRBuilder &builder, RegisterRef cond) {
  builder.create<rocisa::SCmpLgU32>(toInput(Operand(cond)), 0);
}

// The aligned pair is written by one instruction, so every operand is read
// before dst is written and aliasing is safe.
void emitScalarSelect64(IRBuilder &builder, RegisterRef dst, RegisterRef cond,
                        Operand a, Operand b) {
  Temporaries temps;
  if (!gfx9::isLegalSalu64Source(a))
    a = copyToTemporary(builder, RegisterKind::sgpr, 2, a, temps);
  if (!gfx9::isLegalSalu64Source(b))
    b = copyToTemporary(builder, RegisterKind::sgpr, 2, b, temps);
  if (!gfx9::isLegalSop2(a, b))
    b = copyToTemporary(builder, RegisterKind::sgpr, 2, b, temps);
  testCondition(builder, cond);
  builder.create<rocisa::SCSelectB64>(dst.container(), toInput64(a),
                                      toInput64(b));
}

// One select per 32-bit word, for a single register or a misaligned pair. The
// condition is read into SCC before any word is written and s_cselect leaves
// SCC alone.
void emitScalarSelectWords(IRBuilder &builder, RegisterRef dst,
                           RegisterRef cond, Operand a, Operand b) {
  Temporaries temps;
  Words words = wordSources(dst.count, a, b);
  for (WordSources &word : words)
    if (!gfx9::isLegalSop2(word.a, word.b))
      word.b = copyToTemporary(builder, RegisterKind::sgpr, 1, word.b, temps);
  bool highFirst = orderHalves(builder, dst, words, temps);
  testCondition(builder, cond);
  emitWords(dst, words, highFirst, [&](RegisterRef part, WordSources word) {
    builder.create<rocisa::SCSelectB32>(part.container(), toInput(word.a),
                                        toInput(word.b));
  });
}

void emitScalarSelect(IRBuilder &builder, RegisterRef dst, RegisterRef cond,
                      Operand a, Operand b) {
  if (dst.count == 2 && dst.start % 2 == 0)
    emitScalarSelect64(builder, dst, cond, a, b);
  else
    emitScalarSelectWords(builder, dst, cond, a, b);
}

// Returns cond if it already is a lane mask. A uniform bool widens to a
// temporary mask; a SALU write followed by a VALU read needs no s_nop.
RegisterRef laneMaskOf(IRBuilder &builder, RegisterRef cond,
                       Temporaries &temps) {
  if (cond.count == 2)
    return cond;
  RegisterRange mask = builder.allocateSgpr(2, 2);
  testCondition(builder, cond);
  builder.create<rocisa::SCSelectB64>(mask.container(), -1, 0);
  temps.push_back(std::move(mask));
  return temps.back().ref();
}

// v_cndmask_b32 takes only VGPRs and inline constants, since the mask already
// uses the constant bus.
void legalizeCndmaskWords(IRBuilder &builder, Words &words,
                          Temporaries &temps) {
  for (WordSources &word : words)
    for (Operand *source : {&word.a, &word.b})
      if (!gfx9::isLegalCndmaskSource(*source))
        *source =
            copyToTemporary(builder, RegisterKind::vgpr, 1, *source, temps);
}

// The mask is read by the cndmasks only, so it cannot alias dst, a VGPR.
void emitVectorSelect(IRBuilder &builder, RegisterRef dst, RegisterRef cond,
                      Operand a, Operand b) {
  Temporaries temps;
  RegisterRef mask = laneMaskOf(builder, cond, temps);
  Words words = wordSources(dst.count, a, b);
  legalizeCndmaskWords(builder, words, temps);
  bool highFirst = orderHalves(builder, dst, words, temps);
  emitWords(dst, words, highFirst, [&](RegisterRef part, WordSources word) {
    // Operand order is src0 = b, src1 = a: a mask bit set selects src1.
    builder.create<rocisa::VCndMaskB32>(part.container(), toInput(word.b),
                                        toInput(word.a), mask.container());
  });
}

} // namespace

void IRBuilder::select(RegisterRef dst, RegisterRef cond, Operand a,
                       Operand b) {
  requireDestination(dst);
  a = validateOperand(dst.count, a);
  b = validateOperand(dst.count, b);
  requireCondition(cond);
  requireScalarSelect(dst, cond, a, b);
  if (dst.kind == RegisterKind::sgpr)
    emitScalarSelect(*this, dst, cond, a, b);
  else
    emitVectorSelect(*this, dst, cond, a, b);
}
