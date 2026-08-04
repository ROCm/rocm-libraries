/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/ir/asm/RegisterKey.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"

namespace stinkytofu {

class BasicBlock;
class Function;
struct DominanceInfo;
struct StinkyInstruction;

using SSAValueID = uint32_t;
using SSAPhiID = uint32_t;

inline constexpr SSAValueID kInvalidSSAValueID = 0;
inline constexpr SSAPhiID kInvalidSSAPhiID = 0;

/// Shape of a graph that was built by hand rather than lifted, and whose
/// agreement with a function therefore cannot be checked.
inline constexpr uint64_t kUnstampedShape = 0;

enum class SSAValueKind : uint8_t {
    LiveIn,
    Undef,
    InstructionDef,
    Phi,
};

/// Exact use of one SSA register unit.
///
/// A normal instruction use has `instruction != nullptr` and an invalid
/// `phi`. A PHI-edge use has a valid `phi` and `predecessor`.
struct SSAUse {
    const StinkyInstruction* instruction = nullptr;
    uint32_t operand = 0;
    uint32_t unit = 0;

    SSAPhiID phi = kInvalidSSAPhiID;
    const BasicBlock* predecessor = nullptr;

    bool isPhiUse() const {
        return phi != kInvalidSSAPhiID;
    }
};

/// One canonical SSA value representing one atomic register unit.
struct SSAValue {
    SSAValueID id = kInvalidSSAValueID;
    SSAValueKind kind = SSAValueKind::Undef;
    RegKey origin{RegType::UNKNOWN, 0, RegHalf::NONE};

    const StinkyInstruction* definingInstruction = nullptr;
    uint32_t definingOperand = 0;
    uint32_t definingUnit = 0;
    SSAPhiID definingPhi = kInvalidSSAPhiID;

    std::vector<SSAUse> uses;
};

/// SSA values corresponding to the atomic units of one physical operand.
struct SSAOperandBinding {
    std::vector<SSAValueID> units;
};

/// Canonical SSA bindings over one existing StinkyInstruction.
struct SSAInstructionInfo {
    std::vector<SSAOperandBinding> sources;
    std::vector<SSAOperandBinding> destinations;
};

struct SSAPhiIncoming {
    const BasicBlock* predecessor = nullptr;
    SSAValueID value = kInvalidSSAValueID;
};

/// Canonical PHI. This lives in the graph; it is never an emitted GFX::PHI.
struct SSAPhi {
    SSAPhiID id = kInvalidSSAPhiID;
    const BasicBlock* block = nullptr;
    RegKey origin{RegType::UNKNOWN, 0, RegHalf::NONE};
    SSAValueID result = kInvalidSSAValueID;
    std::vector<SSAPhiIncoming> incoming;
};

/// Structural fingerprint of everything a canonical SSA graph depends on:
/// block count, CFG edge counts, instruction count and order, opcodes, and
/// every register operand.
///
/// A graph is only valid for the program it was built from, and no revision
/// counter exists because mutation happens on BasicBlock and on instruction
/// operands, neither of which notifies the Function. Comparing fingerprints at
/// the boundaries that matter catches a stale graph without instrumenting every
/// mutation site. Never returns kUnstampedShape.
STINKYTOFU_EXPORT uint64_t computeFunctionShape(const Function& function);

/// Function-local canonical SSA graph, cached as CanonicalSSAAnalysis.
///
/// Values and PHIs use dense, one-based IDs. Public access is read-only;
/// CanonicalSSABuilder owns the mutation API used during construction.
class STINKYTOFU_EXPORT CanonicalSSA {
   public:
    CanonicalSSA() = default;
    ~CanonicalSSA() = default;

    CanonicalSSA(const CanonicalSSA&) = delete;
    CanonicalSSA& operator=(const CanonicalSSA&) = delete;
    CanonicalSSA(CanonicalSSA&&) noexcept = default;
    CanonicalSSA& operator=(CanonicalSSA&&) noexcept = default;

    bool empty() const;
    size_t valueCount() const;
    size_t phiCount() const;

    bool containsValue(SSAValueID id) const;
    bool containsPhi(SSAPhiID id) const;

    const SSAValue& value(SSAValueID id) const;
    const SSAPhi& phi(SSAPhiID id) const;

    const std::vector<SSAValue>& values() const;
    const std::vector<SSAPhi>& phis() const;

    const SSAInstructionInfo* findInstructionInfo(const StinkyInstruction& instruction) const;
    const std::vector<SSAPhiID>& phisForBlock(const BasicBlock& block) const;

    /// Number of instructions carrying operand bindings. Verification compares
    /// this with the instructions actually reachable from the function.
    size_t instructionInfoCount() const;

    /// Number of blocks carrying a PHI list, used for the same cross-check.
    size_t blockPhiListCount() const;

    /// Fingerprint of the function this graph was lifted from, or
    /// kUnstampedShape for a hand-built graph.
    uint64_t shape() const;

   private:
    friend class CanonicalSSABuilder;

    std::vector<SSAValue> values_;
    std::vector<SSAPhi> phis_;
    std::unordered_map<const StinkyInstruction*, SSAInstructionInfo> instructions_;
    std::unordered_map<const BasicBlock*, std::vector<SSAPhiID>> blockPhis_;
    uint64_t shape_ = kUnstampedShape;
};

/// Low-level construction API for CanonicalSSA.
///
/// This builder only manages graph storage and dense IDs. The lifting
/// algorithm, dominance checks, and verification are separate work.
class STINKYTOFU_EXPORT CanonicalSSABuilder {
   public:
    CanonicalSSABuilder() = default;

    SSAValueID addValue(SSAValue value);
    SSAPhiID addPhi(SSAPhi phi);

    SSAValue& value(SSAValueID id);
    SSAPhi& phi(SSAPhiID id);

    void setInstructionInfo(const StinkyInstruction& instruction, SSAInstructionInfo info);
    void addPhiToBlock(const BasicBlock& block, SSAPhiID phi);

    /// Records which program this graph describes; see computeFunctionShape().
    void setShape(uint64_t shape);

    CanonicalSSA take();

   private:
    CanonicalSSA ssa_;
};

/// All invariant violations found in one graph, in deterministic order.
struct STINKYTOFU_EXPORT CanonicalSSAVerificationResult {
    std::vector<std::string> errors;

    bool ok() const {
        return errors.empty();
    }

    /// One diagnostic per line; empty when the graph is valid.
    std::string toString() const;
};

/// Check the canonical SSA invariants of \p ssa against \p function.
///
/// Verified: graph ownership and dense IDs, one definition per value,
/// definition/use symmetry with operand bindings, operand widths and origin
/// agreement with the physical operands, PHI predecessor coverage and
/// ordering, and same-block definition-before-use ordering.
///
/// Cross-block dominance needs dominance info; pass it to check that too.
STINKYTOFU_EXPORT CanonicalSSAVerificationResult verifyCanonicalSSA(const Function& function,
                                                                    const CanonicalSSA& ssa);

/// As above, and additionally checks that every definition dominates its uses
/// and that every PHI input dominates the end of its predecessor block.
STINKYTOFU_EXPORT CanonicalSSAVerificationResult verifyCanonicalSSA(const Function& function,
                                                                    const CanonicalSSA& ssa,
                                                                    const DominanceInfo& dominance);

struct LiftAsmRegistersToSSAOptions {
    /// Verify the constructed graph before handing it back.
    bool verify = true;

    /// Treat a read with no reaching definition as a function live-in.
    ///
    /// Physical input does not say which registers are genuine kernel inputs,
    /// so this conservative default preserves the meaning of the original
    /// program. Set false to require that every read is defined, which is the
    /// strict mode used once entry metadata is available.
    bool allowInferredLiveIns = true;
};

/// Build canonical SSA from the physical register operands of \p function.
///
/// Physical registers are treated as mutable variables: every reaching
/// definition of a register unit becomes its own SSA value, and each value
/// keeps its originating RegKey for legacy replay.
///
/// Values that merge at a control-flow join become canonical PHIs, placed at
/// iterated dominance frontiers and pruned by liveness so no dead PHI is
/// created. Reducible and irreducible CFGs are both supported.
///
/// Current scope is deliberately narrow. Operands must be full-DWORD VGPRs or
/// SGPRs, and every block must be reachable from the entry. Literals, special
/// registers such as EXEC or SCC, and pseudo registers are ignored rather than
/// lifted. Anything else - accumulator classes, unresolved template virtual
/// registers, True16 halves, calls, or leftover analysis PHIs - is reported as
/// an error instead of being silently mishandled.
///
/// The function must already be free of def-use analysis state; a leftover
/// `GFX::PHI` is an error rather than something to clean up,
/// because this reads the function without modifying it. Run
/// RemoveDefUseAnalysisPass first.
///
/// Construction is atomic: on error nothing is returned, so a caller can never
/// receive a partially built graph.
STINKYTOFU_EXPORT Expected<CanonicalSSA> liftAsmRegistersToSSA(
    Function& function, const LiftAsmRegistersToSSAOptions& options = {});

/// As above, reusing dominance information the caller already computed. This is
/// what CanonicalSSAAnalysis runs.
STINKYTOFU_EXPORT Expected<CanonicalSSA> liftAsmRegistersToSSA(
    Function& function, const DominanceInfo& dominance,
    const LiftAsmRegistersToSSAOptions& options = {});

}  // namespace stinkytofu
