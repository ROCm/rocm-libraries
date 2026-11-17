// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// What a choice of register banks costs, for an allocator that knows nothing
// about how banks are selected.
//
// Some costs depend on the whole colouring rather than on where one block sits:
// a switch of the VGPR bank word is paid between two instructions, so it is a
// function of the banks of every operand in the stream. The rules table cannot
// say that (docs/developer/register-allocation.md §14.7). A policy can still plan
// against it through this interface, which keeps the encoding and the chip
// behind it, so the policy never names an architecture.

#include <cstdint>
#include <span>

#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"

namespace stinkytofu {

/// The bank of a value whose register is not chosen yet. No register is in it,
/// so it differs from every real bank and agrees only with another undecided
/// value: pricing a neighbour as undecided favours no bank over another.
inline constexpr uint8_t kUnknownBank = 4;

class BankCostModel {
   public:
    virtual ~BankCostModel() = default;

    /// Values whose bank changes the cost, ascending.
    virtual std::span<const SSAValueID> bankSensitiveValues() const = 0;

    /// The heaviest weight among the uses of \p value that the cost counts, or 0
    /// for a value it does not price.
    virtual double useWeight(SSAValueID value) const = 0;

    /// The cost with each value v in bank bankOfValue[v]. A value past the end
    /// of \p bankOfValue is kUnknownBank.
    virtual double cost(std::span<const uint8_t> bankOfValue) const = 0;

    /// How much cost() changes when every value in \p moved, ascending, takes
    /// \p bank and the others stay where \p bankOfValue has them.
    virtual double delta(std::span<const uint8_t> bankOfValue, std::span<const SSAValueID> moved,
                         uint8_t bank) const = 0;
};

}  // namespace stinkytofu
