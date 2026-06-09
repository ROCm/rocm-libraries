// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Pure C++ port of the smallest *data-only* decisions made by the subtile emit
// leaves in Tensile/Components/Subtile (Kernel.emitMfmaInstruction,
// SubtileGREmit.emitSingleBufferLoad, SubtileLREmit.emitSingleDsRead).
//
// SCOPE / NON-GOALS
// -----------------
// This header is intentionally free of any nanobind / Python / rocisa
// dependency, matching the rest of the tensile_writer scaffold (pure integer/
// float math, no HIP, no rocisa). It therefore does NOT construct rocisa
// instruction objects. Instead it computes the *plan* — the instruction-shape
// decisions (which MFMA instType to emit; the per-load m0 / DS offsets and
// register strides) — and the Python emit functions build the actual rocisa
// Module from that plan. When delegation is disabled or a case is unsupported
// (MX scale offsets, TLU column-major GR/LR, tail masks, …) the Python side
// keeps its native fall-back path. No GR/LR offset assignment, scale swizzle,
// InstructionEmitter.populate, or mainLoop control flow is modeled here.
//
// The single-buffer-load / single-ds-read plan structs are computed as methods
// on ABTileInfoQuery (tile_info.hpp), reusing the already-ported read-only
// query layer rather than duplicating geometry math.

#pragma once

#include <stdexcept>
#include <string>

namespace tw::subtile::emit {

// ---------------------------------------------------------------------------
// MFMA instType selection for the V_MFMA_SCALE_F32_16x16x128_F8F6F4 family
// (miK == 128). Pure port of Kernel._selectF8F6F4InstType.
//
// The boolean predicates (isFloat8 / isBFloat8 / isFloat4 for A and B) are
// resolved on the Python side from the kernel ProblemType data types — this
// keeps the defensive MagicMock handling in Python and the pure mapping here.
// SourceSwap (handled by the caller swapping A/B before calling, mirroring the
// Python code) is applied here for a self-contained, testable mapping.
//
// Returns the rocisa InstType *member name* (e.g. "INST_F8"); the Python
// caller maps it back to ``rocisa.enum.InstType``. Throws std::runtime_error
// for unsupported combinations, mirroring Python's RuntimeError (the caller
// then falls back to / surfaces the Python behavior).
// ---------------------------------------------------------------------------
inline std::string mfma_f8f6f4_inst_type(bool aIsF8, bool aIsBF8, bool aIsF4,
                                         bool bIsF8, bool bIsBF8, bool bIsF4,
                                         bool sourceSwap) {
  // SourceSwap swaps the A/B operands' formats (Python swaps aType/bType).
  if (sourceSwap) {
    std::swap(aIsF8, bIsF8);
    std::swap(aIsBF8, bIsBF8);
    std::swap(aIsF4, bIsF4);
  }

  // Pure types
  if (aIsF8 && bIsF8) return "INST_F8";
  if (aIsBF8 && bIsBF8) return "INST_BF8";
  if (aIsF4 && bIsF4) return "INST_F4";

  // Mixed FP8 / BF8 (8-bit only)
  if (aIsF8 && bIsBF8) return "INST_F8_BF8";
  if (aIsBF8 && bIsF8) return "INST_BF8_F8";

  // Mixed F8 and F4
  if (aIsF8 && bIsF4) return "INST_F8_F4";
  if (aIsF4 && bIsF8) return "INST_F4_F8";

  // Mixed BF8 and F4
  if (aIsBF8 && bIsF4) return "INST_B8_F4";
  if (aIsF4 && bIsBF8) return "INST_F4_B8";

  throw std::runtime_error(
      "Unsupported data types for MFMA instruction (F8F6F4 family)");
}

}  // namespace tw::subtile::emit
