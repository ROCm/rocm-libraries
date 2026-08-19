// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/// Debug-only, gfx1250-only pass: before every MUBUF load/store touching a
/// tracked tensor SRD (SrdA/SrdB/SrdC/SrdD), inserts a bounds check that
/// reuses real ASan shadow memory -- shadow_addr = (addr >> 3) | 0x7fff8000 --
/// which is only valid/poisoned when the launching client process is itself
/// built with -fsanitize=address. On a shadow-poisoned access, records the
/// checking instruction's PC to the kernarg-supplied AsanReportBuf and halts
/// the wave with s_trap. Does not call real __asan_report_*/hostcall.
/// Enabled via Module.hpp's AsanInstrument option; see Gfx1250Backend.cpp for
/// pipeline placement.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createInsertAsanCheckPass();

}  // namespace stinkytofu
