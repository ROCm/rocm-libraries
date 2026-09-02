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
#include "../common/Gfx125xRocisaMaps.hpp"
#include "gfx/InstDefDSL.hpp"

namespace stinkytofu {
// v0 shares v1's rocisa surface exactly; every difference between the two steppings lives in
// Gfx1250v0Instructions.def and Gfx1250v0Formats.def and is limited to instruction cost.
// NOLINTNEXTLINE(misc-use-internal-linkage)
void setGfx1250v0RocisaToArchMap(GpuArch& registry) {
    gfx125x::setRocisaToArchMap(registry);
}

// NOLINTNEXTLINE(misc-use-internal-linkage)
void setGfx1250v0ConversionMap(GpuArch& registry) {
    gfx125x::setConversionMap(registry);
}

}  // namespace stinkytofu
