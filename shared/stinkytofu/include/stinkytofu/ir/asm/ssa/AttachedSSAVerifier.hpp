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

#include <string>
#include <vector>

namespace stinkytofu {

class Function;

/// All attached-SSA invariant violations found in one function, in
/// deterministic order.
struct AttachedSSAVerificationResult {
    std::vector<std::string> errors;

    bool ok() const {
        return errors.empty();
    }

    std::string toString() const;
};

/// Check attached SSA on \p function.
///
/// A function with no attached SSA is valid (pre-lift). Instructions without
/// attached SSA are skipped. Values in the Function arena are always checked
/// for use-list symmetry when any SSA is present.
AttachedSSAVerificationResult verifyAttachedSSA(const Function& function);

}  // namespace stinkytofu
