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

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/PassManager.hpp"

namespace stinkytofu {

/// One named pass in the catalog stinkytofu-opt and the Python binding share.
///
/// The creator receives the comma-separated argument list supplied as
/// `--PassName=arg1,arg2` or `PassName=arg1,arg2`. Passes that take no
/// arguments ignore the vector.
struct PassInfo {
    const char* name;
    std::function<std::unique_ptr<Pass>(const std::vector<std::string>& args)> creator;
};

/// The pass catalog. Creators run when a caller asks for that pass.
STINKYTOFU_EXPORT const std::vector<PassInfo>& passCatalog();

/// The catalog entry for `name`, or null when the name is not in the catalog.
/// Does not run the creator. The pointer stays valid for the life of the process.
STINKYTOFU_EXPORT const PassInfo* findCatalogPass(const std::string& name);

/// Create the named pass. Returns null when the name is unknown or the creator
/// rejects the arguments. `findCatalogPass` tells those apart: a name it finds
/// whose creator then returns null rejected its arguments.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createCatalogPass(const std::string& name,
                                                          const std::vector<std::string>& args);

}  // namespace stinkytofu
