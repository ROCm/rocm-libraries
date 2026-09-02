// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef UNIQUE_PATH_H
#define UNIQUE_PATH_H

#include <miopen/filesystem.hpp>

namespace miopen {

fs::path unique_path(const fs::path& model = "%%%%-%%%%-%%%%-%%%%");

} // namespace miopen

#endif // UNIQUE_PATH_H
