// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// TENSILE_API is intended to control visibility. However, in the new build system there
// is no context where we would ever build tensile as a shared library so we are unconditionally
// defining it as empty and in the build system we define the visibility flags. In the future, if we
// ever build tensile as a shared library, we will need to use CMakes generate_export_header macro
// to provide this functionality rather than using a roll-your-own approach.

#define TENSILE_API
