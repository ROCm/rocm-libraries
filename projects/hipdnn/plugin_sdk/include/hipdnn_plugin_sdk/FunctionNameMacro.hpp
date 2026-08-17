// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/// Expands to the full function signature including template parameters.
/// Useful for diagnostic messages without requiring RTTI.
#ifdef _MSC_VER
#define HIPDNN_FUNCTION_NAME __FUNCSIG__
#else
#define HIPDNN_FUNCTION_NAME __PRETTY_FUNCTION__
#endif
