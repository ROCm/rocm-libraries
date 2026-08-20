// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// PathSanitizer turns an arbitrary string (e.g. a scoped engine name such as
// "hipkernel:Pointwise") into a string that is legal as a single path component on every
// platform hipDNN supports, without needing a global registry of previously-seen names.

#include <string>
#include <string_view>

namespace hipdnn_data_sdk::utilities
{

/// Sanitizes @p raw into a single path component safe to use as a directory or file name
/// component on Linux and Windows alike.
///
/// The result is always `<human-readable sanitized stem>-<fixed-width hash of raw>`: the
/// hash suffix is unconditional, appended to every result regardless of whether the
/// sanitized stem alone would have collided with another input. Because the raw input's
/// own hash is always part of the output, two distinct raw inputs never produce the same
/// result -- the mapping is trivially injective and needs no runtime collision detection
/// or registry of names seen so far.
///
/// The sanitized stem portion handles, at minimum:
///  - the colon (every conforming scoped engine name contains exactly one; colons are
///    replaced, not merely tolerated, since they are illegal in a Windows path component);
///  - Windows-reserved stems (CON, PRN, AUX, NUL, COM1-COM9, LPT1-LPT9), matched
///    case-insensitively against the stem before any extension, and altered so the result
///    is not itself a reserved name;
///  - leading and trailing dots, which are stripped (a lone leading dot makes a Unix
///    dotfile, a lone trailing dot is silently dropped by some Windows APIs);
///  - an overall length cap, so the sanitized stem plus hash suffix never exceeds a
///    filesystem's practical component-length limit.
///
/// @param raw The arbitrary, caller-supplied string to sanitize (e.g. an engine name).
///     May be empty; every std::string_view is a valid input.
/// @return A single, non-empty path component safe to use on Linux and Windows. Never
///     throws.
inline std::string sanitizeForPath(std::string_view raw);
// TODO(Stream A): implement in Phase 2

} // namespace hipdnn_data_sdk::utilities
