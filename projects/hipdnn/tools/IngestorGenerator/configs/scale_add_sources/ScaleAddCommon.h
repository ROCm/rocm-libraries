// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// A header the bundle ships alongside its source. It is here to make one
// property visible: the generator stages EVERY file in the bundle directory,
// not just the `source_file` a descriptor names. A bundle that shipped only
// the named sources would compile on the author's machine -- where the header
// is a sibling of the config -- and fail on the target at the first #include.

#pragma once

#ifndef SCALE_ADD_DTYPE
#error "SCALE_ADD_DTYPE must be supplied by the compile command (see kernel_source.defines)"
#endif

#ifndef SCALE_ADD_BLOCK_SIZE
#error "SCALE_ADD_BLOCK_SIZE must be supplied by the compile command (see kernel_source.defines)"
#endif

using ScaleAddElement = SCALE_ADD_DTYPE;
