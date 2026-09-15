// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The bundle's tag-to-type preprocessor, per hiprtc-mining.md "What belongs where":
// `$kernel.dtype` on the pointwise pack renders the flatbuffers enum spelling (FLOAT /
// HALF), never a C++ type name, because pointwiseKernelMatches compares metadata.dtype to
// EnumNameDataType. Mapping the tag to an element type is therefore the bundle's job --
// the substituter evaluates nothing.
//
// The #error arms are the proof that both defines actually bound: if either token failed
// to resolve, this header stops the hipRTC compile with a message naming the macro rather
// than letting the kernel silently compile against a default.

#pragma once

#ifndef HIPDNN_DROPIN_DTYPE
#error "HIPDNN_DROPIN_DTYPE must be supplied by the compile command (kernel_source.defines)"
#endif

#ifndef HIPDNN_DROPIN_BLOCK
#error "HIPDNN_DROPIN_BLOCK must be supplied by the compile command (kernel_source.defines)"
#endif

#define HIPDNN_DROPIN_T_FLOAT float
#define HIPDNN_DROPIN_T_HALF _Float16

// Two levels, deliberately. `##` suppresses expansion of its operands, so the one-level
// form spelled in hiprtc-mining.md's authoring loop pastes the literal macro NAME and
// yields HIPDNN_DROPIN_T_HIPDNN_DROPIN_DTYPE, which does not exist. HIPDNN_DROPIN_CAT
// expands its argument first and HIPDNN_DROPIN_PASTE does the paste.
#define HIPDNN_DROPIN_PASTE(tag) HIPDNN_DROPIN_T_##tag
#define HIPDNN_DROPIN_CAT(tag) HIPDNN_DROPIN_PASTE(tag)

using DropinElement = HIPDNN_DROPIN_CAT(HIPDNN_DROPIN_DTYPE);
