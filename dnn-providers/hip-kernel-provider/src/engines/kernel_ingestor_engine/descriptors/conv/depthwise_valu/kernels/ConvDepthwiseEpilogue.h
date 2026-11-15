// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The fusion hook of the depthwise kernels (VALU, channel rows, matrix core), over the conv
// engines' epilogue (ConvEpilogue.h). Every depthwise thread owns fixed channels, so a store
// site declares the channel's operands once and applies the epilogue per element before the
// single rounding:
//
//   DW_EPI_DECL(name, ch, ok)   the per-channel operands of channel ch (loaded only when ok)
//   DW_EPI_APPLY(v, name)       the fused fp32 value of v
//
// They read the functor `dwEpi`: a core takes it through DW_EPI_PARAM / DW_EPI_ARG, or binds it
// from the VALU parameter block with DW_EPI_BIND(p). Unset CONV_EPI: every macro expands to
// nothing (DW_EPI_APPLY to v), so the preprocessed source, and the code, are those of an unfused
// build.

#pragma once

#include "ConvEpilogue.h"

#if defined(CONV_EPI) && CONV_EPI
#define DW_EPI_PARAM , const ConvEpiGen& dwEpi
#define DW_EPI_ARG , dwEpi
#define DW_EPI_BIND(p) const ConvEpiGen& dwEpi = (p).epi;
#define DW_EPI_DECL(name, ch, ok) const ConvEpiGen::Chan name = dwEpi.chan((ch), (ok));
#define DW_EPI_APPLY(v, name) dwEpi((v), (name), 0, true)
#else
#define DW_EPI_PARAM
#define DW_EPI_ARG
#define DW_EPI_BIND(p)
#define DW_EPI_DECL(name, ch, ok)
#define DW_EPI_APPLY(v, name) (v)
#endif
