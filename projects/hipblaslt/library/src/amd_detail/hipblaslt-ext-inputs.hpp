// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The fields of hipblaslt_ext::GemmInputs, for the library's own code. `dPeer` (the second D of
// a PartitionedGemm storePeerD slice) is set only by PartitionedGemm and has no public setter.

#pragma once

#include "hipblaslt/hipblaslt-ext.hpp"

namespace hipblaslt_ext
{
    class GemmInputs::GemmInputsImpl
    {
    public:
        const void* a     = nullptr;
        const void* b     = nullptr;
        const void* c     = nullptr;
        const void* d     = nullptr;
        const void* alpha = nullptr;
        const void* beta  = nullptr;
        // Epilogue inputs
        const void* bias          = nullptr;
        const void* scaleA        = nullptr;
        const void* scaleB        = nullptr;
        const void* scaleC        = nullptr;
        const void* scaleD        = nullptr;
        const void* scaleAux      = nullptr;
        const void* scaleAlphaVec = nullptr;
        const void* aux           = nullptr;
        const void* amaxD         = nullptr;
        void*       dPeer         = nullptr;
        // Keep this layout in sync with rocblaslt::RocGemmInputsV2.
        bool deviceScalarAlpha = false;
    };
} // namespace hipblaslt_ext
