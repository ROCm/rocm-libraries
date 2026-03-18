// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "KernelOptions_fwd.hpp"

#include <memory>
#include <ostream>
#include <string>

#include <rocRoller/AssertOpKinds_fwd.hpp>
#include <rocRoller/Operations/Scratch_fwd.hpp>
#include <rocRoller/Utilities/EnumBitset.hpp>
#include <rocRoller/Utilities/Settings_fwd.hpp>

namespace rocRoller
{
    const std::string XLOOP   = "XLoop";
    const std::string YLOOP   = "YLoop";
    const std::string KLOOP   = "KLoop";
    const std::string RECEIVE = "ReceiveTileLoop";

    const int XLOOP_UNROLL = 0;
    const int YLOOP_UNROLL = 1;
    const int KLOOP_UNROLL = 2;

    const std::string KLOOPTAIL = KLOOP + "Tail";

    const std::string SCRATCH = "SCRATCH";
    const std::string NUMWGS  = "numWGs";
    const std::string WGM     = "WGM";

    // Helper to get scratch argument name for a specific policy
    inline std::string getScratchName(Operations::ScratchPolicy policy)
    {
        return rocRoller::SCRATCH + "_" + Operations::toString(policy);
    }

    class KernelOptions
    {
    public:
        KernelOptions();
        // cppcheck-suppress noExplicitConstructor
        KernelOptions(KernelOptionValues&& other);

        KernelOptions(KernelOptions const& other);
        KernelOptions(KernelOptions&& other);

        KernelOptions& operator=(KernelOptions const& other);
        KernelOptions& operator=(KernelOptions&& other);

        KernelOptions& operator=(KernelOptionValues const& other);
        KernelOptions& operator=(KernelOptionValues&& other);

        ~KernelOptions();

        KernelOptionValues* operator->();
        KernelOptionValues& operator*();

        KernelOptionValues const* operator->() const;
        KernelOptionValues const& operator*() const;

        std::string          toString() const;
        friend std::ostream& operator<<(std::ostream&, const KernelOptions&);

    private:
        std::unique_ptr<KernelOptionValues> m_values;
    };
}
