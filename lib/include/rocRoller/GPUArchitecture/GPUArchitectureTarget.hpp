
#pragma once

#include <array>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <rocRoller/Serialization/Base_fwd.hpp>

namespace rocRoller
{

    /**
     * @brief Represents GFX architecture version IDs
     *
     * NOTE: When adding new arches, be sure to
     * - Add the enum
     * - Update the `toString()` method
     * - Update relevant filters like `isGFX9X()`
     * - Double check Observers and other places the filters might be used
     *
     */
    enum class GPUArchitectureGFX : int32_t
    {
        UNKNOWN = 0,
        GFX803,
        GFX900,
        GFX906,
        GFX908,
        GFX90A,
        GFX940,
        GFX941,
        GFX942,
        GFX950,
        GFX1010,
        GFX1011,
        GFX1012,
        GFX1030,

        Count,
    };
    std::string toString(GPUArchitectureGFX const& gfx);

    struct GPUArchitectureFeatures
    {
    public:
        bool sramecc = false;
        bool xnack   = false;

        std::string toString() const;

        // Return a string of features that can be provided as input to the LLVM Assembler.
        // These should have the ON/OFF symbol in front of each feature, and be comma
        // delimmited.
        std::string toLLVMString() const;

        auto operator<=>(const GPUArchitectureFeatures&) const = default;
    };

    struct GPUArchitectureTarget
    {
    public:
        GPUArchitectureGFX      gfx      = GPUArchitectureGFX::UNKNOWN;
        GPUArchitectureFeatures features = {};

        static GPUArchitectureTarget fromString(std::string const& archStr);
        std::string                  toString() const;

        constexpr bool is10XGPU() const
        {
            return gfx == GPUArchitectureGFX::GFX1010 || gfx == GPUArchitectureGFX::GFX1011
                   || gfx == GPUArchitectureGFX::GFX1012 || gfx == GPUArchitectureGFX::GFX1030;
        }

        constexpr bool is9XGPU() const
        {
            return gfx == GPUArchitectureGFX::GFX900 || gfx == GPUArchitectureGFX::GFX906
                   || gfx == GPUArchitectureGFX::GFX908 || gfx == GPUArchitectureGFX::GFX90A
                   || gfx == GPUArchitectureGFX::GFX940 || gfx == GPUArchitectureGFX::GFX941
                   || gfx == GPUArchitectureGFX::GFX942 || gfx == GPUArchitectureGFX::GFX950;
        }

        constexpr bool is908GPU() const
        {
            return gfx == GPUArchitectureGFX::GFX908;
        }

        constexpr bool is90aGPU() const
        {
            return gfx == GPUArchitectureGFX::GFX90A;
        }

        constexpr bool is94XGPU() const
        {
            return gfx == GPUArchitectureGFX::GFX940 || gfx == GPUArchitectureGFX::GFX941
                   || gfx == GPUArchitectureGFX::GFX942;
        }

        constexpr bool is950GPU() const
        {
            return gfx == GPUArchitectureGFX::GFX950;
        }

        auto operator<=>(const GPUArchitectureTarget&) const = default;

    private:
        template <typename T1, typename T2, typename T3>
        friend struct rocRoller::Serialization::MappingTraits;
    };

    inline std::ostream& operator<<(std::ostream& os, GPUArchitectureTarget const& input)
    {
        os << input.toString();
        return os;
    }

    inline std::istream& operator>>(std::istream& is, GPUArchitectureTarget& input)
    {
        std::string recvd;
        is >> recvd;
        input = GPUArchitectureTarget::fromString(recvd);
        return is;
    }

    inline std::string toString(GPUArchitectureFeatures const& feat)
    {
        return feat.toString();
    }

    inline std::string toString(GPUArchitectureTarget const& target)
    {
        return target.toString();
    }

}

#include "GPUArchitectureTarget_impl.hpp"
