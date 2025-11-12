/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <Tensile/Tensile.hpp>

namespace TensileLite
{
    /**
 * \ingroup Hardware
 * Represents a particular AMD GPU in terms of processor model and number of
 * compute units.
 *
 * See subclass in `hip` directory which can create an instance
 * automatically.
 */
    struct TENSILE_API AMDGPU : public Hardware
    {
        static std::string Type()
        {
            return "AMDGPU";
        }
        virtual std::string type() const;

        enum class Processor : int
        {
            // matching enum used in hipGcnArch
            // only including supported types
            gfx000 = 0,
            //gfx701  =  1,
            //gfx801  =  2,
            //gfx802  =  3,
            gfx803  = 803,
            gfx900  = 900,
            gfx906  = 906,
            gfx908  = 908,
            gfx90a  = 910,
            gfx940  = 940,
            gfx941  = 941,
            gfx942  = 942,
            gfx950  = 950,
            gfx1010 = 1010,
            gfx1011 = 1011,
            gfx1012 = 1012,
            gfx1030 = 1030,
            gfx1100 = 1100,
            gfx1101 = 1101,
            gfx1102 = 1102,
            gfx1103 = 1103,
            gfx1150 = 1150,
            gfx1151 = 1151,
            gfx1200 = 1200,
            gfx1201 = 1201
        };

        static Processor toProcessor(std::string archName)
        {
            if(archName.find("gfx803") != std::string::npos)
            {
                return Processor::gfx803;
            }
            else if(archName.find("gfx900") != std::string::npos)
            {
                return Processor::gfx900;
            }
            else if(archName.find("gfx906") != std::string::npos)
            {
                return Processor::gfx906;
            }
            else if(archName.find("gfx908") != std::string::npos)
            {
                return Processor::gfx908;
            }
            else if(archName.find("gfx90a") != std::string::npos)
            {
                return Processor::gfx90a;
            }
            else if(archName.find("gfx940") != std::string::npos)
            {
                return Processor::gfx940;
            }
            else if(archName.find("gfx941") != std::string::npos)
            {
                return Processor::gfx941;
            }
            else if(archName.find("gfx942") != std::string::npos)
            {
                return Processor::gfx942;
            }
            else if(archName.find("gfx950") != std::string::npos)
            {
                return Processor::gfx950;
            }
            else if(archName.find("gfx1010") != std::string::npos)
            {
                return Processor::gfx1010;
            }
            else if(archName.find("gfx1011") != std::string::npos)
            {
                return Processor::gfx1011;
            }
            else if(archName.find("gfx1012") != std::string::npos)
            {
                return Processor::gfx1012;
            }
            else if(archName.find("gfx1030") != std::string::npos)
            {
                return Processor::gfx1030;
            }
            else if(archName.find("gfx1100") != std::string::npos)
            {
                return Processor::gfx1100;
            }
            else if(archName.find("gfx1101") != std::string::npos)
            {
                return Processor::gfx1101;
            }
            else if(archName.find("gfx1102") != std::string::npos)
            {
                return Processor::gfx1102;
            }
            else if(archName.find("gfx1103") != std::string::npos)
            {
                return Processor::gfx1103;
            }
            else if(archName.find("gfx1150") != std::string::npos)
            {
                return Processor::gfx1150;
            }
            else if(archName.find("gfx1151") != std::string::npos)
            {
                return Processor::gfx1151;
            }
            else if(archName.find("gfx1200") != std::string::npos)
            {
                return Processor::gfx1200;
            }
            else if(archName.find("gfx1201") != std::string::npos)
            {
                return Processor::gfx1201;
            }
            return static_cast<Processor>(0);
        }

        static std::string toString(Processor p)
        {
            switch(p)
            {
            case AMDGPU::Processor::gfx803:
                return "gfx803";
            case AMDGPU::Processor::gfx900:
                return "gfx900";
            case AMDGPU::Processor::gfx906:
                return "gfx906";
            case AMDGPU::Processor::gfx908:
                return "gfx908";
            case AMDGPU::Processor::gfx90a:
                return "gfx90a";
            case AMDGPU::Processor::gfx940:
                return "gfx940";
            case AMDGPU::Processor::gfx941:
                return "gfx941";
            case AMDGPU::Processor::gfx942:
                return "gfx942";
            case AMDGPU::Processor::gfx950:
                return "gfx950";
            case AMDGPU::Processor::gfx1010:
                return "gfx1010";
            case AMDGPU::Processor::gfx1011:
                return "gfx1011";
            case AMDGPU::Processor::gfx1012:
                return "gfx1012";
            case AMDGPU::Processor::gfx1030:
                return "gfx1030";
            case AMDGPU::Processor::gfx1100:
                return "gfx1100";
            case AMDGPU::Processor::gfx1101:
                return "gfx1101";
            case AMDGPU::Processor::gfx1102:
                return "gfx1102";
            case AMDGPU::Processor::gfx1103:
                return "gfx1103";
            case AMDGPU::Processor::gfx1150:
                return "gfx1150";
            case AMDGPU::Processor::gfx1151:
                return "gfx1151";
            case AMDGPU::Processor::gfx1200:
                return "gfx1200";
            case AMDGPU::Processor::gfx1201:
                return "gfx1201";
            case AMDGPU::Processor::gfx000:
                return "gfx000";
            }
            return "";
        }

        AMDGPU();
        AMDGPU(Processor p, int computeUnitCount, std::string const& deviceName);
        ~AMDGPU();

        Processor   processor        = Processor::gfx900;
        int         wavefrontSize    = 64;
        int         simdPerCu        = 4;
        int         computeUnitCount = 0;
        int         skDynamicGrid    = 6;
        int         skDynamicWGM     = 0;
        int         fixedWGM         = std::numeric_limits<int>::max();
        int         fixedWGMXCC      = std::numeric_limits<int>::max();
        int         skMaxCUs         = 0;
        int         skGridMultiplier = 1;
        int         skFixedGrid      = 0;
        int         skFullTiles      = 1;
        mutable int isStandardCUs    = -1; // -1: unset, 0:false, 1:true
        std::string deviceName;

        virtual bool   isStandardCU() const;
        virtual bool   runsKernelTargeting(Processor p) const;
        virtual size_t id() const
        {
            return (size_t)processor;
        }

        virtual std::string archName() const
        {
            return toString(processor);
        }

        virtual std::string description() const;

        const int getSKDynamicGrid() const
        {
            static const char* envStr = std::getenv("TENSILE_STREAMK_DYNAMIC_GRID");
            static const int   value  = (envStr == NULL ? 6 : std::atoi(envStr));
            return value;
        }

        const int getSKDynamicWGM() const
        {
            static const char* envStr = std::getenv("TENSILE_STREAMK_DYNAMIC_WGM");
            static const int   value  = (envStr == NULL ? 0 : std::atoi(envStr));
            return value;
        }

        const int getFixedWGM() const
        {
            static const char* envStr = std::getenv("TENSILE_FIXED_WGM");
            static const int   value  = (envStr == NULL ? std::numeric_limits<int>::max() : std::atoi(envStr));
            return value;
        }

        const int getFixedWGMXCC() const
        {
            static const char* envStr = std::getenv("TENSILE_FIXED_WGMXCC");
            static const int   value  = (envStr == NULL ? std::numeric_limits<int>::max() : std::atoi(envStr));
            return value;
        }

        const int getSKMaxCUs() const
        {
            static const char* envStr = std::getenv("TENSILE_STREAMK_MAX_CUS");
            static const int   value  = (envStr == NULL ? 0 : std::atoi(envStr));
            return value;
        }

        const int getSKGridMultiplier() const
        {
            static const char* envStr = std::getenv("TENSILE_STREAMK_GRID_MULTIPLIER");
            static const int   value  = (envStr == NULL ? 1 : std::atoi(envStr));
            return value;
        }

        const int getSKFixedGrid() const
        {
            static const char* envStr = std::getenv("TENSILE_STREAMK_FIXED_GRID");
            static const int   value  = (envStr == NULL ? 0 : std::atoi(envStr));
            return value;
        }

        const int getSKFullTiles() const
        {
            static const char* envStr = std::getenv("TENSILE_STREAMK_FULL_TILES");
            static const int   value  = (envStr == NULL ? 1 : std::atoi(envStr));
            return value;
        }

        bool operator==(AMDGPU const& rhs) const
        {
            return processor == rhs.processor && computeUnitCount == rhs.computeUnitCount;
        }
    };

    inline bool operator<(AMDGPU::Processor l, AMDGPU::Processor r)
    {
        return static_cast<int>(l) < static_cast<int>(r);
    }

    inline bool operator>(AMDGPU::Processor l, AMDGPU::Processor r)
    {
        return static_cast<int>(l) > static_cast<int>(r);
    }

    inline bool operator<=(AMDGPU::Processor l, AMDGPU::Processor r)
    {
        return static_cast<int>(l) <= static_cast<int>(r);
    }

    inline bool operator>=(AMDGPU::Processor l, AMDGPU::Processor r)
    {
        return static_cast<int>(l) >= static_cast<int>(r);
    }

    TENSILE_API std::ostream& operator<<(std::ostream& stream, AMDGPU::Processor p);
    TENSILE_API std::ostream& operator<<(std::ostream& stream, AMDGPU g);
} // namespace TensileLite
