// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Parameters/Solution/LoadOption.hpp>
#include <rocRoller/Utilities/Error.hpp>

#include <string>

namespace rocRoller
{
    namespace Parameters
    {
        namespace Solution
        {
            MemoryType GetMemoryType(LoadPath const& mode)
            {
                switch(mode)
                {
                case LoadPath::BufferToVGPR:
                    return MemoryType::WAVE;
                case LoadPath::BufferToLDSViaVGPR:
                    return MemoryType::WAVE_LDS;
                case LoadPath::BufferToLDS:
                    return MemoryType::WAVE_Direct2LDS;
                case LoadPath::GlobalToVGPR:
                    return MemoryType::WAVE_FROM_GLOBAL;
                case LoadPath::GlobalToLDSViaVGPR:
                    return MemoryType::WAVE_LDS_FROM_GLOBAL;
                case LoadPath::Count:
                    Throw<FatalError>(
                        fmt::format("No valid MemoryType available for mode {}\n", toString(mode)));
                }
            }

            bool IsBufferToLDS(LoadPath const& mode)
            {
                return mode == LoadPath::BufferToLDS;
            }

            bool IsPathToLDS(LoadPath const& mode)
            {
                switch(mode)
                {
                case LoadPath::BufferToLDSViaVGPR:
                case LoadPath::BufferToLDS:
                case LoadPath::GlobalToLDSViaVGPR:
                    return true;
                default:
                    break;
                }
                return false;
            }

            std::string toString(LoadPath mode)
            {
                switch(mode)
                {
                case LoadPath::BufferToVGPR:
                    return "BufferToVGPR";
                case LoadPath::BufferToLDSViaVGPR:
                    return "BufferToLDSViaVGPR";
                case LoadPath::BufferToLDS:
                    return "BufferToLDS";
                case LoadPath::GlobalToVGPR:
                    return "GlobalToVGPR";
                case LoadPath::GlobalToLDSViaVGPR:
                    return "GlobalToLDSViaVGPR";
                default:
                    break;
                }
                return "Invalid";
            }

            std::ostream& operator<<(std::ostream& stream, LoadPath const& mode)
            {
                return stream << toString(mode);
            }

        } // namespace Solution
    } // namespace Parameters
} // namespace rocRoller
