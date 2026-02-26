// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Parameters/Solution/StoreOption.hpp>
#include <rocRoller/Utilities/Error.hpp>

#include <string>

namespace rocRoller
{
    namespace Parameters
    {
        namespace Solution
        {
            MemoryType GetMemoryType(StorePath const& mode)
            {
                switch(mode)
                {
                case StorePath::VGPRToGlobalMemoryWithBuffer:
                    return MemoryType::WAVE;
                case StorePath::VGPRToGlobalMemoryWithGlobal:
                    return MemoryType::WAVE_FROM_GLOBAL;
                case StorePath::VGPRToGlobalMemoryViaLDSWithBuffer:
                    return MemoryType::WAVE_LDS;
                case StorePath::VGPRToGlobalMemoryViaLDSWithGlobal:
                    return MemoryType::WAVE_LDS_FROM_GLOBAL;
                case StorePath::Count:
                    Throw<FatalError>(
                        fmt::format("No valid MemoryType available for mode {}\n", toString(mode)));
                }
            }

            bool IsLDSStore(StorePath const& mode)
            {
                switch(mode)
                {
                case StorePath::VGPRToGlobalMemoryViaLDSWithBuffer:
                case StorePath::VGPRToGlobalMemoryViaLDSWithGlobal:
                    return true;
                default:
                    break;
                }
                return false;
            }

            std::string toString(StorePath mode)
            {
                switch(mode)
                {
                case StorePath::VGPRToGlobalMemoryWithBuffer:
                    return "VGPRToGlobalMemoryWithBuffer";
                case StorePath::VGPRToGlobalMemoryWithGlobal:
                    return "VGPRToGlobalMemoryWithGlobal";
                case StorePath::VGPRToGlobalMemoryViaLDSWithBuffer:
                    return "VGPRToGlobalMemoryViaLDSWithBuffer";
                case StorePath::VGPRToGlobalMemoryViaLDSWithGlobal:
                    return "VGPRToGlobalMemoryViaLDSWithGlobal";
                default:
                    break;
                }
                return "Invalid";
            }

            std::ostream& operator<<(std::ostream& stream, StorePath const& mode)
            {
                return stream << toString(mode);
            }

        } // namespace Solution
    } // namespace Parameters
} // namespace rocRoller
