/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/AMDGPU.hpp>
#include <Tensile/PredicateDebugger.hpp>
#include <Tensile/Predicates.hpp>

#include <map>
#include <set>
#include <sstream>
#include <vector>

namespace TensileLite
{
    /**
     * @brief Registry of known PCI Chip IDs and their fallback relationships.
     *
     * This mirrors the Python-side SUPPORTED_ARCH_DEVICE_IDS and ARCH_DEVICE_ID_FALLBACKS
     * from Architectures.py to ensure consistency between build-time and runtime behavior.
     * @todo Move this definition to a shared configuration file so both the Python-side and
     * runtime side use the same definitions -- single source of truth.
     *
     * Device ID to GFX architecture mapping:
     * - gfx942: 0x74a0, 0x74a1, 0x74a2, 0x74a3, 0x74a5, 0x74a9
     * - gfx950: 0x75a0 (mi350), 0x75a2 (mi355), 0x75a3 (mi355)
     */
    namespace ChipIdRegistry
    {
        // Known/registered chip IDs mapped to their GFX architecture
        inline const std::map<int, AMDGPU::Processor>& knownChipIds()
        {
            static const std::map<int, AMDGPU::Processor> ids = {
                // gfx950 variants
                {0x75a0, AMDGPU::Processor::gfx950}, // mi350
                {0x75b0, AMDGPU::Processor::gfx950}, // mi350
                {0x75a3, AMDGPU::Processor::gfx950}, // mi355x
                {0x75b3, AMDGPU::Processor::gfx950}, // mi355x
                {0x75a8, AMDGPU::Processor::gfx950}, // mi350p
                {0x75b8, AMDGPU::Processor::gfx950}, // mi350p
            };
            return ids;
        }

        // Fallback relationships: chip ID -> fallback chip IDs to try
        // This mirrors ARCH_DEVICE_ID_FALLBACKS from Python side
        inline const std::map<int, std::vector<int>>& chipIdFallbacks()
        {
            static const std::map<int, std::vector<int>> fallbacks = {
                // mi355 chip IDs fall back to mi350
                {0x75a3, {0x75a0}},  // mi355x -> mi350
                {0x75b3, {0x75a0}},  // mi355x -> mi350
                {0x75a8, {0x75a0}},  // mi350p -> mi350
                {0x75b8, {0x75a0}},  // mi350p -> mi350
            };
            return fallbacks;
        }

        // Check if a chip ID is registered/known
        inline bool isKnownChipId(int chipId)
        {
            return knownChipIds().count(chipId) > 0;
        }

        // Get fallback chip IDs for a given chip ID (empty if none)
        inline std::vector<int> getFallbackChipIds(int chipId)
        {
            auto it = chipIdFallbacks().find(chipId);
            if(it != chipIdFallbacks().end())
                return it->second;
            return {};
        }

        // Check if gpuChipId can use a solution targeting solutionChipId
        // Returns true if exact match OR if gpuChipId can fallback to solutionChipId
        inline bool canUseSolution(int gpuChipId, int solutionChipId)
        {
            // Exact match
            if(gpuChipId == solutionChipId)
                return true;

            // Check fallback
            auto fallbacks = getFallbackChipIds(gpuChipId);
            for(int fallbackId : fallbacks)
            {
                if(fallbackId == solutionChipId)
                    return true;
            }
            return false;
        }

        // Check if this is a fallback match (not exact)
        inline bool isFallbackMatch(int gpuChipId, int solutionChipId)
        {
            if(gpuChipId == solutionChipId)
                return false;  // Exact match, not fallback

            auto fallbacks = getFallbackChipIds(gpuChipId);
            for(int fallbackId : fallbacks)
            {
                if(fallbackId == solutionChipId)
                    return true;
            }
            return false;
        }
    } // namespace ChipIdRegistry

    namespace Predicates
    {
        /**
 * \addtogroup Predicates
 * @{
 */
        /**
 * @brief GPU Predicates
 */
        namespace GPU
        {
            struct ProcessorEqual : public Predicate_CRTP<ProcessorEqual, AMDGPU>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                AMDGPU::Processor value;

                ProcessorEqual() = default;
                ProcessorEqual(AMDGPU::Processor p)
                    : value(p)
                {
                }

                static std::string Type()
                {
                    return "Processor";
                }

                virtual bool operator()(AMDGPU const& gpu) const override
                {
                    return gpu.processor == value;
                }

                virtual bool debugEval(AMDGPU const& gpu,
                                       std::ostream& stream) const override
                {
                    return debugEvalCmp(gpu, stream, "gpu", gpu.archName(), "==", "sol", AMDGPU::toString(value));
                }
            };

            struct CUCountEqual : public Predicate_CRTP<CUCountEqual, AMDGPU>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                int value;

                CUCountEqual() = default;
                CUCountEqual(int val)
                    : value(val)
                {
                }

                static std::string Type()
                {
                    return "CUCount";
                }

                virtual bool operator()(AMDGPU const& gpu) const override
                {
                    return gpu.computeUnitCount == value;
                }

                virtual bool debugEval(AMDGPU const& gpu,
                                       std::ostream& stream) const override
                {
                    return debugEvalCmp(gpu, stream, "gpu", gpu.computeUnitCount, "==", "sol", value);
                }
            };

            struct PciChipIDEqual : public Predicate_CRTP<PciChipIDEqual, AMDGPU>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                int value;

                PciChipIDEqual() = default;
                PciChipIDEqual(int val)
                    : value(val)
                {
                }

                static std::string Type()
                {
                    return "PciChipID";
                }

                virtual bool operator()(AMDGPU const& gpu) const override
                {
                    // Must have a chip ID to match
                    if(!gpu.pciChipId().has_value())
                        return false;

                    int gpuChipId = gpu.pciChipId().value();

                    // Check exact match or fallback match
                    return ChipIdRegistry::canUseSolution(gpuChipId, value);
                }

                // Check if this match is a fallback (not exact)
                bool isFallbackMatch(AMDGPU const& gpu) const
                {
                    if(!gpu.pciChipId().has_value())
                        return false;
                    return ChipIdRegistry::isFallbackMatch(gpu.pciChipId().value(), value);
                }

                virtual bool debugEval(AMDGPU const& gpu,
                                       std::ostream& stream) const override
                {
                    bool result = (*this)(gpu);
                    bool isFallback = isFallbackMatch(gpu);

                    std::ostringstream details;
                    details << "[" << gpu.deviceName << "] gpu=";
                    if(gpu.pciChipId().has_value())
                        stream << "0x" << std::hex << gpu.pciChipId().value() << std::dec;
                    else
                        details << "nullopt";
                    details << " == sol=0x" << std::hex << value << std::dec;
                    if(isFallback)
                        details << " (FALLBACK)";

                    PredicateDebugger::printRow(stream, result, this->type(), details.str());

                    if(result && isFallback)
                    {
                        stream << "    [TENSILE FALLBACK] Using fallback kernel: device 0x"
                               << std::hex << gpu.pciChipId().value()
                               << " matched to solution for 0x" << value << std::dec << std::endl;
                    }

                    return result;
                }
            };

            struct RunsKernelTargeting : public Predicate_CRTP<RunsKernelTargeting, AMDGPU>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                AMDGPU::Processor value;

                RunsKernelTargeting() = default;
                RunsKernelTargeting(AMDGPU::Processor p)
                    : value(p)
                {
                }

                static std::string Type()
                {
                    return "TargetProcessor";
                }

                virtual bool operator()(AMDGPU const& gpu) const override
                {
                    return gpu.runsKernelTargeting(value);
                }

                virtual bool debugEval(AMDGPU const& gpu,
                                       std::ostream& stream) const override
                {
                    return debugEvalCmp(gpu, stream, "gpu", gpu.archName(), "can run", "sol", AMDGPU::toString(value));
                }
            };
        } // namespace GPU

        /**
 * @}
 */
    } // namespace Predicates
} // namespace TensileLite
