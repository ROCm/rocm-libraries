/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iostream>
#include <string>
#include <tuple>
#include <unordered_map>

#include <hip/hip_runtime.h>

namespace TensileLite
{
    namespace analytical
    {
        struct MatrixInstruction
        {
            size_t MI_M;
            size_t MI_N;
            size_t MI_K;
            size_t element_size;

            MatrixInstruction()
                : MI_M(0)
                , MI_N(0)
                , MI_K(0)
                , element_size(0)
            {
            }

            MatrixInstruction(size_t m, size_t n, size_t k, size_t element_size)
                : MI_M(m)
                , MI_N(n)
                , MI_K(k)
                , element_size(element_size)
            {
            }

            MatrixInstruction(const MatrixInstruction& other)
                : MI_M(other.MI_M)
                , MI_N(other.MI_N)
                , MI_K(other.MI_K)
                , element_size(other.element_size)
            {
            }

            bool operator<(const MatrixInstruction& other) const
            {
                return std::tie(MI_M, MI_N, MI_K, element_size)
                       < std::tie(other.MI_M, other.MI_N, other.MI_K, other.element_size);
            }

            bool operator==(const MatrixInstruction& other) const
            {
                return MI_M == other.MI_M && MI_N == other.MI_N && MI_K == other.MI_K
                       && element_size == other.element_size;
            }

            std::size_t hash() const
            {
                return std::hash<size_t>()(MI_M) ^ std::hash<size_t>()(MI_N)
                       ^ std::hash<size_t>()(MI_K) ^ std::hash<size_t>()(element_size);
            }
        };
    }
}

// Specialize std::hash for the MatrixInstruction struct to use it as an unordered_map key.
namespace std
{
    template <>
    struct hash<TensileLite::analytical::MatrixInstruction>
    {
        std::size_t operator()(const TensileLite::analytical::MatrixInstruction& k) const
        {
            return k.hash();
        }
    };
}

namespace TensileLite
{
    namespace analytical
    {
        class Hardware
        {
        public:
            enum class Architecture
            {
                gfx942,
                gfx950,
                Count
            };

            static Architecture archNameToEnum(const std::string& str)
            {
                static const std::unordered_map<std::string, Architecture> strToEnumMap
                    = {{"gfx942", Architecture::gfx942}, {"gfx950", Architecture::gfx950}};

                auto it = strToEnumMap.find(str);
                if(it != strToEnumMap.end())
                {
                    return it->second;
                }
                else
                {
                    return Architecture::Count;
                }
            }

            struct ArchitectureConstants
            {
                size_t num_xcds;
                double mem1_perf_ratio;
                double mem2_perf_ratio;
                double mem3_perf_ratio;
                size_t parallel_MI_CU;
                double percent_bw_per_wg;
                double mem_clock_ratio;
                ArchitectureConstants(size_t num_xcds,
                                      double mem1_perf_ratio,
                                      double mem2_perf_ratio,
                                      double mem3_perf_ratio,
                                      size_t parallel_MI_CU,
                                      double percent_bw_per_wg,
                                      double mem_clock_ratio) //Obtained through microbenchmarking
                    : num_xcds(num_xcds)
                    , mem1_perf_ratio(mem1_perf_ratio)
                    , mem2_perf_ratio(mem2_perf_ratio)
                    , mem3_perf_ratio(mem3_perf_ratio)
                    , parallel_MI_CU(parallel_MI_CU)
                    , percent_bw_per_wg(percent_bw_per_wg)
                    , mem_clock_ratio(mem_clock_ratio)
                {
                }
            };

            static const std::unordered_map<Architecture, ArchitectureConstants> ARCH_CONSTANT_MAP;

            static const std::unordered_map<Architecture,
                                            std::unordered_map<MatrixInstruction, size_t>>
                INSTRUCTION_MAP;

            Architecture arch;
            size_t       N_CU; // Number of Compute Units
            size_t       LDS_capacity; // Capacity of LDS
            double       mem1_perf_ratio;
            double       mem2_perf_ratio;
            double       mem3_perf_ratio;
            size_t       L2_capacity; // Capacity of L2 in bytes
            size_t       CU_per_L2; // Number of compute units per L2 domain
            double       compute_clock_ghz;
            size_t       parallel_MI_CU; // The number of parallel MI in a CU
            double       percent_bw_per_wg;
            size_t       NUM_XCD;

            Hardware(Architecture arch,
                     size_t       N_CU,
                     size_t       LDS_capacity,
                     size_t       NUM_XCD,
                     double       mem1_perf_ratio,
                     double       mem2_perf_ratio,
                     double       mem3_perf_ratio,
                     size_t       L2_capacity,
                     double       compute_clock_ghz,
                     size_t       parallel_MI_CU,
                     double       percent_bw_per_wg)
                : arch(arch)
                , N_CU(N_CU)
                , LDS_capacity(LDS_capacity)
                , mem1_perf_ratio(mem1_perf_ratio)
                , mem2_perf_ratio(mem2_perf_ratio)
                , mem3_perf_ratio(mem3_perf_ratio)
                , L2_capacity(L2_capacity)
                , CU_per_L2(N_CU / NUM_XCD)
                , compute_clock_ghz(compute_clock_ghz)
                , parallel_MI_CU(parallel_MI_CU)
                , percent_bw_per_wg(percent_bw_per_wg)
                , NUM_XCD(NUM_XCD)
            {
                if(Hardware::is_debug_enabled())
                {
                    print();
                }
            }

            Hardware(hipDeviceProp_t properties)
                : Hardware(getHardwareForProperties(properties))
            {
            }

            Hardware(const Hardware& other)
                : arch(other.arch)
                , N_CU(other.N_CU)
                , LDS_capacity(other.LDS_capacity)
                , mem1_perf_ratio(other.mem1_perf_ratio)
                , mem2_perf_ratio(other.mem2_perf_ratio)
                , mem3_perf_ratio(other.mem3_perf_ratio)
                , L2_capacity(other.L2_capacity)
                , CU_per_L2(other.CU_per_L2)
                , compute_clock_ghz(other.compute_clock_ghz)
                , parallel_MI_CU(other.parallel_MI_CU)
                , percent_bw_per_wg(other.percent_bw_per_wg)
                , NUM_XCD(other.NUM_XCD)
            {
            }

            static Hardware getHardwareForProperties(hipDeviceProp_t properties)
            {
                auto archName = get_before_first_colon(properties.gcnArchName);
                auto archEnum = archNameToEnum(archName);
                auto it       = ARCH_CONSTANT_MAP.find(archEnum);
                if(it == ARCH_CONSTANT_MAP.end())
                {
                    throw std::runtime_error(
                        "Attempting to retrieve hardware constants for unsupported architecture: "
                        + archName); // Could also return default values here.
                }
                auto constants = it->second;
                return Hardware(archEnum,
                                properties.multiProcessorCount,
                                properties.sharedMemPerBlock,
                                constants.num_xcds,
                                1e9 * constants.mem1_perf_ratio / properties.clockRate,
                                1e9 * constants.mem2_perf_ratio
                                    / (properties.memoryClockRate * constants.mem_clock_ratio),
                                1e9 * constants.mem3_perf_ratio / properties.memoryClockRate,
                                properties.l2CacheSize,
                                properties.clockRate / 1e6,
                                constants.parallel_MI_CU,
                                constants.percent_bw_per_wg);
            }

            static Hardware getHardwareForDevice(int deviceId)
            {
                hipDeviceProp_t prop;
                hipError_t      e = hipGetDeviceProperties(&prop, deviceId);
                if(e)
                {
                    throw std::runtime_error(hipGetErrorString(e));
                }
                return getHardwareForProperties(prop);
            }

            static bool isHardwareSupported(hipDeviceProp_t properties)
            {
                auto archName = get_before_first_colon(properties.gcnArchName);
                auto archEnum = archNameToEnum(archName);
                auto it       = ARCH_CONSTANT_MAP.find(archEnum);
                return it != ARCH_CONSTANT_MAP.end();
            }

            // Function to print hardware details
            void print()
            {
                std::cout << "================== Hardware Configuration ==================\n";
                std::cout << "Number of CUs (N_CU)       : " << N_CU << "\n";
                std::cout << "LDS capacity              : " << LDS_capacity << " bytes\n";
                std::cout << "mem1_perf_ratio           : " << mem1_perf_ratio << "\n";
                std::cout << "mem2_perf_ratio           : " << mem2_perf_ratio << "\n";
                std::cout << "mem3_perf_ratio           : " << mem3_perf_ratio << "\n";
                std::cout << "L2 Cache capacity         : " << L2_capacity << " bytes\n";
                std::cout << "CUs per L2 domain         : " << CU_per_L2 << "\n";
                std::cout << "Compute clock (GHz)       : " << compute_clock_ghz << "\n";
                std::cout << "Parallel MI/CU            : " << parallel_MI_CU << "\n";
                std::cout << "Number of XCDs (NUM_XCD)  : " << NUM_XCD << "\n";
                std::cout << "percent_bw_per_wg         : " << percent_bw_per_wg << "\n\n";

                std::cout << "------------------ Instruction Map -------------------------\n";
                // Loop over the instruction_map and print each entry
                for(const auto& kv : INSTRUCTION_MAP.at(arch))
                {
                    const auto& key  = kv.first;
                    const auto& L_MI = kv.second;

                    std::cout << "Instruction: MI_M=" << key.MI_M << ", MI_N=" << key.MI_N
                              << ", MI_K=" << key.MI_K << ", element_size=" << key.element_size
                              << " bytes\n"
                              << "  -> Latency (L_MI): " << L_MI << "\n";
                }
                std::cout << "===========================================================\n";
            }
            // Debug tracking info
            mutable std::unordered_map<std::string, std::string> debug_info;

            static bool is_debug_enabled()
            {
                static bool debugEnvVar = read_debug_env_var(); //Used to cache the read.
                return debugEnvVar;
            }

            void log_debug(const std::string& key, const std::string& value) const
            {
                debug_info[key] = value;
            }

            void log_debug(const std::string& key, double value) const
            {
                debug_info[key] = std::to_string(value);
            }

            void clear_debug() const
            {
                debug_info.clear();
            }

            void print_debug_info() const
            {
                std::cout << "=== Hardware Debug Info ===\n";
                for(const auto& [key, val] : debug_info)
                {
                    std::cout << key << ": " << val << "\n";
                }
                std::cout << "===========================\n";
            }

            size_t get_MI_latency(size_t MI_M, size_t MI_N, size_t MI_K, size_t element_size) const
            {
                const auto& instruction_map = INSTRUCTION_MAP.at(arch);
                auto        key             = MatrixInstruction(MI_M, MI_N, MI_K, element_size);

                auto it = instruction_map.find(key);
                if(it != instruction_map.end())
                {
                    return it->second / parallel_MI_CU;
                }
                else
                {
                    std::cerr << "Warning: Latency not found for MI_M=" << MI_M << ", MI_N=" << MI_N
                              << ", MI_K=" << MI_K << ", Element_Size=" << element_size
                              << ". Returning latency value of 32 (really slow).\n";
                    return 32 / parallel_MI_CU; // Default latency if instruction is not found
                }
            }

        private:
            static std::string get_before_first_colon(const std::string& input)
            {
                size_t pos = input.find(':');
                if(pos != std::string::npos)
                {
                    return input.substr(0, pos);
                }
                return input; // Return the whole string if ':' is not found
            }

            // Helper function to read the debug environment variable
            static bool read_debug_env_var()
            {
                const char* env = std::getenv("ANALYTICAL_GEMM_DEBUG");
                return env && std::string(env) == "1";
            }
        };

    } // namespace analytical
} // namespace TensileLite
