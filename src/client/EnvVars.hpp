/*
Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef ENVVARS_HPP
#define ENVVARS_HPP

// Helper macro for catching HIP errors
#define HIP_CALL(cmd)                                                           \
  do {                                                                          \
    hipError_t error = (cmd);                                                   \
    if (error != hipSuccess) {                                                  \
      std::cerr << "Encountered HIP error (" << hipGetErrorString(error)        \
                << ") at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
      exit(-1);                                                                 \
    }                                                                           \
  } while (0)

#include <algorithm>
#include <iostream>
#include <numa.h>
#include <random>
#include <time.h>
#include "Client.hpp"

#include "TransferBench.hpp"
using namespace TransferBench;

// Redefinitions for CUDA compatibility
//==========================================================================================
#if defined(__NVCC__)
  #define hipError_t                                         cudaError_t
  #define hipGetErrorString                                  cudaGetErrorString
  #define hipDeviceProp_t                                    cudaDeviceProp
  #define hipDeviceGetPCIBusId                               cudaDeviceGetPCIBusId
  #define hipGetDeviceProperties                             cudaGetDeviceProperties
  #define hipSuccess                                         cudaSuccess
  #define gcnArchName                                        name
  #define hipGetDeviceCount                                  cudaGetDeviceCount
#endif

// This class manages environment variable that affect TransferBench
class EnvVars
{
public:
  // Default configuration values
  int const DEFAULT_SAMPLING_FACTOR = 1;

  // Environment variables
  // General options
  int numIterations;                 // Number of timed iterations to perform.  If negative, run for -numIterations seconds instead
  int numSubIterations;              // Number of subiterations to perform
  int numWarmups;                    // Number of un-timed warmup iterations to perform
  int showIterations;                // Show per-iteration timing info
  int useInteractive;                // Pause for user-input before starting transfer loop

  // Data options
  int alwaysValidate;                // Validate after each iteration instead of once after all iterations
  int blockBytes;                    // Each subexecutor, except the last, gets a multiple of this many bytes to copy
  int byteOffset;                    // Byte-offset for memory allocations
  vector<float> fillPattern;         // Pattern of floats used to fill source data
  int validateDirect;                // Validate GPU destination memory directly instead of staging GPU memory on host
  int validateSource;                // Validate source GPU memory immediately after preparation

  // DMA options
  int useHsaDma;                     // Use hsa_amd_async_copy instead of hipMemcpy for non-targetted DMA executions

  // GFX options
  int gfxBlockSize;                  // Size of each threadblock (must be multiple of 64)
  vector<uint32_t> cuMask;           // Bit-vector representing the CU mask
  vector<vector<int>> prefXccTable;  // Specifies XCC to use for given exe->dst pair
  int gfxUnroll;                     // GFX-kernel unroll factor
  int useHipEvents;                  // Use HIP events for timing GFX/DMA Executor
  int useSingleStream;               // Use a single stream per GPU GFX executor instead of stream per Transfer
  int gfxSingleTeam;                 // Team all subExecutors across the data array
  int gfxWaveOrder;                  // GFX-kernel wavefront ordering

  // Client options
  int hideEnv;                       // Skip printing environment variable
  int minNumVarSubExec;              // Minimum # of subexecutors to use for variable subExec Transfers
  int maxNumVarSubExec;              // Maximum # of subexecutors to use for variable subExec Transfers (0 to use device limit)
  int outputToCsv;                   // Output in CSV format
  int samplingFactor;                // Affects how many different values of N are generated (when N set to 0)

  // NIC options
  int ibGidIndex;                    // GID Index for RoCE NICs
  int roceVersion;                   // RoCE version number
  int ipAddressFamily;               // IP Address Famliy
  uint8_t ibPort;                    // NIC port number to be used
  int nicRelaxedOrder;               // Use relaxed ordering for RDMA
  std::string closestNicStr;         // Holds the user-specified list of closest NICs

  // Developer features
  int gpuMaxHwQueues;                // Tracks GPU_MAX_HW_QUEUES environment variable

  // Constructor that collects values
  EnvVars()
  {
    int numDetectedCpus = TransferBench::GetNumExecutors(EXE_CPU);
    int numDetectedGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);
    int numDeviceCUs    = TransferBench::GetNumSubExecutors({EXE_GPU_GFX, 0});

    hipDeviceProp_t prop;
    HIP_CALL(hipGetDeviceProperties(&prop, 0));
    std::string fullName = prop.gcnArchName;
    std::string archName = fullName.substr(0, fullName.find(':'));

    // Different hardware pick different GPU kernels
    // This performance difference is generally only noticable when executing fewer CUs
    int defaultGfxUnroll = 4;
    if      (archName == "gfx906") defaultGfxUnroll = 8;
    else if (archName == "gfx90a") defaultGfxUnroll = 8;
    else if (archName == "gfx940") defaultGfxUnroll = 6;
    else if (archName == "gfx941") defaultGfxUnroll = 6;
    else if (archName == "gfx942") defaultGfxUnroll = 4;

    alwaysValidate    = GetEnvVar("ALWAYS_VALIDATE"     , 0);
    blockBytes        = GetEnvVar("BLOCK_BYTES"         , 256);
    byteOffset        = GetEnvVar("BYTE_OFFSET"         , 0);
    gfxBlockSize      = GetEnvVar("GFX_BLOCK_SIZE"      , 256);
    gfxSingleTeam     = GetEnvVar("GFX_SINGLE_TEAM"     , 1);
    gfxUnroll         = GetEnvVar("GFX_UNROLL"          , defaultGfxUnroll);
    gfxWaveOrder      = GetEnvVar("GFX_WAVE_ORDER"      , 0);
    hideEnv           = GetEnvVar("HIDE_ENV"            , 0);
    minNumVarSubExec  = GetEnvVar("MIN_VAR_SUBEXEC"     , 1);
    maxNumVarSubExec  = GetEnvVar("MAX_VAR_SUBEXEC"     , 0);
    numIterations     = GetEnvVar("NUM_ITERATIONS"      , 10);
    numSubIterations  = GetEnvVar("NUM_SUBITERATIONS"   , 1);
    numWarmups        = GetEnvVar("NUM_WARMUPS"         , 3);
    outputToCsv       = GetEnvVar("OUTPUT_TO_CSV"       , 0);
    samplingFactor    = GetEnvVar("SAMPLING_FACTOR"     , 1);
    showIterations    = GetEnvVar("SHOW_ITERATIONS"     , 0);
    useHipEvents      = GetEnvVar("USE_HIP_EVENTS"      , 1);
    useHsaDma         = GetEnvVar("USE_HSA_DMA"         , 0);
    useInteractive    = GetEnvVar("USE_INTERACTIVE"     , 0);
    useSingleStream   = GetEnvVar("USE_SINGLE_STREAM"   , 1);
    validateDirect    = GetEnvVar("VALIDATE_DIRECT"     , 0);
    validateSource    = GetEnvVar("VALIDATE_SOURCE"     , 0);

    ibGidIndex        = GetEnvVar("IB_GID_INDEX"        ,-1);
    ibPort            = GetEnvVar("IB_PORT_NUMBER"      , 1);
    roceVersion       = GetEnvVar("ROCE_VERSION"        , 2);
    ipAddressFamily   = GetEnvVar("IP_ADDRESS_FAMILY"   , 4);
    nicRelaxedOrder   = GetEnvVar("NIC_RELAX_ORDER"     , 1);
    closestNicStr     = GetEnvVar("CLOSEST_NIC"         , "");

    gpuMaxHwQueues    = GetEnvVar("GPU_MAX_HW_QUEUES"   , 4);


    // Check for fill pattern
    char* pattern = getenv("FILL_PATTERN");
    if (pattern != NULL) {
      int patternLen = strlen(pattern);
      if (patternLen % 2) {
        printf("[ERROR] FILL_PATTERN must contain an even-number of hex digits\n");
        exit(1);
      }

      // Read in bytes
      std::vector<unsigned char> bytes;
      unsigned char val = 0;
      for (int i = 0; i < patternLen; i++) {
        if ('0' <= pattern[i] && pattern[i] <= '9')
          val += (pattern[i] - '0');
        else if ('A' <= pattern[i] && pattern[i] <= 'F')
          val += (pattern[i] - 'A' + 10);
        else if ('a' <= pattern[i] && pattern[i] <= 'f')
          val += (pattern[i] - 'a' + 10);
        else {
          printf("[ERROR] FILL_PATTERN must contain an even-number of hex digits (0-9'/a-f/A-F).  (not %c)\n", pattern[i]);
          exit(1);
        }

        if (i % 2 == 0)
          val <<= 4;
        else {
          bytes.push_back(val);
          val = 0;
        }
      }

      // Reverse bytes (input is assumed to be given in big-endian)
      std::reverse(bytes.begin(), bytes.end());

      // Figure out how many copies of the pattern are necessary to fill a 4-byte float properly
      int copies;
      switch (patternLen % 8) {
      case 0:  copies = 1; break;
      case 4:  copies = 2; break;
      default: copies = 4; break;
      }

      // Fill floats
      int numFloats = copies * patternLen / 8;
      fillPattern.resize(numFloats);
      unsigned char* rawData = (unsigned char*) fillPattern.data();
      for (int i = 0; i < numFloats * 4; i++)
        rawData[i] = bytes[i % bytes.size()];
    }
    else fillPattern.clear();

    // Check for CU mask
    int numXccs = TransferBench::GetNumExecutorSubIndices({EXE_GPU_GFX, 0});
    cuMask.clear();
    char* cuMaskStr = getenv("CU_MASK");
    if (cuMaskStr != NULL) {
#if defined(__NVCC__)
      printf("[WARN] CU_MASK is not supported in CUDA\n");
#else
      std::vector<std::pair<int, int>> ranges;
      int maxCU = 0;
      char* token = strtok(cuMaskStr, ",");
      while (token) {
        int start, end;
        if (sscanf(token, "%d-%d", &start, &end) == 2) {
          ranges.push_back(std::make_pair(std::min(start, end), std::max(start, end)));
          maxCU = std::max(maxCU, std::max(start, end));
        } else if (sscanf(token, "%d", &start) == 1) {
          ranges.push_back(std::make_pair(start, start));
          maxCU = std::max(maxCU, start);
        } else {
          printf("[ERROR] Unrecognized token [%s]\n", token);
          exit(1);
        }
        token = strtok(NULL, ",");
      }
      cuMask.resize(2 * numXccs, 0);

      for (auto range : ranges) {
        for (int i = range.first; i <= range.second; i++) {
          for (int x = 0; x < numXccs; x++) {
            int targetBit = i * numXccs + x;
            cuMask[targetBit/32] |= (1<<(targetBit%32));
          }
        }
      }
#endif
    }

    // Parse preferred XCC table (if provided)
    char* prefXccStr = getenv("XCC_PREF_TABLE");
    if (prefXccStr) {
      prefXccTable.resize(numDetectedGpus);
      for (int i = 0; i < numDetectedGpus; i++){
        prefXccTable[i].resize(numDetectedGpus, -1);
      }
      char* token = strtok(prefXccStr, ",");
      int tokenCount = 0;
      while (token) {
        int xccId;
        if (sscanf(token, "%d", &xccId) == 1) {
          int src = tokenCount / numDetectedGpus;
          int dst = tokenCount % numDetectedGpus;
          if (xccId < 0 || xccId >= numXccs) {
            printf("[ERROR] XCC index (%d) out of bounds. Expect value less than %d\n", xccId, numXccs);
            exit(1);
          }
          prefXccTable[src][dst] = xccId;

          tokenCount++;
          if (tokenCount == (numDetectedGpus * numDetectedGpus)) break;
        } else {
          printf("[ERROR] Unrecognized token [%s]\n", token);
          exit(1);
        }
        token = strtok(NULL, ",");
      }
    }
  }

  // Display info on the env vars that can be used
  static void DisplayUsage()
  {
    printf("Environment variables:\n");
    printf("======================\n");
    printf(" ALWAYS_VALIDATE   - Validate after each iteration instead of once after all iterations\n");
    printf(" BLOCK_SIZE        - # of threads per threadblock (Must be multiple of 64)\n");
    printf(" BLOCK_BYTES       - Controls granularity of how work is divided across subExecutors\n");
    printf(" BYTE_OFFSET       - Initial byte-offset for memory allocations.  Must be multiple of 4\n");
#if NIC_EXEC_ENABLED
    printf(" CLOSEST_NIC       - Comma-separated list of per-GPU closest NIC (default=auto)\n");
#endif
    printf(" CU_MASK           - CU mask for streams. Can specify ranges e.g '5,10-12,14'\n");
    printf(" FILL_PATTERN      - Big-endian pattern for source data, specified in hex digits. Must be even # of digits\n");
    printf(" GFX_UNROLL        - Unroll factor for GFX kernel (0=auto), must be less than %d\n", TransferBench::GetIntAttribute(ATR_GFX_MAX_UNROLL));
    printf(" GFX_SINGLE_TEAM   - Have subexecutors work together on full array instead of working on disjoint subarrays\n");
    printf(" GFX_WAVE_ORDER    - Stride pattern for GFX kernel (0=UWC,1=UCW,2=WUC,3=WCU,4=CUW,5=CWU)\n");
    printf(" HIDE_ENV          - Hide environment variable value listing\n");
#if NIC_EXEC_ENABLED
    printf(" IB_GID_INDEX      - Required for RoCE NICs (default=-1/auto)\n");
    printf(" IB_PORT_NUMBER    - RDMA port count for RDMA NIC (default=1)\n");
    printf(" IP_ADDRESS_FAMILY - IP address family (4=v4, 6=v6, default=v4)\n");
#endif
    printf(" MIN_VAR_SUBEXEC   - Minumum # of subexecutors to use for variable subExec Transfers\n");
    printf(" MAX_VAR_SUBEXEC   - Maximum # of subexecutors to use for variable subExec Transfers (0 for device limits)\n");
#if NIC_EXEC_ENABLED
    printf(" NIC_RELAX_ORDER   - Set to non-zero to use relaxed ordering");
#endif
    printf(" NUM_ITERATIONS    - # of timed iterations per test. If negative, run for this many seconds instead\n");
    printf(" NUM_SUBITERATIONS - # of sub-iterations to run per iteration. Must be non-negative\n");
    printf(" NUM_WARMUPS       - # of untimed warmup iterations per test\n");
    printf(" OUTPUT_TO_CSV     - Outputs to CSV format if set\n");
#if NIC_EXEC_ENABLED
    printf(" ROCE_VERSION      - RoCE version (default=2)\n");
#endif
    printf(" SAMPLING_FACTOR   - Add this many samples (when possible) between powers of 2 when auto-generating data sizes\n");
    printf(" SHOW_ITERATIONS   - Show per-iteration timing info\n");
    printf(" USE_HIP_EVENTS    - Use HIP events for GFX executor timing\n");
    printf(" USE_HSA_DMA       - Use hsa_amd_async_copy instead of hipMemcpy for non-targeted DMA execution\n");
    printf(" USE_INTERACTIVE   - Pause for user-input before starting transfer loop\n");
    printf(" USE_SINGLE_STREAM - Use a single stream per GPU GFX executor instead of stream per Transfer\n");
    printf(" VALIDATE_DIRECT   - Validate GPU destination memory directly instead of staging GPU memory on host\n");
    printf(" VALIDATE_SOURCE   - Validate GPU src memory immediately after preparation\n");
  }


  void Print(std::string const& name, int32_t const value, const char* format, ...) const
  {
    printf("%-20s%s%12d%s", name.c_str(), outputToCsv ? "," : " = ", value, outputToCsv ? "," : " : ");
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("\n");
  }

  void Print(std::string const& name, std::string const& value, const char* format, ...) const
  {
    printf("%-20s%s%12s%s", name.c_str(), outputToCsv ? "," : " = ", value.c_str(), outputToCsv ? "," : " : ");
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("\n");
  }

  // Display env var settings
  void DisplayEnvVars() const
  {
    int numGpuDevices = TransferBench::GetNumExecutors(EXE_GPU_GFX);
    std::string nicSupport = "";
#if NIC_EXEC_ENABLED
    nicSupport = " (with NIC support)";
#endif
    if (!outputToCsv) {
      printf("TransferBench v%s.%s%s\n", TransferBench::VERSION, CLIENT_VERSION, nicSupport.c_str());
      printf("===============================================================\n");
      if (!hideEnv) printf("[Common]                              (Suppress by setting HIDE_ENV=1)\n");
    }
    else if (!hideEnv)
      printf("EnvVar,Value,Description,(TransferBench Client v%s Backend v%s)\n", CLIENT_VERSION, TransferBench::VERSION);
    if (hideEnv) return;

    Print("ALWAYS_VALIDATE", alwaysValidate,
          "Validating after %s", (alwaysValidate ? "each iteration" : "all iterations"));
    Print("BLOCK_BYTES", blockBytes,
          "Each CU gets a mulitple of %d bytes to copy", blockBytes);
    Print("BYTE_OFFSET", byteOffset,
          "Using byte offset of %d", byteOffset);
#if NIC_EXEC_ENABLED
    Print("CLOSEST_NIC", (closestNicStr == "" ? "auto" : "user-input"),
          "Per-GPU closest NIC is set as %s", (closestNicStr == "" ? "auto" : closestNicStr.c_str()));
#endif
    Print("CU_MASK", getenv("CU_MASK") ? 1 : 0,
          "%s", (cuMask.size() ? GetCuMaskDesc().c_str() : "All"));
    Print("FILL_PATTERN", getenv("FILL_PATTERN") ? 1 : 0,
          "%s", (fillPattern.size() ? getenv("FILL_PATTERN") : TransferBench::GetStrAttribute(ATR_SRC_PREP_DESCRIPTION).c_str()));
    Print("GFX_BLOCK_SIZE", gfxBlockSize,
          "Threadblock size of %d", gfxBlockSize);
    Print("GFX_SINGLE_TEAM", gfxSingleTeam,
          "%s", (gfxSingleTeam ? "Combining CUs to work across entire data array" :
                                 "Each CUs operates on its own disjoint subarray"));
    Print("GFX_UNROLL", gfxUnroll,
          "Using GFX unroll factor of %d", gfxUnroll);
    Print("GFX_WAVE_ORDER", gfxWaveOrder,
          "Using GFX wave ordering of %s", (gfxWaveOrder == 0 ? "Unroll,Wavefront,CU" :
                                            gfxWaveOrder == 1 ? "Unroll,CU,Wavefront" :
                                            gfxWaveOrder == 2 ? "Wavefront,Unroll,CU" :
                                            gfxWaveOrder == 3 ? "Wavefront,CU,Unroll" :
                                            gfxWaveOrder == 4 ? "CU,Unroll,Wavefront" :
                                                                "CU,Wavefront,Unroll"));
#if NIC_EXEC_ENABLED
    Print("IP_ADDRESS_FAMILY", ipAddressFamily,
          "IP address family is set to IPv%d", ipAddressFamily);

    Print("IB_GID_INDEX", ibGidIndex,
          "RoCE GID index is set to %s", (ibGidIndex < 0 ? "auto" : std::to_string(ibGidIndex).c_str()));
    Print("IB_PORT_NUMBER", ibPort,
          "IB port number is set to %d", ibPort);
#endif
    Print("MIN_VAR_SUBEXEC", minNumVarSubExec,
          "Using at least %d subexecutor(s) for variable subExec tranfers", minNumVarSubExec);
    Print("MAX_VAR_SUBEXEC", maxNumVarSubExec,
          "Using up to %s subexecutors for variable subExec transfers",
          maxNumVarSubExec ? std::to_string(maxNumVarSubExec).c_str() : "all available");
#if NIC_EXEC_ENABLED
    Print("NIC_RELAX_ORDER", nicRelaxedOrder,
          "Using %s ordering for NIC RDMA", nicRelaxedOrder ? "relaxed" : "strict");
#endif
    Print("NUM_ITERATIONS", numIterations,
          (numIterations == 0) ? "Running infinitely" :
          "Running %d %s", abs(numIterations), (numIterations > 0 ? " timed iteration(s)" : "seconds(s) per Test"));
    Print("NUM_SUBITERATIONS", numSubIterations,
          "Running %s subiterations", (numSubIterations == 0 ? "infinite" : std::to_string(numSubIterations)).c_str());
    Print("NUM_WARMUPS", numWarmups,
          "Running %d warmup iteration(s) per Test", numWarmups);
#if NIC_EXEC_ENABLED
    Print("ROCE_VERSION", roceVersion,
          "RoCE version is set to %d", roceVersion);
#endif
    Print("SHOW_ITERATIONS", showIterations,
          "%s per-iteration timing", showIterations ? "Showing" : "Hiding");
    Print("USE_HIP_EVENTS", useHipEvents,
          "Using %s for GFX/DMA Executor timing", useHipEvents ? "HIP events" : "CPU wall time");
    Print("USE_HSA_DMA", useHsaDma,
          "Using %s for DMA execution", useHsaDma ? "hsa_amd_async_copy" : "hipMemcpyAsync");
    Print("USE_INTERACTIVE", useInteractive,
          "Running in %s mode", useInteractive ? "interactive" : "non-interactive");
    Print("USE_SINGLE_STREAM", useSingleStream,
          "Using single stream per GFX %s", useSingleStream ? "device" : "Transfer");
    if (getenv("XCC_PREF_TABLE")) {
      printf("%36s: Preferred XCC Table (XCC_PREF_TABLE)\n", "");
      printf("%36s:         ", "");
      for (int i = 0; i < numGpuDevices; i++) printf(" %3d", i); printf(" (#XCCs)\n");
      for (int i = 0; i < numGpuDevices; i++) {
        printf("%36s: GPU %3d ", "", i);
        for (int j = 0; j < numGpuDevices; j++)
          printf(" %3d", prefXccTable[i][j]);
        printf(" %3d\n", TransferBench::GetNumExecutorSubIndices({EXE_GPU_GFX, i}));
      }
    }
    Print("VALIDATE_DIRECT", validateDirect,
          "Validate GPU destination memory %s", validateDirect ? "directly" : "via CPU staging buffer");
    Print("VALIDATE_SOURCE", validateSource,
          validateSource ? "Validate source after preparation" : "Do not perform source validation after prep");
    printf("\n");
  };

  // Helper function that gets parses environment variable or sets to default value
  static int GetEnvVar(std::string const& varname, int defaultValue)
  {
    if (getenv(varname.c_str()))
      return atoi(getenv(varname.c_str()));
    return defaultValue;
  }

  static std::string GetEnvVar(std::string const& varname, std::string const& defaultValue)
  {
    if (getenv(varname.c_str()))
      return getenv(varname.c_str());
    return defaultValue;
  }

  std::string GetCuMaskDesc() const
  {
    std::vector<std::pair<int, int>> runs;
    int numXccs = TransferBench::GetNumExecutorSubIndices({EXE_GPU_GFX, 0});
    bool inRun = false;
    std::pair<int, int> curr;
    int used = 0;
    for (int targetBit = 0; targetBit < cuMask.size() * 32; targetBit += numXccs) {
      if (cuMask[targetBit/32] & (1 << (targetBit%32))) {
        used++;
        if (!inRun) {
          inRun = true;
          curr.first = targetBit / numXccs;
        }
      } else {
        if (inRun) {
          inRun = false;
          curr.second = targetBit / numXccs - 1;
          runs.push_back(curr);
        }
      }
    }
    if (inRun)
      curr.second = (cuMask.size() * 32) / numXccs - 1;

    std::string result = "CUs used: (" + std::to_string(used) + ") ";
    for (int i = 0; i < runs.size(); i++)
    {
      if (i) result += ",";
      if (runs[i].first == runs[i].second) result += std::to_string(runs[i].first);
      else result += std::to_string(runs[i].first) + "-" + std::to_string(runs[i].second);
    }
    return result;
  }

  TransferBench::ConfigOptions ToConfigOptions()
  {
    TransferBench::ConfigOptions cfg;

    cfg.general.numIterations      = numIterations;
    cfg.general.numSubIterations   = numSubIterations;
    cfg.general.numWarmups         = numWarmups;
    cfg.general.recordPerIteration = showIterations;
    cfg.general.useInteractive     = useInteractive;

    cfg.data.alwaysValidate        = alwaysValidate;
    cfg.data.blockBytes            = blockBytes;
    cfg.data.byteOffset            = byteOffset;
    cfg.data.validateDirect        = validateDirect;
    cfg.data.validateSource        = validateSource;
    cfg.data.fillPattern           = fillPattern;

    cfg.dma.useHipEvents           = useHipEvents;
    cfg.dma.useHsaCopy             = useHsaDma;

    cfg.gfx.blockSize              = gfxBlockSize;
    cfg.gfx.cuMask                 = cuMask;
    cfg.gfx.prefXccTable           = prefXccTable;
    cfg.gfx.unrollFactor           = gfxUnroll;
    cfg.gfx.useHipEvents           = useHipEvents;
    cfg.gfx.useMultiStream         = !useSingleStream;
    cfg.gfx.useSingleTeam          = gfxSingleTeam;
    cfg.gfx.waveOrder              = gfxWaveOrder;

    cfg.nic.ibGidIndex            = ibGidIndex;
    cfg.nic.ibPort                = ibPort;
    cfg.nic.ipAddressFamily       = ipAddressFamily;
    cfg.nic.useRelaxedOrder       = nicRelaxedOrder;
    cfg.nic.roceVersion           = roceVersion;

    std::vector<int> closestNics;
    if(closestNicStr != "") {
      std::stringstream ss(closestNicStr);
      std::string item;
      while (std::getline(ss, item, ',')) {
        try {
          int nic = std::stoi(item);
          closestNics.push_back(nic);
        } catch (const std::invalid_argument& e) {
          printf("[ERROR] Invalid NIC index (%s) by user in %s\n", item.c_str(), closestNicStr.c_str());
          exit(1);
        }
      }
      cfg.nic.closestNics = closestNics;
    }
    return cfg;
  }
};

#endif
