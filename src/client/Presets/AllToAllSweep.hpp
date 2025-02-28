/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include "EnvVars.hpp"

void AllToAllSweepPreset(EnvVars&           ev,
                         size_t      const  numBytesPerTransfer,
                         std::string const  presetName)
{
  enum
  {
    A2A_COPY       = 0,
    A2A_READ_ONLY  = 1,
    A2A_WRITE_ONLY = 2,
    A2A_CUSTOM     = 3,
  };
  char a2aModeStr[4][20] = {"Copy", "Read-Only", "Write-Only", "Custom"};

  // Force single-stream mode for all-to-all benchmark
  ev.useSingleStream = 1;

  int numDetectedGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);

  // Collect env vars for this preset
  int a2aDirect     = EnvVars::GetEnvVar("A2A_DIRECT"     , 1);
  int a2aLocal      = EnvVars::GetEnvVar("A2A_LOCAL"      , 0);
  int numGpus       = EnvVars::GetEnvVar("NUM_GPU_DEVICES", numDetectedGpus);
  int useFineGrain  = EnvVars::GetEnvVar("USE_FINE_GRAIN" , 1);
  int useRemoteRead = EnvVars::GetEnvVar("USE_REMOTE_READ", 0);
  int useSpray      = EnvVars::GetEnvVar("USE_SPRAY",       0);
  int verbose       = EnvVars::GetEnvVar("VERBOSE",         0);

  std::vector<int> unrollList = EnvVars::GetEnvVarArray("UNROLLS", {1,2,3,4,6,8});
  std::vector<int> numCusList = EnvVars::GetEnvVarArray("NUM_CUS", {4,8,12,16,24,32});

  // A2A_MODE may be 0,1,2 or else custom numSrcs:numDsts
  int numSrcs, numDsts;
  int a2aMode = 0;
  if (getenv("A2A_MODE") && sscanf(getenv("A2A_MODE"), "%d:%d", &numSrcs, &numDsts) == 2) {
    a2aMode = A2A_CUSTOM;
  } else {
    a2aMode = EnvVars::GetEnvVar("A2A_MODE", 0);
    if (a2aMode < 0 || a2aMode > 2) {
      printf("[ERROR] a2aMode must be between 0 and 2, or else numSrcs:numDsts\n");
      exit(1);
    }
    numSrcs = (a2aMode == A2A_WRITE_ONLY ? 0 : 1);
    numDsts = (a2aMode == A2A_READ_ONLY  ? 0 : 1);
  }

  // Print off environment variables
  ev.DisplayEnvVars();
  if (!ev.hideEnv) {
    if (!ev.outputToCsv) printf("[AllToAll Related]\n");
    ev.Print("A2A_DIRECT"     , a2aDirect        , a2aDirect ? "Only using direct links" : "Full all-to-all");
    ev.Print("A2A_LOCAL"      , a2aLocal         , "%s local transfers", a2aLocal ? "Include" : "Exclude");
    ev.Print("A2A_MODE"       , (a2aMode == A2A_CUSTOM) ?  std::to_string(numSrcs) + ":" + std::to_string(numDsts) : std::to_string(a2aMode),
                                (a2aMode == A2A_CUSTOM) ? (std::to_string(numSrcs) + " read(s) " +
                                                           std::to_string(numDsts) + " write(s)").c_str(): a2aModeStr[a2aMode]);
    ev.Print("NUM_CUS"        , numCusList.size(), EnvVars::ToStr(numCusList).c_str());
    ev.Print("NUM_GPU_DEVICES", numGpus          , "Using %d GPUs", numGpus);
    ev.Print("UNROLLS"        , unrollList.size(), EnvVars::ToStr(unrollList).c_str());
    ev.Print("USE_FINE_GRAIN" , useFineGrain     , "Using %s-grained memory", useFineGrain ? "fine" : "coarse");
    ev.Print("USE_REMOTE_READ", useRemoteRead    , "Using %s as executor", useRemoteRead ? "DST" : "SRC");
    ev.Print("USE_SPRAY"      , useSpray         , "%s per CU", useSpray ? "All targets" : "One target");
    ev.Print("VERBOSE"        , verbose          , verbose ? "Display test results" : "Display summary only");
    printf("\n");
  }

  // Validate env vars
  if (numGpus < 0 || numGpus > numDetectedGpus) {
    printf("[ERROR] Cannot use %d GPUs.  Detected %d GPUs\n", numGpus, numDetectedGpus);
    exit(1);
  }

  if (useSpray && numDsts > 1) {
    printf("[ERROR] Cannot use USE_SPRAY with multiple destination buffers\n");
    exit(1);
  }

  // Collect the number of GPU devices to use
  MemType memType = useFineGrain ? MEM_GPU_FINE : MEM_GPU;
  ExeType exeType = EXE_GPU_GFX;

  std::vector<Transfer> transfers;

  int targetCount = 0;
  if (!useSpray) {
    // Each CU will work on just one target
    for (int i = 0; i < numGpus; i++) {
      targetCount = 0;
      for (int j = 0; j < numGpus; j++) {
        // Check whether or not to execute this pair
        if (i == j) {
          if (!a2aLocal) continue;
        } else if (a2aDirect) {
#if !defined(__NVCC__)
          uint32_t linkType, hopCount;
          HIP_CALL(hipExtGetLinkTypeAndHopCount(i, j, &linkType, &hopCount));
          if (hopCount != 1) continue;
#endif
        }

        // Build Transfer and add it to list
        TransferBench::Transfer transfer;
        targetCount++;
        transfer.numBytes = numBytesPerTransfer;
        for (int x = 0; x < numSrcs; x++) transfer.srcs.push_back({memType, i});

        // When using multiple destinations, the additional destinations are "local"
        if (numDsts) transfer.dsts.push_back({memType, j});
        for (int x = 1; x < numDsts; x++) transfer.dsts.push_back({memType, i});
        transfer.exeDevice = {exeType, (useRemoteRead ? j : i)};
        transfer.exeSubIndex = -1;
        transfers.push_back(transfer);
      }
    }
  } else {
    // Each CU will work on all targets
    for (int i = 0; i < numGpus; i++) {
      TransferBench::Transfer transfer;
      transfer.numBytes = numBytesPerTransfer;
      transfer.exeDevice = {exeType, i};
      transfer.exeSubIndex = -1;
      targetCount = 0;
      for (int j = 0; j < numGpus; j++) {
        // Check whether or not to transfer to this GPU
        if (i == j) {
          if (!a2aLocal) continue;
        } else if (a2aDirect) {
#if !defined(__NVCC__)
          uint32_t linkType, hopCount;
          HIP_CALL(hipExtGetLinkTypeAndHopCount(i, j, &linkType, &hopCount));
          if (hopCount != 1) continue;
#endif
        }
        targetCount++;
        for (int x = 0; x < numSrcs; x++) transfer.srcs.push_back({memType, useRemoteRead ? j : i});

        if (numDsts) transfer.dsts.push_back({memType, j});
        for (int x = 1; x < numDsts; x++) transfer.dsts.push_back({memType, i});
      }
      transfers.push_back(transfer);
    }
  }

  printf("GPU-GFX All-To-All Sweep benchmark:\n");
  printf("==========================\n");
  printf("- Copying %lu bytes between %s pairs of GPUs\n", numBytesPerTransfer, a2aDirect ? "directly connected" : "all");
  if (transfers.size() == 0) {
    printf("[WARN} No transfers requested. Try adjusting A2A_DIRECT or A2A_LOCAL\n");
    return;
  }

  // Execute Transfers
  TransferBench::ConfigOptions cfg = ev.ToConfigOptions();

  // Run tests
  std::map<std::pair<int, int>, TransferBench::TestResults> results;

  // Display summary
  printf("#CUs\\Unroll");
  for (int u : unrollList) {
    printf("  %d(Min) ", u);
    printf("  %d(Max) ", u);
  }
  printf("\n");
  for (int c : numCusList) {
    printf("   %5d   ", c);  fflush(stdout);
    for (int u : unrollList) {
      ev.gfxUnroll = cfg.gfx.unrollFactor = u;
      for (auto& transfer : transfers)
        transfer.numSubExecs = useSpray ? (c * targetCount) : c;

      double minBandwidth = std::numeric_limits<double>::max();
      double maxBandwidth = std::numeric_limits<double>::min();
      TransferBench::TestResults result;
      if (TransferBench::RunTransfers(cfg, transfers, result)) {
        for (auto const& exeResult : result.exeResults) {
          minBandwidth = std::min(minBandwidth, exeResult.second.avgBandwidthGbPerSec);
	  maxBandwidth = std::max(maxBandwidth, exeResult.second.avgBandwidthGbPerSec);
	}
        if (useSpray) {
	  minBandwidth *= targetCount;
	  maxBandwidth *= targetCount;
	}
        results[std::make_pair(c,u)] = result;
      } else {
        minBandwidth = 0.0;
      }
      printf(" %7.2f  %7.2f ", minBandwidth, maxBandwidth); fflush(stdout);
    }
    printf("\n"); fflush(stdout);
  }

  if (verbose) {
    int testNum = 0;
    for (int c : numCusList) {
      for (int u : unrollList) {
        printf("CUs: %d Unroll %d\n", c, u);
        PrintResults(ev, ++testNum, transfers, results[std::make_pair(c,u)]);
      }
    }
  }
}
