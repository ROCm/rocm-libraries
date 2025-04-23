/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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

void AllToAllPreset(EnvVars&           ev,
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

  // Force to gfx unroll 2 unless explicitly set
  ev.gfxUnroll      = EnvVars::GetEnvVar("GFX_UNROLL", 2);

  int numDetectedGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);

  // Collect env vars for this preset
  int a2aDirect     = EnvVars::GetEnvVar("A2A_DIRECT"     , 1);
  int a2aLocal      = EnvVars::GetEnvVar("A2A_LOCAL"      , 0);
  int numGpus       = EnvVars::GetEnvVar("NUM_GPU_DEVICES", numDetectedGpus);
  int numQueuePairs = EnvVars::GetEnvVar("NUM_QUEUE_PAIRS", 0);
  int numSubExecs   = EnvVars::GetEnvVar("NUM_SUB_EXEC"   , 8);
  int useDmaExec    = EnvVars::GetEnvVar("USE_DMA_EXEC"   , 0);
  int useFineGrain  = EnvVars::GetEnvVar("USE_FINE_GRAIN" , 1);
  int useRemoteRead = EnvVars::GetEnvVar("USE_REMOTE_READ", 0);

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
    ev.Print("A2A_DIRECT"     , a2aDirect    , a2aDirect ? "Only using direct links" : "Full all-to-all");
    ev.Print("A2A_LOCAL"      , a2aLocal     , "%s local transfers", a2aLocal ? "Include" : "Exclude");
    ev.Print("A2A_MODE"       , (a2aMode == A2A_CUSTOM) ?  std::to_string(numSrcs) + ":" + std::to_string(numDsts) : std::to_string(a2aMode),
                                (a2aMode == A2A_CUSTOM) ? (std::to_string(numSrcs) + " read(s) " +
                                                           std::to_string(numDsts) + " write(s)").c_str(): a2aModeStr[a2aMode]);
    ev.Print("NUM_GPU_DEVICES", numGpus      , "Using %d GPUs", numGpus);
    ev.Print("NUM_QUEUE_PAIRS", numQueuePairs, "Using %d queue pairs for NIC transfers", numQueuePairs);
    ev.Print("NUM_SUB_EXEC"   , numSubExecs  , "Using %d subexecutors/CUs per Transfer", numSubExecs);
    ev.Print("USE_DMA_EXEC"   , useDmaExec   , "Using %s executor", useDmaExec ? "DMA" : "GFX");
    ev.Print("USE_FINE_GRAIN" , useFineGrain , "Using %s-grained memory", useFineGrain ? "fine" : "coarse");
    ev.Print("USE_REMOTE_READ", useRemoteRead, "Using %s as executor", useRemoteRead ? "DST" : "SRC");
    printf("\n");
  }

  // Validate env vars
  if (numGpus < 0 || numGpus > numDetectedGpus) {
    printf("[ERROR] Cannot use %d GPUs.  Detected %d GPUs\n", numGpus, numDetectedGpus);
    exit(1);
  }
  if (useDmaExec && (numSrcs != 1 || numDsts != 1)) {
    printf("[ERROR] DMA execution can only be used for copies (A2A_MODE=0)\n");
    exit(1);
  }

  // Collect the number of GPU devices to use
  MemType memType = useFineGrain ? MEM_GPU_FINE : MEM_GPU;
  ExeType exeType = useDmaExec ? EXE_GPU_DMA : EXE_GPU_GFX;

  std::map<std::pair<int, int>, int> reIndex;
  std::vector<Transfer> transfers;
  for (int i = 0; i < numGpus; i++) {
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
      transfer.numBytes = numBytesPerTransfer;
      for (int x = 0; x < numSrcs; x++) transfer.srcs.push_back({memType, i});

      // When using multiple destinations, the additional destinations are "local"
      if (numDsts) transfer.dsts.push_back({memType, j});
      for (int x = 1; x < numDsts; x++) transfer.dsts.push_back({memType, i});
      transfer.exeDevice = {exeType, (useRemoteRead ? j : i)};
      transfer.exeSubIndex = -1;
      transfer.numSubExecs = numSubExecs;

      reIndex[std::make_pair(i,j)] = transfers.size();
      transfers.push_back(transfer);
    }
  }

  // Create a ring using NICs
  std::vector<int> nicTransferIdx(numGpus);
  if (numQueuePairs > 0) {
    int numNics = TransferBench::GetNumExecutors(EXE_NIC);
    for (int i = 0; i < numGpus; i++) {
      TransferBench::Transfer transfer;
      transfer.numBytes = numBytesPerTransfer;
      transfer.srcs.push_back({memType, i});
      transfer.dsts.push_back({memType, (i+1) % numGpus});
      transfer.exeDevice = {TransferBench::EXE_NIC_NEAREST, i};
      transfer.exeSubIndex = (i+1) % numGpus;
      transfer.numSubExecs = numQueuePairs;
      nicTransferIdx[i] = transfers.size();
      transfers.push_back(transfer);
    }
  }

  printf("GPU-GFX All-To-All benchmark:\n");
  printf("==========================\n");
  printf("- Copying %lu bytes between %s pairs of GPUs using %d CUs (%lu Transfers)\n",
         numBytesPerTransfer, a2aDirect ? "directly connected" : "all", numSubExecs, transfers.size());
  if (transfers.size() == 0) return;

  // Execute Transfers
  TransferBench::ConfigOptions cfg = ev.ToConfigOptions();
  TransferBench::TestResults results;
  if (!TransferBench::RunTransfers(cfg, transfers, results)) {
    for (auto const& err : results.errResults)
      printf("%s\n", err.errMsg.c_str());
    exit(0);
  } else {
    PrintResults(ev, 1, transfers, results);
  }

  // Print results
  char separator = (ev.outputToCsv ? ',' : ' ');
  printf("\nSummary: [%lu bytes per Transfer]\n", numBytesPerTransfer);
  printf("==========================================================\n");
  printf("SRC\\DST ");
  for (int dst = 0; dst < numGpus; dst++)
    printf("%cGPU %02d    ", separator, dst);
  if (numQueuePairs > 0)
    printf("%cNIC(%02d QP)", separator, numQueuePairs);
  printf("   %cSTotal     %cActual\n", separator, separator);

  double totalBandwidthGpu = 0.0;
  double minActualBandwidth = std::numeric_limits<double>::max();
  double maxActualBandwidth = 0.0;
  std::vector<double> colTotalBandwidth(numGpus+2, 0.0);
  for (int src = 0; src < numGpus; src++) {
    double rowTotalBandwidth = 0;
    int    transferCount = 0;
    double minBandwidth = std::numeric_limits<double>::max();
    printf("GPU %02d", src);
    for (int dst = 0; dst < numGpus; dst++) {
      if (reIndex.count(std::make_pair(src, dst))) {
        int const transferIdx = reIndex[std::make_pair(src,dst)];
        TransferBench::TransferResult const& r = results.tfrResults[transferIdx];
        colTotalBandwidth[dst]  += r.avgBandwidthGbPerSec;
        rowTotalBandwidth       += r.avgBandwidthGbPerSec;
        totalBandwidthGpu       += r.avgBandwidthGbPerSec;
        minBandwidth             = std::min(minBandwidth, r.avgBandwidthGbPerSec);
        transferCount++;
        printf("%c%8.3f  ", separator, r.avgBandwidthGbPerSec);
      } else {
        printf("%c%8s  ", separator, "N/A");
      }
    }

    if (numQueuePairs > 0) {
      TransferBench::TransferResult const& r = results.tfrResults[nicTransferIdx[src]];
      colTotalBandwidth[numGpus]  += r.avgBandwidthGbPerSec;
      rowTotalBandwidth           += r.avgBandwidthGbPerSec;
      totalBandwidthGpu           += r.avgBandwidthGbPerSec;
      minBandwidth                 = std::min(minBandwidth, r.avgBandwidthGbPerSec);
      transferCount++;
      printf("%c%8.3f  ", separator, r.avgBandwidthGbPerSec);
    }
    double actualBandwidth = minBandwidth * transferCount;
    printf("   %c%8.3f   %c%8.3f\n", separator, rowTotalBandwidth, separator, actualBandwidth);
    minActualBandwidth = std::min(minActualBandwidth, actualBandwidth);
    maxActualBandwidth = std::max(maxActualBandwidth, actualBandwidth);
    colTotalBandwidth[numGpus+1] += rowTotalBandwidth;
  }
  printf("\nRTotal");
  for (int dst = 0; dst < numGpus; dst++) {
    printf("%c%8.3f  ", separator, colTotalBandwidth[dst]);
  }
  if (numQueuePairs > 0) {
    printf("%c%8.3f  ", separator, colTotalBandwidth[numGpus]);
  }
  printf("   %c%8.3f   %c%8.3f   %c%8.3f\n", separator, colTotalBandwidth[numGpus+1],
         separator, minActualBandwidth, separator, maxActualBandwidth);
  printf("\n");

  printf("Average   bandwidth (GPU Timed): %8.3f GB/s\n", totalBandwidthGpu / transfers.size());
  printf("Aggregate bandwidth (GPU Timed): %8.3f GB/s\n", totalBandwidthGpu);
  printf("Aggregate bandwidth (CPU Timed): %8.3f GB/s\n", results.avgTotalBandwidthGbPerSec);

  PrintErrors(results.errResults);
}
