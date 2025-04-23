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

void AllToAllRdmaPreset(EnvVars&           ev,
                        size_t      const  numBytesPerTransfer,
                        std::string const  presetName)
{


  int numDetectedGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);

  // Collect env vars for this preset
  int numGpus       = EnvVars::GetEnvVar("NUM_GPU_DEVICES", numDetectedGpus);
  int numQueuePairs = EnvVars::GetEnvVar("NUM_QUEUE_PAIRS", 1);
  int useFineGrain  = EnvVars::GetEnvVar("USE_FINE_GRAIN" , 1);

  // Print off environment variables
  ev.DisplayEnvVars();
  if (!ev.hideEnv) {
    if (!ev.outputToCsv) printf("[AllToAll Network Related]\n");
    ev.Print("NUM_GPU_DEVICES", numGpus      , "Using %d GPUs", numGpus);
    ev.Print("NUM_QUEUE_PAIRS", numQueuePairs, "Using %d queue pairs for NIC transfers", numQueuePairs);
    ev.Print("USE_FINE_GRAIN" , useFineGrain , "Using %s-grained memory", useFineGrain ? "fine" : "coarse");
    printf("\n");
  }

  // Validate env vars
  if (numGpus < 0 || numGpus > numDetectedGpus) {
    printf("[ERROR] Cannot use %d GPUs.  Detected %d GPUs\n", numGpus, numDetectedGpus);
    exit(1);
  }

  MemType memType = useFineGrain ? MEM_GPU_FINE : MEM_GPU;

  std::map<std::pair<int, int>, int> reIndex;
  std::vector<Transfer> transfers;
  for (int i = 0; i < numGpus; i++) {
    for (int j = 0; j < numGpus; j++) {
      // Build Transfer and add it to list
      TransferBench::Transfer transfer;
      transfer.numBytes = numBytesPerTransfer;
      transfer.srcs.push_back({memType, i});
      transfer.dsts.push_back({memType, j});
      transfer.exeDevice = {EXE_NIC_NEAREST, i};
      transfer.exeSubIndex = j;
      transfer.numSubExecs = numQueuePairs;

      reIndex[std::make_pair(i,j)] = transfers.size();
      transfers.push_back(transfer);
    }
  }

  printf("GPU-RDMA All-To-All benchmark:\n");
  printf("==========================\n");
  printf("- Copying %lu bytes between all pairs of GPUs using %d QPs per Transfer (%lu Transfers)\n",
         numBytesPerTransfer, numQueuePairs, transfers.size());
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
  printf("   %c%8.3f   %c%8.3f   %c%8.3f\n", separator, colTotalBandwidth[numGpus+1],
         separator, minActualBandwidth, separator, maxActualBandwidth);
  printf("\n");

  printf("Average   bandwidth (Tx Thread Timed): %8.3f GB/s\n", totalBandwidthGpu / transfers.size());
  printf("Aggregate bandwidth (Tx Thread Timed): %8.3f GB/s\n", totalBandwidthGpu);
  printf("Aggregate bandwidth       (CPU Timed): %8.3f GB/s\n", results.avgTotalBandwidthGbPerSec);

  PrintErrors(results.errResults);
}
