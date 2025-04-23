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

void PeerToPeerPreset(EnvVars&           ev,
                      size_t      const  numBytesPerTransfer,
                      std::string const  presetName)
{
  int numDetectedCpus = TransferBench::GetNumExecutors(EXE_CPU);
  int numDetectedGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);

  // Collect env vars for this preset
  int useDmaCopy     = EnvVars::GetEnvVar("USE_GPU_DMA",     0);
  int numCpuDevices  = EnvVars::GetEnvVar("NUM_CPU_DEVICES", numDetectedCpus);
  int numCpuSubExecs = EnvVars::GetEnvVar("NUM_CPU_SE",      4);
  int numGpuDevices  = EnvVars::GetEnvVar("NUM_GPU_DEVICES", numDetectedGpus);
  int numGpuSubExecs = EnvVars::GetEnvVar("NUM_GPU_SE",      useDmaCopy ? 1 : TransferBench::GetNumSubExecutors({EXE_GPU_GFX, 0}));
  int p2pMode        = EnvVars::GetEnvVar("P2P_MODE",        0);
  int useFineGrain   = EnvVars::GetEnvVar("USE_FINE_GRAIN",  0);
  int useRemoteRead  = EnvVars::GetEnvVar("USE_REMOTE_READ", 0);

  // Display environment variables
  ev.DisplayEnvVars();
  if (!ev.hideEnv) {
    int outputToCsv = ev.outputToCsv;
    if (!outputToCsv) printf("[P2P Related]\n");
    ev.Print("NUM_CPU_DEVICES", numCpuDevices,  "Using %d CPUs", numCpuDevices);
    ev.Print("NUM_CPU_SE",      numCpuSubExecs, "Using %d CPU threads per Transfer", numCpuSubExecs);
    ev.Print("NUM_GPU_DEVICES", numGpuDevices,  "Using %d GPUs", numGpuDevices);
    ev.Print("NUM_GPU_SE",      numGpuSubExecs, "Using %d GPU subexecutors/CUs per Transfer", numGpuSubExecs);
    ev.Print("P2P_MODE",        p2pMode,        "Running %s transfers", p2pMode == 0 ? "Uni + Bi" :
                                                                        p2pMode == 1 ? "Unidirectional"
                                                                                     : "Bidirectional");
    ev.Print("USE_FINE_GRAIN",  useFineGrain,   "Using %s-grained memory", useFineGrain ? "fine" : "coarse");
    ev.Print("USE_GPU_DMA",     useDmaCopy,     "Using GPU-%s as GPU executor", useDmaCopy ? "DMA" : "GFX");
    ev.Print("USE_REMOTE_READ", useRemoteRead,  "Using %s as executor", useRemoteRead ? "DST" : "SRC");
    printf("\n");
  }

  char const separator = ev.outputToCsv ? ',' : ' ';
  printf("Bytes Per Direction%c%lu\n", separator, numBytesPerTransfer);

  TransferBench::ConfigOptions cfg = ev.ToConfigOptions();
  TransferBench::TestResults results;

  // Collect the number of available CPUs/GPUs on this machine
  int const numDevices = numCpuDevices + numGpuDevices;

  // Perform unidirectional / bidirectional
  for (int isBidirectional = 0; isBidirectional <= 1; isBidirectional++) {
    if (p2pMode == 1 && isBidirectional == 1 ||
        p2pMode == 2 && isBidirectional == 0) continue;

    printf("%sdirectional copy peak bandwidth GB/s [%s read / %s write] (GPU-Executor: %s)\n", isBidirectional ? "Bi" : "Uni",
           useRemoteRead ? "Remote" : "Local",
           useRemoteRead ? "Local" : "Remote",
           useDmaCopy    ? "DMA"   : "GFX");

    // Print header
    if (isBidirectional) {
      printf("%12s", "SRC\\DST");
    } else {
      if (useRemoteRead)
        printf("%12s", "SRC\\EXE+DST");
      else
        printf("%12s", "SRC+EXE\\DST");
    }
    if (ev.outputToCsv) printf(",");
    for (int i = 0; i < numCpuDevices; i++) {
      printf("%7s %02d", "CPU", i);
      if (ev.outputToCsv) printf(",");
    }
    if (numCpuDevices > 0) printf("   ");
    for (int i = 0; i < numGpuDevices; i++) {
      printf("%7s %02d", "GPU", i);
      if (ev.outputToCsv) printf(",");
    }
    printf("\n");

    double avgBwSum[2][2] = {};
    int    avgCount[2][2] = {};

    ExeType const gpuExeType = useDmaCopy ? EXE_GPU_DMA : EXE_GPU_GFX;

    // Loop over all possible src/dst pairs
    for (int src = 0; src < numDevices; src++) {
      MemType const srcType  = (src < numCpuDevices ? MEM_CPU : MEM_GPU);
      int     const srcIndex = (srcType == MEM_CPU ? src : src - numCpuDevices);
      MemType const srcTypeActual = ((useFineGrain && srcType == MEM_CPU) ? MEM_CPU_FINE :
                                     (useFineGrain && srcType == MEM_GPU) ? MEM_GPU_FINE :
                                                                               srcType);
      std::vector<std::vector<double>> avgBandwidth(isBidirectional + 1);
      std::vector<std::vector<double>> minBandwidth(isBidirectional + 1);
      std::vector<std::vector<double>> maxBandwidth(isBidirectional + 1);
      std::vector<std::vector<double>> stdDev(isBidirectional + 1);

      if (src == numCpuDevices && src != 0) printf("\n");
      for (int dst = 0; dst < numDevices; dst++) {
        MemType const dstType  = (dst < numCpuDevices ? MEM_CPU : MEM_GPU);
        int     const dstIndex = (dstType == MEM_CPU ? dst : dst - numCpuDevices);
        MemType const dstTypeActual = ((useFineGrain && dstType == MEM_CPU) ? MEM_CPU_FINE :
                                       (useFineGrain && dstType == MEM_GPU) ? MEM_GPU_FINE :
                                                                                 dstType);
        // Prepare Transfers
        std::vector<Transfer> transfers(isBidirectional + 1);

        // SRC -> DST
        transfers[0].numBytes = numBytesPerTransfer;
        transfers[0].srcs.push_back({srcTypeActual, srcIndex});
        transfers[0].dsts.push_back({dstTypeActual, dstIndex});
        transfers[0].exeDevice = {IsGpuMemType(useRemoteRead ? dstType : srcType) ? gpuExeType : EXE_CPU,
                                  (useRemoteRead ? dstIndex : srcIndex)};
        transfers[0].exeSubIndex = -1;
        transfers[0].numSubExecs = (transfers[0].exeDevice.exeType == gpuExeType) ? numGpuSubExecs : numCpuSubExecs;

        // DST -> SRC
        if (isBidirectional) {
          transfers[1].numBytes = numBytesPerTransfer;
          transfers[1].srcs.push_back({dstTypeActual, dstIndex});
          transfers[1].dsts.push_back({srcTypeActual, srcIndex});
          transfers[1].exeDevice = {IsGpuMemType(useRemoteRead ? srcType : dstType) ? gpuExeType : EXE_CPU,
                                    (useRemoteRead ? srcIndex : dstIndex)};
          transfers[1].exeSubIndex = -1;
          transfers[1].numSubExecs = (transfers[1].exeDevice.exeType == gpuExeType) ? numGpuSubExecs : numCpuSubExecs;
        }

        bool skipTest = false;

        // Abort if executing on NUMA node with no CPUs
        for (int i = 0; i <= isBidirectional; i++) {
          if (transfers[i].exeDevice.exeType == EXE_CPU &&
              TransferBench::GetNumSubExecutors(transfers[i].exeDevice) == 0) {
            skipTest = true;
            break;
          }

#if defined(__NVCC__)
          // NVIDIA platform cannot access GPU memory directly from CPU executors
          if (transfers[i].exeDevice.exeType == EXE_CPU && (IsGpuMemType(srcType) || IsGpuMemType(dstType))) {
            skipTest = true;
            break;
          }
#endif
        }

        if (isBidirectional && srcType == dstType && srcIndex == dstIndex) skipTest = true;

        if (!skipTest) {
          if (!TransferBench::RunTransfers(cfg, transfers, results)) {
            for (auto const& err : results.errResults)
              printf("%s\n", err.errMsg.c_str());
            exit(1);
          }

          for (int dir = 0; dir <= isBidirectional; dir++) {
            double const avgBw   = results.tfrResults[dir].avgBandwidthGbPerSec;
            avgBandwidth[dir].push_back(avgBw);

            if (!(srcType == dstType && srcIndex == dstIndex)) {
              avgBwSum[srcType][dstType] += avgBw;
              avgCount[srcType][dstType]++;
            }

            if (ev.showIterations) {
              double minTime = results.tfrResults[dir].perIterMsec[0];
              double maxTime = minTime;
              double varSum  = 0;
              for (int i = 0; i < results.tfrResults[dir].perIterMsec.size(); i++) {
                minTime = std::min(minTime, results.tfrResults[dir].perIterMsec[i]);
                maxTime = std::max(maxTime, results.tfrResults[dir].perIterMsec[i]);
                double const bw  = (transfers[dir].numBytes / 1.0E9) / results.tfrResults[dir].perIterMsec[i] * 1000.0f;
                double const delta = (avgBw - bw);
                varSum += delta * delta;
              }
              double const minBw = (transfers[dir].numBytes / 1.0E9) / maxTime * 1000.0f;
              double const maxBw = (transfers[dir].numBytes / 1.0E9) / minTime * 1000.0f;
              double const stdev = sqrt(varSum / results.tfrResults[dir].perIterMsec.size());
              minBandwidth[dir].push_back(minBw);
              maxBandwidth[dir].push_back(maxBw);
              stdDev[dir].push_back(stdev);
            }
          }
        } else {
          for (int dir = 0; dir <= isBidirectional; dir++) {
            avgBandwidth[dir].push_back(0);
            minBandwidth[dir].push_back(0);
            maxBandwidth[dir].push_back(0);
            stdDev[dir].push_back(-1.0);
          }
        }
      }

      for (int dir = 0; dir <= isBidirectional; dir++) {
        printf("%5s %02d %3s", (srcType == MEM_CPU) ? "CPU" : "GPU", srcIndex, dir ? "<- " : " ->");
        if (ev.outputToCsv) printf(",");

        for (int dst = 0; dst < numDevices; dst++) {
          if (dst == numCpuDevices && dst != 0) printf("   ");
          double const avgBw = avgBandwidth[dir][dst];

          if (avgBw == 0.0)
            printf("%10s", "N/A");
          else
            printf("%10.2f", avgBw);
          if (ev.outputToCsv) printf(",");
        }
        printf("\n");

        if (ev.showIterations) {
          // minBw
          printf("%5s %02d %3s", (srcType == MEM_CPU) ? "CPU" : "GPU", srcIndex, "min");
          if (ev.outputToCsv) printf(",");
          for (int i = 0; i < numDevices; i++) {
            double const minBw = minBandwidth[dir][i];
            if (i == numCpuDevices && i != 0) printf("   ");
            if (minBw == 0.0)
              printf("%10s", "N/A");
            else
              printf("%10.2f", minBw);
            if (ev.outputToCsv) printf(",");
          }
          printf("\n");

          // maxBw
          printf("%5s %02d %3s", (srcType == MEM_CPU) ? "CPU" : "GPU", srcIndex, "max");
          if (ev.outputToCsv) printf(",");
          for (int i = 0; i < numDevices; i++) {
            double const maxBw = maxBandwidth[dir][i];
            if (i == numCpuDevices && i != 0) printf("   ");
            if (maxBw == 0.0)
              printf("%10s", "N/A");
            else
              printf("%10.2f", maxBw);
            if (ev.outputToCsv) printf(",");
          }
          printf("\n");

          // stddev
          printf("%5s %02d %3s", (srcType == MEM_CPU) ? "CPU" : "GPU", srcIndex, " sd");
          if (ev.outputToCsv) printf(",");
          for (int i = 0; i < numDevices; i++) {
            double const sd = stdDev[dir][i];
            if (i == numCpuDevices && i != 0) printf("   ");
            if (sd == -1.0)
              printf("%10s", "N/A");
            else
              printf("%10.2f", sd);
            if (ev.outputToCsv) printf(",");
          }
          printf("\n");
        }
        fflush(stdout);
      }

      if (isBidirectional) {
        printf("%5s %02d %3s", (srcType == MEM_CPU) ? "CPU" : "GPU", srcIndex, "<->");
        if (ev.outputToCsv) printf(",");
        for (int dst = 0; dst < numDevices; dst++) {
          double const sumBw = avgBandwidth[0][dst] + avgBandwidth[1][dst];
          if (dst == numCpuDevices && dst != 0) printf("   ");
          if (sumBw == 0.0)
            printf("%10s", "N/A");
          else
            printf("%10.2f", sumBw);
          if (ev.outputToCsv) printf(",");
        }
        printf("\n");
        if (src < numDevices - 1) printf("\n");
      }
    }

    if (!ev.outputToCsv) {
      printf("                         ");
      for (int srcType : {MEM_CPU, MEM_GPU})
        for (int dstType : {MEM_CPU, MEM_GPU})
          printf("  %cPU->%cPU", srcType == MEM_CPU ? 'C' : 'G', dstType == MEM_CPU ? 'C' : 'G');
      printf("\n");

      printf("Averages (During %s):",  isBidirectional ? " BiDir" : "UniDir");
      for (int srcType : {MEM_CPU, MEM_GPU})
        for (int dstType : {MEM_CPU, MEM_GPU}) {
          if (avgCount[srcType][dstType])
            printf("%10.2f", avgBwSum[srcType][dstType] / avgCount[srcType][dstType]);
          else
            printf("%10s", "N/A");
        }
      printf("\n\n");
    }
  }
}
