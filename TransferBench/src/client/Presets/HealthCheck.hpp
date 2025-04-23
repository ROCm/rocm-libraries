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


void HealthCheckPreset(EnvVars&           ev,
                       size_t      const  numBytesPerTransfer,
                       std::string const  presetName)
{
  // Check for supported platforms
#if defined(__NVCC__)
  printf("[WARN] healthcheck preset not supported on NVIDIA hardware\n");
  return;
#endif

  bool hasFail = false;

  // Force use of single stream
  ev.useSingleStream = 1;

  TransferBench::TestResults results;
  int numGpuDevices = TransferBench::GetNumExecutors(EXE_GPU_GFX);

  if (numGpuDevices != 8) {
    printf("[WARN] healthcheck preset is currently only supported on 8-GPU MI300X hardware\n");
    exit(1);
  }

  for (int gpuId = 0; gpuId < numGpuDevices; gpuId++) {
    hipDeviceProp_t prop;
    HIP_CALL(hipGetDeviceProperties(&prop, gpuId));
    std::string fullName = prop.gcnArchName;
    std::string archName = fullName.substr(0, fullName.find(':'));
    if (!(archName == "gfx940" || archName == "gfx941" || archName == "gfx942"))
    {
      printf("[WARN] healthcheck preset is currently only supported on 8-GPU MI300X hardware\n");
      exit(1);
    }
  }

  // Pass limits
  double udirLimit = getenv("LIMIT_UDIR") ? atof(getenv("LIMIT_UDIR")) : (int)(48 * 0.95);
  double bdirLimit = getenv("LIMIT_BDIR") ? atof(getenv("LIMIT_BDIR")) : (int)(96 * 0.95);
  double a2aLimit  = getenv("LIMIT_A2A")  ? atof(getenv("LIMIT_A2A"))  : (int)(45 * 0.95);

  // Run CPU to GPU

  // Run unidirectional read from CPU to GPU
  printf("Testing unidirectional reads from CPU ");
  {
    ev.gfxUnroll = 4;
    TransferBench::ConfigOptions cfg = ev.ToConfigOptions();
    std::vector<std::pair<int, double>> fails;
    for (int gpuId = 0; gpuId < numGpuDevices; gpuId++) {
      printf("."); fflush(stdout);

      int memIndex = TransferBench::GetClosestCpuNumaToGpu(gpuId);
      if (memIndex == -1) {
        printf("[ERROR] Unable to detect closest CPU NUMA node to GPU %d\n", gpuId);
        exit(1);
      }

      std::vector<Transfer> transfers(1);
      Transfer& t = transfers[0];
      t.exeDevice = {EXE_GPU_GFX, gpuId};
      t.numBytes  = 64*1024*1024;
      t.srcs      = {{MEM_CPU, memIndex}};
      t.dsts      = {};

      // Loop over number of CUs to use
      bool passed = false;
      double bestResult = 0;
      for (int cu = 7; cu <= 10; cu++) {
        t.numSubExecs = cu;
        if (TransferBench::RunTransfers(cfg, transfers, results)) {
          bestResult = std::max(bestResult, results.tfrResults[0].avgBandwidthGbPerSec);
        } else {
          PrintErrors(results.errResults);
        }
        if (results.tfrResults[0].avgBandwidthGbPerSec >= udirLimit) {
          passed = true;
          break;
        }
      }
      if (!passed) fails.push_back(std::make_pair(gpuId, bestResult));
    }
    if (fails.size() == 0) {
      printf("PASS\n");
    } else {
      hasFail = true;
      printf("FAIL (%lu test(s))\n", fails.size());
      for (auto p : fails) {
        printf(" GPU %02d: Measured: %6.2f GB/s      Criteria: %6.2f GB/s\n", p.first, p.second, udirLimit);
      }
    }
  }

  // Run unidirectional write from GPU to CPU
  printf("Testing unidirectional writes to  CPU ");
  {
    ev.gfxUnroll = 4;
    TransferBench::ConfigOptions cfg = ev.ToConfigOptions();

    std::vector<std::pair<int, double>> fails;
    for (int gpuId = 0; gpuId < numGpuDevices; gpuId++) {
      printf("."); fflush(stdout);

      int memIndex = TransferBench::GetClosestCpuNumaToGpu(gpuId);
      if (memIndex == -1) {
        printf("[ERROR] Unable to detect closest CPU NUMA node to GPU %d\n", gpuId);
        exit(1);
      }

      std::vector<Transfer> transfers(1);
      Transfer& t = transfers[0];
      t.exeDevice = {EXE_GPU_GFX, gpuId};
      t.numBytes  = 64*1024*1024;
      t.srcs      = {};
      t.dsts      = {{MEM_CPU, memIndex}};

      // Loop over number of CUs to use
      bool passed = false;
      double bestResult = 0;
      for (int cu = 7; cu <= 10; cu++) {
        t.numSubExecs = cu;
        if (TransferBench::RunTransfers(cfg, transfers, results)) {
          bestResult = std::max(bestResult, results.tfrResults[0].avgBandwidthGbPerSec);
        } else {
          PrintErrors(results.errResults);
        }
        if (results.tfrResults[0].avgBandwidthGbPerSec >= udirLimit) {
          passed = true;
          break;
        }
      }
      if (!passed) fails.push_back(std::make_pair(gpuId, bestResult));
    }
    if (fails.size() == 0) {
      printf("PASS\n");
    } else {
      hasFail = true;
      printf("FAIL (%lu test(s))\n", fails.size());
      for (auto p : fails) {
        printf(" GPU %02d: Measured: %6.2f GB/s      Criteria: %6.2f GB/s\n", p.first, p.second, udirLimit);
      }
    }
  }

  // Run bidirectional tests
  printf("Testing bidirectional  reads + writes ");
  {
    ev.gfxUnroll = 4;
    TransferBench::ConfigOptions cfg = ev.ToConfigOptions();

    std::vector<std::pair<int, double>> fails;
    for (int gpuId = 0; gpuId < numGpuDevices; gpuId++) {
      printf("."); fflush(stdout);

      int memIndex = TransferBench::GetClosestCpuNumaToGpu(gpuId);
      if (memIndex == -1) {
        printf("[ERROR] Unable to detect closest CPU NUMA node to GPU %d\n", gpuId);
        exit(1);
      }

      std::vector<Transfer> transfers(2);
      Transfer& t0 = transfers[0];
      Transfer& t1 = transfers[1];

      t0.exeDevice = {EXE_GPU_GFX, gpuId};
      t0.numBytes  = 64*1024*1024;
      t0.srcs      = {{MEM_CPU, memIndex}};
      t0.dsts      = {};

      t1.exeDevice = {EXE_GPU_GFX, gpuId};
      t1.numBytes  = 64*1024*1024;
      t1.srcs      = {};
      t1.dsts      = {{MEM_CPU, memIndex}};

      // Loop over number of CUs to use
      bool passed = false;
      double bestResult = 0;
      for (int cu = 7; cu <= 10; cu++) {
        t0.numSubExecs = cu;
        t1.numSubExecs = cu;

        if (TransferBench::RunTransfers(cfg, transfers, results)) {
          double sum = (results.tfrResults[0].avgBandwidthGbPerSec +
                        results.tfrResults[1].avgBandwidthGbPerSec);
          bestResult = std::max(bestResult, sum);
          if (sum >= bdirLimit) {
            passed = true;
            break;
          }
        } else {
          PrintErrors(results.errResults);
        }
      }
      if (!passed) fails.push_back(std::make_pair(gpuId, bestResult));
    }
    if (fails.size() == 0) {
      printf("PASS\n");
    } else {
      hasFail = true;
      printf("FAIL (%lu test(s))\n", fails.size());
      for (auto p : fails) {
        printf(" GPU %02d: Measured: %6.2f GB/s      Criteria: %6.2f GB/s\n", p.first, p.second, bdirLimit);
      }
    }
  }

  // Run XGMI tests:
  printf("Testing all-to-all XGMI copies        "); fflush(stdout);
  {
    // Force GFX unroll to 2 for MI300
    ev.gfxUnroll = 2;
    TransferBench::ConfigOptions cfg = ev.ToConfigOptions();

    std::vector<Transfer> transfers;
    for (int i = 0; i < numGpuDevices; i++) {
      for (int j = 0; j < numGpuDevices; j++) {
        if (i == j) continue;
        Transfer t;
        t.numBytes    = 64*1024*1024;
        t.numSubExecs = 8;
        t.exeDevice   = {EXE_GPU_GFX, i};
        t.srcs        = {{MEM_GPU_FINE, i}};
        t.dsts        = {{MEM_GPU_FINE, j}};
        transfers.push_back(t);
      }
    }
    std::vector<std::pair<std::pair<int,int>, double>> fails;

    if (TransferBench::RunTransfers(cfg, transfers, results)) {
      int transferIdx = 0;
      for (int i = 0; i < numGpuDevices; i++) {
        printf("."); fflush(stdout);
        for (int j = 0; j < numGpuDevices; j++) {
          if (i == j) continue;
          double bw = results.tfrResults[transferIdx].avgBandwidthGbPerSec;
          if (bw < a2aLimit) {
            fails.push_back(std::make_pair(std::make_pair(i,j), bw));
          }
          transferIdx++;
        }
      }
    }
    if (fails.size() == 0) {
      printf("PASS\n");
    } else {
      hasFail = true;
      printf("FAIL (%lu test(s))\n", fails.size());
      for (auto p : fails) {
        printf(" GPU %02d to GPU %02d: %6.2f GB/s      Criteria: %6.2f GB/s\n", p.first.first, p.first.second, p.second, a2aLimit);
      }
    }
  }
  exit(hasFail ? 1 : 0);
}
