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

#pragma once

#include "TransferBench.hpp"

static int RemappedCpuIndex(int origIdx)
{
  static std::vector<int> remappingCpu;

  // Build CPU remapping on first use
  // Skip numa nodes that are not configured
  if (remappingCpu.empty()) {
    for (int node = 0; node <= numa_max_node(); node++)
      if (numa_bitmask_isbitset(numa_get_mems_allowed(), node))
        remappingCpu.push_back(node);
  }
  return remappingCpu[origIdx];
}

static void PrintNicToGPUTopo(bool outputToCsv)
{
#ifdef NIC_EXEC_ENABLED
  printf(" NIC | Device Name | Active | PCIe Bus ID  | NUMA | Closest GPU(s) | GID Index | GID Descriptor\n");
  if(!outputToCsv)
    printf("-----+-------------+--------+--------------+------+----------------+-----------+-------------------\n");

  int numGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);
  auto const& ibvDeviceList = GetIbvDeviceList();
  for (int i = 0; i < ibvDeviceList.size(); i++) {

    std::string closestGpusStr = "";
    for (int j = 0; j < numGpus; j++) {
      if (TransferBench::GetClosestNicToGpu(j) == i) {
        if (closestGpusStr != "") closestGpusStr += ",";
        closestGpusStr += std::to_string(j);
      }
    }

    printf(" %-3d | %-11s | %-6s | %-12s | %-4d | %-14s | %-9s | %-20s\n",
           i, ibvDeviceList[i].name.c_str(),
           ibvDeviceList[i].hasActivePort ? "Yes" : "No",
           ibvDeviceList[i].busId.c_str(),
           ibvDeviceList[i].numaNode,
           closestGpusStr.c_str(),
           ibvDeviceList[i].isRoce && ibvDeviceList[i].hasActivePort?  std::to_string(ibvDeviceList[i].gidIndex).c_str() : "N/A",
           ibvDeviceList[i].isRoce && ibvDeviceList[i].hasActivePort?  ibvDeviceList[i].gidDescriptor.c_str() : "N/A"
          );
  }
  printf("\n");
#endif
}

void DisplayTopology(bool outputToCsv)
{
  int numCpus = TransferBench::GetNumExecutors(EXE_CPU);
  int numGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);
  int numNics = TransferBench::GetNumExecutors(EXE_NIC);
  char sep = (outputToCsv ? ',' : '|');

  if (outputToCsv) {
    printf("NumCpus,%d\n", numCpus);
    printf("NumGpus,%d\n", numGpus);
    printf("NumNics,%d\n", numNics);
  } else {
    printf("\nDetected Topology:\n");
    printf("==================\n");
    printf("  %d configured CPU NUMA node(s) [%d total]\n", numCpus, numa_max_node() + 1);
    printf("  %d GPU device(s)\n", numGpus);
    printf("  %d Supported NIC device(s)\n", numNics);
  }

  // Print out detected CPU topology
  printf("\n            %c", sep);
  for (int j = 0; j < numCpus; j++)
    printf("NUMA %02d%c", j, sep);
  printf(" #Cpus %c Closest GPU(s)\n", sep);

  if (!outputToCsv) {
    printf("------------+");
    for (int j = 0; j <= numCpus; j++)
      printf("-------+");
    printf("---------------\n");
  }

  for (int i = 0; i < numCpus; i++) {
    int nodeI = RemappedCpuIndex(i);
    printf("NUMA %02d (%02d)%c", i, nodeI, sep);
    for (int j = 0; j < numCpus; j++) {
      int nodeJ = RemappedCpuIndex(j);
      int numaDist = numa_distance(nodeI, nodeJ);
      printf(" %5d %c", numaDist, sep);
    }

    int numCpuCores = 0;
    for (int j = 0; j < numa_num_configured_cpus(); j++)
      if (numa_node_of_cpu(j) == nodeI) numCpuCores++;
    printf(" %5d %c", numCpuCores, sep);

    for (int j = 0; j < numGpus; j++) {
      if (TransferBench::GetClosestCpuNumaToGpu(j) == nodeI) {
        printf(" %d", j);
      }
    }
    printf("\n");
  }
  printf("\n");

  // Print out detected NIC topology
  PrintNicToGPUTopo(outputToCsv);

  // Print out detected GPU topology
#if defined(__NVCC__)
  for (int i = 0; i < numGpus; i++) {
    hipDeviceProp_t prop;
    HIP_CALL(hipGetDeviceProperties(&prop, i));
    printf(" GPU %02d | %s\n", i, prop.name);
  }
  // No further topology detection done for NVIDIA platforms
  return;
#else
  // Print headers
  if (!outputToCsv) {
    printf("        |");
    for (int j = 0; j < numGpus; j++) {
      hipDeviceProp_t prop;
      HIP_CALL(hipGetDeviceProperties(&prop, j));
      std::string fullName = prop.gcnArchName;
      std::string archName = fullName.substr(0, fullName.find(':'));
      printf(" %6s |", archName.c_str());
    }
    printf("\n");
  }

  printf("        %c", sep);
  for (int j = 0; j < numGpus; j++)
    printf(" GPU %02d %c", j, sep);
  printf(" PCIe Bus ID  %c #CUs %c NUMA %c #DMA %c #XCC %c NIC\n", sep, sep, sep, sep, sep);

  if (!outputToCsv) {
    for (int j = 0; j <= numGpus; j++)
      printf("--------+");
    printf("--------------+------+------+------+------+------\n");
  }

  // Loop over each GPU device
  for (int i = 0; i < numGpus; i++) {
    printf(" GPU %02d %c", i, sep);

    // Print off link information
    for (int j = 0; j < numGpus; j++) {
      if (i == j) {
        printf("    N/A %c", sep);
      } else {
        uint32_t linkType, hopCount;
        HIP_CALL(hipExtGetLinkTypeAndHopCount(i, j, &linkType, &hopCount));
        printf(" %s-%d %c",
               linkType == HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT ? "  HT" :
               linkType == HSA_AMD_LINK_INFO_TYPE_QPI            ? " QPI" :
               linkType == HSA_AMD_LINK_INFO_TYPE_PCIE           ? "PCIE" :
               linkType == HSA_AMD_LINK_INFO_TYPE_INFINBAND      ? "INFB" :
               linkType == HSA_AMD_LINK_INFO_TYPE_XGMI           ? "XGMI" : "????",
               hopCount, sep);
      }
    }

    char pciBusId[20];
    HIP_CALL(hipDeviceGetPCIBusId(pciBusId, 20, i));
    printf(" %-11s %c %-4d %c %-4d %c %-4d %c %-4d %c %-4d\n",
           pciBusId, sep,
           TransferBench::GetNumSubExecutors({EXE_GPU_GFX, i}), sep,
           TransferBench::GetClosestCpuNumaToGpu(i), sep,
           TransferBench::GetNumExecutorSubIndices({EXE_GPU_DMA, i}), sep,
           TransferBench::GetNumExecutorSubIndices({EXE_GPU_GFX, i}), sep,
           TransferBench::GetClosestNicToGpu(i));
  }
#endif
}
