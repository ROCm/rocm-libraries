/*
Copyright (c) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include "Client.hpp"
#include "Presets.hpp"
#include "Topology.hpp"
#include <fstream>

int main(int argc, char **argv) {

  // Collect environment variables
  EnvVars ev;

  // Display usage instructions and detected topology
  if (argc <= 1) {
    if (!ev.outputToCsv) {
      DisplayUsage(argv[0]);
      DisplayPresets();
    }
    DisplayTopology(ev.outputToCsv);
    exit(0);
  }

  // Determine number of bytes to run per Transfer
  size_t numBytesPerTransfer = argc > 2 ? atoll(argv[2]) : DEFAULT_BYTES_PER_TRANSFER;
  if (argc > 2) {
    // Adjust bytes if unit specified
    char units = argv[2][strlen(argv[2])-1];
    switch (units) {
    case 'G': case 'g': numBytesPerTransfer *= 1024;
    case 'M': case 'm': numBytesPerTransfer *= 1024;
    case 'K': case 'k': numBytesPerTransfer *= 1024;
    }
  }
  if (numBytesPerTransfer % 4) {
    printf("[ERROR] numBytesPerTransfer (%lu) must be a multiple of 4\n", numBytesPerTransfer);
    exit(1);
  }

  // Run preset benchmark if requested
  if (RunPreset(ev, numBytesPerTransfer, argc, argv)) exit(0);

  // Read input from command line or configuration file
  std::vector<std::string> lines;
  {
    std::string line;
    if (!strcmp(argv[1], "cmdline")) {
      for (int i = 3; i < argc; i++)
        line += std::string(argv[i]) + " ";
      lines.push_back(line);
    } else {
      std::ifstream cfgFile(argv[1]);
      if (!cfgFile.is_open()) {
        printf("[ERROR] Unable to open transfer configuration file: [%s]\n", argv[1]);
        exit(1);
      }
      while (std::getline(cfgFile, line))
        lines.push_back(line);
      cfgFile.close();
    }
  }

  // Print environment variables and CSV header
  ev.DisplayEnvVars();
  if (ev.outputToCsv)
    printf("Test#,Transfer#,NumBytes,Src,Exe,Dst,CUs,BW(GB/s),Time(ms),SrcAddr,DstAddr\n");

  TransferBench::ConfigOptions cfgOptions = ev.ToConfigOptions();
  TransferBench::TestResults results;
  std::vector<ErrResult> errors;

  // Process each line as a Test
  int testNum = 0;
  for (std::string const &line : lines) {
    // Check if line is a comment to be echoed to output (starts with ##)
    if (!ev.outputToCsv && line[0] == '#' && line[1] == '#') printf("%s\n", line.c_str());

    // Parse set of parallel Transfers to execute
    std::vector<Transfer> transfers;
    CheckForError(TransferBench::ParseTransfers(line, transfers));
    if (transfers.empty()) continue;

    // Check for variable sub-executors Transfers
    int numVariableTransfers = 0;
    int maxVarCount = 0;
    {
      std::map<ExeDevice, int> varTransferCount;
      for (auto const& t : transfers) {
        if (t.numSubExecs == 0) {
          if (t.exeDevice.exeType != EXE_GPU_GFX) {
            printf("[ERROR] Variable number of subexecutors is only supported on GFX executors\n");
            exit(1);
          }
          numVariableTransfers++;
          varTransferCount[t.exeDevice]++;
          maxVarCount = max(maxVarCount, varTransferCount[t.exeDevice]);
        }
      }
      if (numVariableTransfers > 0 && numVariableTransfers != transfers.size()) {
        printf("[ERROR] All or none of the Transfers in the Test must use variable number of Subexecutors\n");
        exit(1);
      }
    }

    // Track which transfers have already numBytes specified
    std::vector<bool> bytesSpecified(transfers.size());
    int hasUnspecified = false;
    for (int i = 0; i < transfers.size(); i++) {
      bytesSpecified[i] = (transfers[i].numBytes != 0);
      if (transfers[i].numBytes == 0) hasUnspecified = true;
    }

    // Run the specified numbers of bytes otherwise generate a range of values
    for (size_t bytes = (1<<10); bytes <= (1<<29); bytes *= 2) {
      size_t deltaBytes = std::max(1UL, bytes / ev.samplingFactor);
      size_t currBytes = (numBytesPerTransfer == 0) ? bytes : numBytesPerTransfer;
      do {
        for (int i = 0; i < transfers.size(); i++) {
          if (!bytesSpecified[i])
            transfers[i].numBytes = currBytes;
        }

        if (maxVarCount == 0) {
          if (TransferBench::RunTransfers(cfgOptions, transfers, results)) {
            PrintResults(ev, ++testNum, transfers, results);
          }
          PrintErrors(results.errResults);
        } else {
          // Variable subexecutors - Determine how many subexecutors to sweep up to
          int maxNumVarSubExec = ev.maxNumVarSubExec;
          if (maxNumVarSubExec == 0) {
            maxNumVarSubExec = TransferBench::GetNumSubExecutors({EXE_GPU_GFX, 0}) / maxVarCount;
          }

          TransferBench::TestResults bestResults;
          std::vector<Transfer> bestTransfers;
          for (int numSubExecs = ev.minNumVarSubExec; numSubExecs <= maxNumVarSubExec; numSubExecs++) {
            std::vector<Transfer> tempTransfers = transfers;
            for (auto& t : tempTransfers) {
              if (t.numSubExecs == 0) t.numSubExecs = numSubExecs;
            }

            TransferBench::TestResults tempResults;
            if (!TransferBench::RunTransfers(cfgOptions, tempTransfers, tempResults)) {
              PrintErrors(tempResults.errResults);
            } else {
              if (tempResults.avgTotalBandwidthGbPerSec > bestResults.avgTotalBandwidthGbPerSec) {
                bestResults = tempResults;
                bestTransfers = tempTransfers;
              }
            }
          }
          PrintResults(ev, ++testNum, bestTransfers, bestResults);
          PrintErrors(bestResults.errResults);
        }
        if (numBytesPerTransfer != 0 || !hasUnspecified) break;
        currBytes += deltaBytes;
      } while (currBytes < bytes * 2);
      if (numBytesPerTransfer != 0 || !hasUnspecified) break;
    }
  }
}

void DisplayUsage(char const* cmdName)
{
  std::string nicSupport = "";
#if NIC_EXEC_ENABLED
  nicSupport = " (with NIC support)";
#endif
  printf("TransferBench v%s.%s%s\n", TransferBench::VERSION, CLIENT_VERSION, nicSupport.c_str());
  printf("========================================\n");

  if (numa_available() == -1) {
    printf("[ERROR] NUMA library not supported. Check to see if libnuma has been installed on this system\n");
    exit(1);
  }

  printf("Usage: %s config <N>\n", cmdName);
  printf("  config: Either:\n");
  printf("          - Filename of configFile containing Transfers to execute (see example.cfg for format)\n");
  printf("          - Name of preset config:\n");
  printf("  N     : (Optional) Number of bytes to copy per Transfer.\n");
  printf("          If not specified, defaults to %lu bytes. Must be a multiple of 4 bytes\n",
         DEFAULT_BYTES_PER_TRANSFER);
  printf("          If 0 is specified, a range of Ns will be benchmarked\n");
  printf("          May append a suffix ('K', 'M', 'G') for kilobytes / megabytes / gigabytes\n");
  printf("\n");

  EnvVars::DisplayUsage();
}

std::string MemDevicesToStr(std::vector<MemDevice> const& memDevices) {
  if (memDevices.empty()) return "N";
  std::stringstream ss;
  for (auto const& m : memDevices)
    ss << TransferBench::MemTypeStr[m.memType] << m.memIndex;
  return ss.str();
}

void PrintResults(EnvVars const& ev, int const testNum,
                  std::vector<Transfer> const& transfers,
                  TransferBench::TestResults const& results)
{
  char sep = ev.outputToCsv ? ',' : '|';
  size_t numTimedIterations = results.numTimedIterations;

  if (!ev.outputToCsv) printf("Test %d:\n", testNum);

  // Loop over each executor
  for (auto exeInfoPair : results.exeResults) {
    ExeDevice const& exeDevice = exeInfoPair.first;
    ExeResult const& exeResult = exeInfoPair.second;
    ExeType const    exeType   = exeDevice.exeType;
    int32_t const    exeIndex  = exeDevice.exeIndex;

    printf(" Executor: %3s %02d %c %8.3f GB/s %c %8.3f ms %c %12lu bytes %c %-7.3f GB/s (sum)\n",
           ExeTypeName[exeType], exeIndex, sep, exeResult.avgBandwidthGbPerSec, sep,
           exeResult.avgDurationMsec, sep, exeResult.numBytes, sep, exeResult.sumBandwidthGbPerSec);

    // Loop over each executor
    for (int idx : exeResult.transferIdx) {
      Transfer const& t = transfers[idx];
      TransferResult const& r = results.tfrResults[idx];

      char exeSubIndexStr[32] = "";
      if (t.exeSubIndex != -1)
        sprintf(exeSubIndexStr, ".%d", t.exeSubIndex);
      printf("     Transfer %02d  %c %8.3f GB/s %c %8.3f ms %c %12lu bytes %c %s -> %c%03d%s:%03d -> %s\n",
             idx,                    sep,
             r.avgBandwidthGbPerSec, sep,
             r.avgDurationMsec,      sep,
             r.numBytes,             sep,
             MemDevicesToStr(t.srcs).c_str(),
             TransferBench::ExeTypeStr[t.exeDevice.exeType], t.exeDevice.exeIndex,
             exeSubIndexStr, t.numSubExecs,
             MemDevicesToStr(t.dsts).c_str());

      // Show per-iteration timing information
      if (ev.showIterations) {

        // Check that per-iteration information exists
        if (r.perIterMsec.size() != numTimedIterations) {
          printf("[ERROR] Per iteration timing data unavailable: Expected %lu data points, but have %lu\n",
                 numTimedIterations, r.perIterMsec.size());
          exit(1);
        }

        // Compute standard deviation and track iterations by speed
        std::set<std::pair<double, int>> times;
        double stdDevTime = 0;
        double stdDevBw = 0;
        for (int i = 0; i < numTimedIterations; i++) {
          times.insert(std::make_pair(r.perIterMsec[i], i+1));
          double const varTime = fabs(r.avgDurationMsec - r.perIterMsec[i]);
          stdDevTime += varTime * varTime;

          double iterBandwidthGbs = (t.numBytes / 1.0E9) / r.perIterMsec[i] * 1000.0f;
          double const varBw = fabs(iterBandwidthGbs - r.avgBandwidthGbPerSec);
          stdDevBw += varBw * varBw;
        }
        stdDevTime = sqrt(stdDevTime / numTimedIterations);
        stdDevBw = sqrt(stdDevBw / numTimedIterations);

        // Loop over iterations (fastest to slowest)
        for (auto& time : times) {
          double iterDurationMsec = time.first;
          double iterBandwidthGbs = (t.numBytes / 1.0E9) / iterDurationMsec * 1000.0f;
          printf("      Iter %03d    %c %8.3f GB/s %c %8.3f ms %c", time.second, sep, iterBandwidthGbs, sep, iterDurationMsec, sep);

          std::set<int> usedXccs;
          if (time.second - 1 < r.perIterCUs.size()) {
            printf(" CUs:");
            for (auto x : r.perIterCUs[time.second - 1]) {
              printf(" %02d:%02d", x.first, x.second);
              usedXccs.insert(x.first);
            }
          }

          printf(" XCCs:");
          for (auto x : usedXccs)
            printf(" %02d", x);
          printf("\n");
        }
        printf("      StandardDev %c %8.3f GB/s %c %8.3f ms %c\n", sep, stdDevBw, sep, stdDevTime, sep);
      }
    }
  }
  printf(" Aggregate (CPU)  %c %8.3f GB/s %c %8.3f ms %c %12lu bytes %c Overhead: %.3f ms\n",
         sep, results.avgTotalBandwidthGbPerSec,
         sep, results.avgTotalDurationMsec,
         sep, results.totalBytesTransferred,
         sep, results.overheadMsec);
}

void CheckForError(ErrResult const& error)
{
  switch (error.errType) {
  case ERR_NONE: return;
  case ERR_WARN:
    printf("[WARN] %s\n", error.errMsg.c_str());
    return;
  case ERR_FATAL:
    printf("[ERROR] %s\n", error.errMsg.c_str());
    exit(1);
  default:
    break;
  }
}

void PrintErrors(std::vector<ErrResult> const& errors)
{
  bool isFatal = false;
  for (auto const& err : errors) {
    printf("[%s] %s\n", err.errType == ERR_FATAL ? "ERROR" : "WARN", err.errMsg.c_str());
    isFatal |= (err.errType == ERR_FATAL);
  }
  if (isFatal) exit(1);
}
