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

void OneToAllPreset(EnvVars&           ev,
                    size_t      const  numBytesPerTransfer,
                    std::string const  presetName)
{
  int numDetectedGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);
  if (numDetectedGpus < 2) {
    printf("[ERROR] One-to-all benchmark requires machine with at least 2 GPUs\n");
    exit(1);
  }

  // Collect env vars for this preset
  int         numGpuDevices = EnvVars::GetEnvVar("NUM_GPU_DEVICES", numDetectedGpus);
  int         numSubExecs   = EnvVars::GetEnvVar("NUM_GPU_SE", 4);
  int         exeIndex      = EnvVars::GetEnvVar("EXE_INDEX",   0);
  int         sweepDir      = EnvVars::GetEnvVar("SWEEP_DIR",  0);
  std::string sweepDst      = EnvVars::GetEnvVar("SWEEP_DST",  "G");
  std::string sweepExe      = EnvVars::GetEnvVar("SWEEP_EXE",  "G");
  std::string sweepSrc      = EnvVars::GetEnvVar("SWEEP_SRC",  "G");
  int         sweepMin      = EnvVars::GetEnvVar("SWEEP_MIN",  1);
  int         sweepMax      = EnvVars::GetEnvVar("SWEEP_MAX",  numGpuDevices);

  // Display environment variables
  ev.DisplayEnvVars();
  if (!ev.hideEnv) {
    if (!ev.outputToCsv) printf("[One-To-All Related]\n");
    ev.Print("NUM_GPU_DEVICES", numGpuDevices,    "Using %d GPUs", numGpuDevices);
    ev.Print("NUM_GPU_SE",      numSubExecs,      "Using %d subExecutors/CUs per Transfer", numSubExecs);
    ev.Print("EXE_INDEX",       exeIndex,         "Executing on GPU %d", exeIndex);
    ev.Print("SWEEP_DIR",       sweepDir,         "Direction of transfer");
    ev.Print("SWEEP_DST",       sweepDst.c_str(), "DST memory types to sweep");
    ev.Print("SWEEP_EXE",       sweepExe.c_str(), "Executor type to use");
    ev.Print("SWEEP_MAX",       sweepMax,         "Maximum number of peers");
    ev.Print("SWEEP_MIN",       sweepMin,         "Minimum number of peers");
    ev.Print("SWEEP_SRC",       sweepSrc.c_str(), "SRC memory types to sweep");
    printf("\n");
  }

  // Perform validation
  for (auto ch : sweepExe) {
    if (ch != 'G' && ch != 'D') {
      printf("[ERROR] Unrecognized executor type '%c' specified\n", ch);
      exit(1);
    }
  }

  TransferBench::ConfigOptions cfg = ev.ToConfigOptions();
  TransferBench::TestResults results;

  char const sep = (ev.outputToCsv ? ',' : ' ');

  for (char src : sweepSrc) for (char exe : sweepExe) for (char dst : sweepDst) {
    // Skip invalid configurations
    if ((exe == 'D' && (src == 'N' || dst == 'N')) || (src == 'N' && dst == 'N')) continue;

    printf("Executing (%c%s -> %c%d -> %c%s)\n",
           src, src == 'N' ? "" : (sweepDir == 0 ? std::to_string(exeIndex).c_str() : std::string("*").c_str()),
           exe, exeIndex,
           dst, dst == 'N' ? "" : sweepDir == 0 ? std::string("*").c_str() : std::to_string(exeIndex).c_str());

    for (int i = 0; i < numGpuDevices; i++) {
      if (i == exeIndex) continue;
      printf("   GPU %-3d  %c", i, sep);
    }
    printf("\n");
    if (!ev.outputToCsv) {
      for (int i = 0; i < numGpuDevices-1; i++)
        printf("-------------");
      printf("\n");
    }

    for (int p = sweepMin; p <= sweepMax; p++) {
      for (int bitmask = 0; bitmask < (1<<numGpuDevices); bitmask++) {
        if (bitmask & (1<<exeIndex) || __builtin_popcount(bitmask) != p) continue;

        std::vector<Transfer> transfers;
        for (int i = 0; i < numGpuDevices; i++) {
          if (bitmask & (1<<i)) {
            Transfer t;
            CheckForError(TransferBench::CharToExeType(exe, t.exeDevice.exeType));
            t.exeDevice.exeIndex = exeIndex;
            t.exeSubIndex = -1;
            t.numSubExecs = numSubExecs;
            t.numBytes    = numBytesPerTransfer;

            if (src == 'N') {
              t.srcs.clear();
            } else {
              t.srcs.resize(1);
              CheckForError(TransferBench::CharToMemType(src, t.srcs[0].memType));
              t.srcs[0].memIndex = sweepDir == 0 ? exeIndex : i;
            }

            if (dst == 'N') {
              t.dsts.clear();
            } else {
              t.dsts.resize(1);
              CheckForError(TransferBench::CharToMemType(dst, t.dsts[0].memType));
              t.dsts[0].memIndex = sweepDir == 0 ? i : exeIndex;
            }
            transfers.push_back(t);
          }
        }
        if (!TransferBench::RunTransfers(cfg, transfers, results)) {
          PrintErrors(results.errResults);
          exit(1);
        }

        int counter = 0;
        for (int i = 0; i < numGpuDevices; i++) {
          if (bitmask & (1<<i))
            printf("  %8.3f  %c", results.tfrResults[counter++].avgBandwidthGbPerSec, sep);
          else if (i != exeIndex)
            printf("            %c", sep);
        }

        printf(" %d %d", p, numSubExecs);
        for (auto i = 0; i < transfers.size(); i++) {
          printf(" (%s %c%d %s)",
                 MemDevicesToStr(transfers[i].srcs).c_str(),
                 ExeTypeStr[transfers[i].exeDevice.exeType], transfers[i].exeDevice.exeIndex,
                 MemDevicesToStr(transfers[i].dsts).c_str());
        }
        printf("\n");
      }
    }
  }
}
