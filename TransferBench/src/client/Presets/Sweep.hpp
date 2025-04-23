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

void LogTransfers(FILE *fp, int const testNum, std::vector<Transfer> const& transfers)
{
  if (fp) {
    fprintf(fp, "# Test %d\n", testNum);
    fprintf(fp, "%d", -1 * (int)transfers.size());
    for (auto const& transfer : transfers)
    {
      fprintf(fp, " (%s->%c%d->%s %d %lu)",
              MemDevicesToStr(transfer.srcs).c_str(),
              ExeTypeStr[transfer.exeDevice.exeType], transfer.exeDevice.exeIndex,
              MemDevicesToStr(transfer.dsts).c_str(),
              transfer.numSubExecs,
              transfer.numBytes);
    }
    fprintf(fp, "\n");
    fflush(fp);
  }
}

void SweepPreset(EnvVars&           ev,
                 size_t      const  numBytesPerTransfer,
                 std::string const  presetName)
{
  bool const isRandom = (presetName == "rsweep");

  int numDetectedCpus = TransferBench::GetNumExecutors(EXE_CPU);
  int numDetectedGpus = TransferBench::GetNumExecutors(EXE_GPU_GFX);

  // Collect env vars and set defaults
  int         continueOnErr  = EnvVars::GetEnvVar("CONTINUE_ON_ERROR"   , 0);
  int         numCpuDevices  = EnvVars::GetEnvVar("NUM_CPU_DEVICES"     , numDetectedCpus);
  int         numCpuSubExecs = EnvVars::GetEnvVar("NUM_CPU_SE"          , 4);
  int         numGpuDevices  = EnvVars::GetEnvVar("NUM_GPU_DEVICES"     , numDetectedGpus);
  int         numGpuSubExecs = EnvVars::GetEnvVar("NUM_GPU_SE"          , 4);
  std::string sweepDst       = EnvVars::GetEnvVar("SWEEP_DST"           , "CG");
  std::string sweepExe       = EnvVars::GetEnvVar("SWEEP_EXE"           , "CDG");
  std::string sweepFile      = EnvVars::GetEnvVar("SWEEP_FILE"          , "/tmp/lastSweep.cfg");
  int         sweepMax       = EnvVars::GetEnvVar("SWEEP_MAX"           , 24);
  int         sweepMin       = EnvVars::GetEnvVar("SWEEP_MIN"           , 1);
  int         sweepRandBytes = EnvVars::GetEnvVar("SWEEP_RAND_BYTES"    , 0);
  int         sweepSeed      = EnvVars::GetEnvVar("SWEEP_SEED"          , time(NULL));
  std::string sweepSrc       = EnvVars::GetEnvVar("SWEEP_SRC"           , "CG");
  int         sweepTestLimit = EnvVars::GetEnvVar("SWEEP_TEST_LIMIT"    , 0);
  int         sweepTimeLimit = EnvVars::GetEnvVar("SWEEP_TIME_LIMIT"    , 0);
  int         sweepXgmiMin   = EnvVars::GetEnvVar("SWEEP_XGMI_MIN"      , 0);
  int         sweepXgmiMax   = EnvVars::GetEnvVar("SWEEP_XGMI_MAX"      , -1);

  auto generator = new std::default_random_engine(sweepSeed);

  // Display env var settings
  ev.DisplayEnvVars();
  if (!ev.hideEnv) {
    int outputToCsv = ev.outputToCsv;
    if (!outputToCsv) printf("[Sweep Related]\n");
    ev.Print("CONTINUE_ON_ERROR", continueOnErr,    continueOnErr ? "Continue on mismatch error" : "Stop after first error");
    ev.Print("NUM_CPU_DEVICES",   numCpuDevices,    "Using %d CPUs", numCpuDevices);
    ev.Print("NUM_CPU_SE",        numCpuSubExecs,   "Using %d CPU threads per CPU executed Transfer", numCpuSubExecs);
    ev.Print("NUM_GPU_DEVICES",   numGpuDevices,    "Using %d GPUs", numGpuDevices);
    ev.Print("NUM_GPU_SE",        numGpuSubExecs,   "Using %d subExecutors/CUs per GPU executed Transfer", numGpuSubExecs);
    ev.Print("SWEEP_DST",         sweepDst.c_str(), "Destination Memory Types to sweep");
    ev.Print("SWEEP_EXE",         sweepExe.c_str(), "Executor Types to sweep");
    ev.Print("SWEEP_FILE",        sweepFile.c_str(),"File to store the executing sweep configuration");
    ev.Print("SWEEP_MAX",         sweepMax,         "Max simultaneous transfers (0 = no limit)");
    ev.Print("SWEEP_MIN",         sweepMin,         "Min simultaenous transfers");
    ev.Print("SWEEP_RAND_BYTES",  sweepRandBytes,   "Using %s number of bytes per Transfer", (sweepRandBytes ? "random" : "constant"));
    ev.Print("SWEEP_SEED",        sweepSeed,        "Random seed set to %d", sweepSeed);
    ev.Print("SWEEP_SRC",         sweepSrc.c_str(), "Source Memory Types to sweep");
    ev.Print("SWEEP_TEST_LIMIT",  sweepTestLimit,   "Max number of tests to run during sweep (0 = no limit)");
    ev.Print("SWEEP_TIME_LIMIT",  sweepTimeLimit,   "Max number of seconds to run sweep for  (0 = no limit)");
    ev.Print("SWEEP_XGMI_MAX",    sweepXgmiMax,     "Max number of XGMI hops for Transfers  (-1 = no limit)");
    ev.Print("SWEEP_XGMI_MIN",    sweepXgmiMin,     "Min number of XGMI hops for Transfers");
    printf("\n");
  }

  // Validate env vars
  for (auto ch : sweepSrc) {
    if (!strchr(MemTypeStr, ch)) {
      printf("[ERROR] Unrecognized memory type '%c' specified for sweep source\n", ch);
      exit(1);
    }
    if (strchr(sweepSrc.c_str(), ch) != strrchr(sweepSrc.c_str(), ch)) {
      printf("[ERROR] Duplicate memory type '%c' specified for sweep source\n", ch);
      exit(1);
    }
  }

  for (auto ch : sweepDst) {
    if (!strchr(MemTypeStr, ch)) {
      printf("[ERROR] Unrecognized memory type '%c' specified for sweep destination\n", ch);
      exit(1);
    }
    if (strchr(sweepDst.c_str(), ch) != strrchr(sweepDst.c_str(), ch)) {
      printf("[ERROR] Duplicate memory type '%c' specified for sweep destination\n", ch);
      exit(1);
    }
  }

  for (auto ch : sweepExe) {
    if (!strchr(ExeTypeStr, ch)) {
      printf("[ERROR] Unrecognized executor type '%c' specified for sweep executor\n", ch);
      exit(1);
    }
    if (strchr(sweepExe.c_str(), ch) != strrchr(sweepExe.c_str(), ch)) {
      printf("[ERROR] Duplicate executor type '%c' specified for sweep executor\n", ch);
      exit(1);
    }
  }

  TransferBench::ConfigOptions cfg = ev.ToConfigOptions();
  TransferBench::TestResults results;

  // Compute how many possible Transfers are permitted (unique SRC/EXE/DST triplets)
  std::vector<ExeDevice> exeList;
  for (auto exe : sweepExe) {
    ExeType exeType;
    CharToExeType(exe, exeType);
    if (IsGpuExeType(exeType)) {
      for (int exeIndex = 0; exeIndex < numGpuDevices; ++exeIndex)
        exeList.push_back({exeType, exeIndex});
    }
    else if (IsCpuExeType(exeType)) {
      for (int exeIndex = 0; exeIndex < numCpuDevices; ++exeIndex) {
        // Skip NUMA nodes that have no CPUs (e.g. CXL)
        if (TransferBench::GetNumSubExecutors({EXE_CPU, exeIndex}) == 0) continue;
        exeList.push_back({exeType, exeIndex});
      }
    }
  }
  int numExes = exeList.size();

  std::vector<MemDevice> srcList;
  for (auto src : sweepSrc) {
    MemType srcType;
    CharToMemType(src, srcType);
    int const numDevices = (srcType == MEM_NULL) ? 1 : IsGpuMemType(srcType) ? numGpuDevices : numCpuDevices;

    for (int srcIndex = 0; srcIndex < numDevices; ++srcIndex)
      srcList.push_back({srcType, srcIndex});
  }
  int numSrcs = srcList.size();


  std::vector<MemDevice> dstList;
  for (auto dst : sweepDst) {
    MemType dstType;
    CharToMemType(dst, dstType);
    int const numDevices = (dstType == MEM_NULL) ? 1 : IsGpuMemType(dstType) ? numGpuDevices : numCpuDevices;

    for (int dstIndex = 0; dstIndex < numDevices; ++dstIndex)
      dstList.push_back({dstType, dstIndex});
  }
  int numDsts = dstList.size();

  // Build array of possibilities, respecting any additional restrictions (e.g. XGMI hop count)
  struct TransferInfo
  {
    MemDevice srcMem;
    ExeDevice exeDevice;
    MemDevice dstMem;
  };

  // If either XGMI minimum is non-zero, or XGMI maximum is specified and non-zero then both links must be XGMI
  bool const useXgmiOnly = (sweepXgmiMin > 0 || sweepXgmiMax > 0);

  std::vector<TransferInfo> possibleTransfers;
  TransferInfo tinfo;
  for (int i = 0; i < numExes; ++i) {
    // Skip CPU executors if XGMI link must be used
    if (useXgmiOnly && !IsGpuExeType(exeList[i].exeType)) continue;
    tinfo.exeDevice = exeList[i];

    bool isXgmiSrc  = false;
    int  numHopsSrc = 0;
    for (int j = 0; j < numSrcs; ++j) {
      if (IsGpuExeType(exeList[i].exeType) && IsGpuMemType(srcList[j].memType)) {
        if (exeList[i].exeIndex != srcList[j].memIndex) {
#if defined(__NVCC__)
          isXgmiSrc = false;
#else
          uint32_t exeToSrcLinkType, exeToSrcHopCount;
          HIP_CALL(hipExtGetLinkTypeAndHopCount(exeList[i].exeIndex,
                                                srcList[j].memIndex,
                                                &exeToSrcLinkType,
                                                &exeToSrcHopCount));
          isXgmiSrc = (exeToSrcLinkType == HSA_AMD_LINK_INFO_TYPE_XGMI);
          if (isXgmiSrc) numHopsSrc = exeToSrcHopCount;
#endif
        } else {
          isXgmiSrc = true;
          numHopsSrc = 0;
        }

        // Skip this SRC if it is not XGMI but only XGMI links may be used
        if (useXgmiOnly && !isXgmiSrc) continue;

        // Skip this SRC if XGMI distance is already past limit
        if (sweepXgmiMax >= 0 && isXgmiSrc && numHopsSrc > sweepXgmiMax) continue;
      } else if (srcList[j].memType != MEM_NULL && useXgmiOnly) continue;

      tinfo.srcMem = srcList[j];

      bool isXgmiDst = false;
      int  numHopsDst = 0;
      for (int k = 0; k < numDsts; ++k) {
        if (IsGpuExeType(exeList[i].exeType) && IsGpuMemType(dstList[k].memType)) {
          if (exeList[i].exeIndex != dstList[k].memIndex) {
#if defined(__NVCC__)
            isXgmiSrc = false;
#else
            uint32_t exeToDstLinkType, exeToDstHopCount;
            HIP_CALL(hipExtGetLinkTypeAndHopCount(exeList[i].exeIndex,
                                                  dstList[k].memIndex,
                                                  &exeToDstLinkType,
                                                  &exeToDstHopCount));
            isXgmiDst = (exeToDstLinkType == HSA_AMD_LINK_INFO_TYPE_XGMI);
            if (isXgmiDst) numHopsDst = exeToDstHopCount;
#endif
          } else {
            isXgmiDst = true;
            numHopsDst = 0;
          }
        }

        // Skip this DST if it is not XGMI but only XGMI links may be used
        if (dstList[k].memType != MEM_NULL && useXgmiOnly && !isXgmiDst) continue;

        // Skip this DST if total XGMI distance (SRC + DST) is less than min limit
        if (sweepXgmiMin > 0 && (numHopsSrc + numHopsDst < sweepXgmiMin)) continue;

        // Skip this DST if total XGMI distance (SRC + DST) is greater than max limit
        if (sweepXgmiMax >= 0 && (numHopsSrc + numHopsDst) > sweepXgmiMax) continue;

#if defined(__NVCC__)
        // Skip CPU executors on GPU memory on NVIDIA platform
        if (IsCpuExeType(exeList[i].exeType) && (IsGpuMemType(dstList[j].memType) || IsGpuMemType(dstList[k].memType)))
          continue;
#endif

        tinfo.dstMem = dstList[k];

        // Skip if there is no src and dst
        if (tinfo.srcMem.memType == MEM_NULL && tinfo.dstMem.memType == MEM_NULL) continue;

        possibleTransfers.push_back(tinfo);
      }
    }
  }

  int const numPossible = (int)possibleTransfers.size();
  int maxParallelTransfers = (sweepMax == 0 ? numPossible : sweepMax);

  if (sweepMin > numPossible) {
    printf("No valid test configurations exist\n");
    return;
  }

  if (ev.outputToCsv) {
    printf("\nTest#,Transfer#,NumBytes,Src,Exe,Dst,CUs,BW(GB/s),Time(ms),"
           "ExeToSrcLinkType,ExeToDstLinkType,SrcAddr,DstAddr\n");
  }

  int numTestsRun = 0;
  int M = sweepMin;
  std::uniform_int_distribution<int> randSize(1, numBytesPerTransfer / sizeof(float));
  std::uniform_int_distribution<int> distribution(sweepMin, maxParallelTransfers);

  // Log sweep to configuration file
  char absPath[1024];
  auto const res = realpath(sweepFile.c_str(), absPath);

  FILE *fp = fopen(sweepFile.c_str(), "w");
  if (!fp) {
    printf("[WARN] Unable to open %s.  Skipping output of sweep configuration file\n", res ? absPath : sweepFile.c_str());
  } else {
    printf("Sweep configuration saved to: %s\n", res ? absPath : sweepFile.c_str());
  }

  // Create bitmask of numPossible triplets, of which M will be chosen
  std::string bitmask(M, 1);  bitmask.resize(numPossible, 0);
  auto cpuStart = std::chrono::high_resolution_clock::now();
  while (1)  {
    if (isRandom) {
      // Pick random number of simultaneous transfers to execute
      // NOTE: This currently skews distribution due to some #s having more possibilities than others
      M = distribution(*generator);

      // Generate a random bitmask
      for (int i = 0; i < numPossible; i++)
        bitmask[i] = (i < M) ? 1 : 0;
      std::shuffle(bitmask.begin(), bitmask.end(), *generator);
    }

    // Convert bitmask to list of Transfers
    std::vector<Transfer> transfers;
    for (int value = 0; value < numPossible; ++value) {
      if (bitmask[value]) {
        // Convert integer value to (SRC->EXE->DST) triplet
        Transfer transfer;
        if (possibleTransfers[value].srcMem.memType != MEM_NULL)
          transfer.srcs.push_back(possibleTransfers[value].srcMem);
        transfer.exeDevice      = possibleTransfers[value].exeDevice;
        if (possibleTransfers[value].dstMem.memType != MEM_NULL)
          transfer.dsts.push_back(possibleTransfers[value].dstMem);
        transfer.exeSubIndex    = -1;
        transfer.numSubExecs    = IsGpuExeType(transfer.exeDevice.exeType) ? numGpuSubExecs : numCpuSubExecs;
        transfer.numBytes       = sweepRandBytes ? randSize(*generator) * sizeof(float) : numBytesPerTransfer;
        transfers.push_back(transfer);
      }
    }

    LogTransfers(fp, ++numTestsRun, transfers);

    if (!TransferBench::RunTransfers(cfg, transfers, results)) {
      PrintErrors(results.errResults);
      if (!continueOnErr) exit(1);
    } else {
      PrintResults(ev, numTestsRun, transfers, results);
    }

    // Check for test limit
    if (numTestsRun == sweepTestLimit) {
      printf("Sweep Test limit reached\n");
      break;
    }

    // Check for time limit
    auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
    double totalCpuTime = std::chrono::duration_cast<std::chrono::duration<double>>(cpuDelta).count();
    if (sweepTimeLimit && totalCpuTime > sweepTimeLimit) {
      printf("Sweep Time limit exceeded\n");
      break;
    }

    // Increment bitmask if not random sweep
    if (!isRandom && !std::prev_permutation(bitmask.begin(), bitmask.end())) {
      M++;
      // Check for completion
      if (M > maxParallelTransfers) {
        printf("Sweep complete\n");
        break;
      }
      for (int i = 0; i < numPossible; i++)
        bitmask[i] = (i < M) ? 1 : 0;
    }
  }
  if (fp) fclose(fp);
}
