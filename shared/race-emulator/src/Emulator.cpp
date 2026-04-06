// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/Emulator.h"
#include "race-emulator/EmulatorException.h"
#include "race-emulator/Parsing.h"
#include "race-emulator/RaceDetector.h"
#include "race-emulator/Util.h"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace raceemulator {

void Emulator::appendStr(std::ostream &os) const {
  parsedAsm->appendStr(os);
  parsedAsm->appendTokensStr(os);
}

std::string Emulator::getProfileReport(double minPercentage) const {
  std::string report;
  if (emulatorProfiler.isEnabled()) {
    report += "=== Emulator ===\n";
    report += emulatorProfiler.reportStr(minPercentage);
    report += "\n=== Workgroups ===\n";
    report += workgroupProfiler.reportStr(minPercentage);
  }
  return report;
}

std::string Emulator::str() const {
  std::ostringstream oss;
  appendStr(oss);
  return oss.str();
}

std::string Emulator::getName() const { return parsedAsm->name; }

int Emulator::getKernargSegmentSize() const {
  return parsedAsm->kernargSegmentSize;
}

int Emulator::getNumKernargs() const {
  return static_cast<int>(parsedAsm->args.size());
}

int Emulator::getKernargOffset(int argNumber) const {
  assert(argNumber >= 0 && argNumber < getNumKernargs());
  return parsedAsm->args[argNumber].offset;
}

int Emulator::getKernargSize(int argNumber) const {
  assert(argNumber >= 0 && argNumber < getNumKernargs());
  return parsedAsm->args[argNumber].size;
}

std::string Emulator::getKernargValueKind(int argNumber) const {
  assert(argNumber >= 0 && argNumber < getNumKernargs());
  return parsedAsm->args[argNumber].valueKind;
}

std::string Emulator::getKernargAddressSpace(int argNumber) const {
  assert(argNumber >= 0 && argNumber < getNumKernargs());
  return parsedAsm->args[argNumber].addressSpace;
}

std::string Emulator::getKernargName(int argNumber) const {
  assert(argNumber >= 0 && argNumber < getNumKernargs());
  return parsedAsm->args[argNumber].name;
}

Emulator::Emulator(std::string_view a, std::shared_ptr<Architecture> arch)
    : arch(std::move(arch)) {
  parsedAsm = std::make_unique<ParsedAsm>(a);

  // Validate that the provided architecture matches the assembly's target.
  if (!parsedAsm->target.empty()) {
    auto detected = architectureFromTarget(parsedAsm->target);
    if (detected->getName() != this->arch->getName()) {
      throw std::runtime_error("Architecture mismatch: assembly targets '" +
                               detected->getName() +
                               "' but emulator was constructed with '" +
                               this->arch->getName() + "'");
    }
  }

  // Pad the kernarg segment so that multi-dword scalar loads
  // (s_load_dwordx4, etc.) that straddle the end don't read out of bounds.
  // The GPU silently ignores reads past the kernarg size. We fill the padding
  // with 0xFF so that any code that accidentally uses a padding value will
  // fail fast: 0xFFFFFFFF is NaN as float and -1/UINT32_MAX as integer.
  // 64 bytes covers the widest load (s_load_dwordx16).
  constexpr int kMaxScalarLoadBytes = 64;
  constexpr char kPoisonByte = static_cast<char>(0xFF);
  int paddedSize = parsedAsm->kernargSegmentSize + kMaxScalarLoadBytes;
  kernargSegment.resize(paddedSize, kPoisonByte);
  kernargIsSet.resize(parsedAsm->args.size(), false);
}

Emulator Emulator::createGfx942(std::string_view assembly) {
  return Emulator(assembly, std::make_shared<Gfx942>());
}

Emulator Emulator::createGfx950(std::string_view assembly) {
  return Emulator(assembly, std::make_shared<Gfx950>());
}

Emulator Emulator::createGfx1151(std::string_view assembly) {
  return Emulator(assembly, std::make_shared<Gfx1151>());
}

WorkgroupConfig Emulator::buildWorkgroupConfig(int nWaves,
                                               const RunConfig &config) const {
  int nextFreeVgpr{-1};
  int accumOffset{-1};
  int nextFreeSgpr{-1};
  int ldsSize{0};

  if (!parsedAsm->amdhsa.empty()) {
    for (const auto &[key, val] : parsedAsm->amdhsa) {
      if (key == ".amdhsa_next_free_sgpr") {
        nextFreeSgpr = val;
      }
      if (key == ".amdhsa_next_free_vgpr") {
        nextFreeVgpr = val;
      } else if (key == ".amdhsa_accum_offset") {
        accumOffset = val;
      } else if (key == ".amdhsa_group_segment_fixed_size") {
        if (val > arch->getMaxLdsSize()) {
          throw std::runtime_error("LDS size " + std::to_string(val) +
                                   " exceeds max for arch '" + arch->getName() +
                                   "' (" +
                                   std::to_string(arch->getMaxLdsSize()) + ")");
        }
        ldsSize = arch->getMaxLdsSize();
      }
    }
  }

  assert(nextFreeVgpr >= 0 && "nextFreeVgpr must be set in AMDHSA metadata");
  assert(nextFreeSgpr >= 0 && "nextFreeSgpr must be set in AMDHSA metadata");

  WorkgroupConfig wgConfig;
  wgConfig.nWaves = nWaves;
  wgConfig.vgprCount = nextFreeVgpr;
  wgConfig.agprOffset = accumOffset; // -1 handled by Workgroup ctor
  wgConfig.sgprCount = nextFreeSgpr;
  wgConfig.waveSize = parsedAsm->wavefrontSize;
  wgConfig.ldsSize = ldsSize;
  wgConfig.raceChecks = config.raceChecks;
  wgConfig.completeEmulation = config.completeEmulation;
  wgConfig.waveSchedule = config.waveSchedule;
  wgConfig.labels = &parsedAsm->labels;
  wgConfig.macros = &parsedAsm->macros;
  return wgConfig;
}

void Emulator::initializeWaveState(Workgroup &workgroup, Dim3d wgId,
                                   Dim3d blockDim, int nWaves) {
  for (int i = 0; i < nWaves; ++i) {
    auto &r = workgroup.getWave(i);
    for (const auto &[key, mapping] :
         parsedAsm->initialRegisterAllocation.registers) {
      if (key == ".amdhsa_user_sgpr_kernarg_segment_ptr") {
        r.setSgpr64(mapping.start_register,
                    reinterpret_cast<uint64_t>(kernargSegment.data()));
      } else if (key == ".amdhsa_system_sgpr_workgroup_id_x") {
        r.setSgpr(mapping.start_register, wgId.x);
      } else if (key == ".amdhsa_system_sgpr_workgroup_id_y") {
        r.setSgpr(mapping.start_register, wgId.y);
      } else if (key == ".amdhsa_system_sgpr_workgroup_id_z") {
        r.setSgpr(mapping.start_register, wgId.z);
      } else {
        throw std::runtime_error(
            "Unhandled SGPR register initialization for key: " + key);
      }
    }
  }

  if (parsedAsm->kernargPreloadLength > 0) {
    auto *srcPtr = reinterpret_cast<const uint32_t *>(
        kernargSegment.data() + parsedAsm->kernargPreloadOffset);
    for (int i = 0; i < nWaves; ++i) {
      auto &r = workgroup.getWave(i);
      for (int j = 0; j < parsedAsm->kernargPreloadLength; ++j) {
        r.setSgpr(2 + j, srcPtr[j]);
      }
    }
  }

  int threadId = {0};
  for (int i = 0; i < nWaves; ++i) {
    auto &r = workgroup.getWave(i);
    for (int lane = 0; lane < r.getWaveSize(); ++lane) {
      int tid_x = threadId % blockDim.x;
      int tid_y = (threadId / blockDim.x) % blockDim.y;
      int tid_z = threadId / (blockDim.x * blockDim.y);
      uint32_t packedThreadId =
          (tid_x & 0x3FF) | ((tid_y & 0x3FF) << 10) | ((tid_z & 0x3FF) << 20);
      r.setVgpr(0, lane, packedThreadId);
      threadId++;
    }
  }

  auto start = parsedAsm->labels.find(parsedAsm->name);
  if (start == parsedAsm->labels.end()) {
    throw std::runtime_error(
        "Kernel start label not found. Expected to find the label '" +
        parsedAsm->name + "' in labels.");
  }

  int labelIndex = start->second;
  for (int i = 0; i < nWaves; ++i) {
    workgroup.getWave(i).setPc(labelIndex);
  }
}

void Emulator::run(const std::vector<Dim3d> &wgIds, Dim3d blockDim,
                   const RunConfig &config) {

  // Validate block dimensions are within hardware limits (10 bits per
  // dimension)
  assert(blockDim.x >= 1 && blockDim.x <= 1024 &&
         "blockDim.x must be in [1, 1024]");
  assert(blockDim.y >= 1 && blockDim.y <= 1024 &&
         "blockDim.y must be in [1, 1024]");
  assert(blockDim.z >= 1 && blockDim.z <= 1024 &&
         "blockDim.z must be in [1, 1024]");

  const WaveSize wavefrontSize = parsedAsm->wavefrontSize;
  const int totalThreads = blockDim.x * blockDim.y * blockDim.z;

  // For early stage development, assert that totalThreads is divisible by
  // wavefrontSize
  assert(totalThreads % wavefrontSize == 0 &&
         "totalThreads must be divisible by wavefrontSize (sanity check for "
         "early stage)");

  const int nWaves = totalThreads / wavefrontSize;

  emulatorProfiler = Profiler(config.profiling);
  workgroupProfiler = Profiler(config.profiling);

  // Set the hidden kernargs (group_size_x/y/z). These are the same for all
  // workgroups so we write them once before the loop.
  auto threadsInX = blockDim.x;
  auto threadsInY = blockDim.y;
  auto threadsInZ = blockDim.z;
  for (size_t i = 0; i < parsedAsm->args.size(); ++i) {
    if (parsedAsm->args[i].valueKind == "hidden_group_size_x") {
      addKernarg(i, &threadsInX);
    } else if (parsedAsm->args[i].valueKind == "hidden_group_size_y") {
      addKernarg(i, &threadsInY);
    } else if (parsedAsm->args[i].valueKind == "hidden_group_size_z") {
      addKernarg(i, &threadsInZ);
    }
  }

  // Validate that all non-hidden kernargs have been set.
  for (size_t i = 0; i < kernargIsSet.size(); ++i) {
    if (!kernargIsSet[i]) {
      if (parsedAsm->args[i].valueKind.find("hidden") != std::string::npos) {
        continue;
      }
      throw std::runtime_error(
          "Kernarg " + std::to_string(i) + " name=(" + parsedAsm->args[i].name +
          ") not set! There are " + std::to_string(kernargIsSet.size()) +
          " kernarg(s). All (non hidden) kernargs must "
          "be set before running the kernel!");
    }
  }

  auto runScope = emulatorProfiler.scopedStopwatch("run");

  const auto &tokens = parsedAsm->tokens;
  const int nWorkgroups = static_cast<int>(wgIds.size());

  // Each workgroup gets its own Profiler (not thread-safe).
  // After joining, merge into workgroupProfiler.
  std::vector<Profiler> wgProfilers(nWorkgroups, Profiler(config.profiling));

  std::vector<std::exception_ptr> exceptions(nWorkgroups);
  std::vector<std::thread> threads;
  threads.reserve(nWorkgroups);

  auto wgConfig = buildWorkgroupConfig(nWaves, config);
  wgConfig.raceHandler = [](RaceViolation v) {
    throw RaceConditionException(v);
  };

  for (int i = 0; i < nWorkgroups; ++i) {
    threads.emplace_back([&, i]() {
      try {
        auto perWgConfig = wgConfig;
        perWgConfig.workgroupId = wgIds[i];
        perWgConfig.profiler = &wgProfilers[i];
        Workgroup workgroup(perWgConfig);
        {
          auto sw = wgProfilers[i].scopedStopwatch("initializeWaveState");
          initializeWaveState(workgroup, wgIds[i], blockDim, nWaves);
        }
        try {
          workgroup.run(tokens);
        } catch (RaceConditionException &e) {
          auto &wave = workgroup.getWave(e.violation.wave);
          auto *detector = workgroup.getRaceDetector();
          auto msg = detector->decorateException(
              e.violation, wave.getPc(),
              workgroup.getRaceState(e.violation.wave),
              static_cast<int>(parsedAsm->tokens.size()),
              [&](int j) -> std::string_view {
                return parsedAsm->tokens[j].originalLine;
              });
          throw RaceConditionException(msg, e.violation);
        }
      } catch (...) {
        exceptions[i] = std::current_exception();
      }
    });
  }

  for (auto &t : threads) {
    t.join();
  }

  // Merge per-workgroup profiling data.
  for (auto &p : wgProfilers) {
    workgroupProfiler.merge(p);
  }

  // Count and rethrow the first exception, if any.
  int nExceptions = 0;
  int firstIdx = -1;
  for (int i = 0; i < nWorkgroups; ++i) {
    if (exceptions[i]) {
      nExceptions++;
      if (firstIdx < 0) {
        firstIdx = i;
      }
    }
  }

  if (nExceptions > 0) {
    try {
      std::rethrow_exception(exceptions[firstIdx]);
    } catch (RaceConditionException &e) {
      if (nExceptions > 1) {
        std::string msg = "Races detected in " + std::to_string(nExceptions) +
                          " of " + std::to_string(nWorkgroups) +
                          " workgroups. Showing first:\n" + e.what();
        throw RaceConditionException(msg, e.violation);
      }
      std::cerr << e.what() << std::endl;
      throw;
    } catch (const EmulatorException &e) {
      std::cerr << "\nRuntime Error: " << e.what() << "\n\n";
      throw;
    }
  }
}

void Emulator::addKernarg(int argNumber, const void *argValue) {
  assert(argNumber >= 0 &&
         argNumber < static_cast<int64_t>(parsedAsm->args.size()) &&
         "Invalid argument number");
  const auto &arg = parsedAsm->args[argNumber];
  assert(arg.offset + arg.size <= parsedAsm->kernargSegmentSize &&
         "Kernarg exceeds segment size");
  kernargIsSet[argNumber] = true;
  std::memcpy(&kernargSegment[arg.offset], argValue, arg.size);
}

void Emulator::addAllKernargs(const void *args) {
  // First, set all 'is set' to true:
  for (size_t i = 0; i < kernargIsSet.size(); ++i) {
    kernargIsSet[i] = true;
  }
  // Copy only the declared kernarg size, not the full (padded) buffer.
  std::memcpy(kernargSegment.data(), args, parsedAsm->kernargSegmentSize);
}

Emulator::Emulator(Emulator &&other) noexcept = default;
Emulator::~Emulator() = default;
} // namespace raceemulator
