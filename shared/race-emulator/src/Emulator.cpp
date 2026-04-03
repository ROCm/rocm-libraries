// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/Emulator.h"
#include "race-emulator/EmulatorException.h"
#include "race-emulator/Parsing.h"
#include "race-emulator/Util.h"
#include <algorithm>
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
#include <vector>

namespace raceemulator {

void Emulator::appendStr(std::ostream &os) const {
  parsedAsm->appendStr(os);
  parsedAsm->appendTokensStr(os);
}

std::string Emulator::getProfileReport(double minPercentage) const {
  return profiler.reportStr(minPercentage);
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

void Emulator::initializeWorkgroup(Workgroup &workgroup, Dim3d wgId,
                                   Dim3d blockDim, int nWaves,
                                   const RunConfig &config) {

  workgroup.clear();

  int nextFreeVgpr{-1};
  int accumOffset{-1};
  int nextFreeSgpr{-1};

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
        workgroup.resizeLds(arch->getMaxLdsSize());
      }
    }
  }

  assert(nextFreeVgpr >= 0 && "nextFreeVgpr must be set in AMDHSA metadata");
  assert(nextFreeSgpr >= 0 && "nextFreeSgpr must be set in AMDHSA metadata");
  if (accumOffset < 0) {
    accumOffset = nextFreeVgpr;
  }

  const auto &labels = parsedAsm->labels;
  const auto &macros = parsedAsm->macros;

  workgroup.setRaceChecks(config.raceChecks);
  workgroup.setCompleteEmulation(config.completeEmulation);
  workgroup.setProfiler(&profiler);
  workgroup.setWaveSchedule(config.waveSchedule);

  for (int i = 0; i < nWaves; ++i) {
    workgroup.addWave(Wave(nextFreeVgpr, accumOffset, nextFreeSgpr,
                           parsedAsm->wavefrontSize, WaveId{i}, workgroup,
                           &labels, &macros));
  }

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

  // Kernarg preload: if the kernel specifies preload_length > 0, the hardware
  // preloads that many dwords from the kernarg segment into SGPRs starting
  // right after the kernarg segment pointer (i.e., at s[2]).
  if (parsedAsm->kernargPreloadLength > 0) {
    auto *srcPtr = reinterpret_cast<const uint32_t *>(
        kernargSegment.data() + parsedAsm->kernargPreloadOffset);
    for (int i = 0; i < nWaves; ++i) {
      auto &r = workgroup.getWave(i);
      for (int j = 0; j < parsedAsm->kernargPreloadLength; ++j) {
        // Preloaded SGPRs start at s[2] (right after the kernarg pointer
        // s[0:1]).
        r.setSgpr(2 + j, srcPtr[j]);
      }
    }
  }

  // Pack 3D thread IDs into VGPR0: x in bits [0:9], y in [10:19], z in [20:29].
  // This packing is inferred from hipcc output (see tests/asm/test_3d.s), and
  // may need verification for architectures other than gfx942.
  int threadId = {0};
  for (int i = 0; i < nWaves; ++i) {
    auto &r = workgroup.getWave(i);
    for (int lane = 0; lane < r.getWaveSize(); ++lane) {
      // Calculate 3D coordinates from flat thread ID
      int tid_x = threadId % blockDim.x;
      int tid_y = (threadId / blockDim.x) % blockDim.y;
      int tid_z = threadId / (blockDim.x * blockDim.y);

      // Pack into v0 according to AMD GPU format
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

  profiler.setEnabled(config.profiling);

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
      throw std::runtime_error("Kernarg " + std::to_string(i) + " name=(" +
                               parsedAsm->args[i].name +
                               ") not set! There are " +
                               std::to_string(kernargIsSet.size()) +
                               " kernarg(s). All (non hidden) kernargs must "
                               "be set before running the kernel!");
    }
  }

  auto reportRaceCondition = [&](Workgroup &workgroup,
                                 const RaceConditionException &e, int waveId,
                                 int pc) -> std::string {
    auto &regs = workgroup.getWave(waveId);
    auto getAllVgprEvents = [&](const RaceConditionException &e) {
      assert(e.space == RaceConditionException::Space::VGPR);
      return regs.getVgprMemoryEvents(e.index);
    };

    struct LdsEventInfo {
      WaveId waveId;
      EventId eventId;
    };
    auto getAllLdsEvents = [&](const RaceConditionException &e) {
      assert(e.space == RaceConditionException::Space::LDS);
      std::vector<LdsEventInfo> result;
      auto scanEvents = [&](const std::vector<EventId> &events) {
        for (EventId eventId : events) {
          if (workgroup.getEventIntervals(eventId).contains(e.index)) {
            WaveId evWaveId = workgroup.getEventWaveId(eventId);
            result.push_back({evWaveId, eventId});
          }
        }
      };
      scanEvents(workgroup.getLdsWriteEvents());
      scanEvents(workgroup.getLdsReadEvents());
      return result;
    };

    const int nBefore = 1;
    const int nAfter = 1;

    auto printCodeBlock = [&](std::ostringstream &oss, int startLine,
                              int endLine, const std::vector<int> &arrowLines) {
      for (int i = startLine; i <= endLine; ++i) {
        if (i < 0 || i >= static_cast<int>(parsedAsm->tokens.size())) {
          continue;
        }
        const auto &t = parsedAsm->tokens[i];
        bool isArrowLine = std::find(arrowLines.begin(), arrowLines.end(), i) !=
                           arrowLines.end();
        if (isArrowLine) {
          oss << i << " --> | " << t.originalLine << "\n";
        } else {
          oss << i << "     | " << t.originalLine << "\n";
        }
      }
    };

    if (e.space == RaceConditionException::Space::VGPR) {
      std::ostringstream oss;
      oss << "\nVGPR race detected on line " << pc << " (wave " << e.wave
          << ", lane " << e.lane << "). Conflicting events:\n\n";

      std::vector<int> eventPcs{pc};
      auto vgprEvents = getAllVgprEvents(e);
      for (EventId evtId : vgprEvents) {
        eventPcs.push_back(workgroup.getEventPc(evtId));
      }
      std::sort(eventPcs.begin(), eventPcs.end());

      if (!eventPcs.empty()) {
        int currentBlockStart = eventPcs[0] - nBefore;
        int currentBlockEnd = eventPcs[0] + nAfter;
        std::vector<int> currentArrows = {eventPcs[0]};

        for (size_t i = 1; i < eventPcs.size(); ++i) {
          int nextStart = eventPcs[i] - nBefore;
          int nextEnd = eventPcs[i] + nAfter;
          if (nextStart <= currentBlockEnd + 1) {
            currentBlockEnd = std::max(currentBlockEnd, nextEnd);
            currentArrows.push_back(eventPcs[i]);
          } else {
            printCodeBlock(oss, currentBlockStart, currentBlockEnd,
                           currentArrows);
            oss << "\n";
            currentBlockStart = nextStart;
            currentBlockEnd = nextEnd;
            currentArrows = {eventPcs[i]};
          }
        }
        printCodeBlock(oss, currentBlockStart, currentBlockEnd, currentArrows);
        oss << "\n";
      }
      return oss.str();
    }

    if (e.space == RaceConditionException::Space::LDS) {
      std::ostringstream oss;
      oss << "\nLDS race in byte " << e.index
          << " detected. Race between a pair in:\n\n";

      std::vector<std::tuple<int, int, int>> pcWaveLane{{pc, e.wave, e.lane}};
      auto ldsEvents = getAllLdsEvents(e);
      for (const auto &evt : ldsEvents) {
        pcWaveLane.push_back(
            {workgroup.getEventPc(evt.eventId), evt.waveId.value, -1});
      }
      std::sort(pcWaveLane.begin(), pcWaveLane.end());

      for (auto [localPc, localWaveId, lane] : pcWaveLane) {
        oss << "Wave " << localWaveId;
        if (lane >= 0) {
          oss << " Lane " << lane;
        }
        oss << ":\n";
        printCodeBlock(oss, localPc - nBefore, localPc + nAfter, {localPc});
        oss << "\n";
      }
      return oss.str();
    }

    std::ostringstream oss;
    oss << "\nRace detector for SGPR coming soon\n";
    return oss.str();
  };

  for (const auto &wgId : wgIds) {
    Workgroup workgroup;
    {
      auto sw = profiler.scopedStopwatch("initializeWorkgroup");
      initializeWorkgroup(workgroup, wgId, blockDim, nWaves, config);
    }

    try {
      workgroup.run(parsedAsm->tokens);
    } catch (RaceConditionException &e) {
      int pc = workgroup.getWave(e.wave).getPc();
      auto newMessage = reportRaceCondition(workgroup, e, e.wave, pc);
      RaceConditionException updated = RaceConditionException(
          newMessage, e.space, e.index, e.wave, e.lane, e.isWrite);
      std::cerr << updated.what() << std::endl;
      throw std::move(updated);
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
