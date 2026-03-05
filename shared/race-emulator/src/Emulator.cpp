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
#include <numeric>
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

void Emulator::enableRaceChecks(bool enable) { raceChecks = enable; }

std::string Emulator::str() const {
  std::ostringstream oss;
  appendStr(oss);
  return oss.str();
}

std::string Emulator::name() const { return parsedAsm->name; }

int Emulator::kernargSegmentSize() const {
  return parsedAsm->kernargSegmentSize;
}

int Emulator::nKernargs() const {
  return static_cast<int>(parsedAsm->args.size());
}

int Emulator::kernargOffset(int argNumber) const {
  assert(argNumber >= 0 && argNumber < nKernargs());
  return parsedAsm->args[argNumber].offset;
}

int Emulator::kernargSize(int argNumber) const {
  assert(argNumber >= 0 && argNumber < nKernargs());
  return parsedAsm->args[argNumber].size;
}

std::string Emulator::kernargValueKind(int argNumber) const {
  assert(argNumber >= 0 && argNumber < nKernargs());
  return parsedAsm->args[argNumber].valueKind;
}

std::string Emulator::kernargAddressSpace(int argNumber) const {
  assert(argNumber >= 0 && argNumber < nKernargs());
  return parsedAsm->args[argNumber].addressSpace;
}

std::string Emulator::kernargName(int argNumber) const {
  assert(argNumber >= 0 && argNumber < nKernargs());
  return parsedAsm->args[argNumber].name;
}

Emulator::Emulator(std::string_view a, std::shared_ptr<Architecture> arch)
    : arch(std::move(arch)) {
  parsedAsm = std::make_unique<ParsedAsm>(a);

  // Validate that the provided architecture matches the assembly's target.
  if (!parsedAsm->target.empty()) {
    auto detected = architectureFromTarget(parsedAsm->target);
    if (detected->name() != this->arch->name()) {
      throw std::runtime_error(
          "Architecture mismatch: assembly targets '" + detected->name() +
          "' but emulator was constructed with '" + this->arch->name() + "'");
    }
  }

  kernargSegment.resize(parsedAsm->kernargSegmentSize, 0);
  kernargIsSet.resize(parsedAsm->args.size(), false);
}

Emulator Emulator::createGfx942(std::string_view assembly) {
  return Emulator(assembly, std::make_shared<Gfx942>());
}

Emulator Emulator::createGfx950(std::string_view assembly) {
  return Emulator(assembly, std::make_shared<Gfx950>());
}

void Emulator::initializeForRun(Dim3d wgId, Dim3d blockDim, int nWaves) {

  auto threadsInX = blockDim.x;
  auto threadsInY = blockDim.y;
  auto threadsInZ = blockDim.z;

  // Set the hidden kernargs from wgId. Look for valueKind of
  //  hidden_group_size_x
  //  hidden_group_size_y
  //  hidden_group_size_z
  for (size_t i = 0; i < parsedAsm->args.size(); ++i) {
    if (parsedAsm->args[i].valueKind == "hidden_group_size_x") {
      addKernarg(i, &threadsInX);
    } else if (parsedAsm->args[i].valueKind == "hidden_group_size_y") {
      addKernarg(i, &threadsInY);
    } else if (parsedAsm->args[i].valueKind == "hidden_group_size_z") {
      addKernarg(i, &threadsInZ);
    }
  }

  // If any of kernargIsSet is false, error out.
  for (size_t i = 0; i < kernargIsSet.size(); ++i) {
    if (!kernargIsSet[i]) {

      // If 'hidden' is in the value_kind, skip the error.
      if (parsedAsm->args[i].valueKind.find("hidden") != std::string::npos) {
        continue;
      } else {
        throw std::runtime_error("Kernarg " + std::to_string(i) + " name=(" +
                                 parsedAsm->args[i].name +
                                 ") not set! There are " +
                                 std::to_string(kernargIsSet.size()) +
                                 " kernarg(s). All (non hidden) kernargs must "
                                 "be set before running the kernel!");
      }
    }
  }

  registers.clear();
  lds.clear();

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
        if (val > arch->maxLdsSize()) {
          throw std::runtime_error(
              "LDS size " + std::to_string(val) + " exceeds max for arch '" +
              arch->name() + "' (" + std::to_string(arch->maxLdsSize()) + ")");
        }
        lds.resize(arch->maxLdsSize());
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

  for (int i = 0; i < nWaves; ++i) {

    registers.push_back(Wave(nextFreeVgpr, accumOffset, nextFreeSgpr,
                             parsedAsm->wavefrontSize, i, &lds, &labels,
                             &macros));
    if (raceChecks) {
      registers.back().enableRaceChecks(true);
    }
  }

  if (raceChecks) {
    lds.enableRaceChecks(true);
  }

  for (auto &r : registers) {
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

  // Pack 3D thread IDs into VGPR0: x in bits [0:9], y in [10:19], z in [20:29].
  // This packing is inferred from hipcc output (see tests/asm/test_3d.s), and
  // may need verification for architectures other than gfx942.
  int threadId = {0};
  for (auto &r : registers) {
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
    registers[i].setPc(labelIndex);
  }
}

void Emulator::run(Dim3d wgId, Dim3d blockDim) {

  // Validate block dimensions are within hardware limits (10 bits per
  // dimension)
  assert(blockDim.x >= 1 && blockDim.x < 1024 &&
         "blockDim.x must be in [1, 1024)");
  assert(blockDim.y >= 1 && blockDim.y < 1024 &&
         "blockDim.y must be in [1, 1024)");
  assert(blockDim.z >= 1 && blockDim.z < 1024 &&
         "blockDim.z must be in [1, 1024)");

  const int wavefrontSize = parsedAsm->wavefrontSize;
  const int totalThreads = blockDim.x * blockDim.y * blockDim.z;

  // For early stage development, assert that totalThreads is divisible by
  // wavefrontSize
  assert(totalThreads % wavefrontSize == 0 &&
         "totalThreads must be divisible by wavefrontSize (sanity check for "
         "early stage)");

  const int nWaves = totalThreads / wavefrontSize;

  initializeForRun(wgId, blockDim, nWaves);

  std::vector<bool> waveActive(nWaves, true);
  int nActiveWaves = nWaves;

  std::vector<bool> waveIsWaiting(nWaves, false);
  int nWaitingWaves = 0;

  // If the front wave can run, run it.
  std::vector<int> prefenceOrder(nWaves, 0);
  std::iota(prefenceOrder.begin(), prefenceOrder.end(), 0);

  auto getNextWaveToRun = [&]() -> int {
    for (int waveId : prefenceOrder) {
      if (waveActive[waveId] && !waveIsWaiting[waveId]) {
        return waveId;
      }
    }
    throw std::runtime_error(
        "didn't expect to fail to get wave in this function");
  };

  // If all the active waves are waiting at an s_barrier (not necessarily the
  // same s_barrier!), then release them all.
  auto tryReleaseBarrier = [&]() {
    if (nWaitingWaves == nActiveWaves) {
      for (int w = 0; w < nWaves; ++w) {
        if (waveIsWaiting[w]) {
          waveIsWaiting[w] = false;
        }
      }
      nWaitingWaves = 0;
    }
  };

  // while there exists a wave that has not terminated
  while (nActiveWaves != 0) {
    auto waveId = getNextWaveToRun();

    // TODO: add to token field for 'isEndPgm' and 'isBarrier' to avoid
    // reparsing again.
    const auto &token = parsedAsm->tokens[registers[waveId].getPc()];
    std::string_view trimmedAndCommentFree = trim(token.commentFreeLine);

    if (trimmedAndCommentFree == "s_endpgm") {
      waveActive[waveId] = false;
      nActiveWaves--;
      tryReleaseBarrier();
    }

    if (trimmedAndCommentFree == "s_barrier") {
      waveIsWaiting[waveId] = true;
      nWaitingWaves++;
      tryReleaseBarrier();
    }

    std::string_view line = token.originalLine;

    auto reportRaceCondition = [&](const RaceConditionException &e,
                                   int pc) -> std::string {
      auto &regs = registers[waveId];
      auto &wme = regs.getWaveMemoryEvents();

      auto getAllVgprEvents = [&](const RaceConditionException &e) {
        assert(e.space == RaceConditionException::Space::VGPR);
        std::vector<WaveMemoryEvent> wmes;
        const auto &events = regs.getVgprMemoryEvents(e.index, e.lane);
        for (const auto &regEvent : events) {
          auto it = std::find_if(wme.begin(), wme.end(),
                                 [&](const WaveMemoryEvent &we) {
                                   return we.eventId == regEvent.eventId;
                                 });
          if (it == wme.end()) {
            throw std::runtime_error("failed to find event");
          }
          wmes.push_back(*it);
        }
        return wmes;
      };

      auto getAllLdsEvents = [&](const RaceConditionException &e) {
        assert(e.space == RaceConditionException::Space::LDS);
        std::vector<std::pair<LDSMemoryEvent, WaveMemoryEvent>> wmes;
        const auto &events = lds.getByteMemoryEvents(e.index);
        for (const auto &ldsEvent : events) {
          const auto &wme2 = registers[ldsEvent.waveId].getWaveMemoryEvents();
          auto it = std::find_if(wme2.begin(), wme2.end(),
                                 [&](const WaveMemoryEvent &we) {
                                   return we.eventId == ldsEvent.eventId;
                                 });
          const auto &wcme2 =
              registers[ldsEvent.waveId].getWaveCompleteMemoryEvents();

          if (it == wme2.end()) {
            it = std::find_if(wcme2.begin(), wcme2.end(),
                              [&](const WaveMemoryEvent &we) {
                                return we.eventId == ldsEvent.eventId;
                              });
          }
          assert(it != wcme2.end() && "failed to find event");
          wmes.push_back({ldsEvent, *it});
        }
        return wmes;
      };

      const int nBefore = 1;
      const int nAfter = 1;

      auto printCodeBlock = [&](std::ostringstream &oss, int startLine,
                                int endLine,
                                const std::vector<int> &arrowLines) {
        for (int i = startLine; i <= endLine; ++i) {
          if (i < 0 || i >= static_cast<int>(parsedAsm->tokens.size())) {
            continue;
          }
          const auto &t = parsedAsm->tokens[i];

          // Check if the current line 'i' is in our list of lines that need an
          // arrow
          bool isArrowLine = std::find(arrowLines.begin(), arrowLines.end(),
                                       i) != arrowLines.end();

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
        for (const auto &evt : vgprEvents) {
          eventPcs.push_back(evt.pc);
        }
        std::sort(eventPcs.begin(), eventPcs.end());

        if (!eventPcs.empty()) {
          int currentBlockStart = eventPcs[0] - nBefore;
          int currentBlockEnd = eventPcs[0] + nAfter;
          std::vector<int> currentArrows = {eventPcs[0]};

          for (size_t i = 1; i < eventPcs.size(); ++i) {
            int nextStart = eventPcs[i] - nBefore;
            int nextEnd = eventPcs[i] + nAfter;

            // Check if ranges overlap or touch.
            if (nextStart <= currentBlockEnd + 1) {
              currentBlockEnd = std::max(currentBlockEnd, nextEnd);
              currentArrows.push_back(eventPcs[i]);
            } else {
              // NO OVERLAP: Print the accumulated block and start a new one
              printCodeBlock(oss, currentBlockStart, currentBlockEnd,
                             currentArrows);
              oss << "\n";
              // Reset for next block
              currentBlockStart = nextStart;
              currentBlockEnd = nextEnd;
              currentArrows = {eventPcs[i]};
            }
          }
          // Print the final remaining block
          printCodeBlock(oss, currentBlockStart, currentBlockEnd,
                         currentArrows);
          oss << "\n";
        }
        return oss.str();
      }

      else if (e.space == RaceConditionException::Space::LDS) {
        std::ostringstream oss;
        oss << "\nLDS race in byte " << e.index
            << " detected. Race between a pair in:\n\n";

        std::vector<std::tuple<int, int, int>> pcId{{pc, e.wave, e.lane}};
        auto ldsEvents = getAllLdsEvents(e);
        for (const auto &evt : ldsEvents) {
          pcId.push_back({evt.second.pc, evt.first.waveId, evt.first.lane});
        }
        std::sort(pcId.begin(), pcId.end());

        for (auto t : pcId) {
          oss << "Wave " << std::get<1>(t) << " Lane " << std::get<2>(t)
              << ":\n";

          int localPc = std::get<0>(t);
          printCodeBlock(oss, localPc - nBefore, localPc + nAfter, {localPc});

          oss << "\n";
        }
        return oss.str();
      } else {
        std::ostringstream oss;
        oss << "\nRace detector for SGPR coming soon" << "\n";
        return oss.str();
      }
    };

    try {
      auto l = token.commentFreeLine;

      // If 'v_' is in the line, print it.
      bool isVector = (l.find("v_") != std::string::npos);
      bool isDs = (l.find("ds_") != std::string::npos);
      bool isBufferOrGlobal = (l.find("buffer") != std::string::npos ||
                               l.find("global") != std::string::npos);
      bool isScalar = (l.find("s_") != std::string::npos);
      bool isWaitcnt = (l.find("s_waitcnt") != std::string::npos);

      (void)isVector;
      (void)isDs;
      (void)isBufferOrGlobal;
      (void)isScalar;
      (void)isWaitcnt;
      registers[waveId].tryExecute(l, true);
    }

    catch (RaceConditionException &e) {
      auto newMessage = reportRaceCondition(e, registers[waveId].getPc());
      RaceConditionException updated = RaceConditionException(
          newMessage, e.space, e.index, e.wave, e.lane, e.isWrite);

      // TODO: throwing exceptions across libraries flunks, why?
      std::cerr << updated.what() << std::endl;
      throw std::move(updated);
    } catch (const EmulatorException &e) {
      std::cerr << "\nRuntime Error at PC " << registers[waveId].getPc() << ": "
                << line << "\n";
      std::cerr << "  " << e.what() << "\n\n";
      throw;
    }
  }
}

void Emulator::addKernarg(uint64_t argNumber, const void *argValue) {
  assert(argNumber < parsedAsm->args.size() && "Invalid argument number");
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
  // Now, just do a memcpy of the whole thing:
  std::memcpy(kernargSegment.data(), args, kernargSegment.size());
}

Emulator::Emulator(Emulator &&other) noexcept = default;
Emulator::~Emulator() = default;
} // namespace raceemulator
