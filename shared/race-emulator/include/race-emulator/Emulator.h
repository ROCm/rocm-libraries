#pragma once
#include "Arch.h"
#include "LDS.h"
#include "Wave.h"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>

namespace raceemulator {

struct ParsedAsm;

class Emulator {

public:
  // TODO(newling) handle case of multiple kernels in single assembly file.
  Emulator(std::string_view assembly, std::shared_ptr<Architecture> arch);

  static Emulator createGfx942(std::string_view assembly);
  static Emulator createGfx950(std::string_view assembly);

  Emulator(const Emulator &other);
  Emulator &operator=(const Emulator &other);
  Emulator(Emulator &&other) noexcept;
  Emulator &operator=(Emulator &&other) = delete;
  ~Emulator();

  // If enabled, race condition checks will be performed during execution.
  // Currently, by default they are disabled.
  void enableRaceChecks(bool);

  void enableCompleteEmulation(bool enable) { completeEmulation = enable; }

  // Write a kernel argument into this emulator's kernarg segment.
  void addKernarg(uint64_t argNumber, const void *argValue);

  void addAllKernargs(const void *args);

  // Initialize registers for each wave.
  // completeEmulation = false: don't do buffer loads or stores.
  void run(Dim3d wgId, Dim3d blockDim);

  // The name of the kernel being emulated.
  std::string name() const;
  int kernargSegmentSize() const;
  int nKernargs() const;
  int kernargOffset(int argNumber) const;
  int kernargSize(int argNumber) const;
  std::string kernargValueKind(int argNumber) const;
  std::string kernargAddressSpace(int argNumber) const;
  std::string kernargName(int argNumber) const;

  const Architecture &getArch() const { return *arch; }

  void appendStr(std::ostream &) const;
  std::string str() const;

private:
  void initializeForRun(Dim3d wgId, Dim3d blockDim, int nWaves);
  std::shared_ptr<Architecture> arch;
  std::unique_ptr<ParsedAsm> parsedAsm;
  std::vector<Wave> registers;
  std::vector<char> kernargSegment;
  std::vector<bool> kernargIsSet;
  LDS lds;
  bool raceChecks{false};
  bool completeEmulation{true};
};

} // namespace raceemulator
