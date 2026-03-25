// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "Workgroup.h"
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace raceemulator {

/// GPU architecture properties used to configure the emulator.
class Architecture {
public:
  virtual ~Architecture() = default;
  /// Maximum LDS (Local Data Share) size in bytes.
  virtual int getMaxLdsSize() const = 0;
  /// Number of lanes per wave (32 for RDNA, 64 for CDNA).
  virtual WaveSize getWaveSize() const = 0;
  /// Number of compute units on the device.
  virtual int getCuCount() const = 0;
  virtual std::string getName() const = 0;
};

/// MI300X (CDNA3).
class Gfx942 : public Architecture {
public:
  int getMaxLdsSize() const override { return 65536; }
  WaveSize getWaveSize() const override { return WaveSize{64}; }
  int getCuCount() const override { return 304; } // MI300X: 8 XCDs × 38 CUs
  std::string getName() const override { return "gfx942"; }
};

/// MI350X (CDNA4).
class Gfx950 : public Architecture {
public:
  int getMaxLdsSize() const override { return 163840; }
  WaveSize getWaveSize() const override { return WaveSize{64}; }
  int getCuCount() const override { return 256; } // MI350X: 8 XCDs × 32 CUs
  std::string getName() const override { return "gfx950"; }
};

/// Strix (RDNA 3.5).
class Gfx1151 : public Architecture {
public:
  int getMaxLdsSize() const override { return 65536; }
  WaveSize getWaveSize() const override { return WaveSize{32}; }
  int getCuCount() const override { return 20; } // TODO: verify
  std::string getName() const override { return "gfx1151"; }
};

/// Create an Architecture from a target string like
/// "amdgcn-amd-amdhsa--gfx942". Throws on unrecognized target.
inline std::shared_ptr<Architecture>
architectureFromTarget(std::string_view target) {
  if (target.find("gfx942") != std::string_view::npos) {
    return std::make_shared<Gfx942>();
  }
  if (target.find("gfx950") != std::string_view::npos) {
    return std::make_shared<Gfx950>();
  }
  if (target.find("gfx1151") != std::string_view::npos) {
    return std::make_shared<Gfx1151>();
  }
  throw std::runtime_error("Unrecognized target architecture: " +
                           std::string(target));
}

} // namespace raceemulator
