// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace raceemulator {

class Architecture {
public:
  virtual ~Architecture() = default;
  virtual int getMaxLdsSize() const = 0;
  virtual int getWaveSize() const = 0;
  virtual int getCuCount() const = 0;
  virtual std::string getName() const = 0;
};

class Gfx942 : public Architecture {
public:
  int getMaxLdsSize() const override { return 65536; }
  int getWaveSize() const override { return 64; }
  int getCuCount() const override { return 304; } // MI300X: 8 XCDs × 38 CUs
  std::string getName() const override { return "gfx942"; }
};

class Gfx950 : public Architecture {
public:
  int getMaxLdsSize() const override { return 163840; }
  int getWaveSize() const override { return 64; }
  int getCuCount() const override { return 256; } // MI350X: 8 XCDs × 32 CUs
  std::string getName() const override { return "gfx950"; }
};

class Gfx1151 : public Architecture {
public:
  int getMaxLdsSize() const override { return 65536; }
  int getWaveSize() const override { return 32; }
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
