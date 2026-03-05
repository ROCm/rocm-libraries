#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace raceemulator {

class Architecture {
public:
  virtual ~Architecture() = default;
  virtual int maxLdsSize() const = 0;
  virtual std::string name() const = 0;
};

class Gfx942 : public Architecture {
public:
  int maxLdsSize() const override { return 65536; }
  std::string name() const override { return "gfx942"; }
};

class Gfx950 : public Architecture {
public:
  int maxLdsSize() const override { return 163840; }
  std::string name() const override { return "gfx950"; }
};

/// Create an Architecture from a target string like
/// "amdgcn-amd-amdhsa--gfx942". Throws on unrecognized target.
inline std::shared_ptr<Architecture>
architectureFromTarget(std::string_view target) {
  if (target.find("gfx942") != std::string_view::npos)
    return std::make_shared<Gfx942>();
  if (target.find("gfx950") != std::string_view::npos)
    return std::make_shared<Gfx950>();
  throw std::runtime_error("Unrecognized target architecture: " +
                           std::string(target));
}

} // namespace raceemulator
