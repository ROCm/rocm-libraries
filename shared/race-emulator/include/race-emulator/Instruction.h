// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <cassert>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>

namespace raceemulator {

class Wave;

/// Base class for instruction implementations. Each subclass emulates one
/// assembly mnemonic (e.g., v_add_f32, ds_write_b32).
class Instruction {
public:
  /// Parse operands from the assembly line and return a callable that
  /// executes the instruction, returning the next program counter.
  virtual std::function<int()> getExecutor(Wave &registers,
                                           std::string_view line) const = 0;
  virtual ~Instruction() = default;
};

/// Singleton registry mapping instruction mnemonics to Instruction instances.
/// Populated at static initialization time via Register<InstT>.
class InstructionRegistry {
public:
  static InstructionRegistry &instance() {
    static InstructionRegistry instance;
    return instance;
  }
  void add(std::string name, std::unique_ptr<Instruction> inst) {
    map.insert({name, std::move(inst)});
  }
  const std::map<std::string, std::unique_ptr<Instruction>> &
  getInstructions() const {
    return map;
  }

private:
  InstructionRegistry();
  std::map<std::string, std::unique_ptr<Instruction>> map;
};

/// Auto-registration helper. A static instance registers an Instruction
/// subclass with the InstructionRegistry at program startup.
template <typename InstT> struct Register {
  template <typename... Args>
  Register(const std::string &name, Args &&...args) {
    auto instr = std::make_unique<InstT>(std::forward<Args>(args)...);
    InstructionRegistry::instance().add(name, std::move(instr));
  }
};

const std::map<std::string, std::unique_ptr<Instruction>> &getInstructions();

} // namespace raceemulator
