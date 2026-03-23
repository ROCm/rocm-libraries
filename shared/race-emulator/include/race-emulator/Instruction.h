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

class Instruction {
public:
  // Given a specific line in the assembly, return a function that executes it.
  virtual std::function<int()> getExecutor(Wave &registers,
                                           std::string_view line) const = 0;
  virtual ~Instruction() = default;
};

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

template <typename InstT> struct Register {
  template <typename... Args>
  Register(const std::string &name, Args &&...args) {
    auto instr = std::make_unique<InstT>(std::forward<Args>(args)...);
    InstructionRegistry::instance().add(name, std::move(instr));
  }
};

const std::map<std::string, std::unique_ptr<Instruction>> &getInstructions();

} // namespace raceemulator
