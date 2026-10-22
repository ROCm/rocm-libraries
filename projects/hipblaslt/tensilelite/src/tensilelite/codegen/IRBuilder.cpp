// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/codegen/IRBuilder.hpp>

#include "format.hpp"

#include <stdexcept>
#include <string>

using namespace rocm::tensilelite::codegen;

namespace {

// rocisa's SWaitCnt uses -1 for a counter that is not waited on.
constexpr int noWait = -1;

} // namespace

IRBuilder::InsertionGuard::InsertionGuard(IRBuilder &builder)
    : builder(builder), savedInsertionPoint(&builder.insertionPoint()) {}

IRBuilder::InsertionGuard::~InsertionGuard() {
  builder.setInsertionPoint(*savedInsertionPoint);
}

IRBuilder::IRBuilder(const Target &target, RegisterAllocator &allocator,
                     rocisa::Module &insertionPoint)
    : targetInfo(&target), registers(&allocator), module(&insertionPoint) {
  target.selectForCurrentThread();
}

void IRBuilder::insertItem(const std::shared_ptr<rocisa::Item> &item) {
  if (!item)
    throw std::invalid_argument("cannot insert a null item");
  if (item->parent)
    throw std::logic_error("item is already in a module");
  module->add(item);
}

std::shared_ptr<rocisa::Label> IRBuilder::createLabel(std::string_view name) {
  std::string unique = std::string(name) + "_" + std::to_string(nextLabelId++);
  return std::make_shared<rocisa::Label>(unique, "");
}

std::shared_ptr<rocisa::TextBlock>
IRBuilder::createComment(std::string_view text) {
  return create<rocisa::TextBlock>(rocisa::slash(std::string(text)));
}

std::shared_ptr<rocisa::SWaitCnt>
IRBuilder::createWaitCnt(WaitCounts counts, std::string_view comment) {
  return create<rocisa::SWaitCnt>(
      counts.vectorLoads.value_or(noWait), counts.vectorStores.value_or(noWait),
      counts.lds.value_or(noWait), counts.scalarMemory.value_or(noWait),
      std::string(comment));
}

std::shared_ptr<rocisa::SEndpgm> IRBuilder::createEndProgram() {
  return create<rocisa::SEndpgm>();
}

RegisterRange IRBuilder::allocateVgpr(int count, int alignment,
                                      std::string_view tag) {
  return registers->allocate(RegisterKind::vgpr, count, alignment, tag);
}

RegisterRange IRBuilder::allocateSgpr(int count, int alignment,
                                      std::string_view tag) {
  return registers->allocate(RegisterKind::sgpr, count, alignment, tag);
}

RegisterRange IRBuilder::allocateAgpr(int count, int alignment,
                                      std::string_view tag) {
  return registers->allocate(RegisterKind::agpr, count, alignment, tag);
}
