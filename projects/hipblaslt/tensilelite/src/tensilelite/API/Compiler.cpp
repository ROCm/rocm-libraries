// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tensilelite/API/Compiler.hpp>

#include <amd_comgr/amd_comgr.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace rocm::tensilelite;

namespace {

// Releases the COMGR handle a guard points at; COMGR handles are plain
// structs, so guards point at local handles.
template <auto destroy>
struct ComgrDeleter {
  template <typename Handle>
  void operator()(Handle *handle) const {
    destroy(*handle);
  }
};

} // namespace

using DataGuard =
    std::unique_ptr<amd_comgr_data_t, ComgrDeleter<amd_comgr_release_data>>;
using DataSetGuard = std::unique_ptr<amd_comgr_data_set_t,
                                     ComgrDeleter<amd_comgr_destroy_data_set>>;
using ActionInfoGuard =
    std::unique_ptr<amd_comgr_action_info_t,
                    ComgrDeleter<amd_comgr_destroy_action_info>>;

static std::string statusString(amd_comgr_status_t status) {
  const char *text = nullptr;
  if (amd_comgr_status_string(status, &text) != AMD_COMGR_STATUS_SUCCESS ||
      !text)
    return "unknown status";
  return text;
}

static void check(amd_comgr_status_t status, std::string_view call) {
  if (status == AMD_COMGR_STATUS_SUCCESS)
    return;
  throw std::runtime_error(std::string(call) +
                           " failed: " + statusString(status));
}

static void addData(amd_comgr_data_set_t set, amd_comgr_data_kind_t kind,
                    std::string_view contents, const char *name) {
  amd_comgr_data_t data;
  check(amd_comgr_create_data(kind, &data), "amd_comgr_create_data");
  DataGuard dataGuard(&data);
  check(amd_comgr_set_data(data, contents.size(), contents.data()),
        "amd_comgr_set_data");
  check(amd_comgr_set_data_name(data, name), "amd_comgr_set_data_name");
  check(amd_comgr_data_set_add(set, data), "amd_comgr_data_set_add");
}

static amd_comgr_status_t readData(amd_comgr_data_set_t set,
                                   amd_comgr_data_kind_t kind, int64_t index,
                                   std::vector<uint8_t> &bytes) {
  amd_comgr_data_t data;
  amd_comgr_status_t status = amd_comgr_action_data_get_data(
      set, kind, static_cast<size_t>(index), &data);
  if (status != AMD_COMGR_STATUS_SUCCESS)
    return status;
  DataGuard dataGuard(&data);
  size_t size = 0;
  status = amd_comgr_get_data(data, &size, nullptr);
  if (status != AMD_COMGR_STATUS_SUCCESS)
    return status;
  bytes.resize(size);
  return amd_comgr_get_data(data, &size,
                            reinterpret_cast<char *>(bytes.data()));
}

static void runAction(amd_comgr_action_kind_t kind, std::string_view actionName,
                      amd_comgr_action_info_t info, std::string_view target,
                      amd_comgr_data_set_t input, amd_comgr_data_set_t output) {
  std::string isaName(target);
  check(amd_comgr_action_info_set_isa_name(info, isaName.c_str()),
        "amd_comgr_action_info_set_isa_name for target '" + isaName + "'");
  check(amd_comgr_action_info_set_logging(info, true),
        "amd_comgr_action_info_set_logging");
  amd_comgr_status_t status = amd_comgr_do_action(kind, info, input, output);
  if (status == AMD_COMGR_STATUS_SUCCESS)
    return;
  std::string message = "comgr action '" + std::string(actionName) +
                        "' failed: " + statusString(status) + "\n";
  // The log is best effort, so a failed read never masks the action failure.
  size_t count = 0;
  if (amd_comgr_action_data_count(output, AMD_COMGR_DATA_KIND_LOG, &count) !=
      AMD_COMGR_STATUS_SUCCESS)
    count = 0;
  std::vector<uint8_t> log;
  for (int64_t i = 0; i < static_cast<int64_t>(count); ++i) {
    if (readData(output, AMD_COMGR_DATA_KIND_LOG, i, log) !=
        AMD_COMGR_STATUS_SUCCESS)
      break;
    message.append(log.begin(), log.end());
  }
  throw std::runtime_error(message);
}

static void codegen(amd_comgr_action_kind_t kind, std::string_view actionName,
                    std::string_view llvmIR, std::string_view target,
                    const CompileOptions &options,
                    amd_comgr_data_set_t output) {
  amd_comgr_data_set_t input;
  check(amd_comgr_create_data_set(&input), "amd_comgr_create_data_set");
  DataSetGuard inputGuard(&input);
  // COMGR selects the input type from the name's extension; textual IR needs
  // .ll.
  addData(input, AMD_COMGR_DATA_KIND_BC, llvmIR, "input.ll");

  // COMGR defaults to -O0, so the level is always passed explicitly.
  std::vector<std::string> args = {
      "-O" + std::to_string(static_cast<int32_t>(options.optLevel)),
      "-mcode-object-version=" +
          std::to_string(static_cast<int32_t>(options.codeObjectVersion))};
  // Extra options go last so callers can override the generated ones.
  args.insert(args.end(), options.extraOptions.begin(),
              options.extraOptions.end());
  std::vector<const char *> argPointers;
  for (const std::string &arg : args)
    argPointers.push_back(arg.c_str());

  amd_comgr_action_info_t info;
  check(amd_comgr_create_action_info(&info), "amd_comgr_create_action_info");
  ActionInfoGuard infoGuard(&info);
  check(amd_comgr_action_info_set_language(info, AMD_COMGR_LANGUAGE_LLVM_IR),
        "amd_comgr_action_info_set_language");
  // COMGR would otherwise link the OpenCL device libraries into the module.
  check(amd_comgr_action_info_set_device_lib_linking(info, false),
        "amd_comgr_action_info_set_device_lib_linking");
  check(amd_comgr_action_info_set_option_list(info, argPointers.data(),
                                              argPointers.size()),
        "amd_comgr_action_info_set_option_list");
  runAction(kind, actionName, info, target, input, output);
}

// Code generation options do not apply to linking.
static std::vector<uint8_t> linkExecutable(amd_comgr_data_set_t relocatables,
                                           std::string_view target) {
  amd_comgr_action_info_t info;
  check(amd_comgr_create_action_info(&info), "amd_comgr_create_action_info");
  ActionInfoGuard infoGuard(&info);
  amd_comgr_data_set_t output;
  check(amd_comgr_create_data_set(&output), "amd_comgr_create_data_set");
  DataSetGuard outputGuard(&output);
  runAction(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
            "link relocatable to executable", info, target, relocatables,
            output);
  std::vector<uint8_t> codeObject;
  check(readData(output, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, codeObject),
        "reading the linked code object");
  return codeObject;
}

std::vector<uint8_t>
rocm::tensilelite::compileToASM(std::string_view llvmIR,
                                std::string_view target,
                                const CompileOptions &options) {
  amd_comgr_data_set_t output;
  check(amd_comgr_create_data_set(&output), "amd_comgr_create_data_set");
  DataSetGuard outputGuard(&output);
  codegen(AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY, "codegen bc to assembly",
          llvmIR, target, options, output);
  std::vector<uint8_t> assembly;
  check(readData(output, AMD_COMGR_DATA_KIND_SOURCE, 0, assembly),
        "reading the generated assembly");
  return assembly;
}

std::vector<uint8_t>
rocm::tensilelite::compileToHSACO(std::string_view llvmIR,
                                  std::string_view target,
                                  const CompileOptions &options) {
  amd_comgr_data_set_t relocatables;
  check(amd_comgr_create_data_set(&relocatables), "amd_comgr_create_data_set");
  DataSetGuard relocatablesGuard(&relocatables);
  codegen(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
          "codegen bc to relocatable", llvmIR, target, options, relocatables);
  return linkExecutable(relocatables, target);
}

std::vector<uint8_t>
rocm::tensilelite::assembleToHSACO(std::string_view assembly,
                                   std::string_view target) {
  amd_comgr_data_set_t input;
  check(amd_comgr_create_data_set(&input), "amd_comgr_create_data_set");
  DataSetGuard inputGuard(&input);
  addData(input, AMD_COMGR_DATA_KIND_SOURCE, assembly, "input.s");
  amd_comgr_action_info_t info;
  check(amd_comgr_create_action_info(&info), "amd_comgr_create_action_info");
  ActionInfoGuard infoGuard(&info);
  amd_comgr_data_set_t relocatables;
  check(amd_comgr_create_data_set(&relocatables), "amd_comgr_create_data_set");
  DataSetGuard relocatablesGuard(&relocatables);
  runAction(AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE,
            "assemble source to relocatable", info, target, input,
            relocatables);
  return linkExecutable(relocatables, target);
}
