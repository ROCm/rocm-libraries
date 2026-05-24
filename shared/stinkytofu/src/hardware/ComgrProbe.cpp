// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "stinkytofu/hardware/ComgrProbe.hpp"

#include "stinkytofu/Config/Config.h"

#ifdef STINKYTOFU_HAS_COMGR
#include <amd_comgr/amd_comgr.h>

#include <vector>
#endif

namespace stinkytofu {

#ifdef STINKYTOFU_HAS_COMGR

namespace {

struct ComgrData {
    amd_comgr_data_t handle{};
    bool valid = false;

    ComgrData() = default;
    ~ComgrData() {
        if (valid) amd_comgr_release_data(handle);
    }
    ComgrData(const ComgrData&) = delete;
    ComgrData& operator=(const ComgrData&) = delete;

    bool create() {
        valid = (amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &handle) ==
                 AMD_COMGR_STATUS_SUCCESS);
        return valid;
    }
};

struct ComgrDataSet {
    amd_comgr_data_set_t handle{};
    bool valid = false;

    ComgrDataSet() = default;
    ~ComgrDataSet() {
        if (valid) amd_comgr_destroy_data_set(handle);
    }
    ComgrDataSet(const ComgrDataSet&) = delete;
    ComgrDataSet& operator=(const ComgrDataSet&) = delete;

    bool create() {
        valid = (amd_comgr_create_data_set(&handle) == AMD_COMGR_STATUS_SUCCESS);
        return valid;
    }
};

struct ComgrActionInfo {
    amd_comgr_action_info_t handle{};
    bool valid = false;

    ComgrActionInfo() = default;
    ~ComgrActionInfo() {
        if (valid) amd_comgr_destroy_action_info(handle);
    }
    ComgrActionInfo(const ComgrActionInfo&) = delete;
    ComgrActionInfo& operator=(const ComgrActionInfo&) = delete;

    bool create() {
        valid = (amd_comgr_create_action_info(&handle) == AMD_COMGR_STATUS_SUCCESS);
        return valid;
    }
};

}  // namespace

bool tryAssembleWithComgr(const std::string& asmString, const std::string& isaName,
                          uint32_t wavefrontSize) {
    ComgrData data;
    if (!data.create()) return false;

    if (!asmString.empty()) {
        if (amd_comgr_set_data(data.handle, asmString.size(), asmString.c_str()) !=
            AMD_COMGR_STATUS_SUCCESS)
            return false;
    }
    if (amd_comgr_set_data_name(data.handle, "probe.s") != AMD_COMGR_STATUS_SUCCESS) return false;

    ComgrDataSet inputSet, outputSet;
    if (!inputSet.create() || !outputSet.create()) return false;
    if (amd_comgr_data_set_add(inputSet.handle, data.handle) != AMD_COMGR_STATUS_SUCCESS)
        return false;

    ComgrActionInfo actionInfo;
    if (!actionInfo.create()) return false;
    if (amd_comgr_action_info_set_language(actionInfo.handle, AMD_COMGR_LANGUAGE_NONE) !=
        AMD_COMGR_STATUS_SUCCESS)
        return false;
    if (amd_comgr_action_info_set_isa_name(actionInfo.handle, isaName.c_str()) !=
        AMD_COMGR_STATUS_SUCCESS)
        return false;

    std::vector<const char*> options;
    if (wavefrontSize == 64) options.push_back("-mwavefrontsize64");
    if (!options.empty()) {
        if (amd_comgr_action_info_set_option_list(actionInfo.handle, options.data(),
                                                  options.size()) != AMD_COMGR_STATUS_SUCCESS)
            return false;
    }

    auto status = amd_comgr_do_action(AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE,
                                      actionInfo.handle, inputSet.handle, outputSet.handle);
    return status == AMD_COMGR_STATUS_SUCCESS;
}

bool hasComgrSupport() {
    return true;
}

#else  // !STINKYTOFU_HAS_COMGR

bool tryAssembleWithComgr(const std::string& /*asmString*/, const std::string& /*isaName*/,
                          uint32_t /*wavefrontSize*/) {
    return false;
}

bool hasComgrSupport() {
    return false;
}

#endif  // STINKYTOFU_HAS_COMGR

}  // namespace stinkytofu
