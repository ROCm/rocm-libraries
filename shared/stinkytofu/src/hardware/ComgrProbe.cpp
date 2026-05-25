// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "stinkytofu/hardware/ComgrProbe.hpp"

#include "stinkytofu/Config/Config.h"

#ifdef STINKYTOFU_HAS_COMGR
#include <amd_comgr/amd_comgr.h>

#include <vector>

#ifdef _WIN32
#include <Windows.h>
#include <delayimp.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

std::string getRocmBinDir() {
    char* rocmPath = nullptr;
    size_t len = 0;
    _dupenv_s(&rocmPath, &len, "ROCM_PATH");
    if (!rocmPath) return {};
    std::string binDir = std::string(rocmPath) + "\\bin";
    free(rocmPath);
    return binDir;
}

FARPROC WINAPI delayLoadHook(unsigned dliNotify, PDelayLoadInfo pdli) {
    if (dliNotify != dliNotePreLoadLibrary) return nullptr;

    std::string binDir = getRocmBinDir();
    if (binDir.empty()) return nullptr;

    std::string fullPath = binDir + "\\" + pdli->szDll;
    HMODULE mod =
        LoadLibraryExA(fullPath.c_str(), nullptr,
                       LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    return reinterpret_cast<FARPROC>(mod);
}

}  // namespace

extern "C" const PfnDliHook __pfnDliNotifyHook2 = delayLoadHook;

bool isRocmPathSet() {
    static int cached = -1;
    if (cached >= 0) return cached;
    std::string binDir = getRocmBinDir();
    if (binDir.empty()) {
        fprintf(stderr, "stinkytofu: ROCM_PATH not set, comgr probing disabled\n");
        cached = 0;
        return false;
    }
    cached = 1;
    return true;
}
#endif  // _WIN32

#endif  // STINKYTOFU_HAS_COMGR

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
#ifdef _WIN32
    if (!isRocmPathSet()) return false;
#endif
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
#ifdef _WIN32
    return isRocmPathSet();
#else
    return true;
#endif
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
