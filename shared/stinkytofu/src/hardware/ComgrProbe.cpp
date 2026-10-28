// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "stinkytofu/hardware/ComgrProbe.hpp"

#include "stinkytofu/Config/Config.h"

#ifdef STINKYTOFU_HAS_COMGR
#include <amd_comgr/amd_comgr.h>

#include <cstdio>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#ifdef _WIN32
#include <Windows.h>
#include <delayimp.h>

#include <cstdint>
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

/// The first log object comgr added to \p set, or empty.
std::string readLog(amd_comgr_data_set_t set) {
    size_t count = 0;
    if (amd_comgr_action_data_count(set, AMD_COMGR_DATA_KIND_LOG, &count) !=
            AMD_COMGR_STATUS_SUCCESS ||
        count == 0)
        return {};
    amd_comgr_data_t log{};
    if (amd_comgr_action_data_get_data(set, AMD_COMGR_DATA_KIND_LOG, 0, &log) !=
        AMD_COMGR_STATUS_SUCCESS)
        return {};
    std::string text;
    size_t size = 0;
    if (amd_comgr_get_data(log, &size, nullptr) == AMD_COMGR_STATUS_SUCCESS && size > 0) {
        text.resize(size);
        if (amd_comgr_get_data(log, &size, text.data()) != AMD_COMGR_STATUS_SUCCESS) text.clear();
    }
    amd_comgr_release_data(log);
    return text;
}

int duplicateFd(int fd) {
#ifdef _WIN32
    return _dup(fd);
#else
    return dup(fd);
#endif
}

int replaceFd(int from, int to) {
#ifdef _WIN32
    return _dup2(from, to);
#else
    return dup2(from, to);
#endif
}

void closeFd(int fd) {
#ifdef _WIN32
    _close(fd);
#else
    close(fd);
#endif
}

int fileFd(std::FILE* file) {
#ifdef _WIN32
    return _fileno(file);
#else
    return fileno(file);
#endif
}

/// While active, points stderr (file descriptor 2) at a temporary file.
/// comgr prints some assembler errors, such as an undefined label, there
/// instead of in the action log. stderr belongs to the whole process, so
/// another thread's output during the capture lands in it too.
class StderrCapture {
   public:
    explicit StderrCapture(bool active) {
        if (!active) return;
        std::fflush(stderr);
        file_ = std::tmpfile();
        if (file_ == nullptr) return;
        saved_ = duplicateFd(2);
        if (saved_ >= 0 && replaceFd(fileFd(file_), 2) >= 0) return;
        if (saved_ >= 0) closeFd(saved_);
        saved_ = -1;
        std::fclose(file_);
        file_ = nullptr;
    }

    ~StderrCapture() {
        restore();
        if (file_ != nullptr) std::fclose(file_);
    }

    StderrCapture(const StderrCapture&) = delete;
    StderrCapture& operator=(const StderrCapture&) = delete;

    /// Put stderr back and return what was written to it.
    std::string finish() {
        restore();
        if (file_ == nullptr) return {};
        std::string text;
        std::rewind(file_);
        char buffer[4096];
        size_t count = 0;
        while ((count = std::fread(buffer, 1, sizeof(buffer), file_)) > 0)
            text.append(buffer, count);
        return text;
    }

   private:
    void restore() {
        if (saved_ < 0) return;
        std::fflush(stderr);
        replaceFd(saved_, 2);
        closeFd(saved_);
        saved_ = -1;
    }

    std::FILE* file_ = nullptr;
    int saved_ = -1;
};

/// Logging is off when \p log is null; the toolchain probes run many of these.
bool assemble(const std::string& asmString, const std::string& isaName, uint32_t wavefrontSize,
              std::string* log) {
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
    if (log != nullptr &&
        amd_comgr_action_info_set_logging(actionInfo.handle, true) != AMD_COMGR_STATUS_SUCCESS)
        return false;

    std::vector<const char*> options;
    if (wavefrontSize == 64) options.push_back("-mwavefrontsize64");
    if (!options.empty()) {
        if (amd_comgr_action_info_set_option_list(actionInfo.handle, options.data(),
                                                  options.size()) != AMD_COMGR_STATUS_SUCCESS)
            return false;
    }

    StderrCapture capture(log != nullptr);
    auto status = amd_comgr_do_action(AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE,
                                      actionInfo.handle, inputSet.handle, outputSet.handle);
    const std::string printed = capture.finish();
    if (log != nullptr) *log = readLog(outputSet.handle) + printed;
    return status == AMD_COMGR_STATUS_SUCCESS;
}

}  // namespace

bool tryAssembleWithComgr(const std::string& asmString, const std::string& isaName,
                          uint32_t wavefrontSize) {
    return assemble(asmString, isaName, wavefrontSize, nullptr);
}

bool assembleWithComgr(const std::string& asmString, const std::string& isaName,
                       uint32_t wavefrontSize, std::string& log) {
    log.clear();
    return assemble(asmString, isaName, wavefrontSize, &log);
}

bool hasComgrSupport() {
#ifdef _WIN32
    return isRocmPathSet();
#else
    return true;
#endif
}

bool comgrSupportsIsa(const std::string& isaName) {
#ifdef _WIN32
    if (!isRocmPathSet()) return false;
#endif
    size_t count = 0;
    if (amd_comgr_get_isa_count(&count) != AMD_COMGR_STATUS_SUCCESS) return false;
    for (size_t i = 0; i < count; ++i) {
        const char* name = nullptr;
        if (amd_comgr_get_isa_name(i, &name) != AMD_COMGR_STATUS_SUCCESS) continue;
        if (name && isaName == name) return true;
    }
    return false;
}

#else  // !STINKYTOFU_HAS_COMGR

bool tryAssembleWithComgr(const std::string& /*asmString*/, const std::string& /*isaName*/,
                          uint32_t /*wavefrontSize*/) {
    return false;
}

bool assembleWithComgr(const std::string& /*asmString*/, const std::string& /*isaName*/,
                       uint32_t /*wavefrontSize*/, std::string& log) {
    log.clear();
    return false;
}

bool hasComgrSupport() {
    return false;
}

bool comgrSupportsIsa(const std::string& /*isaName*/) {
    return false;
}

#endif  // STINKYTOFU_HAS_COMGR

}  // namespace stinkytofu
