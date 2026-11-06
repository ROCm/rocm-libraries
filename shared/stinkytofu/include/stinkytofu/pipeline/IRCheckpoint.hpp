/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */
#pragma once

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/PassManager.hpp"

namespace stinkytofu {
class StinkyAsmModule;

/// Where a checkpoint sits: after the `occurrence`-th run of `passName` in the pass managers
/// labelled `label`. Spelled `<label>:<passName>[#n]`.
struct IRCheckpointSpec {
    std::string label;
    std::string passName;
    unsigned occurrence = 1;

    std::string str() const;
};

/// Null on a malformed spec, with the reason in \p error.
STINKYTOFU_EXPORT std::optional<IRCheckpointSpec> parseIRCheckpointSpec(
    std::string_view text, std::string* error = nullptr);

/// A Function printed the way a checkpoint bundle stores it: AsmPrinter's `st.func` form, one
/// line per printed IR node. The checkpoint id of a node is its index in `nodes`.
///
/// LABEL instructions print nothing (the block header names them), so they get no id and the
/// loader keeps them where they are.
struct IRCheckpointRendering {
    std::string text;
    std::vector<IRBase*> nodes;
    std::vector<size_t> blockOf;
    /// The line of each node without its id or indentation.
    std::vector<std::string> nodeText;
    /// Block header names in order, without the `^`.
    std::vector<std::string> blockLabels;
};

/// With \p withIds every node line carries `mod.ckpt = { id = N }`; without, the text is what
/// the base fingerprint is taken over.
STINKYTOFU_EXPORT IRCheckpointRendering renderCheckpointRegion(const Function& function,
                                                               bool withIds);

/// Remove the `mod.ckpt` id from one line. Returns the id, or -1 when the line has none.
STINKYTOFU_EXPORT long takeCheckpointId(std::string& line);

/// Remove every `mod.ckpt` id from a bundle, so the text parses as plain Stinky IR.
STINKYTOFU_EXPORT std::string stripCheckpointIds(std::string_view text);

STINKYTOFU_EXPORT std::string sha256Hex(std::string_view data);

struct IRCheckpointConfig {
    IRCheckpointSpec spec;
    std::string dumpDir;
    std::string loadDir;
    /// Names the bundle directory; Tensile sets it per kernel through setOutputName().
    std::string kernelName;
    std::array<int, 3> arch{};
    /// Group endpoints come from here when set.
    const StinkyAsmModule* module = nullptr;
};

/// One kernel's checkpoint: dumps or loads the IR at the spec's pass boundary.
///
/// install() hooks it into every pass manager carrying the spec's label. The pass managers
/// count runs of the named pass together, and after the matching one the session's
/// IRCheckpointPass runs and every analysis is invalidated. A spec that never matched is
/// reported when the session is destroyed, after the pipeline ran: a warning for a dump, an
/// error for a load, which must not quietly build the unedited kernel.
class STINKYTOFU_EXPORT IRCheckpointSession
    : public PassInsertionPoint,
      public std::enable_shared_from_this<IRCheckpointSession> {
   public:
    /// Null when \p module requests no checkpoint. A malformed spec, both directories, or a
    /// directory without a spec is fatal.
    static std::shared_ptr<IRCheckpointSession> create(const StinkyAsmModule& module);

    /// Same, for a driver without a module (stinkytofu-opt running named passes).
    static std::shared_ptr<IRCheckpointSession> create(const std::string& spec,
                                                       const std::string& dumpDir,
                                                       const std::string& loadDir,
                                                       const std::string& kernelName,
                                                       const std::array<int, 3>& arch);

    explicit IRCheckpointSession(IRCheckpointConfig config);
    ~IRCheckpointSession() override;

    /// Hook into \p pm when \p label is the spec's label.
    void install(PassManager& pm, const std::string& label);

    Pass* passAfter(const std::string& passName, Function& function) override;

    const IRCheckpointConfig& getConfig() const {
        return config;
    }

   private:
    IRCheckpointConfig config;
    std::unique_ptr<Pass> checkpointPass;
    std::vector<std::string> labelsOffered;
    std::vector<std::string> passesSeen;
    unsigned matches = 0;
    bool installed = false;
    bool ran = false;
    bool fired = false;
};
}  // namespace stinkytofu
