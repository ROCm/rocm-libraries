// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Stage-3 resolution at runtime: an ArtifactStore indexes a shipped per-arch
// kernel bundle (offline-produced) by cache_key, and materializes a Kernel from
// either a prebuilt HSACO (path a) or LLVM IR to be comgr-compiled (path b).
//
// Bundle layout (flat): for each kernel <id> (== cache_key or kernel_name):
//   <id>.manifest.json   (required)
//   <id>.hsaco           (optional; if present -> Fast Mode)
//   <id>.ll              (optional; if present -> JIT-fallback source)
// `<id>` is derived from the manifest; the hsaco/ll basenames are resolved from
// the manifest's `kernel_name`/`hsaco` fields, so legacy single-kernel
// write_artifact() output (manifest.json + <name>.hsaco + <name>.ll) also loads.
#pragma once

#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "ck_dsl_runtime/kernel.hpp"
#include "ck_dsl_runtime/manifest.hpp"

namespace ck_dsl {

class ArtifactStore {
   public:
    struct Entry {
        Manifest manifest;
        std::string hsaco_path;  // empty if absent
        std::string ll_path;     // empty if absent
        bool has_hsaco() const {
            return !hsaco_path.empty();
        }
        bool has_ll() const {
            return !ll_path.empty();
        }
    };

    // Index every manifest under `dir` (recursively). Returns count added.
    size_t add_bundle(const std::string& dir) {
        namespace fs = std::filesystem;
        size_t added = 0;
        std::error_code ec;
        if (!fs::exists(dir, ec)) return 0;
        for (auto& de : fs::recursive_directory_iterator(dir, ec)) {
            if (ec) break;
            if (!de.is_regular_file()) continue;
            const auto path = de.path();
            const auto name = path.filename().string();
            bool is_manifest =
                (name == "manifest.json") ||
                (name.size() > 14 && name.rfind(".manifest.json") == name.size() - 14);
            if (!is_manifest) continue;
            Manifest m;
            try {
                m = Manifest::parse(read_text(path.string()));
            } catch (...) {
                continue;  // skip malformed manifests
            }
            Entry e;
            e.manifest = m;
            const auto pdir = path.parent_path();
            // hsaco: prefer manifest.hsaco basename, else <kernel_name>.hsaco
            std::string hbase = !m.hsaco.empty() ? m.hsaco : (m.kernel_name + ".hsaco");
            auto hpath = pdir / hbase;
            if (fs::exists(hpath, ec)) e.hsaco_path = hpath.string();
            auto lpath = pdir / (m.kernel_name + ".ll");
            if (fs::exists(lpath, ec)) e.ll_path = lpath.string();
            entries_[m.id()] = std::move(e);
            ++added;
        }
        return added;
    }

    bool has(const std::string& id) const {
        return entries_.count(id) != 0;
    }
    const Entry& at(const std::string& id) const {
        return entries_.at(id);
    }
    const std::map<std::string, Entry>& entries() const {
        return entries_;
    }

    // Materialize a Kernel for `id`. Prefers prebuilt HSACO; otherwise sets up
    // lazy comgr compilation from .ll for `isa`.
    Kernel make_kernel(const std::string& id, const std::string& isa) const {
        const Entry& e = entries_.at(id);
        if (e.has_hsaco()) return Kernel::from_hsaco(read_bytes(e.hsaco_path), e.manifest);
        if (e.has_ll()) return Kernel::from_llvm_ir(read_text(e.ll_path), e.manifest, isa);
        throw std::runtime_error("artifact '" + id + "' has neither .hsaco nor .ll");
    }

    static std::string read_text(const std::string& p) {
        std::ifstream f(p, std::ios::binary);
        std::stringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }
    static std::vector<std::byte> read_bytes(const std::string& p) {
        std::ifstream f(p, std::ios::binary | std::ios::ate);
        std::streamsize n = f.tellg();
        // A failed open or tellg() returns -1; without this guard the size would
        // be reinterpreted as an enormous (or negative) allocation.
        if (!f || n < 0) throw std::runtime_error("artifact_store: cannot read '" + p + "'");
        f.seekg(0);
        std::vector<std::byte> b(n);
        f.read(reinterpret_cast<char*>(b.data()), n);
        return b;
    }

   private:
    std::map<std::string, Entry> entries_;
};

}  // namespace ck_dsl
