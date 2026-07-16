// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "dispatcher/AotCatalog.hpp"
#include "dispatcher/AotInstance.hpp"

namespace rocke_client::dispatcher
{
namespace
{

namespace fs = std::filesystem;

// A well-formed single-entry bundle manifest for gfx942 matching the parser
// contract in AotCatalog.cpp (compile_spec, selection, launch, args_signature).
const char* kValidManifest = R"JSON(
{
  "arch": "gfx942",
  "kpack": "rocke_client_gfx942.kpack",
  "entries": [
    {
      "name": "sdpa_fwd_d64",
      "op": "sdpa_fwd",
      "family": "fmha_fwd_mfma",
      "compile_spec": {
        "dtype": "fp16",
        "canonical_layout": "BSHD",
        "seqlen_q": 64,
        "seqlen_k": 64,
        "num_query_heads": 4,
        "num_kv_heads": 4,
        "head_size": 64,
        "block_size_q": 64,
        "block_size_k": 64,
        "mask_mode": "none"
      },
      "selection": {
        "batch": { "min": 1, "max": 256 },
        "attribute_constraints": {}
      },
      "cache_key": "cache-abc",
      "toc_key": "sdpa_fwd/fmha_fwd_mfma/sdpa_fwd_d64",
      "symbol": "sdpa_fwd_kernel",
      "launch": {
        "grid_formula": {
          "x": { "ceil_div": ["seqlen_q", "block_size_q"] },
          "y": "num_query_heads",
          "z": "batch"
        },
        "block": [256, 1, 1],
        "shared_mem_bytes": 0
      },
      "args_signature": [
        { "name": "Q", "type": "ptr", "kind": "pointer", "size_bytes": 8, "alignment": 8 },
        { "name": "seqlen_q", "type": "i32", "kind": "scalar", "size_bytes": 4 }
      ]
    }
  ]
}
)JSON";

void writeFile(const fs::path& path, const std::string& content)
{
    fs::create_directories(path.parent_path());
    std::ofstream stream(path);
    stream << content;
}

// Creates and removes a unique temp directory tree for a single test.
class TempTree
{
public:
    explicit TempTree(const std::string& name)
        : _root(fs::temp_directory_path() / ("rocke_aot_loader_" + name))
    {
        fs::remove_all(_root);
    }

    ~TempTree()
    {
        std::error_code ec;
        fs::remove_all(_root, ec);
    }

    TempTree(const TempTree&) = delete;
    TempTree& operator=(const TempTree&) = delete;

    const fs::path& root() const
    {
        return _root;
    }

private:
    fs::path _root;
};

TEST(TestAotManifestLoader, ParsesValidBundleManifest)
{
    const TempTree tree("valid");
    writeFile(tree.root() / "gfx942" / "rocke_client_gfx942.json", kValidManifest);
    writeFile(tree.root() / "gfx942" / "rocke_client_gfx942.kpack", "stub");

    const auto instances = loadManifestsFromDirectory(tree.root());
    ASSERT_EQ(instances.size(), 1u);

    const AotInstance& instance = instances.front();
    EXPECT_EQ(instance.name, "sdpa_fwd_d64");
    EXPECT_EQ(instance.op, "sdpa_fwd");
    EXPECT_EQ(instance.family, "fmha_fwd_mfma");
    EXPECT_EQ(instance.arch, "gfx942");
    EXPECT_EQ(instance.compileSpec.headSize, 64);
    EXPECT_EQ(instance.compileSpec.maskMode, "none");
    EXPECT_EQ(instance.batch.min, 1);
    EXPECT_EQ(instance.batch.max, 256);
    EXPECT_EQ(instance.runtime.cacheKey, "cache-abc");
    EXPECT_EQ(instance.runtime.symbol, "sdpa_fwd_kernel");
    EXPECT_TRUE(instance.runtime.kpackPath.ends_with("rocke_client_gfx942.kpack"));

    const auto& launch = instance.runtime.launch;
    EXPECT_EQ(launch.block[0], 256u);
    EXPECT_EQ(launch.grid.x.kind, GridAxis::Kind::CEIL_DIV);
    EXPECT_EQ(launch.grid.z.kind, GridAxis::Kind::VALUE);
    ASSERT_TRUE(launch.grid.z.value.symbol.has_value());
    EXPECT_EQ(*launch.grid.z.value.symbol, "batch");
    ASSERT_EQ(launch.argsSignature.size(), 2u);
    EXPECT_EQ(launch.argsSignature[0].kind, ArgKind::POINTER);
    EXPECT_EQ(launch.argsSignature[1].kind, ArgKind::SCALAR);
    ASSERT_TRUE(launch.argsSignature[1].scalarType.has_value());
    EXPECT_EQ(*launch.argsSignature[1].scalarType, ScalarType::I32);
    // size/alignment are derived from kind+dtype (scalar i32 -> 4 bytes).
    EXPECT_EQ(argSizeBytes(launch.argsSignature[1]), 4u);
}

TEST(TestAotManifestLoader, SkipsMalformedManifestButKeepsValidOnes)
{
    const TempTree tree("skip");
    writeFile(tree.root() / "gfx942" / "rocke_client_gfx942.json", kValidManifest);
    writeFile(tree.root() / "gfx942" / "rocke_client_gfx942.kpack", "stub");
    // A second bundle that fails to parse (no "arch") must not drop the first.
    writeFile(tree.root() / "gfx950" / "rocke_client_gfx950.json", R"({ "not": "valid" })");

    const auto instances = loadManifestsFromDirectory(tree.root());
    ASSERT_EQ(instances.size(), 1u);
    EXPECT_EQ(instances.front().arch, "gfx942");
}

TEST(TestAotManifestLoader, MissingKpackFileSkipsBundle)
{
    const TempTree tree("nokpack");
    // Manifest references a kpack that does not exist on disk.
    writeFile(tree.root() / "gfx942" / "rocke_client_gfx942.json", kValidManifest);

    const auto instances = loadManifestsFromDirectory(tree.root());
    EXPECT_TRUE(instances.empty());
}

TEST(TestAotManifestLoader, MissingDirectoryYieldsEmpty)
{
    const TempTree tree("missing");
    const auto instances = loadManifestsFromDirectory(tree.root() / "does_not_exist");
    EXPECT_TRUE(instances.empty());
}

TEST(TestAotManifestLoader, RejectsUnknownArgKind)
{
    const TempTree tree("badkind");
    std::string manifest = kValidManifest;
    const std::string needle = R"("kind": "pointer")";
    const auto pos = manifest.find(needle);
    ASSERT_NE(pos, std::string::npos);
    manifest.replace(pos, needle.size(), R"("kind": "matrix")");
    writeFile(tree.root() / "gfx942" / "rocke_client_gfx942.json", manifest);
    writeFile(tree.root() / "gfx942" / "rocke_client_gfx942.kpack", "stub");

    // An unknown args_signature kind fails parsing, so the bundle is dropped.
    const auto instances = loadManifestsFromDirectory(tree.root());
    EXPECT_TRUE(instances.empty());
}

TEST(TestAotManifestLoader, RejectsUnknownScalarType)
{
    const TempTree tree("badtype");
    std::string manifest = kValidManifest;
    const std::string needle = R"("type": "i32")";
    const auto pos = manifest.find(needle);
    ASSERT_NE(pos, std::string::npos);
    manifest.replace(pos, needle.size(), R"("type": "f16")");
    writeFile(tree.root() / "gfx942" / "rocke_client_gfx942.json", manifest);
    writeFile(tree.root() / "gfx942" / "rocke_client_gfx942.kpack", "stub");

    // An unknown scalar dtype fails parsing, so the bundle is dropped.
    const auto instances = loadManifestsFromDirectory(tree.root());
    EXPECT_TRUE(instances.empty());
}

TEST(TestAotManifestLoader, RejectsSizeBytesMismatch)
{
    const TempTree tree("badsize");
    std::string manifest = kValidManifest;
    // The scalar i32 argument declares 4 bytes; 8 contradicts the dtype width.
    const std::string needle = R"("size_bytes": 4)";
    const auto pos = manifest.find(needle);
    ASSERT_NE(pos, std::string::npos);
    manifest.replace(pos, needle.size(), R"("size_bytes": 8)");
    writeFile(tree.root() / "gfx942" / "rocke_client_gfx942.json", manifest);
    writeFile(tree.root() / "gfx942" / "rocke_client_gfx942.kpack", "stub");

    // A size_bytes that disagrees with the dtype width fails parsing.
    const auto instances = loadManifestsFromDirectory(tree.root());
    EXPECT_TRUE(instances.empty());
}

} // namespace
} // namespace rocke_client::dispatcher
