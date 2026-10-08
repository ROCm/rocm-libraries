// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-source-bundle.hpp"
#include "test_helpers.hpp"
#include <catch2/catch_test_macros.hpp>
#include <functional>
#include <iostream>

namespace a  = hipblaslt_jit::source_bundle;
namespace fs = std::filesystem;

using hipblaslt_jit_test::writeFile;

void reject(const std::function<void()>& f)
{
    hipblaslt_jit_test::reject(f, "Malformed artifact was accepted");
}
std::vector<std::string> names(const std::vector<hipblaslt_jit::Artifact>& files)
{
    std::vector<std::string> result;
    for(const auto& file : files)
        result.push_back(file.name);
    return result;
}

TEST_CASE("the source bundle reader accepts only a contained directory", "[jit-cpu]")
{
    const auto root = fs::u8path(HIPBLASLT_JIT_SCRATCH);
    fs::remove_all(root);
    fs::create_directories(root);

    const auto        library = root / fs::u8path("library π.dat");
    const std::string payload = std::string(200000, 'x') + "serialized solution bytes";
    writeFile(library, payload);
    const auto bytes = a::readArtifact(library);
    {
        INFO(("Artifact bytes changed"));
        REQUIRE((std::string(bytes.begin(), bytes.end()) == payload));
    }
    writeFile(library, "");
    reject([&] { a::readArtifact(library); });
    const auto raw = root / "raw.dat";
    writeFile(raw, payload);
    std::cout << "PASS native Unicode paths and complete, non-empty artifacts\n";

    reject([&] { a::artifact(root, "../outside.co"); });
    reject([&] { a::artifact(root, fs::absolute(raw).u8string()); });
    reject([&] { a::artifact(root, "absent.co"); });
    {
        INFO(("Valid artifact path rejected"));
        REQUIRE((a::artifact(root, "raw.dat") == fs::canonical(raw)));
    }
#ifdef _WIN32
    reject([&] { a::artifact(root, "C:raw.dat", false); });
    reject([&] { a::artifact(root, "\\raw.dat", false); });
#endif
#ifndef _WIN32
    fs::create_symlink(fs::absolute(root.parent_path()), root / "escape");
    reject([&] { a::artifact(root, "escape/outside.co", false); });
#endif
    std::cout << "PASS relative paths, missing artifacts and containment\n";

    const auto bundle = root / fs::u8path("bundle π");
    fs::create_directories(bundle / "library");
    fs::create_directories(bundle / "sources");
    writeFile(bundle / "library/TensileLibrary.dat", payload);
    for(const auto* name : {"a.s", "scale.hip", "add.cpp", "kernel.h", "types.h"})
        writeFile(bundle / "sources" / name, std::string("// ") + name + "\n");
    const auto sources = a::readSourceBundle(bundle);
    {
        INFO(("The main kernel is not the assembly source"));
        REQUIRE((sources.mainKernel.name == "a.s"
                 && sources.mainKernel.kind == hipblaslt_jit::Artifact::Kind::Assembly));
    }
    {
        INFO(("HIP sources are not helpers by name"));
        REQUIRE((names(sources.helpers) == std::vector<std::string>({"add.cpp", "scale.hip"})));
    }
    {
        INFO(("Headers are not the bundle includes"));
        REQUIRE((names(sources.includes) == std::vector<std::string>({"kernel.h", "types.h"})));
    }
    {
        INFO(("Source bytes changed"));
        REQUIRE((std::string(sources.mainKernel.bytes.begin(), sources.mainKernel.bytes.end())
                 == "// a.s\n"));
    }
    writeFile(bundle / "sources/b.s", "// b.s\n");
    reject([&] { a::readSourceBundle(bundle); });
    fs::remove(bundle / "sources/b.s");
    fs::rename(bundle / "sources/a.s", root / "a.s");
    {
        INFO(("A bundle of HIP sources was rejected"));
        REQUIRE((a::readSourceBundle(bundle).mainKernel.kind == hipblaslt_jit::Artifact::Kind::Hip
                 && a::readSourceBundle(bundle).mainKernel.name == "add.cpp"
                 && names(a::readSourceBundle(bundle).helpers)
                        == std::vector<std::string>{"scale.hip"}));
    }
    fs::rename(root / "a.s", bundle / "sources/a.s");
    fs::remove(bundle / "sources/scale.hip");
    fs::remove(bundle / "sources/add.cpp");
    {
        INFO(("A bundle of assembly was rejected"));
        REQUIRE((a::readSourceBundle(bundle).helpers.empty()
                 && a::readSourceBundle(bundle).mainKernel.kind
                        == hipblaslt_jit::Artifact::Kind::Assembly));
    }
    auto damage = [&](const std::function<void()>& change, const std::function<void()>& undo) {
        change();
        reject([&] { a::readSourceBundle(bundle); });
        undo();
        a::readSourceBundle(bundle);
    };
    damage([&] { fs::rename(bundle / "sources", root / "moved"); },
           [&] { fs::rename(root / "moved", bundle / "sources"); });
    damage([&] { fs::rename(bundle / "sources/a.s", root / "a.s"); },
           [&] { fs::rename(root / "a.s", bundle / "sources/a.s"); });
    damage([&] { fs::create_directory(bundle / "sources/nested"); },
           [&] { fs::remove(bundle / "sources/nested"); });
    damage([&] { writeFile(bundle / "sources/empty.h", ""); },
           [&] { fs::remove(bundle / "sources/empty.h"); });
    damage(
        [&] {
            writeFile(bundle / "sources/large.h", "x");
            fs::resize_file(bundle / "sources/large.h", 64 * 1024 * 1024 + 1);
        },
        [&] { fs::remove(bundle / "sources/large.h"); });
    damage(
        [&] {
            for(int i = 0; i < 1024; ++i)
                writeFile(bundle / "sources" / ("extra" + std::to_string(i) + ".h"), "x");
        },
        [&] {
            for(int i = 0; i < 1024; ++i)
                fs::remove(bundle / "sources" / ("extra" + std::to_string(i) + ".h"));
        });
#ifndef _WIN32
    writeFile(root / "outside.h", "outside");
    damage(
        [&] { fs::create_symlink(fs::absolute(root / "outside.h"), bundle / "sources/outside.h"); },
        [&] { fs::remove(bundle / "sources/outside.h"); });
#endif
    std::cout << "PASS source bundle convention, containment, file and count caps\n";
}
