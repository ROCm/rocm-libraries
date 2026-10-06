// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-source-bundle.hpp"
#include "test_helpers.hpp"
#include <functional>
#include <iostream>

namespace a  = hipblaslt_jit::source_bundle;
namespace fs = std::filesystem;

using hipblaslt_jit_test::require;
using hipblaslt_jit_test::writeFile;

void reject(const std::function<void()>& f)
{
    hipblaslt_jit_test::reject(f, "Malformed artifact was accepted");
}
std::vector<std::string> names(const std::vector<a::SourceFile>& files)
{
    std::vector<std::string> result;
    for(const auto& file : files)
        result.push_back(file.name);
    return result;
}

int main(int argc, char** argv)
try
{
    require(argc == 2, "Usage: hipblaslt-jit-source-bundle-test SCRATCH");
    const auto root = fs::u8path(argv[1]);
    fs::remove_all(root);
    fs::create_directories(root);

    const auto        library = root / fs::u8path("library π.dat");
    const std::string payload = std::string(200000, 'x') + "serialized solution bytes";
    writeFile(library, payload);
    const auto bytes = a::readArtifact(library);
    require(std::string(bytes.begin(), bytes.end()) == payload, "Artifact bytes changed");
    writeFile(library, "");
    reject([&] { a::readArtifact(library); });
    const auto raw = root / "raw.dat";
    writeFile(raw, payload);
    std::cout << "PASS native Unicode paths and complete, non-empty artifacts\n";

    reject([&] { a::artifact(root, "../outside.co"); });
    reject([&] { a::artifact(root, fs::absolute(raw).u8string()); });
    reject([&] { a::artifact(root, "absent.co"); });
    require(a::artifact(root, "raw.dat") == fs::canonical(raw), "Valid artifact path rejected");
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
    for(const auto* name : {"b.s", "a.s", "scale.hip", "add.cpp", "kernel.h", "types.h"})
        writeFile(bundle / "sources" / name, std::string("// ") + name + "\n");
    const auto sources = a::readSourceBundle(bundle);
    require(std::string(sources.library.begin(), sources.library.end()) == payload,
            "Source bundle library changed");
    require(names(sources.assembly) == std::vector<std::string>({"a.s", "b.s"}),
            "Assembly is not every sources/*.s by name");
    require(names(sources.hip) == std::vector<std::string>({"add.cpp", "scale.hip"}),
            "HIP sources are not every sources/*.hip and *.cpp by name");
    require(names(sources.headers) == std::vector<std::string>({"kernel.h", "types.h"}),
            "Headers are not the other sources/ files by name");
    require(std::string(sources.assembly[0].bytes.begin(), sources.assembly[0].bytes.end())
                == "// a.s\n",
            "Source bytes changed");
    fs::rename(bundle / "sources/a.s", root / "a.s");
    fs::rename(bundle / "sources/b.s", root / "b.s");
    require(a::readSourceBundle(bundle).assembly.empty(), "A bundle of HIP sources was rejected");
    fs::rename(root / "a.s", bundle / "sources/a.s");
    fs::rename(root / "b.s", bundle / "sources/b.s");
    fs::remove(bundle / "sources/scale.hip");
    fs::remove(bundle / "sources/add.cpp");
    require(a::readSourceBundle(bundle).hip.empty(), "A bundle of assembly was rejected");
    auto damage = [&](const std::function<void()>& change, const std::function<void()>& undo) {
        change();
        reject([&] { a::readSourceBundle(bundle); });
        undo();
        a::readSourceBundle(bundle);
    };
    damage([&] { fs::rename(bundle / "sources", root / "moved"); },
           [&] { fs::rename(root / "moved", bundle / "sources"); });
    damage(
        [&] {
            fs::rename(bundle / "library/TensileLibrary.dat",
                       bundle / "library/TensileLibrary.yaml");
        },
        [&] {
            fs::rename(bundle / "library/TensileLibrary.yaml",
                       bundle / "library/TensileLibrary.dat");
        });
    damage([&] { fs::rename(bundle / "library", root / "moved"); },
           [&] { fs::rename(root / "moved", bundle / "library"); });
    damage([&] { writeFile(bundle / "library/TensileLibrary.dat", ""); },
           [&] { writeFile(bundle / "library/TensileLibrary.dat", payload); });
    damage(
        [&] {
            fs::rename(bundle / "sources/a.s", root / "a.s");
            fs::rename(bundle / "sources/b.s", root / "b.s");
        },
        [&] {
            fs::rename(root / "a.s", bundle / "sources/a.s");
            fs::rename(root / "b.s", bundle / "sources/b.s");
        });
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
    return 0;
}
catch(const std::exception& error)
{
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
