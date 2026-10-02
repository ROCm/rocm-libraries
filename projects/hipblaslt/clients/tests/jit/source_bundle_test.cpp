// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-source-bundle.hpp"
#include <functional>
#include <iostream>

namespace a  = hipblaslt_jit::source_bundle;
namespace fs = std::filesystem;

void check(bool value, const char* message)
{
    if(!value)
        throw std::runtime_error(message);
}
void reject(const std::function<void()>& f)
{
    bool failed = false;
    try
    {
        f();
    }
    catch(const std::exception&)
    {
        failed = true;
    }
    check(failed, "Malformed artifact was accepted");
}
void write(const fs::path& path, const std::string& bytes)
{
    std::ofstream file(path, std::ios::binary);
    check(bool(file.write(bytes.data(), bytes.size())), "Cannot write fixture");
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
    check(argc == 2, "Usage: artifact-test NEW_DIRECTORY");
    const auto root = fs::u8path(argv[1]);
    check(fs::create_directory(root), "Fixture directory already exists");

    const auto        library = root / fs::u8path("library π.dat.zlib");
    const std::string payload = std::string(200000, 'x') + "serialized solution bytes";
    uLongf            size    = compressBound(payload.size());
    std::string       compressed(size, '\0');
    check(compress(reinterpret_cast<Bytef*>(compressed.data()),
                   &size,
                   reinterpret_cast<const Bytef*>(payload.data()),
                   payload.size())
              == Z_OK,
          "Cannot compress fixture");
    compressed.resize(size);
    write(library, compressed);
    const auto decoded = a::readLibrary(library);
    check(std::string(decoded.begin(), decoded.end()) == payload,
          "Library decompression changed bytes");
    write(library, compressed.substr(0, compressed.size() - 1));
    reject([&] { a::readLibrary(library); });
    write(library, compressed + "trailing");
    reject([&] { a::readLibrary(library); });
    write(library, "not zlib");
    reject([&] { a::readLibrary(library); });
    const auto raw = root / "raw.dat";
    write(raw, payload);
    check(a::readLibrary(raw) == decoded, "Raw library bytes changed");
    std::cout << "PASS native Unicode paths and raw/compressed library validation\n";

    reject([&] { a::artifact(root, "../outside.co"); });
    reject([&] { a::artifact(root, fs::absolute(raw).u8string()); });
    reject([&] { a::artifact(root, "absent.co"); });
    check(a::artifact(root, "raw.dat") == fs::canonical(raw), "Valid artifact path rejected");
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
    write(bundle / "library/TensileLibrary.dat.zlib", compressed);
    for(const auto* name : {"b.s", "a.s", "Kernels.cpp", "Kernels.h", "TensileTypes.h"})
        write(bundle / "sources" / name, std::string("// ") + name + "\n");
    const auto sources = a::readSourceBundle(bundle);
    check(std::string(sources.library.begin(), sources.library.end()) == payload,
          "Source bundle library changed");
    check(names(sources.assembly) == std::vector<std::string>({"a.s", "b.s"}),
          "Main assembly is not every sources/*.s by name");
    check(names(sources.helpers) == std::vector<std::string>({"Kernels.cpp"}),
          "Helper source is not sources/Kernels.cpp");
    check(names(sources.headers) == std::vector<std::string>({"Kernels.h", "TensileTypes.h"}),
          "Headers are not the other sources/ files by name");
    check(std::string(sources.assembly[0].bytes.begin(), sources.assembly[0].bytes.end())
              == "// a.s\n",
          "Source bytes changed");
    fs::remove(bundle / "sources/Kernels.cpp");
    check(a::readSourceBundle(bundle).helpers.empty(), "A bundle without helpers was rejected");
    auto damage = [&](const std::function<void()>& change, const std::function<void()>& undo) {
        change();
        reject([&] { a::readSourceBundle(bundle); });
        undo();
        a::readSourceBundle(bundle);
    };
    damage([&] { fs::rename(bundle / "sources", root / "moved"); },
           [&] { fs::rename(root / "moved", bundle / "sources"); });
    damage([&] { write(bundle / "library/TensileLibrary.yaml", "second library"); },
           [&] { fs::remove(bundle / "library/TensileLibrary.yaml"); });
    damage([&] { fs::rename(bundle / "library", root / "moved"); },
           [&] { fs::rename(root / "moved", bundle / "library"); });
    damage([&] { write(bundle / "library/TensileLibrary.dat.zlib", "not zlib"); },
           [&] { write(bundle / "library/TensileLibrary.dat.zlib", compressed); });
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
    damage([&] { write(bundle / "sources/empty.h", ""); },
           [&] { fs::remove(bundle / "sources/empty.h"); });
    damage(
        [&] {
            write(bundle / "sources/large.h", "x");
            fs::resize_file(bundle / "sources/large.h", 64 * 1024 * 1024 + 1);
        },
        [&] { fs::remove(bundle / "sources/large.h"); });
    damage(
        [&] {
            for(int i = 0; i < 1024; ++i)
                write(bundle / "sources" / ("extra" + std::to_string(i) + ".h"), "x");
        },
        [&] {
            for(int i = 0; i < 1024; ++i)
                fs::remove(bundle / "sources" / ("extra" + std::to_string(i) + ".h"));
        });
#ifndef _WIN32
    write(root / "outside.h", "outside");
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
