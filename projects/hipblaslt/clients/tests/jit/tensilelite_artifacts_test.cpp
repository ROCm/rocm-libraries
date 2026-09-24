// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-tensilelite-artifacts.hpp"
#include <functional>
#include <iostream>

namespace a  = hipblaslt_ext::experimental::jit::tensilelite::detail::artifacts;
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
void number(std::string& output, uint32_t value)
{
    for(int i = 0; i < 4; ++i)
        output.push_back(static_cast<char>(value >> (8 * i)));
}
void string(std::string& output, const std::string& value)
{
    number(output, value.size());
    output += value;
}
void write(const fs::path& path, const std::string& bytes)
{
    std::ofstream file(path, std::ios::binary);
    check(bool(file.write(bytes.data(), bytes.size())), "Cannot write fixture");
}

int main(int argc, char** argv)
try
{
    check(argc == 2, "Usage: artifact-test NEW_DIRECTORY");
    const auto root = fs::u8path(argv[1]);
    check(fs::create_directory(root), "Fixture directory already exists");
    const auto  file  = root / fs::u8path("loader space π.bin");
    std::string valid = "TLJIT001";
    number(valid, 14);
    const std::vector<std::string> fields = {"2",
                                             "1",
                                             "1",
                                             "0",
                                             "main",
                                             "main",
                                             "solution",
                                             "gfx950",
                                             "gfx950",
                                             "gfx950",
                                             "msgpack",
                                             "main.co",
                                             "library.dat.zlib",
                                             "library.dat"};
    for(const auto& value : fields)
        string(valid, value);
    number(valid, 2);
    string(valid, "main.co");
    string(valid, "helper.hsaco");
    auto read = [&] {
        std::vector<std::string> objects;
        const auto               tree = a::readEnvelope(file, objects);
        check(tree.at("solution.name") == "solution", "Envelope field ordering changed");
        check(objects == std::vector<std::string>({"main.co", "helper.hsaco"}),
              "Lost helper object");
    };
    write(file, valid);
    read();
    for(size_t end = 0; end < valid.size(); ++end)
    {
        write(file, valid.substr(0, end));
        reject(read);
    }
    auto damaged = valid;
    damaged[0]   = 'X';
    write(file, damaged);
    reject(read);
    damaged    = valid;
    damaged[8] = 13;
    write(file, damaged);
    reject(read);
    damaged = valid;
    damaged.replace(12, 4, std::string(4, '\xff'));
    write(file, damaged);
    reject(read);
    damaged     = valid;
    damaged[16] = '\0';
    write(file, damaged);
    reject(read);
    write(file, valid + "trailing");
    reject(read);
    std::cout << "PASS loader version, fields, helpers, every truncation, length bounds and EOF\n";

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
    return 0;
}
catch(const std::exception& error)
{
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
