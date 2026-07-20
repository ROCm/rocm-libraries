#ifndef RPP_TEST_TENSOR_LOADER_H
#define RPP_TEST_TENSOR_LOADER_H

#include <cstdio>
#include <string>
#include <vector>

namespace rpptest {

// Read a raw binary golden/input file into a byte buffer. Returns empty on failure.
inline std::vector<char> read_binary_file(const std::string& path) {
    std::vector<char> data;
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return data;
    std::fseek(fp, 0, SEEK_END);
    long size = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    if (size > 0) {
        data.resize(static_cast<std::size_t>(size));
        std::fread(data.data(), 1, data.size(), fp);
    }
    std::fclose(fp);
    return data;
}

}  // namespace rpptest

#endif  // RPP_TEST_TENSOR_LOADER_H
