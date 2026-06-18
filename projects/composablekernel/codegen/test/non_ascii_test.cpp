// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <non_ascii_test.hpp>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

int main()
{
    auto data = non_ascii_test();
    auto it   = data.find("non_ascii_test.txt");
    assert(it != data.end());

    auto content = it->second;
    assert(!content.empty());
    assert(content.find("\xe2\x80\x94") != std::string_view::npos);

    std::ifstream file(std::string(SOURCE_DIR) + "/non_ascii_test.txt", std::ios::binary);
    assert(file.is_open());

    std::vector<char> original((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());

    assert(content.size() == original.size());
    assert(std::memcmp(content.data(), original.data(), content.size()) == 0);

    std::cout << "Non-ASCII embed regression test passed." << std::endl;
    return 0;
}
