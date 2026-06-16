// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstring>
#include <iostream>
#include <string>

int sweepMain(const std::string& shapesPath, const std::string& outPath,
              const std::string& dtype);

static void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " --shapes <shapes.csv> --out <out.csv>\n"
              << "\n"
              << "  --shapes   CSV of conv shapes to sweep (N,G,C,K,Hi,Wi,Y,X,...)\n"
              << "  --out      Output CSV path; header is written if the file is new,\n"
              << "             rows are appended if it already exists.\n";
}

int main(int argc, char** argv) {
    std::string shapesPath, outPath, dtype = "fp16";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--shapes") == 0 && i + 1 < argc)
            shapesPath = argv[++i];
        else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc)
            outPath = argv[++i];
        else if (std::strcmp(argv[i], "--dtype") == 0 && i + 1 < argc)
            dtype = argv[++i];
        else {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
            usage(argv[0]);
            return 1;
        }
    }

    if (shapesPath.empty() || outPath.empty()) {
        usage(argv[0]);
        return 1;
    }

    return sweepMain(shapesPath, outPath, dtype);
}
