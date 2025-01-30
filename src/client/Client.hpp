/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

// TransferBench client version
#define CLIENT_VERSION "00"

#include "TransferBench.hpp"
#include "EnvVars.hpp"

size_t const DEFAULT_BYTES_PER_TRANSFER = (1<<28);

char const ExeTypeName[5][4] = {"CPU", "GPU", "DMA", "NIC", "NIC"};

// Display detected hardware
void DisplayTopology(bool outputToCsv);

// Display usage instructions
void DisplayUsage(char const* cmdName);

// Print TransferBench test results
void PrintResults(EnvVars const& ev, int const testNum,
                  std::vector<Transfer> const& transfers,
                  TransferBench::TestResults const& results);

// Helper function that converts MemDevices to a string
std::string MemDevicesToStr(std::vector<MemDevice> const& memDevices);

// Helper function to print warning / exit on fatal error
void CheckForError(ErrResult const& error);

// Helper function to print list of errors
void PrintErrors(std::vector<ErrResult> const& errors);
