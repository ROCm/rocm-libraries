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

// Included after EnvVars and Executors
#include "AllToAll.hpp"
#include "AllToAllN.hpp"
#include "AllToAllSweep.hpp"
#include "HealthCheck.hpp"
#include "OneToAll.hpp"
#include "PeerToPeer.hpp"
#include "Scaling.hpp"
#include "Schmoo.hpp"
#include "Sweep.hpp"
#include <map>

typedef void (*PresetFunc)(EnvVars&          ev,
                           size_t      const numBytesPerTransfer,
                           std::string const presetName);

std::map<std::string, std::pair<PresetFunc, std::string>> presetFuncMap =
{
  {"a2a",         {AllToAllPreset,      "Tests parallel transfers between all pairs of GPU devices"}},
  {"a2a_n",       {AllToAllRdmaPreset,  "Tests parallel transfers between all pairs of GPU devices using Nearest NIC RDMA transfers"}},
  {"a2asweep",    {AllToAllSweepPreset, "Test GFX-based all-to-all transfers swept across different CU and GFX unroll counts"}},
  {"healthcheck", {HealthCheckPreset,   "Simple bandwidth health check (MI300X series only)"}},
  {"one2all",     {OneToAllPreset,      "Test all subsets of parallel transfers from one GPU to all others"}},
  {"p2p"   ,      {PeerToPeerPreset,    " Peer-to-peer device memory bandwidth test"}},
  {"rsweep",      {SweepPreset,         "Randomly sweep through sets of Transfers"}},
  {"scaling",     {ScalingPreset,       "Run scaling test from one GPU to other devices"}},
  {"schmoo",      {SchmooPreset,        "Scaling tests for local/remote read/write/copy"}},
  {"sweep",       {SweepPreset,         "Ordered sweep through sets of Transfers"}},
};

void DisplayPresets()
{
  printf("\nAvailable Preset Benchmarks:\n");
  printf("============================\n");
  for (auto const& x : presetFuncMap)
    printf("   %15s - %s\n", x.first.c_str(), x.second.second.c_str());
}

int RunPreset(EnvVars&       ev,
              size_t   const numBytesPerTransfer,
              int      const argc,
              char**   const argv)
{
  std::string preset = (argc > 1 ? argv[1] : "");
  if (presetFuncMap.count(preset)) {
    (presetFuncMap[preset].first)(ev, numBytesPerTransfer, preset);
    return 1;
  }
  return 0;
}
