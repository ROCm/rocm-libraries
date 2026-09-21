// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#define HIP_CHECK(expression)                                                                               \
  {                                                                                                         \
    const hipError_t status = expression;                                                                   \
    if (status != hipSuccess)                                                                               \
    {                                                                                                       \
      std::cerr << "HIP error " << status << ": " << hipGetErrorString(status) << " at " << __FILE__ << ":" \
                << __LINE__ << std::endl;                                                                   \
      std::exit(EXIT_FAILURE);                                                                              \
    }                                                                                                       \
  }

// This program queries available AMD GPUs in the system and outputs
// a JSON structure that conforms to CTest's resource spec file format.
// See https://cmake.org/cmake/help/latest/manual/ctest.1.html#resource-specification-file
// for more information about the format.
// The JSON structure maps gfxIDs to device ids. This output can be passed to CTest
// via the --resource-spec-file option.
//
// Usage: hipcc generate_resource_spec.cpp -o generate_resource_spec
// ./generate_resource_spec <output file>
//
//  Sample output:
// 	{
//   "version": {
//     "major": 1,
//     "minor": 0
//   },
//   "local": [
//     {
//       "gfx1100": [
//         {
//           "id": 0
//         }
//       ],
//       "gfx1200": [
//         {
//           "id": 1
//         },
//         {
//           "id": 2
//         }
//       ],
//       "gpus": [
//         {
//           "id": 0
//         },
//         {
//           "id": 1
//         },
//         {
//           "id": 2
//         }
//       ]
//     }
//   ]
// }
int main(int argc, char* argv[])
{
  if (argc != 2)
  {
    std::cout << "Usage: ./generate_resource_spec <output file path>" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  std::ofstream out_file;
  out_file.open(argv[1]);

  // Figure out how many devices are in the system.
  int dev_count = 0;
  HIP_CHECK(hipGetDeviceCount(&dev_count));

  // There could be more than one device of each type.
  // Build a mapping of gfxID (string) to a vector of device IDs (unsigned ints).
  std::map<std::string, std::vector<unsigned int>> names_to_ids;
  // This entry will hold all device IDs.
  names_to_ids["gpus"] = {};

  // Populate the map
  for (unsigned int dev_id = 0; dev_id < static_cast<unsigned int>(dev_count); ++dev_id)
  {
    hipDeviceProp_t dev_prop;
    HIP_CHECK(hipGetDeviceProperties(&dev_prop, dev_id));
    std::string name(dev_prop.gcnArchName);
    auto pos = name.find_first_of(':');
    if (pos != std::string::npos)
    {
      name = name.substr(0, pos);
    }

    if (names_to_ids.find(name) == names_to_ids.end())
    {
      names_to_ids[name] = std::vector<unsigned int>({dev_id});
    }
    else
    {
      names_to_ids[name].push_back(dev_id);
    }

    // Add every device ID to the "gpus" entry
    names_to_ids["gpus"].push_back(dev_id);
  }

  // Begin the JSON output. The first few lines are boilerplate:
  out_file
    << "{" << std::endl
    << "  \"version\": {" << std::endl
    << "    \"major\": 1," << std::endl
    << "    \"minor\": 0" << std::endl
    << "  }," << std::endl
    << "  \"local\": [" << std::endl
    << "    {" << std::endl;

  // Add one object for each gfxID.
  // Each gfxID-keyed object will contain an array of device IDs.
  unsigned int key_index = 0;
  for (auto& name_it : names_to_ids)
  {
    out_file << "      \"" << name_it.first << "\": [" << std::endl;

    // The gfxIDs (names) are already sorted by the map, but the
    // device IDs may not be in order.
    // Sorting them is not technically necessary, but it's nice
    // to have a consistent output on each run so that the resource
    // spec file stays the same.
    std::sort(name_it.second.begin(), name_it.second.end());
    unsigned int id_index = 0;
    for (const auto& id_it : name_it.second)
    {
      out_file << "        {" << std::endl;
      // Note: ctest expects ids to be specified as a string (quoted), not an integer.
      out_file << "          \"id\": \"" << id_it << "\"" << std::endl;
      out_file << "        }";

      if (id_index < name_it.second.size() - 1)
      {
        out_file << ",";
      }
      out_file << std::endl;
      id_index++;
    }

    out_file << "      ]";

    if (key_index < names_to_ids.size() - 1)
    {
      out_file << ",";
    }
    out_file << std::endl;
    key_index++;
  }

  // Close out the remaining open objects and arrays.
  out_file << "    }" << std::endl << "  ]" << std::endl << "}" << std::endl;

  out_file.close();

  return EXIT_SUCCESS;
}
