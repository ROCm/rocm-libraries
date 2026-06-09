/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <dirent.h>
#include <omp.h>
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <map>
#include <cstdio>
#include <cstdlib>

#define DEBUG_MODE 0
using namespace std;

// Test-suite convenience macro used by the HIP test binaries. On any non-zero
// return the macro prints the failing expression to stderr and aborts the test
// process. The main RPP library no longer uses this macro: library code must
// propagate HIP failures via RppStatus (see RPP_HIP_RETURN_IF_ERROR in
// api/rppdefs.h).
#ifndef CHECK_RETURN_STATUS
#define CHECK_RETURN_STATUS(x) do {                                                     \
    int retval = (x);                                                                   \
    if (retval != 0) {                                                                  \
        fprintf(stderr, "Runtime error: %s returned %d at %s:%d", #x, retval,           \
                __FILE__, __LINE__);                                                    \
        exit(-1);                                                                       \
    }                                                                                   \
} while (0)
#endif

// This is a test-suite enum to specify the Bit-Depth Testing Mode. 
// RPP supports different combinations of the following 9 testing modes
// of source and destination tensor bit depths, depending on the functionality being tested.
// The capability of each functionality is listed in the RPP API documentation.
enum BitDepthTestMode
{
    U8_TO_U8   = 0,  // Input: U8  -> Output: U8
    F16_TO_F16 = 1,  // Input: F16 -> Output: F16
    F32_TO_F32 = 2,  // Input: F32 -> Output: F32
    U8_TO_F16  = 3,  // Input: U8  -> Output: F16
    U8_TO_F32  = 4,  // Input: U8  -> Output: F32
    I8_TO_I8   = 5,  // Input: I8  -> Output: I8
    U8_TO_I8   = 6,  // Input: U8  -> Output: I8
    I16_TO_I16 = 7,  // Input: I16 -> Output: I16
    U16_TO_U16 = 8,  // Input: U16 -> Output: U16
    I32_TO_I32 = 9,  // Input: I32 -> Output: I32
    U32_TO_U32 = 10, // Input: U32 -> Output: U32
    I8_TO_F32  = 11, // Input: I8  -> Output: F32
    I16_TO_F32 = 12, // Input: I16 -> Output: F32
    U16_TO_F32 = 13, // Input: U16 -> Output: F32
    U32_TO_F32 = 14, // Input: U32 -> Output: F32
    I32_TO_F32 = 15  // Input: I32 -> Output: F32
};

// Enum representing different test types
enum testType
{
    UNIT_TEST = 0,
    PERFORMANCE_TEST = 1
};

// Create a map from RppStatus to string representation
inline std::map<RppStatus, std::string> rppStatusToString = {
    {RPP_SUCCESS,                               "RPP_SUCCESS"},
    {RPP_ERROR,                                 "RPP_ERROR"},
    {RPP_ERROR_INVALID_ARGUMENTS,               "RPP_ERROR_INVALID_ARGUMENTS"},
    {RPP_ERROR_LOW_OFFSET,                      "RPP_ERROR_LOW_OFFSET"},
    {RPP_ERROR_ZERO_DIVISION,                   "RPP_ERROR_ZERO_DIVISION"},
    {RPP_ERROR_HIGH_SRC_DIMENSION,              "RPP_ERROR_HIGH_SRC_DIMENSION"},
    {RPP_ERROR_NOT_IMPLEMENTED,                 "RPP_ERROR_NOT_IMPLEMENTED"},
    {RPP_ERROR_INVALID_SRC_CHANNELS,            "RPP_ERROR_INVALID_SRC_CHANNELS"},
    {RPP_ERROR_INVALID_DST_CHANNELS,            "RPP_ERROR_INVALID_DST_CHANNELS"},
    {RPP_ERROR_INVALID_SRC_LAYOUT,              "RPP_ERROR_INVALID_SRC_LAYOUT"},
    {RPP_ERROR_INVALID_DST_LAYOUT,              "RPP_ERROR_INVALID_DST_LAYOUT"},
    {RPP_ERROR_INVALID_SRC_DATATYPE,            "RPP_ERROR_INVALID_SRC_DATATYPE"},
    {RPP_ERROR_INVALID_DST_DATATYPE,            "RPP_ERROR_INVALID_DST_DATATYPE"},
    {RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE,     "RPP_ERROR_INVALID_SRC_OR_DST_DATATYPE"},
    {RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH,  "RPP_ERROR_INSUFFICIENT_DST_BUFFER_LENGTH"},
    {RPP_ERROR_INVALID_PARAMETER_DATATYPE,      "RPP_ERROR_INVALID_PARAMETER_DATATYPE"},
    {RPP_ERROR_NOT_ENOUGH_MEMORY,               "RPP_ERROR_NOT_ENOUGH_MEMORY"},
    {RPP_ERROR_OUT_OF_BOUND_SRC_ROI,            "RPP_ERROR_OUT_OF_BOUND_SRC_ROI"},
    {RPP_ERROR_LAYOUT_MISMATCH,                 "RPP_ERROR_LAYOUT_MISMATCH"},
    {RPP_ERROR_INVALID_CHANNELS,                "RPP_ERROR_INVALID_CHANNELS"},
    {RPP_ERROR_INVALID_OUTPUT_TILE_LENGTH,      "RPP_ERROR_INVALID_OUTPUT_TILE_LENGTH"},
    {RPP_ERROR_OUT_OF_BOUND_SHARED_MEMORY_SIZE, "RPP_ERROR_OUT_OF_BOUND_SHARED_MEMORY_SIZE"},
    {RPP_ERROR_OUT_OF_BOUND_SCRATCH_MEMORY_SIZE,"RPP_ERROR_OUT_OF_BOUND_SCRATCH_MEMORY_SIZE"},
    {RPP_ERROR_INVALID_SRC_DIMS,                "RPP_ERROR_INVALID_SRC_DIMS"},
    {RPP_ERROR_INVALID_DST_DIMS,                "RPP_ERROR_INVALID_DST_DIMS"},
    {RPP_ERROR_INVALID_DIM_LENGTHS,             "RPP_ERROR_INVALID_DIM_LENGTHS"},
    {RPP_ERROR_INVALID_AXIS,                    "RPP_ERROR_INVALID_AXIS"}
};

// True if basename extension (substring after the last '.') matches extension without its leading dot (e.g. ".yuv" -> foo.nv12.yuv yes, foo.info no).
inline bool filename_matches_requested_extension(const std::string& fileName, const std::string& extension)
{
    if (extension.empty() || extension[0] != '.') {
        std::cout << "ERROR: Extension is empty: " << extension << std::endl;
        return false;
    }
    const std::string ext = extension.substr(1);
    if (ext.empty()) {
        std::cout << "ERROR: Extension is empty: " << extension << std::endl;
        return false;
    }
    size_t dot = fileName.find_last_of('.');
    if (dot == std::string::npos || dot + 1 >= fileName.size()) {
        std::cout << "ERROR: File name does not contain a valid extension: " << fileName << std::endl;
        return false;
    }
    return fileName.compare(dot + 1, std::string::npos, ext) == 0;
}

// Opens a folder and recursively search for files with given extension (e.g. ".jpg", ".wav", ".yuv").
void open_folder(const string& folderPath, vector<string>& imageNames, vector<string>& imageNamesPath, string extension)
{
    auto src_dir = opendir(folderPath.c_str());
    struct dirent* entity;
    std::string fileName = " ";

    if (src_dir == nullptr)
        std::cerr << "\n ERROR: Failed opening the directory at " <<folderPath;

    while((entity = readdir(src_dir)) != nullptr)
    {
        string entry_name(entity->d_name);
        if (entry_name == "." || entry_name == "..")
            continue;
        fileName = entity->d_name;
        std::string filePath = folderPath;
        filePath.append("/");
        filePath.append(entity->d_name);
        fs::path pathObj(filePath);
        if(fs::exists(pathObj) && fs::is_directory(pathObj))
            open_folder(filePath, imageNames, imageNamesPath, extension);

        if (filename_matches_requested_extension(fileName, extension))
        {
            imageNamesPath.push_back(filePath);
            imageNames.push_back(entity->d_name);
        }
    }
    if(imageNames.empty())
        std::cerr << "\n Did not load any file from " << folderPath;

    closedir(src_dir);
}

// Searches for files with the provided extensions in input folders.
void search_files_recursive(const string& folder_path, vector<string>& imageNames, vector<string>& imageNamesPath, string extension)
{
    vector<string> entry_list;
    string full_path = folder_path;
    auto sub_dir = opendir(folder_path.c_str());
    if (!sub_dir)
    {
        std::cerr << "ERROR: Failed opening the directory at "<< folder_path << std::endl;
        exit(0);
    }

    struct dirent* entity;
    while ((entity = readdir(sub_dir)) != nullptr)
    {
        string entry_name(entity->d_name);
        if (entry_name == "." || entry_name == "..")
            continue;
        entry_list.push_back(entry_name);
    }
    closedir(sub_dir);
    sort(entry_list.begin(), entry_list.end());

    for (unsigned dir_count = 0; dir_count < entry_list.size(); ++dir_count)
    {
        string subfolder_path = full_path + "/" + entry_list[dir_count];
        fs::path pathObj(subfolder_path);
        if (fs::exists(pathObj) && fs::is_regular_file(pathObj))
        {
            // ignore files with extensions .tar, .zip, .7z
            auto file_extension_idx = subfolder_path.find_last_of(".");
            if (file_extension_idx != std::string::npos)
            {
                std::string file_extension = subfolder_path.substr(file_extension_idx+1);
                if ((file_extension == "tar") || (file_extension == "zip") || (file_extension == "7z") || (file_extension == "rar"))
                    continue;
            }
            if (filename_matches_requested_extension(entry_list[dir_count], extension))
            {
                imageNames.push_back(entry_list[dir_count]);
                imageNamesPath.push_back(subfolder_path);
            }
        }
        else if (fs::exists(pathObj) && fs::is_directory(pathObj))
            open_folder(subfolder_path, imageNames, imageNamesPath, extension);
    }
}

// replicates the last image in a batch to fill the remaining images in a batch
void replicate_last_file_to_fill_batch(const string& lastFilePath, vector<string>& imageNamesPath, vector<string>& imageNames, const string& lastFileName, int noOfImages, int batchCount)
{
    int remainingImages = batchCount - (noOfImages % batchCount);
    std::string filePath = lastFilePath;
    std::string fileName = lastFileName;
    if (noOfImages > 0 && ( noOfImages < batchCount || noOfImages % batchCount != 0 ))
    {
        for (int i = 0; i < remainingImages; i++)
        {
            imageNamesPath.push_back(filePath);
            imageNames.push_back(fileName);
        }
    }
}

template <typename T>
inline void read_bin_file(string refFile, T *binaryContent)
{
    FILE *fp;
    fp = fopen(refFile.c_str(), "rb");
    if(!fp)
    {
        std::cout << "\n unable to open file : "<<refFile;
        exit(0);
    }

    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    if (fsize == 0)
    {
        std::cout << "File is empty";
        exit(0);
    }

    fseek(fp, 0, SEEK_SET);
    fread(binaryContent, fsize, 1, fp);
    fclose(fp);
}