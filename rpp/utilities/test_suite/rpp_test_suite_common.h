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

#define DEBUG_MODE 0
using namespace std;

// Opens a folder and recursively search for files with given extension
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

        if (fileName.size() > 4 && fileName.substr(fileName.size() - 4) == extension)
        {
            imageNamesPath.push_back(filePath);
            imageNames.push_back(entity->d_name);
        }
    }
    if(imageNames.empty())
        std::cerr << "\n Did not load any file from " << folderPath;

    closedir(src_dir);
}

// Searches for files with the provided extensions in input folders
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
            if (entry_list[dir_count].size() > 4 && entry_list[dir_count].substr(entry_list[dir_count].size() - 4) == extension)
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