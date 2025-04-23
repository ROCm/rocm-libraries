/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include "meta_data/label_reader_folders.h"

#include <string.h>

#include <algorithm>
#include "pipeline/commons.h"
#include "pipeline/exception.h"
#include "pipeline/filesystem.h"

using namespace std;

LabelReaderFolders::LabelReaderFolders() {
    _src_dir = nullptr;
    _entity = nullptr;
    _sub_dir = nullptr;
}

void LabelReaderFolders::init(const MetaDataConfig& cfg, pMetaDataBatch meta_data_batch) {
    _path = cfg.path();
    _output = meta_data_batch;
}
bool LabelReaderFolders::exists(const std::string& image_name) {
    return _map_content.find(image_name) != _map_content.end();
}
void LabelReaderFolders::add(std::string image_name, int label) {
    pMetaData info = std::make_shared<Label>(label);
    if (exists(image_name)) {
        WRN("Entity with the same name exists")
        return;
    }
    _map_content.insert(pair<std::string, std::shared_ptr<Label>>(image_name, info));
}

void LabelReaderFolders::print_map_contents() {
    std::cerr << "\nMap contents: \n";
    for (auto& elem : _map_content) {
        std::cerr << "Name :\t " << elem.first << "\t ID:  " << elem.second->get_labels()[0] << std::endl;
    }
}

void LabelReaderFolders::release() {
    _map_content.clear();
}

void LabelReaderFolders::release(std::string image_name) {
    if (!exists(image_name)) {
        WRN("ERROR: Given not present in the map" + image_name);
        return;
    }
    _map_content.erase(image_name);
}

void LabelReaderFolders::lookup(const std::vector<std::string>& image_names) {
    if (image_names.empty()) {
        WRN("No image names passed")
        return;
    }
    if (image_names.size() != (unsigned)_output->size())
        _output->resize(image_names.size());

    for (unsigned i = 0; i < image_names.size(); i++) {
        auto image_name = image_names[i];
        auto it = _map_content.find(image_name);
        if (_map_content.end() == it)
            THROW("ERROR: Given name not present in the map" + image_name)
        _output->get_labels_batch()[i] = it->second->get_labels();
    }
}

void LabelReaderFolders::read_all(const std::string& _path) {
    std::string _folder_path = _path;
    if ((_sub_dir = opendir(_folder_path.c_str())) == nullptr)
        THROW("ERROR: Failed opening the directory at " + _folder_path);

    std::vector<std::string> entry_name_list;
    std::string _full_path = _folder_path;

    while ((_entity = readdir(_sub_dir)) != nullptr) {
        std::string entry_name(_entity->d_name);
        if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0) continue;
        entry_name_list.push_back(entry_name);
    }
    std::sort(entry_name_list.begin(), entry_name_list.end());
    closedir(_sub_dir);
    uint label_counter = 0;
    for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count) {
        std::string subfolder_path = _full_path + "/" + entry_name_list[dir_count];
        filesys::path pathObj(subfolder_path);
        if (filesys::exists(pathObj) && filesys::is_regular_file(pathObj)) {
            // ignore files with non-image extensions
            auto file_extension_idx = subfolder_path.find_last_of(".");
            if (file_extension_idx != std::string::npos) {
                std::string file_extension = subfolder_path.substr(file_extension_idx + 1);
                std::transform(file_extension.begin(), file_extension.end(), file_extension.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                if ((file_extension != "jpg") && (file_extension != "jpeg") && (file_extension != "png") && (file_extension != "ppm") && (file_extension != "bmp") && (file_extension != "pgm") && (file_extension != "tif") && (file_extension != "tiff") && (file_extension != "webp") && (file_extension != "wav"))
                    continue;
            }
            read_files(_folder_path);
            for (unsigned i = 0; i < _subfolder_file_names.size(); i++) {
                add(_subfolder_file_names[i], 0);
            }
            break;  // assume directory has only files.
        } else if (filesys::exists(pathObj) && filesys::is_directory(pathObj)) {
            _folder_path = subfolder_path;
            _subfolder_file_names.clear();
            read_files(_folder_path);
            for (unsigned i = 0; i < _subfolder_file_names.size(); i++) {
                add(_subfolder_file_names[i], label_counter);
            }
            label_counter++;
        }
    }
}

void LabelReaderFolders::read_files(const std::string& _path) {
    if ((_src_dir = opendir(_path.c_str())) == nullptr)
        THROW("ERROR: Failed opening the directory at " + _path);

    while ((_entity = readdir(_src_dir)) != nullptr) {
        if (_entity->d_type != DT_REG)
            continue;

        std::string file_path = _path;
        file_path.append("/");
        std::string filename(_entity->d_name);
        auto file_extension_idx = filename.find_last_of(".");
        if (file_extension_idx != std::string::npos) {
            std::string file_extension = filename.substr(file_extension_idx + 1);
            std::transform(file_extension.begin(), file_extension.end(), file_extension.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            if ((file_extension != "jpg") && (file_extension != "jpeg") && (file_extension != "png") && (file_extension != "ppm") && (file_extension != "bmp") && (file_extension != "pgm") && (file_extension != "tif") && (file_extension != "tiff") && (file_extension != "webp") && (file_extension != "wav"))
                continue;
        }
        file_path.append(_entity->d_name);
        _file_names.push_back(file_path);
        _subfolder_file_names.push_back(_entity->d_name);
    }
    if (_file_names.empty())
        WRN("LabelReader: Could not find any file in " + _path)
    closedir(_src_dir);
}
