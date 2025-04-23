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

#include "meta_data/cifar10_meta_data_reader.h"

#include <string.h>

#include <algorithm>
#include "pipeline/commons.h"
#include "pipeline/exception.h"
#include "pipeline/filesystem.h"

using namespace std;

Cifar10MetaDataReader::Cifar10MetaDataReader()
{
    _src_dir = nullptr;
    _entity = nullptr;
    _sub_dir = nullptr;
}

void Cifar10MetaDataReader::init(const MetaDataConfig& cfg, pMetaDataBatch meta_data_batch) {
    _path = cfg.path();
    _file_prefix = cfg.file_prefix();
    _output = meta_data_batch;
    _raw_file_size = 32 * 32 * 3 + 1;  // 1 extra byte is label
}
bool Cifar10MetaDataReader::exists(const std::string& image_name) {
    return _map_content.find(image_name) != _map_content.end();
}
void Cifar10MetaDataReader::add(std::string image_name, int label) {
    pMetaData info = std::make_shared<Label>(label);
    if (exists(image_name)) {
        WRN("Entity with the same name exists")
        return;
    }
    _map_content.insert(pair<std::string, std::shared_ptr<Label>>(image_name, info));
}

void Cifar10MetaDataReader::print_map_contents() {
    std::cerr << "\nMap contents: \n";
    for (auto& elem : _map_content) {
        std::cerr << "Name :\t " << elem.first << "\t ID:  " << elem.second->get_labels()[0] << std::endl;
    }
}

void Cifar10MetaDataReader::release() {
    _map_content.clear();
}

void Cifar10MetaDataReader::release(std::string image_name) {
    if (!exists(image_name)) {
        WRN("ERROR: Given not present in the map" + image_name);
        return;
    }
    _map_content.erase(image_name);
}

void Cifar10MetaDataReader::lookup(const std::vector<std::string>& image_names) {
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

void Cifar10MetaDataReader::read_all(const std::string& _path) {
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

    std::string subfolder_path = _full_path + "/" + entry_name_list[0];
    filesys::path pathObj(subfolder_path);
    if (filesys::exists(pathObj) && filesys::is_regular_file(pathObj)) {
        read_files(_folder_path);
        // print_map_contents();
    } else if (filesys::exists(pathObj) && filesys::is_directory(pathObj)) {
        for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count) {
            std::string subfolder_path = _full_path + "/" + entry_name_list[dir_count];
            _folder_path = subfolder_path;
            _subfolder_file_names.clear();
            read_files(_folder_path);
            // print_map_contents();
        }
    }
    closedir(_sub_dir);
}

void Cifar10MetaDataReader::read_files(const std::string& _path) {
    if ((_src_dir = opendir(_path.c_str())) == nullptr)
        THROW("ERROR: Failed opening the directory at " + _path);

    while ((_entity = readdir(_src_dir)) != nullptr) {
        if (_entity->d_type != DT_REG)
            continue;
        std::string file_path = _path;
        // check if the filename has the _file_name_prefix
        std::string data_file_name = std::string(_entity->d_name);
        if (data_file_name.find(_file_prefix) != std::string::npos) {
            file_path.append("/");
            file_path.append(_entity->d_name);
            FILE* fp = fopen(file_path.c_str(), "rb");  // Open the file,
            fseek(fp, 0, SEEK_END);                     // Take the file read pointer to the end
            size_t total_file_size = ftell(fp);
            size_t num_of_raw_files = _raw_file_size ? total_file_size / _raw_file_size : 0;
            unsigned file_offset = 0;
            std::string file_id;
            unsigned char label;
            for (unsigned i = 0; i < num_of_raw_files; i++) {
                _file_names.push_back(file_path);
                _file_offsets.push_back(file_offset);
                _file_idx.push_back(i);
                // read first byte for each bin as label and add entry
                // generate image_name and label
                file_id = file_path;
                auto last_slash_idx = file_id.find_last_of("\\/");
                if (std::string::npos != last_slash_idx) {
                    file_id.erase(0, last_slash_idx + 1);
                }
                // add file_idx to last_id so the loader knows the index within the same master file
                file_id.append("_");
                file_id.append(std::to_string(i));
                // read first byte for each bin as label and add entry
                fseek(fp, file_offset, SEEK_SET);
                size_t n = fread((void*)&label, sizeof(unsigned char), 1, fp);
                if ((n != 1) || (label < 0) || (label > 9))
                    WRN("Cifar10MetaDataReader:: Invalid label " + TOSTR(label) + "read");
                add(file_id, (int)label);
                // LOG("Cifar10MetaDataReader:: Added record ID: " + file_id + " label: " + TOSTR(label));
                file_offset += _raw_file_size;
            }
            LOG("Cifar10MetaDataReader:: Added " + TOSTR(num_of_raw_files) + " Meta map ");
            fclose(fp);
        }
    }
    if (_file_names.empty())
        WRN("LabelReader: Could not find any file in " + _path)
    closedir(_src_dir);
}
