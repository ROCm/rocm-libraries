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

#include "meta_data/mxnet_meta_data_reader.h"

#include <memory.h>
#include <stdint.h>
#include "pipeline/filesystem.h"

using namespace std;

void MXNetMetaDataReader::init(const MetaDataConfig &cfg, pMetaDataBatch meta_data_batch) {
    _path = cfg.path();
    _output = meta_data_batch;
    _src_dir = nullptr;
    _entity = nullptr;
}

bool MXNetMetaDataReader::exists(const std::string &_image_name) {
    return _map_content.find(_image_name) != _map_content.end();
}

void MXNetMetaDataReader::add(std::string image_name, int label) {
    pMetaData info = std::make_shared<Label>(label);
    if (exists(image_name)) {
        WRN("Entity with the same name exists")
        return;
    }
    _map_content.insert(std::pair<std::string, std::shared_ptr<Label>>(image_name, info));
}

void MXNetMetaDataReader::lookup(const std::vector<std::string> &_image_names) {
    if (_image_names.empty()) {
        WRN("No image names passed")
        return;
    }
    if (_image_names.size() != (unsigned)_output->size())
        _output->resize(_image_names.size());

    for (unsigned i = 0; i < _image_names.size(); i++) {
        auto _image_name = _image_names[i];
        auto it = _map_content.find(_image_name);
        if (_map_content.end() == it)
            THROW("MXNetMetaDataReader ERROR: Given name not present in the map" + _image_name)
        _output->get_labels_batch()[i] = it->second->get_labels();
    }
}

void MXNetMetaDataReader::print_map_contents() {
    std::cerr << "\nMap contents: \n";
    for (auto &elem : _map_content) {
        std::cerr << "Name :\t " << elem.first << "\t ID:  " << elem.second->get_labels()[0] << std::endl;
    }
}

void MXNetMetaDataReader::read_all(const std::string &_path) {
    std::string _rec_file, _idx_file;
    if ((_src_dir = opendir(_path.c_str())) == nullptr)
        THROW("MXNetMetaDataReader ERROR: Failed opening the directory at " + _path);

    while ((_entity = readdir(_src_dir)) != nullptr) {
        std::string file_name = _path + "/" + _entity->d_name;
        filesys::path pathObj(file_name);
        if (filesys::exists(pathObj) && filesys::is_regular_file(pathObj)) {
            auto file_extension_idx = file_name.find_last_of(".");
            if (file_extension_idx != std::string::npos) {
                std::string file_extension = file_name.substr(file_extension_idx + 1);
                if (file_extension == "rec")
                    _rec_file = file_name;
                else if (file_extension == "idx")
                    _idx_file = file_name;
                else
                    continue;
            }
        }
    }
    closedir(_src_dir);
    uint rec_size;
    _file_contents.open(_rec_file);
    if (!_file_contents)
        THROW("MXNetMetaDataReader ERROR: Failed opening the file " + _rec_file);
    _file_contents.seekg(0, ifstream::end);
    rec_size = _file_contents.tellg();
    _file_contents.seekg(0, ifstream::beg);

    ifstream index_file(_idx_file);
    if (!index_file)
        THROW("MXNetMetaDataReader ERROR: Could not open RecordIO index file. Provided path: " + _idx_file);

    std::vector<size_t> _index_list;
    size_t _index, _offset;
    while (index_file >> _index >> _offset)
        _index_list.push_back(_offset);
    if (_index_list.empty())
        THROW("MXNetMetaDataReader ERROR: RecordIO index file doesn't contain any indices. Provided path: " + _idx_file);
    _index_list.push_back(rec_size);
    std::sort(_index_list.begin(), _index_list.end());
    for (size_t i = 0; i < _index_list.size() - 1; ++i)
        _indices.emplace_back(_index_list[i], _index_list[i + 1] - _index_list[i]);
    read_images();
}

void MXNetMetaDataReader::release(std::string _image_name) {
    if (!exists(_image_name)) {
        WRN("ERROR: Given not present in the map" + _image_name);
        return;
    }
    _map_content.erase(_image_name);
}

void MXNetMetaDataReader::release() {
    _map_content.clear();
}

void MXNetMetaDataReader::read_images() {
    for (int current_index = 0; current_index < (int)_indices.size(); current_index++) {
        uint32_t _magic, _length_flag;
        int64_t _seek_pos, _data_size_to_read;
        std::tie(_seek_pos, _data_size_to_read) = _indices[current_index];
        _file_contents.seekg(_seek_pos, ifstream::beg);
        uint8_t *_data = new uint8_t[_data_size_to_read];
        uint8_t *_data_ptr = _data;
        auto ret = _file_contents.read((char *)_data_ptr, _data_size_to_read).gcount();
        if (ret == -1 || ret != _data_size_to_read)
            THROW("MXNetMetaDataReader ERROR:  Unable to read the data from the file ");
        _magic = *((uint32_t *)_data_ptr);
        _data_ptr += sizeof(_magic);
        if (_magic != _kMagic)
            THROW("MXNetMetaDataReader ERROR: Invalid RecordIO: wrong magic number");
        _length_flag = *((uint32_t *)_data_ptr);
        _data_ptr += sizeof(_length_flag);
        _hdr = *((ImageRecordIOHeader *)_data_ptr);

        if (_hdr.flag == 0) {
            add((to_string(_hdr.image_id[0])), _hdr.label);
        } else {
            WRN("\nMultiple record reading has not supported");
            continue;
        }
        delete[] _data;
    }
}

MXNetMetaDataReader::MXNetMetaDataReader() {
}
