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

#include "meta_data/text_file_meta_data_reader.h"

#include <string.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>

#include "pipeline/commons.h"
#include "pipeline/exception.h"

void TextFileMetaDataReader::init(const MetaDataConfig &cfg, pMetaDataBatch meta_data_batch) {
    _path = cfg.path();
    _output = meta_data_batch;
}

bool TextFileMetaDataReader::exists(const std::string &image_name) {
    return _map_content.find(image_name) != _map_content.end();
}

void TextFileMetaDataReader::add(std::string image_name, int label) {
    pMetaData info = std::make_shared<Label>(label);
    if (exists(image_name)) {
        WRN("Entity with the same name exists")
        return;
    }
    _map_content.insert(std::pair<std::string, std::shared_ptr<Label>>(image_name, info));
}

void TextFileMetaDataReader::lookup(const std::vector<std::string> &image_names) {
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

void TextFileMetaDataReader::read_all(const std::string &path) {
    std::ifstream text_file(path.c_str());
    if (text_file.good()) {
        std::string line;
        while (std::getline(text_file, line)) {
            std::istringstream line_ss(line);
            int label;
            std::string file_name;
            if (!(line_ss >> file_name >> label))
                continue;
            _relative_file_path.push_back(file_name); // to be used in file source reader to reduce I/O operations
            auto last_id = file_name;
            auto last_slash_idx = last_id.find_last_of("\\/");
            if (std::string::npos != last_slash_idx) {
                last_id.erase(0, last_slash_idx + 1);
            }
            add(last_id, label);
        }
    } else {
        THROW("Can't open the metadata file at " + path)
    }
}

void TextFileMetaDataReader::release(std::string image_name) {
    if (!exists(image_name)) {
        WRN("ERROR: Given not present in the map" + image_name);
        return;
    }
    _map_content.erase(image_name);
}

void TextFileMetaDataReader::release() {
    _map_content.clear();
}

TextFileMetaDataReader::TextFileMetaDataReader() {
}
//
// Created by mvx on 3/31/20.
//
