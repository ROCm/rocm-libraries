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

#include "meta_data/caffe_meta_data_reader.h"

#include <string.h>

#include <algorithm>
#include "pipeline/commons.h"
#include "pipeline/exception.h"

using caffe_protos::Datum;
using std::string;
using namespace std;

CaffeMetaDataReader::CaffeMetaDataReader() {
}

void CaffeMetaDataReader::init(const MetaDataConfig& cfg, pMetaDataBatch meta_data_batch) {
    _path = cfg.path();
    _output = meta_data_batch;
}

bool CaffeMetaDataReader::exists(const std::string& image_name) {
    return _map_content.find(image_name) != _map_content.end();
}

void CaffeMetaDataReader::add(std::string image_name, int label) {
    pMetaData info = std::make_shared<Label>(label);
    if (exists(image_name)) {
        WRN("Entity with the same name exists")
        return;
    }
    _map_content.insert(pair<std::string, std::shared_ptr<Label>>(image_name, info));
}

void CaffeMetaDataReader::print_map_contents() {
    std::cout << "\nMap contents: \n";
    for (auto& elem : _map_content) {
        std::cout << "Name :\t " << elem.first << "\tsize: " << elem.first.size() << "\t ID:  " << elem.second->get_labels()[0] << std::endl;
    }
}

void CaffeMetaDataReader::release() {
    _map_content.clear();
}

void CaffeMetaDataReader::release(std::string image_name) {
    if (!exists(image_name)) {
        WRN("ERROR: Given not present in the map" + image_name);
        return;
    }
    _map_content.erase(image_name);
}

void CaffeMetaDataReader::lookup(const std::vector<std::string>& image_names) {
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

void CaffeMetaDataReader::read_all(const std::string& _path) {
    string tmp1 = _path + "/data.mdb";
    string tmp2 = _path + "/lock.mdb";
    uint file_size, file_size1, file_bytes;

    ifstream in_file(tmp1, ios::binary);
    in_file.seekg(0, ios::end);
    file_size = in_file.tellg();
    ifstream in_file1(tmp2, ios::binary);
    in_file1.seekg(0, ios::end);
    file_size1 = in_file1.tellg();
    file_bytes = file_size + file_size1;
    read_lmdb_record(_path, file_bytes);
    // print_map_contents();
}

void CaffeMetaDataReader::read_lmdb_record(std::string _path, uint file_byte_size) {
    int rc;
    // Creating an LMDB environment handle
    CHECK_LMDB_RETURN_STATUS(mdb_env_create(&_mdb_env));
    // Setting the size of the memory map to use for this environment.
    // The size of the memory map is also the maximum size of the database.
    CHECK_LMDB_RETURN_STATUS(mdb_env_set_mapsize(_mdb_env, file_byte_size));
    // Opening an environment handle.
    CHECK_LMDB_RETURN_STATUS(mdb_env_open(_mdb_env, _path.c_str(), MDB_RDONLY, 0664));
    // Creating a transaction for use with the environment
    CHECK_LMDB_RETURN_STATUS(mdb_txn_begin(_mdb_env, NULL, MDB_RDONLY, &_mdb_txn));
    // Opening a database in the environment.
    CHECK_LMDB_RETURN_STATUS(mdb_open(_mdb_txn, NULL, 0, &_mdb_dbi));
    // Creating a cursor handle.
    // A cursor is associated with a specific transaction and database
    CHECK_LMDB_RETURN_STATUS(mdb_cursor_open(_mdb_txn, _mdb_dbi, &_mdb_cursor));

    Datum datum;
    // Retrieve by cursor. It retrieves key/data pairs from the database
    while ((rc = mdb_cursor_get(_mdb_cursor, &_mdb_key, &_mdb_value, MDB_NEXT)) == 0) {
        std::string file_name = string((char*)_mdb_key.mv_data);
        datum.ParseFromArray((const void*)_mdb_value.mv_data, _mdb_value.mv_size);
        add(file_name.c_str(), datum.label());
    }

    // Closing all the LMDB environment and cursor handles
    mdb_cursor_close(_mdb_cursor);
    mdb_close(_mdb_env, _mdb_dbi);
    mdb_txn_abort(_mdb_txn);
    mdb_env_close(_mdb_env);
}
