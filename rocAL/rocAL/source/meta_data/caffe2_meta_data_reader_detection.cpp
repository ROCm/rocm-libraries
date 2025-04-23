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

#include "meta_data/caffe2_meta_data_reader_detection.h"

#include <google/protobuf/message_lite.h>
#include <stdint.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>

#include "caffe2_protos.pb.h"
#include <lmdb.h>

using namespace std;

void Caffe2MetaDataReaderDetection::init(const MetaDataConfig &cfg, pMetaDataBatch meta_data_batch) {
    _path = cfg.path();
    _output = meta_data_batch;
}

bool Caffe2MetaDataReaderDetection::exists(const std::string &_image_name) {
    return _map_content.find(_image_name) != _map_content.end();
}

void Caffe2MetaDataReaderDetection::add(std::string image_name, BoundingBoxCords bb_coords, Labels bb_labels, ImgSize image_size) {
    if (exists(image_name)) {
        auto it = _map_content.find(image_name);
        it->second->get_bb_cords().push_back(bb_coords[0]);
        it->second->get_labels().push_back(bb_labels[0]);
        return;
    }
    pMetaDataBox info = std::make_shared<BoundingBox>(bb_coords, bb_labels, image_size);
    _map_content.insert(pair<std::string, std::shared_ptr<BoundingBox>>(image_name, info));
}

void Caffe2MetaDataReaderDetection::lookup(const std::vector<std::string> &_image_names) {
    if (_image_names.empty()) {
        WRN("No image names passed")
        return;
    }
    if (_image_names.size() != (unsigned)_output->size())
        _output->resize(_image_names.size());

    for (unsigned i = 0; i < _image_names.size(); i++) {
        auto image_name = _image_names[i];
        auto it = _map_content.find(image_name);
        if (_map_content.end() == it)
            THROW("ERROR: Given name not present in the map" + image_name)
        _output->get_bb_cords_batch()[i] = it->second->get_bb_cords();
        _output->get_labels_batch()[i] = it->second->get_labels();
        _output->get_img_sizes_batch()[i] = it->second->get_img_size();
    }
}

void Caffe2MetaDataReaderDetection::print_map_contents() {
    BoundingBoxCords bb_coords;
    Labels bb_labels;

    std::cerr << "\nMap contents: \n";
    for (auto &elem : _map_content) {
        std::cerr << "Name :\t " << elem.first;
        bb_coords = elem.second->get_bb_cords();
        bb_labels = elem.second->get_labels();
        std::cerr << "\nsize of the element  : " << bb_coords.size() << std::endl;
        for (unsigned int i = 0; i < bb_coords.size(); i++) {
            std::cerr << " l : " << bb_coords[i].l << " t: :" << bb_coords[i].t << " r : " << bb_coords[i].r << " b: :" << bb_coords[i].b << std::endl;
            std::cerr << "Label Id : " << bb_labels[i] << std::endl;
        }
    }
}

void Caffe2MetaDataReaderDetection::read_all(const std::string &path) {
    string tmp1 = path + "/data.mdb";
    string tmp2 = path + "/lock.mdb";
    uint file_size, file_size1, file_bytes;

    ifstream in_file(tmp1, ios::binary);
    in_file.seekg(0, ios::end);
    file_size = in_file.tellg();
    ifstream in_file1(tmp2, ios::binary);
    in_file1.seekg(0, ios::end);
    file_size1 = in_file1.tellg();
    file_bytes = file_size + file_size1;
    read_lmdb_record(path, file_bytes);
    // print_map_contents();
}

void Caffe2MetaDataReaderDetection::read_lmdb_record(std::string file_name, uint file_byte_size) {
    int rc;
    MDB_env *env;
    MDB_dbi dbi;
    MDB_val key, data;
    MDB_txn *txn;
    MDB_cursor *cursor;
    string str_key;

    // Creating an LMDB environment handle
    CHECK_LMDB_RETURN_STATUS(mdb_env_create(&env));
    // Setting the size of the memory map to use for this environment.
    // The size of the memory map is also the maximum size of the database.
    CHECK_LMDB_RETURN_STATUS(mdb_env_set_mapsize(env, file_byte_size));
    // Opening an environment handle.
    CHECK_LMDB_RETURN_STATUS(mdb_env_open(env, file_name.c_str(), 0, 0664));
    // Creating a transaction for use with the environment.
    CHECK_LMDB_RETURN_STATUS(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn));
    // Opening a database in the environment.
    CHECK_LMDB_RETURN_STATUS(mdb_dbi_open(txn, NULL, 0, &dbi));

    // Creating a cursor handle.
    // A cursor is associated with a specific transaction and database
    CHECK_LMDB_RETURN_STATUS(mdb_cursor_open(txn, dbi, &cursor));

    // Retrieve by cursor. It retrieves key/data pairs from the database
    while ((rc = mdb_cursor_get(cursor, &key, &data, MDB_NEXT)) == 0) {
        // Reading the key value for each record from LMDB
        str_key = string((char *)key.mv_data);

        // Parsing Image and Label Protos using the key and data values
        // read from LMDB records
        caffe2_protos::TensorProtos tens_protos;
        tens_protos.ParseFromArray((char *)data.mv_data, data.mv_size);
        int protos_size = tens_protos.protos_size();
        if (protos_size != 0) {
            caffe2_protos::TensorProto label_proto = tens_protos.protos(1);
            caffe2_protos::TensorProto boundingBox_proto = tens_protos.protos(2);

            // Parsing bounding box size for the image
            int boundBox_size = boundingBox_proto.dims_size();

            BoundingBoxCords bb_coords;
            Labels bb_labels;
            BoundingBoxCord box;

            ImgSize img_size;
            if (boundBox_size != 0) {
                int boundIter = 0;
                for (int i = 0; i < boundBox_size >> 2; i++) {
                    // Parsing the bounding box points using Iterator
                    box.l = boundingBox_proto.dims(boundIter);
                    box.t = boundingBox_proto.dims(boundIter + 1);
                    box.r = box.l + boundingBox_proto.dims(boundIter + 2);
                    box.b = box.t + boundingBox_proto.dims(boundIter + 3);

                    boundIter += 4;

                    // Parsing the image label using Iterator
                    int label = label_proto.int32_data(i);

                    bb_coords.push_back(box);
                    bb_labels.push_back(label);
                    add(str_key.c_str(), bb_coords, bb_labels, img_size);
                    bb_coords.clear();
                    bb_labels.clear();
                }
            } else {
                box.l = box.t = 0;
                box.r = box.b = 1;
                bb_coords.push_back(box);
                bb_labels.push_back(0);
                add(str_key.c_str(), bb_coords, bb_labels, img_size);
            }
        } else {
            THROW("Parsing Protos Failed");
        }
    }

    // Closing all the LMDB environment and cursor handles
    mdb_cursor_close(cursor);
    mdb_txn_abort(txn);
    mdb_close(env, dbi);
    mdb_env_close(env);
}

void Caffe2MetaDataReaderDetection::release(std::string _image_name) {
    if (!exists(_image_name)) {
        WRN("ERROR: Given not present in the map" + _image_name);
        return;
    }
    _map_content.erase(_image_name);
}

void Caffe2MetaDataReaderDetection::release() {
    _map_content.clear();
}

Caffe2MetaDataReaderDetection::Caffe2MetaDataReaderDetection() {
}
