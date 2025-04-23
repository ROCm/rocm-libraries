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

#include <iostream>
#include <fstream>
#include <utility>
#include <algorithm>
#include <fstream>
#include <string>
#include <stdint.h>
#include <google/protobuf/message_lite.h>
#include <lmdb.h>
#include "meta_data/caffe_meta_data_reader_detection.h"
#include "caffe_protos.pb.h"

using namespace std;

void CaffeMetaDataReaderDetection::init(const MetaDataConfig &cfg, pMetaDataBatch meta_data_batch) {
    _path = cfg.path();
    _output = meta_data_batch;
}

bool CaffeMetaDataReaderDetection::exists(const std::string &_image_name) {
    return _map_content.find(_image_name) != _map_content.end();
}

void CaffeMetaDataReaderDetection::add(std::string image_name, BoundingBoxCords bb_coords, Labels bb_labels, ImgSize image_size) {
    if (exists(image_name)) {
        auto it = _map_content.find(image_name);
        it->second->get_bb_cords().push_back(bb_coords[0]);
        it->second->get_labels().push_back(bb_labels[0]);
        return;
    }
    pMetaDataBox info = std::make_shared<BoundingBox>(bb_coords, bb_labels, image_size);
    _map_content.insert(pair<std::string, std::shared_ptr<BoundingBox>>(image_name, info));
}

void CaffeMetaDataReaderDetection::lookup(const std::vector<std::string> &_image_names) {
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

void CaffeMetaDataReaderDetection::print_map_contents() {
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

void CaffeMetaDataReaderDetection::read_all(const std::string &path) {
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

void CaffeMetaDataReaderDetection::read_lmdb_record(std::string file_name, uint file_byte_size) {
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

    // Retrieve by cursor. It retrieves key/data pairs from the database
    while ((rc = mdb_cursor_get(_mdb_cursor, &_mdb_key, &_mdb_value, MDB_NEXT)) == 0) {
        std::string file_name = string((char *)_mdb_key.mv_data);

        caffe_protos::AnnotatedDatum annotatedDatum_protos;
        annotatedDatum_protos.ParseFromArray((char *)_mdb_value.mv_data, _mdb_value.mv_size);
        caffe_protos::AnnotationGroup annotGrp_protos = annotatedDatum_protos.annotation_group(0);

        int boundBox_size = annotGrp_protos.annotation_size();

        BoundingBoxCords bb_coords;
        Labels bb_labels;
        BoundingBoxCord box;
        ImgSize img_size;

        if (boundBox_size > 0) {
            for (int i = 0; i < boundBox_size; i++) {
                caffe_protos::Annotation annot_protos = annotGrp_protos.annotation(i);
                caffe_protos::NormalizedBBox bbox_protos = annot_protos.bbox();

                // Parsing the bounding box points using Iterator & converting the bbox values to ltrb format
                box.l = bbox_protos.xmin();
                box.t = bbox_protos.ymin();
                box.r = (bbox_protos.xmin() + bbox_protos.xmax());
                box.b = (bbox_protos.ymin() + bbox_protos.ymax());

                int label = bbox_protos.label();

                bb_coords.push_back(box);
                bb_labels.push_back(label);
                add(file_name.c_str(), bb_coords, bb_labels, img_size);
                bb_coords.clear();
                bb_labels.clear();
            }
        } else {
            box.l = box.t = 0;
            box.r = box.b = 1;
            bb_coords.push_back(box);
            bb_labels.push_back(0);
            add(file_name.c_str(), bb_coords, bb_labels, img_size);
        }
    }
    // Closing all the LMDB environment and cursor handles
    mdb_cursor_close(_mdb_cursor);
    mdb_close(_mdb_env, _mdb_dbi);
    mdb_txn_abort(_mdb_txn);
    mdb_env_close(_mdb_env);
}

void CaffeMetaDataReaderDetection::release(std::string _image_name) {
    if (!exists(_image_name)) {
        WRN("ERROR: Given not present in the map" + _image_name);
        return;
    }
    _map_content.erase(_image_name);
}

void CaffeMetaDataReaderDetection::release() {
    _map_content.clear();
}

CaffeMetaDataReaderDetection::CaffeMetaDataReaderDetection() {
}