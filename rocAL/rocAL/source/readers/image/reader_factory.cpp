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

#include "readers/image/reader_factory.h"

#include <memory>
#include <stdexcept>

#include "readers/file_source_reader.h"
#include "readers/image/caffe2_lmdb_record_reader.h"
#include "readers/image/caffe_lmdb_record_reader.h"
#include "readers/image/cifar10_data_reader.h"
#include "readers/image/coco_file_source_reader.h"
#include "readers/image/external_source_reader.h"
#include "readers/image/mxnet_recordio_reader.h"
#include "readers/video/sequence_file_source_reader.h"
#include "readers/image/tf_record_reader.h"
#include "readers/webdataset_source_reader.h"
#include "readers/image/numpy_data_reader.h"

std::shared_ptr<Reader> create_reader(ReaderConfig config) {
    switch (config.type()) {
        case StorageType ::FILE_SYSTEM: {
            auto ret = std::make_shared<FileSourceReader>();
            if (ret->initialize(config) != Reader::Status::OK)
                throw std::runtime_error("File reader cannot access the storage");
            return ret;
        } break;
        case StorageType ::SEQUENCE_FILE_SYSTEM: {
            auto ret = std::make_shared<SequenceFileSourceReader>();
            if (ret->initialize(config) != Reader::Status::OK)
                throw std::runtime_error("File reader cannot access the storage");
            return ret;
        } break;
        case StorageType ::COCO_FILE_SYSTEM: {
            auto ret = std::make_shared<COCOFileSourceReader>();
            if (ret->initialize(config) != Reader::Status::OK)
                throw std::runtime_error("COCO File reader cannot access the storage");
            return ret;
        } break;
        case StorageType::TF_RECORD: {
            auto ret = std::make_shared<TFRecordReader>();
            if (ret->initialize(config) != Reader::Status::OK)
                throw std::runtime_error("File reader cannot access the storage");
            return ret;
        } break;
        case StorageType::UNCOMPRESSED_BINARY_DATA: {
            auto ret = std::make_shared<CIFAR10DataReader>();
            if (ret->initialize(config) != Reader::Status::OK)
                throw std::runtime_error("CFar10 data reader cannot access the storage");
            return ret;
        } break;
        case StorageType::CAFFE_LMDB_RECORD: {
            auto ret = std::make_shared<CaffeLMDBRecordReader>();
            if (ret->initialize(config) != Reader::Status::OK)
                throw std::runtime_error("CaffeLMDBRecordReader cannot access the storage");
            return ret;
        } break;
        case StorageType::CAFFE2_LMDB_RECORD: {
            auto ret = std::make_shared<Caffe2LMDBRecordReader>();
            if (ret->initialize(config) != Reader::Status::OK)
                throw std::runtime_error("Caffe2LMDBRecordReader cannot access the storage");
            return ret;
        } break;
        case StorageType::MXNET_RECORDIO: {
            auto ret = std::make_shared<MXNetRecordIOReader>();
            if (ret->initialize(config) != Reader::Status::OK)
                throw std::runtime_error("MXNetRecordIOReader cannot access the storage");
            return ret;
        } break;
        case StorageType::EXTERNAL_FILE_SOURCE: {
            auto ret = std::make_shared<ExternalSourceReader>();
            if (ret->initialize(config) != Reader::Status::OK)
                throw std::runtime_error("ExternalSourceReader cannot access the storage");
            return ret;
        } break;
#ifdef ENABLE_WDS
        case StorageType::WEBDATASET_RECORDS: {
            auto ret = std::make_shared<WebDatasetSourceReader>();
            if (ret->initialize(config) != Reader::Status::OK)
                throw std::runtime_error("WebDatasetSourceReader cannot access the storage");
            return ret;
        } break;
#endif
        case StorageType::NUMPY_DATA: {
            auto ret = std::make_shared<NumpyDataReader>();
            if (ret->initialize(config) != Reader::Status::OK)
                throw std::runtime_error("NumpyDataReader cannot access the storage");
            return ret;
        } break;
        default:
            throw std::runtime_error("Reader type is unsupported");
    }
}
