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

#pragma once
#include <dirent.h>
#include <google/protobuf/message_lite.h>

#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "caffe2_protos.pb.h"
#include <lmdb.h>
#include "readers/image/image_reader.h"
#include "pipeline/timing_debug.h"

class Caffe2LMDBRecordReader : public Reader {
   public:
    //! Reads the Caffe2LMDB File, and loads the image ids and other necessary info
    /*!
     \param desc  User provided descriptor containing the files' path.
    */
    Reader::Status initialize(ReaderConfig desc) override;
    //! Reads the next resource item
    /*!
     \param buf User's provided buffer to receive the loaded images
     \return Size of the loaded resource
    */
    size_t read_data(unsigned char* buf, size_t max_size) override;
    //! Opens the next file in the folder
    /*!
     \return The size of the next file, 0 if couldn't access it
    */
    size_t open() override;
    //! Resets the object's state to read from the first file in the folder
    void reset() override;

    //! Returns the id of the latest file opened
    std::string id() override { return _last_id; };

    ~Caffe2LMDBRecordReader() override;

    int close() override;

    Caffe2LMDBRecordReader();

   private:
    //! opens the folder containing the images
    Reader::Status Caffe2_LMDB_reader();
    Reader::Status folder_reading();
    std::string _folder_path;
    std::string _path;
    DIR* _src_dir;
    DIR* _sub_dir;
    struct dirent* _entity;
    std::vector<std::string> _file_names;
    std::map<std::string, unsigned int> _file_size;
    unsigned _current_file_size;
    std::string _last_id;
    std::string _last_file_name;
    unsigned int _last_file_size;
    bool _last_rec;
    uint _file_byte_size;
    void incremenet_read_ptr();
    int release();
    //!< _file_count_all_shards total_number of files in to figure out the max_batch_size (usually needed for distributed training).
    void read_image(unsigned char* buff, std::string file_name);
    void read_image_names();
    std::map<std::string, uint> _image_record_starting;
    int _open_env = 1;
    MDB_env* _read_mdb_env;
    MDB_dbi _read_mdb_dbi;
    MDB_val _read_mdb_key, _read_mdb_value;
    MDB_txn* _read_mdb_txn;
    MDB_cursor* _read_mdb_cursor;
    void open_env_for_read_image();
};
