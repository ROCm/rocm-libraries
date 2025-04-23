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

#include <algorithm>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "readers/image/image_reader.h"
#include "pipeline/timing_debug.h"

class MXNetRecordIOReader : public Reader {
   public:
    //! Reads the MXNet Record File, and loads the image ids and other necessary info
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

    ~MXNetRecordIOReader() override;

    int close() override;

    MXNetRecordIOReader();

   private:
    //! opens the folder containing the images
    Reader::Status record_reading();
    Reader::Status MXNet_reader();
    std::string _path;
    DIR* _src_dir;
    struct dirent* _entity;
    std::string _image_key;
    std::vector<std::string> _file_names;
    std::map<std::string, std::tuple<unsigned int, int64_t, int64_t>> _record_properties;
    unsigned _current_file_size;
    std::string _last_id, _last_file_name;
    unsigned int _last_file_size;
    int64_t _last_seek_pos;
    int64_t _last_data_size;
    //!< _file_count_all_shards total_number of files in to figure out the max_batch_size (usually needed for distributed training).
    void incremenet_read_ptr();
    int release();
    void read_image(unsigned char* buff, int64_t seek_position, int64_t data_size);
    void read_image_names();
    uint32_t DecodeFlag(uint32_t rec) { return (rec >> 29U) & 7U; };
    uint32_t DecodeLength(uint32_t rec) { return rec & ((1U << 29U) - 1U); };
    std::vector<std::tuple<int64_t, int64_t>> _indices;  // used to store seek position and record size for a particular record.
    std::ifstream _file_contents;
    const uint32_t _kMagic = 0xced7230a;
    int64_t _seek_pos, _data_size_to_read;
    ImageRecordIOHeader _hdr;
};
