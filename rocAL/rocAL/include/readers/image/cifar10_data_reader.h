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

#include <memory>
#include <string>
#include <vector>

#include "readers/image/image_reader.h"

class CIFAR10DataReader : public Reader {
   public:
    //! Looks up the folder which contains the CIFAR10 training/test data which is uncompressed, amd makes up image names
    /*!
     \param desc  User provided descriptor containing the files' path.
    */
    Reader::Status initialize(ReaderConfig desc) override;
    //! Reads the next resource item
    /*!
     \param buf User's provided buffer to receive the loaded images
     \return Size of the loaded resource
    */
    size_t read_data(unsigned char *buf, size_t max_size) override;
    //! Opens the next file in the folder
    /*!
     \return The size of the next file, 0 if couldn't access it
    */
    size_t open() override;

    //! Resets the object's state to read from the first file in the folder
    void reset() override;

    //! Returns the name of the latest data_id opened
    std::string id() override { return _last_id; };

    ~CIFAR10DataReader() override;

    int close() override;

    CIFAR10DataReader();

    unsigned get_file_index() { return _last_file_idx; }

   private:
    //! opens the folder containing the images
    Reader::Status open_folder();
    Reader::Status subfolder_reading();
    std::string _folder_path;
    DIR *_src_dir;
    DIR *_sub_dir;
    struct dirent *_entity;
    std::vector<std::string> _file_names;
    std::vector<unsigned> _file_offsets;
    std::vector<unsigned> _file_idx;
    FILE *_current_fPtr;
    unsigned _current_file_size;
    std::string _last_id;
    std::string _last_file_name;
    unsigned _last_file_idx;  // index of individual raw file in a batched file
    // hard_coding the following for now. Eventually needs to add in the ReaderConfig
    //!< file_name_prefix tells the reader to read only files with the prefix:: eventually needs to be passed through ReaderConfig
    std::string _file_name_prefix;  // = "data_batch_";
    //!< _raw_file_size of each file to read
    const size_t _raw_file_size = (32 * 32 * 3 + 1);  // todo:: need to add an option in reader config to take this.
    size_t _total_file_size;
    void incremenet_read_ptr();
    int release();
};
