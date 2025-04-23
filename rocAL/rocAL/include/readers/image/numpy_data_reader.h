/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "pipeline/commons.h"
#include "pipeline/timing_debug.h"
#include "readers/image/image_reader.h"

class NumpyDataReader : public Reader {
   public:
    //! Looks up the folder which contains the files, and loads the numpy files
    /*!
     \param desc  User provided descriptor containing the files' path.
    */
    Reader::Status initialize(ReaderConfig desc) override;
    //! Reads the next resource item
    /*!
     \param buf User's provided buffer to receive the loaded numpy data
     \return Size of the loaded resource
    */
    size_t read_data(unsigned char* buf, size_t max_size) override;
    //! Opens the next file in the folder
    /*!
     \return The size of the next file, 0 if couldn't access it
    */
    size_t open() override;

    //! Returns the numpy header data 
    /*!
     \return The numpy header data of the current file
    */
    const NumpyHeaderData get_numpy_header_data() override;

    //! Reads the next numpy file
    /*!
     \param buf User's provided buffer to receive the loaded numpy data
     \param raad_size Requested read size by the user
     \param max_shape Max dimension strides of the output tensor used for padding along the dimensions
     \return Size of the loaded resource
    */
    size_t read_numpy_data(void* buf, size_t read_size, std::vector<unsigned>& max_shape) override;

    //! Resets the object's state to read from the first file in the folder
    void reset() override;

    //! Returns the name of the latest file opened
    std::string id() override { return _last_id; };

    //! Returns the name of the latest file_path opened
    const std::string file_path() override { return _last_file_path; }

    ~NumpyDataReader() override;

    int close() override;

    NumpyDataReader();

    //! Returns the root folder path
    std::string get_root_folder_path() override;

   private:
    //! opens the folder containing the numpy arrays
    Reader::Status open_folder();
    Reader::Status subfolder_reading();
    std::string _folder_path;
    std::string _file_list_path;
    DIR *_src_dir = nullptr;
    DIR *_sub_dir = nullptr;
    struct dirent *_entity = nullptr;
    FILE *_current_file_ptr = nullptr;
    std::vector<std::string> _file_names;
    std::vector<std::string> _files;
    std::vector<NumpyHeaderData> _file_headers;
    NumpyHeaderData _curr_file_header;
    std::string _last_id;
    std::string _last_file_name, _last_file_path, _absolute_file_path;
    unsigned _seed = 0;
    std::mutex _cache_mutex;
    std::map<std::string, NumpyHeaderData> _header_cache;
    std::shared_ptr<MetaDataReader> _meta_data_reader = nullptr;
    bool _header_parsing_failed = false;
    std::unordered_map<std::string, RocalTensorDataType> _numpy_str_to_rocal_dtype =
    {
        {"u1", RocalTensorDataType::UINT8},
        {"u4", RocalTensorDataType::UINT32},
        {"i1", RocalTensorDataType::INT8},
        {"i2", RocalTensorDataType::INT16},
        {"i4", RocalTensorDataType::INT32},
#if defined(AMD_FP16_SUPPORT)
        {"f2", RocalTensorDataType::FP16},
#endif
        {"f4", RocalTensorDataType::FP32}
    };
    //! Parse the header data and save header metadata info
    void parse_header_data(NumpyHeaderData& target, const std::string& header);
    //! Skips the specific character(s) in a char array
    template <size_t N>
    void skip_char(const char*& ptr, const char (&what)[N]);
    //! Skips the specific character(s) in a char array and returns true if present else returns false
    template <size_t N>
    bool try_skip_char(const char*& ptr, const char (&what)[N]);
    //! Skips the specific field name in the numpy header string
    template <size_t N>
    void skip_field(const char*& ptr, const char (&name)[N]);
    //! Converts the given char array to a long integer
    template <typename T = int64_t>
    T parse_int(const char*& ptr);
    //! Parses the numpy header string to fetch the data type and endianness
    std::string parse_string(const char*& input, char delim_start = '\'', char delim_end = '\'');
    //! Reads the npy file, parses the numpy header data and stores the metadata info
    void parse_header(NumpyHeaderData& parsed_header, std::string file_path);
    //! Reads the numpy data present in the npy array and stores the data in the circular buffer
    size_t parse_numpy_data(unsigned char* buf, std::vector<unsigned>& strides, std::vector<unsigned>& shapes, size_t dtype_size, unsigned dim = 0);
    //! Fetches cached header data if its already parsed before
    bool get_header_from_cache(const std::string& file_name, NumpyHeaderData& target);
    //! Stores parsed header data for a specific npy file
    void update_header_cache(const std::string& file_name, const NumpyHeaderData& value);
    void incremenet_read_ptr();
    int release();
    Reader::Status generate_file_names();            // Function that would generate _file_names containing all the samples in the dataset
};
