/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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
#ifdef ENABLE_WDS
#include "meta_data/webdataset_meta_data_reader.h"
#include "pipeline/timing_debug.h"
#include "readers/image/image_reader.h"

class WebDatasetSourceReader : public Reader {
  public:
    //! Reads the TFRecord File, and loads the image ids and other necessary
    //! info
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

    //! Returns the id of the latest file opened
    std::string id() override { return _last_id; };

    ~WebDatasetSourceReader() override;

    int close() override;

    WebDatasetSourceReader();

  private:
    //! opens the folder containnig the images
    Reader::Status webdataset_record_reader();
    Reader::Status folder_reading();
    std::string _folder_path;
    std::string _path;
    std::string _paths, _index_paths;
    DIR *_sub_dir = nullptr;
    struct dirent *_entity = nullptr;
    std::vector<std::string> _file_names, _all_shard_file_names_padded;
    std::map<std::string, unsigned int> _file_wds_shard_idx_mapping, _file_size, _file_offset;
    unsigned _current_file_size;
    std::string _last_id;
    std::string _last_file_name;
    std::vector<std::string> _index_name_list;
    void incremenet_read_ptr();
    int release();
    void parse_tar_files(std::vector<SampleDescription> &samples_container, std::vector<ComponentDescription> &components_container, std::unique_ptr<std::ifstream> &tar_file);
    void parse_index_files(std::vector<SampleDescription> &samples_container,
                      std::vector<ComponentDescription> &components_container,
                      const std::string &index_path);
    void parse_sample_description(
        std::vector<SampleDescription> &samples_container,
        std::vector<ComponentDescription> &components_container,
        std::ifstream &index_file, const std::string &index_path, int64_t line,
        int index_version);
    Reader::Status webdataset_record_reader_from_components(ComponentDescription component, unsigned wds_shard_index);
    std::shared_ptr<MetaDataReader> _meta_data_reader = nullptr;
    std::vector<std::unique_ptr<std::ifstream>> _wds_shards;
    Reader::Status read_web_dataset_at_offset(unsigned char *buff,
                                              std::string file_name,
                                              uint file_size, uint offset,
                                              uint wds_shard_index);
    void increment_shard_id();
};
#endif
