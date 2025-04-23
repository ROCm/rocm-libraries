/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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
#include <condition_variable>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "pipeline/commons.h"
#include "readers/image/image_reader.h"
#include "readers/image/external_source.h"
#include "pipeline/timing_debug.h"
#include "pipeline/filesystem.h"

class ExternalSourceReader : public Reader, public ExternalSourceImageReader {
   public:
    //! Looks up the folder which contains the files, amd loads the image names
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

    //! Resets the object's state to read from the first file in the list
    void reset() override;

    //! Returns the name of the latest file opened
    std::string id() override { return _last_id; }

    //! Return batch_size() for count_items unless end_of_sequence has been signalled
    unsigned count_items() override;

    ~ExternalSourceReader() override;

    int close() override;

    ExternalSourceReader();

    //! receive next set of filenames from external source
    void feed_file_names(const std::vector<std::string>& file_names, size_t num_images, bool eos = false) override;

    //! receive next set of file data from external source
    void feed_data(const std::vector<unsigned char*>& images, const std::vector<size_t>& image_size, ExternalSourceFileMode mode, bool eos = false, const std::vector<unsigned> roi_width = {}, const std::vector<unsigned> roi_height = {}, unsigned int width = 0, unsigned int height = 0, unsigned int channels = 0) override;

    // mode(): returs the mode for the reader
    ExternalSourceFileMode mode() { return _file_mode; }

    // get image_dims
    void get_dims(int cur_idx, int& width, int& height, int& channels, unsigned& roi_width, unsigned& roi_height);

   private:
    //! opens the folder containnig the images
    std::string _folder_path;
    std::queue<std::string> _file_names_queue;
    std::vector<ExternalSourceImageInfo> _file_data;
    std::queue<ExternalSourceImageInfo> _images_data_queue;
    std::mutex _lock;
    std::condition_variable _wait_for_input;

    unsigned _curr_file_idx;
    FILE* _current_fPtr;
    unsigned _current_file_size;
    std::string _last_id;
    size_t _shard_id = 0;
    size_t _shard_count = 1;  // equivalent of batch size
    //!< _batch_count Defines the quantum count of the images to be read. It's usually equal to the user's batch size.
    /// The loader will repeat images if necessary to be able to have images available in multiples of the load_batch_count,
    /// for instance if there are 10 images in the dataset and _batch_count is 3, the loader repeats 2 images as if there are 12 images available.
    size_t _batch_count = 1;
    size_t _file_id = 0;
    bool _loop;
    bool _shuffle;
    int _read_counter = 0;
    volatile bool _end_of_sequence;
    ExternalSourceFileMode _file_mode;
    //!< _file_count_all_shards total_number of files in to figure out the max_batch_size (usually needed for distributed training).
    size_t _file_count_all_shards;
    void push_file_name(const std::string& image_name);
    bool pop_file_name(std::string& file_name);
    void push_file_data(ExternalSourceImageInfo& image);
    bool pop_file_data(ExternalSourceImageInfo& image);
    void increment_read_ptr();
    int release();
    size_t get_file_shard_id();
    void increment_file_id() { _file_id++; }
    void replicate_last_image_to_fill_last_shard();
    void replicate_last_batch_to_pad_partial_shard();
};
