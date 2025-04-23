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
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <lmdb.h>
#include "meta_data/meta_data_reader.h"
#include "readers/video/video_properties.h"
#include "pipeline/tensor.h"

#define CHECK_LMDB_RETURN_STATUS(status)                                                          \
    do {                                                                                          \
        if (status != MDB_SUCCESS)                                                                \
            THROW("LMDB error, " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " + \
                  #status + ":" + std::string(mdb_strerror(status)));                             \
    } while (0)

enum class StorageType {
    FILE_SYSTEM = 0,
    TF_RECORD = 1,
    UNCOMPRESSED_BINARY_DATA = 2,  // experimental: added for supporting cifar10 data set
    CAFFE_LMDB_RECORD = 3,
    CAFFE2_LMDB_RECORD = 4,
    COCO_FILE_SYSTEM = 5,
    SEQUENCE_FILE_SYSTEM = 6,
    MXNET_RECORDIO = 7,
    VIDEO_FILE_SYSTEM = 8,
    EXTERNAL_FILE_SOURCE = 9,      // to support reading from external source
    WEBDATASET_RECORDS = 10, // tar files - webdataset format
    NUMPY_DATA = 11          // to support reading from numpy files
};

enum class ExternalSourceFileMode {
    FILENAME = 0,
    RAWDATA_COMPRESSED = 1,
    RAWDATA_UNCOMPRESSED = 2,
    NONE = 3,
};

struct ShardingInfo {
    RocalBatchPolicy last_batch_policy;
    bool pad_last_batch_repeated;
    bool stick_to_shard;
    int32_t shard_size;

    // Constructor with default values
    ShardingInfo()
        : last_batch_policy(RocalBatchPolicy::FILL),
          pad_last_batch_repeated(false),
          stick_to_shard(true),
          shard_size(-1) {}

    // Parameterized constructor
    ShardingInfo(RocalBatchPolicy policy, bool pad_repeated, bool stick, int32_t size)
        : last_batch_policy(policy),
          pad_last_batch_repeated(pad_repeated),
          stick_to_shard(stick),
          shard_size(size) {}
};

struct ReaderConfig {
    explicit ReaderConfig(StorageType type, std::string path = "", std::string json_path = "",
                          const std::map<std::string, std::string> feature_key_map = std::map<std::string, std::string>(),
                          bool shuffle = false, bool loop = false, std::string index_path = "") : _type(type), _path(path), _json_path(json_path), _feature_key_map(feature_key_map), _shuffle(shuffle), _loop(loop), _index_path(index_path) {}
    virtual StorageType type() { return _type; };
    void set_path(const std::string &path) { _path = path; }
    void set_shard_id(size_t shard_id) { _shard_id = shard_id; }
    void set_shard_count(size_t shard_count) { _shard_count = shard_count; }
    void set_cpu_num_threads(size_t cpu_num_threads) { _cpu_num_threads = cpu_num_threads; }
    void set_json_path(const std::string &json_path) { _json_path = json_path; }
    void set_index_path(const std::string &index_path) { _index_path = index_path; } // Index path - optional arg for webdataset reader - corresponding to each tar archive files
    /// \param read_batch_count Tells the reader it needs to read the images in multiples of load_batch_count. If available images not divisible to load_batch_count,
    /// the reader will repeat images to make available images an even multiple of this load_batch_count
    void set_batch_count(size_t read_batch_count) { _batch_count = read_batch_count; }
    /// \param loop if True the reader's available images still the same no matter how many images have been read
    bool shuffle() { return _shuffle; }
    bool loop() { return _loop; }
    void set_shuffle(bool shuffle) { _shuffle = shuffle; }
    void set_loop(bool loop) { _loop = loop; }
    void set_meta_data_reader(std::shared_ptr<MetaDataReader> meta_data_reader) { _meta_data_reader = meta_data_reader; }
    void set_sequence_length(unsigned sequence_length) { _sequence_length = sequence_length; }
    void set_frame_step(unsigned step) { _sequence_frame_step = step; }
    void set_frame_stride(unsigned stride) { _sequence_frame_stride = stride; }
    void set_external_filemode(ExternalSourceFileMode mode) { _file_mode = mode; }
    void set_sharding_info(const ShardingInfo& sharding_info) {
        _sharding_info = sharding_info;
    }
    void set_files_list(const std::vector<std::string> &files) { _file_names = files; }
    void set_seed(unsigned seed) { _seed = seed; }
    size_t get_shard_count() { return _shard_count; }
    size_t get_shard_id() { return _shard_id; }
    size_t get_cpu_num_threads() { return _cpu_num_threads; }
    size_t get_batch_size() { return _batch_count; }
    size_t get_sequence_length() { return _sequence_length; }
    size_t get_frame_step() { return _sequence_frame_step; }
    size_t get_frame_stride() { return _sequence_frame_stride; }
    std::vector<std::string> get_files() { return _file_names; }
    std::string path() { return _path; }
    unsigned seed() { return _seed; }
#ifdef ROCAL_VIDEO
    void set_video_properties(VideoProperties video_prop) { _video_prop = video_prop; }
    VideoProperties get_video_properties() { return _video_prop; }
#endif
    std::string json_path() { return _json_path; }
    std::string index_path() { return _index_path; }
    std::map<std::string, std::string> feature_key_map() { return _feature_key_map; }
    void set_file_prefix(const std::string &prefix) { _file_prefix = prefix; }
    std::string file_prefix() { return _file_prefix; }
    void set_file_list_path(const std::string &file_list_path) { _file_list_path = file_list_path; }
    std::string file_list_path() { return _file_list_path; }
    std::shared_ptr<MetaDataReader> meta_data_reader() { return _meta_data_reader; }
    ExternalSourceFileMode mode() { return _file_mode; }
    const ShardingInfo& get_sharding_info() { return _sharding_info; }

   private:
    StorageType _type = StorageType::FILE_SYSTEM;
    std::string _path = "";
    std::string _json_path = "";
    std::map<std::string, std::string> _feature_key_map;
    size_t _shard_count = 1;
    size_t _shard_id = 0;
    size_t _cpu_num_threads = 1;
    size_t _batch_count = 1;      //!< The reader will repeat images if necessary to be able to have images in multiples of the _batch_count.
    size_t _sequence_length = 1;  // Video reader module sequence length
    size_t _sequence_frame_step;
    size_t _sequence_frame_stride = 1;
    bool _shuffle = false;
    bool _loop = false;
    std::string _file_prefix;  //!< to read only files with prefix. supported only for cifar10_data_reader and tf_record_reader
    std::string _file_list_path;  //!< to read only files present in the file list
    std::shared_ptr<MetaDataReader> _meta_data_reader = nullptr;
    ExternalSourceFileMode _file_mode = ExternalSourceFileMode::NONE;
    ShardingInfo _sharding_info;
    std::vector<std::string> _file_names;
    unsigned _seed = 0;
#ifdef ROCAL_VIDEO
    VideoProperties _video_prop;
#endif
    std::string _index_path = "";

};

// MXNet image recordio struct - used to read the contents from the MXNet recordIO files.
struct ImageRecordIOHeader {
    uint32_t flag;        // flag of the header
    float label;          // label field that returns label of images
    uint64_t image_id[2]; /* unique image index
                           *  image_id[1] is always set to 0,
                           *  reserved for future purposes for 128bit id
                           *  image_id[0] is used to store image id
                           */
};

struct NumpyHeaderData {
   public:
    std::vector<unsigned> array_shape; // Shape of the numpy array
    RocalTensorDataType type_info;     // Dtype of the numpy array
    bool fortran_order = false;        // Indicates whether the data is present in fortran ordering
    int64_t data_offset = 0;           // Offset indicating the start of the data 

    RocalTensorDataType type() const { return type_info; }  // Returns the dtype of the numpy array

    size_t size() const {              // Returns the size of the numpy array
        size_t num_elements = 1;
        for (const auto &dim : array_shape)
            num_elements *= dim;
        return num_elements;
    };

    // Returns the entire data size of the numpy array in bytes
    size_t numpy_data_nbytes() const { return tensor_data_size(type_info) * size(); }
    
    // Returns the shape of the numpy array
    std::vector<unsigned> shape() const { return array_shape; }
};

// The VectorView Class - to refer to a portion of a vector and avoiding copies of the data.
template <typename T>
class VectorView {
    private:
    std::vector<T>* _data = nullptr;
    
    public:
    size_t start = 0;
    size_t num = 0;
    VectorView() = default;

    explicit inline VectorView(std::vector<T>& data, size_t start_idx = 0, size_t count = 0)
        : _data(&data), start(start_idx), num(count) {}

    inline T* begin() {
        return _data->data() + start;
    }

    inline T* end() {
        return begin() + num;
    }

    // Operator[] overload to access elements within the view
    T& operator[](size_t index) {
        if (index >= num) {
            throw std::out_of_range("Index out of range in VectorView");
        }
        return (*_data)[start + index];
    }
};

// The contents of the component of a tar file
struct ComponentDescription {
    std::string filename, ext;
    size_t size = 0;
    int64_t offset = 0;
    VectorView<size_t> outputs;
    ComponentDescription() = default;
};

// The contents of the sample of a tar file
struct SampleDescription {
    VectorView<ComponentDescription> components;
    VectorView<size_t> empty_outputs;
    size_t wds_shard_index;
    int64_t line_number;
};

class Reader {
   public:
    enum class Status {
        OK = 0
    };

    // TODO: change method names to open_next, read_next , ...

    //! Initializes the resource which it's spec is defined by the desc argument
    /*!
     \param desc description of the resource infor. It's exact fields are defind by the derived class.
     \return status of the being able to locate the resource pointed to by the desc
    */
    virtual Status initialize(ReaderConfig desc) = 0;
    //! Reads the next resource item
    /*!
     \param buf User's provided buffer to receive the loaded items
     \return Size of the loaded resource
    */

    //! Opens the next item and returns it's size
    /*!
     \return Size of the item, if 0 failed to access it
    */
    virtual size_t open() = 0;

    //! Copies the data of the opened item to the buf
    virtual size_t read_data(unsigned char *buf, size_t read_size) = 0;

    //! Returns the numpy header data information used containing shape, size and dtype
    virtual const NumpyHeaderData get_numpy_header_data() { return {}; }

    //! Reads the data present in numpy arrays to the buffer
    virtual size_t read_numpy_data(void *buf, size_t read_size, std::vector<unsigned>& strides_in_dims) { return 0; }

    //! Closes the opened item
    virtual int close() = 0;

    //! Starts reading from the first item in the resource
    virtual void reset() = 0;

    //! Returns the name/identifier of the last item opened in this resource
    virtual std::string id() = 0;
    //! Returns the number of items remained in this resource

     //! Returns the path of the last item opened in this resource
    virtual const std::string file_path() { THROW("File path is not set by the reader") }

    virtual unsigned count_items();

    virtual ~Reader() = default;

    virtual std::string get_root_folder_path() { return {}; }

    virtual std::vector<std::string> get_file_paths_from_meta_data_reader() { return {}; }

    //! Returns the number of images in the last batch
    size_t last_batch_padded_size() { return _last_batch_padded_size; }

   protected:
    ShardingInfo _sharding_info = ShardingInfo();  // The members of ShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
    std::vector<unsigned> _shard_start_idx_vector, _shard_end_idx_vector;   // Holds the start and end idx of the file names vector for each shard
    unsigned _curr_file_idx = 0;    // Tracks the current file idx being processed
    size_t _file_count_all_shards;  // Total number of files present in all shards
    size_t _last_batch_padded_size = 0; // Number of files padded in the last batch
    size_t _shard_id = 0;   // Shard ID of the current shard
    size_t _shard_count = 1;    // Total number of shards in the pipeline
    bool _stick_to_shard = false;   // Determines whether the reader should stick to a data shard instead of going through the entire dataset.
    bool _pad_last_batch_repeated = false;  // Determines if last file is to be repeated for padding the batch.
    int32_t _shard_size = -1;   // Size of the shard
    size_t _batch_size = 1;
    bool _loop;
    bool _shuffle;
    int _read_counter = 0;

    //! Modified the file idx, and sets the current file idx to be processed
    void increment_curr_file_idx(size_t dataset_size);

    //! Increments the shard id, to process data from next shard
    void increment_shard_id();

    //! Computes the start and end index of the file names vector for each shard
    void compute_start_and_end_idx_of_all_shards();

    //! Returns the shard size without padding
    size_t actual_shard_size_without_padding();

    //! Returns the largest shard size without padding
    size_t largest_shard_size_without_padding();

    //! Returns the maximum size of the current shard
    size_t get_max_size_of_shard(size_t batch_size, bool loop);

    //! Modifies the file names vector with files to be padded
    void update_filenames_with_padding(std::vector<std::string> &file_names, size_t batch_size);
};
