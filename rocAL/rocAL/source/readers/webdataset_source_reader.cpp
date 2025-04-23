
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

#ifdef ENABLE_WDS
#include "readers/webdataset_source_reader.h"
#include <cassert>

using namespace std;

#define BLOCKSIZE 10240

constexpr int create_version_number(int major, int minor, int patch = 0) {
  if (major < 0 || minor < 0 || patch < 0) {
    return -1;
  }
    return major * 1000 + minor * 10 + patch;
}

inline std::tuple<std::string, std::string> split_name(const std::string& file_path) {
  size_t dot_pos = file_path.find('.', file_path.rfind('/') + 1);
  return {file_path.substr(0, dot_pos), file_path.substr(dot_pos + 1)};
}

WebDatasetSourceReader::WebDatasetSourceReader() {
    _curr_file_idx = 0;
    _current_file_size = 0;
    _loop = false;
    _shuffle = false;
    _file_count_all_shards = 0;
}

Reader::Status WebDatasetSourceReader::initialize(ReaderConfig desc) {
    auto ret = Reader::Status::OK;
    _folder_path = desc.path();
    _path = desc.path();
    _index_paths = desc.index_path();
    _wds_shards.reserve(_path.size());
    _shard_id = desc.get_shard_id();
    _shard_count = desc.get_shard_count();
    _batch_size = desc.get_batch_size();
    _loop = desc.loop();
    _meta_data_reader = desc.meta_data_reader();
    _sharding_info = desc.get_sharding_info();
    _pad_last_batch_repeated = _sharding_info.pad_last_batch_repeated;
    _stick_to_shard = _sharding_info.stick_to_shard;
    _shard_size = _sharding_info.shard_size;
    _shuffle = desc.shuffle();
    ret = folder_reading();
    _curr_file_idx = _shard_start_idx_vector[_shard_id]; // shard's start_idx would vary for every shard in the vector
    // shuffle dataset if set
    if (ret == Reader::Status::OK && _shuffle)
        std::random_shuffle(_file_names.begin() + _shard_start_idx_vector[_shard_id],
                            _file_names.begin() + _shard_end_idx_vector[_shard_id]);
    return ret;
}

void WebDatasetSourceReader::incremenet_read_ptr() {
    _read_counter++;
    increment_curr_file_idx(_file_names.size());
}

size_t WebDatasetSourceReader::open() {
    auto file_path = _file_names[_curr_file_idx];  // Get next file name
    _last_id = file_path;
    auto last_slash_idx = _last_id.find_last_of("\\/");
    if (std::string::npos != last_slash_idx) {
        _last_id.erase(0, last_slash_idx + 1);
    }
    _current_file_size = _file_size[_file_names[_curr_file_idx]];
    return _current_file_size;
}

size_t WebDatasetSourceReader::read_data(unsigned char* buf, size_t read_size) {
    auto ret = read_web_dataset_at_offset(buf, _file_names[_curr_file_idx], _file_size[_file_names[_curr_file_idx]], _file_offset[_file_names[_curr_file_idx]], _file_wds_shard_idx_mapping[_file_names[_curr_file_idx]]);
    if (ret != Reader::Status::OK)
        THROW("WebDatasetSourceReader: Error in reading tar records of the web  dataset reader");
    incremenet_read_ptr();
    return read_size;
}

int WebDatasetSourceReader::close() {
    return release();
}

WebDatasetSourceReader::~WebDatasetSourceReader() {
    release();
}

int WebDatasetSourceReader::release() {
    return 0;
}

void WebDatasetSourceReader::reset() {
    if (_shuffle)
        std::random_shuffle(_file_names.begin() + _shard_start_idx_vector[_shard_id],
                            _file_names.begin() + _shard_start_idx_vector[_shard_id] + actual_shard_size_without_padding());

    if (_stick_to_shard == false)  // Pick elements from the next shard - hence increment shard_id
        increment_shard_id();      // Should work for both single and multiple shards

    _read_counter = 0;

    if (_sharding_info.last_batch_policy == RocalBatchPolicy::DROP) {  // Skipping the dropped batch in next epoch
        for (uint i = 0; i < _batch_size; i++)
            increment_curr_file_idx(_file_names.size());
    }
}

void WebDatasetSourceReader::increment_shard_id() {
    _shard_id = (_shard_id + 1) % _shard_count;
}

void WebDatasetSourceReader::parse_sample_description(
    std::vector<SampleDescription>& samples_container,
    std::vector<ComponentDescription>& components_container,
    std::ifstream& index_file, const std::string& index_path, int64_t line,
    int index_version) {
    samples_container.emplace_back();
    samples_container.back().components = VectorView<ComponentDescription>(
        components_container, components_container.size());
    samples_container.back().line_number = line;

    // Getting the components data
    std::string components_metadata;
    std::getline(index_file, components_metadata);
    std::stringstream components_stream(components_metadata);

    // Reading consecutive components
    ComponentDescription component;
    while (components_stream >> component.ext) {
        if (index_version == create_version_number(1, 0)) {
            if (!(components_stream >> component.offset >> component.size >>
                  component.filename)) {
                THROW( "Could not find all necessary component parameters "
                      "(offset, size or filename). Every record in the index "
                      "file should look like: `<ext> <offset> <size> "
                      "<filename>`." );
            }
        } else {
            if (!(components_stream >> component.offset >> component.size))
                THROW( "Could not find all necessary component parameters "
                      "(offset or size). Every record in the index file should "
                      "look like: `<ext> <offset> <size>`" );
        }

        if (component.filename.empty())  // Use line number as file number
            component.filename = std::to_string(line);
        else {
            // Find the position of the last period
            auto last_period_pos = component.filename.find_last_of('.');
            // If a period is found, truncate everything after it
            if (last_period_pos != std::string::npos) {
                component.filename.erase(last_period_pos);
            }
        }

        if (!(component.offset % kBlockSize == 0))
            THROW( "tar offset is not a multiple of tar block size kBlockSize, "
                  "perhaps the size value is exported before offset?" );

        components_container.emplace_back(std::move(component));
        samples_container.back().components.num++;
    }

    if ((!samples_container.back().components.num))
        THROW("No extensions provided for the sample");
}

inline int parse_index_version(const string& idx_version_in_str) {
  const char* c_string_ptr = idx_version_in_str.c_str();
  assert(*c_string_ptr == 'v');
  c_string_ptr++;
  auto major = atoi(c_string_ptr);
  c_string_ptr = strchr(c_string_ptr, '.');
  assert(c_string_ptr);
  c_string_ptr++;
  auto minor = atoi(c_string_ptr);
  return create_version_number(major, minor);
}

void WebDatasetSourceReader::parse_index_files(
    std::vector<SampleDescription>& samples_container,
    std::vector<ComponentDescription>& components_container,
    const std::string& index_path) {
    std::ifstream index_file(index_path);
    std::string global_meta;
    getline(index_file, global_meta);
    std::stringstream global_meta_stream(global_meta);
    std::string index_version_str;
    if (!(global_meta_stream >> index_version_str))
        THROW("Unsupported version of the index file")

    int index_version = parse_index_version(index_version_str);

    int64_t sample_desc_num_signed;
    if (!(global_meta_stream >> sample_desc_num_signed))
        THROW("no sample count found")
    if (!(sample_desc_num_signed > 0))
        THROW("sample count must be positive")

    const size_t sample_desc_num = sample_desc_num_signed;
    samples_container.reserve(samples_container.size() + sample_desc_num);
    for (size_t sample_index = 0; sample_index < sample_desc_num;
         sample_index++) {
        parse_sample_description(samples_container, components_container, index_file,
                        index_path, sample_index + 1, index_version);
    }
}

void WebDatasetSourceReader::parse_tar_files(std::vector<SampleDescription>& samples_vector,
                                              std::vector<ComponentDescription>& components_vector,
                                              std::unique_ptr<std::ifstream>& tar_file) {
    TarArchive tar_archive(std::move(tar_file));

    std::string last_filename;
    for (; !tar_archive.at_end_of_archive(); tar_archive.advance_to_next_file_in_tar()) {
        if (tar_archive.get_current_file_type() == TarArchive::ENTRY_FILE) {
            std::tie(last_filename, std::ignore) = split_name(tar_archive.get_current_file_name());
            break;
        }
    }
    size_t last_components_size = components_vector.size();
    for (; !tar_archive.at_end_of_archive(); tar_archive.advance_to_next_file_in_tar()) {
        if (tar_archive.get_current_file_type() != TarArchive::ENTRY_FILE) {
            continue;
        }

        std::string basename, ext;
        std::tie(basename, ext) = split_name(tar_archive.get_current_file_name());
        if (basename.empty()) {
            continue;
        }

        if (basename != last_filename) {
            samples_vector.emplace_back();
            samples_vector.back().components = VectorView<ComponentDescription>(components_vector, last_components_size, components_vector.size() - last_components_size);
            last_filename = basename;
            last_components_size = components_vector.size();
        }
        components_vector.emplace_back();
        components_vector.back().size = tar_archive.get_current_file_size();
        components_vector.back().offset = tar_archive.get_current_archive_offset() + tar_archive.get_current_header_size();
        components_vector.back().ext = std::move(ext);
        auto _last_id = basename;
        auto last_slash_idx = _last_id.find_last_of("\\/");
        if (std::string::npos != last_slash_idx) {
            _last_id.erase(0, last_slash_idx + 1);
        }
        components_vector.back().filename = _last_id;
    }
    samples_vector.emplace_back();
    samples_vector.back().components = VectorView<ComponentDescription>(components_vector, last_components_size, components_vector.size() - last_components_size);

    tar_file = tar_archive.release_file_stream();
}

Reader::Status WebDatasetSourceReader::folder_reading() {
    auto ret = Reader::Status::OK;
    std::string _full_path;
    std::vector<std::string> entry_name_list;
    if (_index_paths.empty()) { 
        _folder_path = _path;
        if ((_sub_dir = opendir(_folder_path.c_str())) == nullptr)
            THROW("WebDatasetSourceReader ShardID [" + TOSTR(_shard_id) + "] ERROR: Failed opening the directory at " + _folder_path);
        _full_path = _folder_path;  
        while ((_entity = readdir(_sub_dir)) != nullptr) {
            std::string entry_name(_entity->d_name);
            if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0)
                continue;
            entry_name_list.push_back(entry_name);
        }
        std::sort(entry_name_list.begin(), entry_name_list.end());
        _wds_shards.reserve(entry_name_list.size());
        // Create n such std-streams for n paths
        for (auto& path : entry_name_list)
         {
            auto file = std::make_unique<std::ifstream>(_path + path, std::ios::binary);
            if (!file->is_open()) {
                std::cerr << "Failed to open file: " << _path + path << std::endl;
            } else {
                _wds_shards.emplace_back(std::move(file));
            }
        }
    } else {
        _folder_path = _index_paths;
        if ((_sub_dir = opendir(_folder_path.c_str())) == nullptr)
            THROW("WebDatasetSourceReader ShardID [" + TOSTR(_shard_id) + "] ERROR: Failed opening the directory at " + _folder_path);
        _full_path = _folder_path;
        while ((_entity = readdir(_sub_dir)) != nullptr) {
            std::string entry_name(_entity->d_name);
            if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0)
                continue;
            _index_name_list.push_back(entry_name);
        }
        std::sort(_index_name_list.begin(), _index_name_list.end());
        // Tar file path
        if ((_sub_dir = opendir(_path.c_str())) == nullptr)
            THROW("WebDatasetSourceReader ShardID [" + TOSTR(_shard_id) + "] ERROR: Failed opening the directory at " + _path);
        std::string _full_path = _path;
        while ((_entity = readdir(_sub_dir)) != nullptr) {
            std::string entry_name(_entity->d_name);
            if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0)
                continue;
            entry_name_list.push_back(entry_name);
        }
        std::sort(entry_name_list.begin(), entry_name_list.end());
        _wds_shards.reserve(entry_name_list.size());
        // Create n such std-streams for n paths
        for (auto& path : entry_name_list) {
            auto file = std::make_unique<std::ifstream>(_path + path, std::ios::binary);
            if (!file->is_open()) {
                std::cerr << "Failed to open file: " << _path + path << std::endl;
            } else {
                _wds_shards.emplace_back(std::move(file));
            }
        }
    }
    std::vector<SampleDescription> unfiltered_samples;
    std::vector<ComponentDescription> unfiltered_components;
    
    for (unsigned wds_shard_index = 0; wds_shard_index < entry_name_list.size(); ++wds_shard_index) {
            unfiltered_samples.resize(0);
            unfiltered_components.resize(0);
            if (_index_paths.size() == 0)
            parse_tar_files(unfiltered_samples, unfiltered_components, _wds_shards[wds_shard_index]);
            else
            parse_index_files(unfiltered_samples, unfiltered_components, _folder_path + _index_name_list[wds_shard_index]);
            // After parsing add the contents to the map
        for (auto& sample : unfiltered_samples) {
            for (auto& component : sample.components) {
                if (!_meta_data_reader || _meta_data_reader->exists(component.filename)) {
                    if (webdataset_record_reader_from_components(component, wds_shard_index) != Reader::Status::OK)
                        WRN("WebDatasetSourceReader ShardID [" + TOSTR(_shard_id) + "] WebDataset File reader cannot access the storage at " + _folder_path + component.filename);
                }
            }
        }
    }

    size_t padded_samples = ((_shard_size > 0) ? _shard_size : largest_shard_size_without_padding()) % _batch_size;
    _last_batch_padded_size = ((_batch_size > 1) && (padded_samples > 0)) ? (_batch_size - padded_samples) : 0;

    // Pad the _file_names with last element of the shard in the vector when _pad_last_batch_repeated is True
    if (_pad_last_batch_repeated == true) {
        update_filenames_with_padding(_file_names, _batch_size);
    }
    _last_file_name = _file_names[_file_names.size() - 1];
     compute_start_and_end_idx_of_all_shards();
    
    return ret;
}

inline bool isJPEG(const std::string& ext) {
    std::string lowerExt = ext;
    std::transform(lowerExt.begin(), lowerExt.end(), lowerExt.begin(), ::tolower);
    return lowerExt == "jpg" || lowerExt == "jpeg" || lowerExt == "jpe";
}

Reader::Status WebDatasetSourceReader::webdataset_record_reader_from_components(ComponentDescription component, unsigned wds_shard_index) {
    auto ret = Reader::Status::OK;
    if (isJPEG(component.ext)) {
        // Update file path and size
        std::string file_path = _folder_path;
        file_path.append("/");
        file_path.append(component.filename);
        _file_names.push_back(file_path);
        _last_file_name = file_path;
        _file_size.insert(std::pair<std::string, unsigned int>(file_path, component.size));
        _file_wds_shard_idx_mapping.insert(std::pair<std::string, unsigned int>(file_path, wds_shard_index));
        _file_offset.insert(std::pair<std::string, off_t>(file_path, component.offset));
        _file_count_all_shards++;
    }  // Case for jpg's. - add for more extensions when encoutered
    return ret;
}

Reader::Status WebDatasetSourceReader::read_web_dataset_at_offset(unsigned char* buff, std::string file_name, uint file_size, uint offset, uint wds_shard_index) {
    auto ret = Reader::Status::OK;
    auto& current_tar_file_stream = _wds_shards[wds_shard_index];
    current_tar_file_stream->seekg(offset, std::ios::beg);
    current_tar_file_stream->read(reinterpret_cast<char*>(buff), file_size);
    return ret;
}
#endif
