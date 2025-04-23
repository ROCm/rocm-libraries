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
#include <list>
#include <map>
#include <memory>
#include <variant>

#include "pipeline/graph.h"
#include "meta_data/meta_data_graph.h"
#include "meta_data/meta_data_reader.h"
#include "pipeline/node.h"
#include "loaders/image/node_cifar10_loader.h"
#include "loaders/image/node_fused_jpeg_crop.h"
#include "loaders/image/node_fused_jpeg_crop_single_shard.h"
#include "loaders/image/node_image_loader.h"
#include "loaders/image/node_image_loader_single_shard.h"
#include "loaders/video/node_video_loader.h"
#include "loaders/video/node_video_loader_single_shard.h"
#include "loaders/image/node_numpy_loader.h"
#include "loaders/image/node_numpy_loader_single_shard.h"
#ifdef ROCAL_AUDIO
#include "loaders/audio/node_audio_loader.h"
#include "loaders/audio/node_audio_loader_single_shard.h"
#endif
#include "pipeline/ring_buffer.h"
#include "pipeline/timing_debug.h"
#if ENABLE_HIP
#include "box_encoder_hip.h"
#include "device/device_manager_hip.h"
#endif
#include "meta_data/randombboxcrop_meta_data_reader.h"
#include "rocal_api_types.h"
#define MAX_STRING_LENGTH 100
#define MAX_OBJECTS 50                // Setting an arbitrary value 50.(Max number of objects/image in COCO dataset is 93)
#define BBOX_COUNT 4
#define MAX_SSD_ANCHORS 8732          // Num of bbox achors used in SSD training
#define MAX_MASK_BUFFER 10000
#define MAX_RETINANET_ANCHORS 120087  // Num of bbox achors used in Retinanet training
#define MAX_ASCII_BUFFER 200        // Max Number of ASCII characters that can be present in any particular extension file for webdataset reader

#if ENABLE_SIMD
#if _WIN32
#include <intrin.h>
#else
#include <immintrin.h>
#include <smmintrin.h>
#include <x86intrin.h>
#endif
#endif

#if (ENABLE_SIMD && __AVX2__)
const __m256i avx_pkdMaskR = _mm256_setr_epi32(0x80808000, 0x80808003, 0x80808006, 0x80808009, 0x80808000,
                                               0x80808003, 0x80808006, 0x80808009);
const __m256i avx_pkdMaskG = _mm256_setr_epi32(0x80808001, 0x80808004, 0x80808007, 0x8080800A, 0x80808001,
                                               0x80808004, 0x80808007, 0x8080800A);
const __m256i avx_pkdMaskB = _mm256_setr_epi32(0x80808002, 0x80808005, 0x80808008, 0x8080800B, 0x80808002,
                                               0x80808005, 0x80808008, 0x8080800B);
#endif

class MasterGraph {
   public:
    enum class Status { OK = 0,
                        NOT_RUNNING = 1,
                        NO_MORE_DATA = 2,
                        NOT_IMPLEMENTED = 3,
                        INVALID_ARGUMENTS };
    MasterGraph(size_t batch_size, RocalAffinity affinity, size_t cpu_thread_count, int gpu_id, size_t prefetch_queue_depth, RocalTensorDataType output_tensor_data_type);
    ~MasterGraph();
    Status reset();
    size_t remaining_count();
    MasterGraph::Status to_tensor(void *out_ptr, RocalTensorlayout format, float multiplier0, float multiplier1, float multiplier2,
                                  float offset0, float offset1, float offset2, bool reverse_channels, RocalTensorDataType output_data_type, RocalOutputMemType output_mem_type, uint max_roi_height = 0, uint max_roi_width = 0);
    Status copy_output(unsigned char *out_ptr, size_t out_size_in_bytes);
    Status copy_out_tensor_planar(void *out_ptr, RocalTensorlayout format, float multiplier0, float multiplier1, float multiplier2,
                                  float offset0, float offset1, float offset2, bool reverse_channels, RocalTensorDataType output_data_type);
    TensorList *get_output_tensors();
    size_t output_width();
    size_t output_height();
    void sequence_start_frame_number(std::vector<size_t> &sequence_start_framenum);             // Returns the starting frame number of the sequences
    void sequence_frame_timestamps(std::vector<std::vector<float>> &sequence_frame_timestamp);  // Returns the timestamps of the frames in the sequences
    size_t augmentation_branch_count();                                                         // Returns the number of output tensors from the pipeline
    RocalColorFormat output_color_format();
    Status build();
    Status run();
    Timing timing();
    RocalMemType mem_type();
    size_t last_batch_padded_size();
    void release();
    vx_context get_vx_context() { return _context; }
    template <typename T>
    std::shared_ptr<T> add_node(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    template <typename T, typename M>
    std::shared_ptr<T> meta_add_node(std::shared_ptr<M> node);
    Tensor *create_tensor(const TensorInfo &info, bool is_output);
    Tensor *create_loader_output_tensor(const TensorInfo &info);
    TensorListVector * create_label_reader(const char *source_path, MetaDataReaderType reader_type);
    TensorListVector * create_video_label_reader(const char *source_path, MetaDataReaderType reader_type, unsigned sequence_length, unsigned frame_step, unsigned frame_stride, bool file_list_frame_num = true);
    TensorListVector * create_coco_meta_data_reader(const char *source_path, bool is_output, MetaDataReaderType reader_type, MetaDataType label_type, bool ltrb_bbox = true, bool is_box_encoder = false,
                                                    bool avoid_class_remapping = false, bool aspect_ratio_grouping = false, bool is_box_iou_matcher = false, float sigma = 0.0, unsigned pose_output_width = 0, unsigned pose_output_height = 0);
    TensorListVector * create_tf_record_meta_data_reader(const char *source_path, MetaDataReaderType reader_type, MetaDataType label_type, const std::map<std::string, std::string> feature_key_map);
    TensorListVector * create_caffe_lmdb_record_meta_data_reader(const char *source_path, MetaDataReaderType reader_type, MetaDataType label_type);
    TensorListVector * create_caffe2_lmdb_record_meta_data_reader(const char *source_path, MetaDataReaderType reader_type, MetaDataType label_type);
    TensorListVector * create_cifar10_label_reader(const char *source_path, const char *file_prefix);
    TensorListVector * create_mxnet_label_reader(const char *source_path, bool is_output);
    TensorListVector * create_webdataset_reader(const char *source_path, const char* index_path, std::vector<std::set<std::string>> extensions , MetaDataReaderType reader_type, MissingComponentsBehaviour missing_component_behaviour);
    void box_encoder(std::vector<float> &anchors, float criteria, const std::vector<float> &means, const std::vector<float> &stds, bool offset, float scale);
    void box_iou_matcher(std::vector<float> &anchors, float high_threshold, float low_threshold, bool allow_low_quality_matches);
    void create_randombboxcrop_reader(RandomBBoxCrop_MetaDataReaderType reader_type, RandomBBoxCrop_MetaDataType label_type, bool all_boxes_overlap, bool no_crop, FloatParam *aspect_ratio, bool has_shape, int crop_width, int crop_height, int num_attempts, FloatParam *scaling, int total_num_attempts, int64_t seed = 0);
    const std::pair<ImageNameBatch, pMetaDataBatch> &meta_data();
    TensorList *labels_meta_data();
    TensorList *bbox_meta_data();
    TensorList *mask_meta_data();
    TensorList *matched_index_meta_data();
    TensorListVector * ascii_values_meta_data(); // Gets the pointer to a batch of ASCII values of all samples in the batch
    void set_loop(bool val) { _loop = val; }
    void set_output(Tensor *output_tensor);
    size_t calculate_cpu_num_threads(size_t shard_count);
    bool empty() { return (remaining_count() < (_is_sequence_reader_output ? _sequence_batch_size : _user_batch_size)); }
    size_t sequence_batch_size() { return _sequence_batch_size; }
    std::shared_ptr<MetaDataGraph> meta_data_graph() { return _meta_data_graph; }
    std::shared_ptr<MetaDataReader> meta_data_reader() { return _meta_data_reader; }
    bool is_random_bbox_crop() { return _is_random_bbox_crop; }
    bool is_sequence_reader_output() { return _is_sequence_reader_output; }
    void set_sequence_reader_output() { _is_sequence_reader_output = true; }
    void set_sequence_batch_size(size_t sequence_length) { _sequence_batch_size = _user_batch_size * sequence_length; }
    TensorListVector * get_bbox_encoded_buffers(size_t num_encoded_boxes);
    void feed_external_input(const std::vector<std::string>& input_images_names, bool labels, const std::vector<unsigned char *>& input_buffer,
                             const std::vector<ROIxywh>& roi_xywh, unsigned int max_width, unsigned int max_height, unsigned int channels, ExternalSourceFileMode mode,
                             RocalTensorlayout layout, bool eos);
    void set_external_source_reader_flag() { _external_source_reader = true; }
    size_t bounding_box_batch_count(pMetaDataBatch meta_data_batch);
#if ENABLE_OPENCL
    cl_command_queue get_ocl_cmd_q() { return _device.resources()->cmd_queue; }
#endif
   private:
    Status update_node_parameters();
    void create_single_graph();
    void start_processing();
    void stop_processing();
    void output_routine();
    void decrease_image_count();
    /// notify_user_thread() is called when the internal processing thread is done with processing all available tensors
    void notify_user_thread();
    /// no_more_processed_data() is logically linked to the notify_user_thread() and is used to tell the user they've already consumed all the processed tensors
    bool no_more_processed_data();
    RingBuffer _ring_buffer;                                                      //!< The queue that keeps the tensors that have benn processed by the internal thread (_output_thread) asynchronous to the user's thread
    pMetaDataBatch _augmented_meta_data = nullptr;                                //!< The output of the meta_data_graph,
    std::shared_ptr<CropCordBatch> _random_bbox_crop_cords_data = nullptr;
    std::thread _output_thread;
    TensorList _internal_tensor_list;                                             //!< Keeps a list of ovx tensors that are used to store the augmented outputs (there is an augmentation output batch per element in the list)
    TensorList _output_tensor_list;                                               //!< Keeps a list of ovx tensors(augmented outputs) that are to be passed to the user (there is an augmentation output batch per element in the list)
    std::list<Tensor *> _internal_tensors;                                        //!< Keeps all the ovx tensors (virtual/non-virtual) either intermediate tensors, or input tensors that feed the graph
    std::list<std::shared_ptr<Node>> _nodes;                                      //!< List of all the nodes
    std::list<std::shared_ptr<Node>> _root_nodes;                                 //!< List of all root nodes (image/video loaders)
    std::list<std::shared_ptr<Node>> _meta_data_nodes;                            //!< List of nodes where meta data has to be updated after augmentation
    std::map<Tensor *, std::shared_ptr<Node>> _tensor_map;                        //!< key: tensor, value : Parent node
    void *_output_tensor_buffer = nullptr;                                        //!< In the GPU processing case , is used to convert the U8 samples to float32 before they are being transfered back to host
    TensorListVector _metadata_output_tensor_list;                                //!< Keeps a list of all the Metadata output TensorList
    TensorListVector _bbox_encoded_output;                                        //!< Keeps a list of label and bounding box metadata TensorList for box encoder
    TensorListVector _webdataset_output_tensor_list;                              //!< Keeps a list of ascii metadata TensorList for the Webdataset reader
    TensorList _labels_tensor_list;
    std::vector<TensorList> _ascii_tensor_list; // TensorList to store the ASCII values of all samples in a batch
    TensorList _bbox_tensor_list;
    TensorList _mask_tensor_list;
    TensorList _matches_tensor_list;                                                  //!< The count of total number of extensions used in the webdataset reader
    std::vector<size_t> _meta_data_buffer_size;
#if ENABLE_HIP
    DeviceManagerHip _device;                                                     //!< Keeps the device related constructs needed for running on GPU
#elif ENABLE_OPENCL
    DeviceManager _device;                                                        //!< Keeps the device related constructs needed for running on GPU
#endif
    std::shared_ptr<Graph> _graph = nullptr;
    RocalAffinity _affinity;
    size_t _cpu_num_threads;                                                      //!< Defines the number of CPU threads used for processing
    const int _gpu_id;                                                            //!< Defines the device id used for processing
    pLoaderModule _loader_module;                                                 //!< Keeps the loader module used to feed the input the tensors of the graph
    TimingDbg _convert_time, _process_time, _bencode_time;
    const size_t _user_batch_size;                                                //!< Batch size provided by the user
    vx_context _context;
    const RocalMemType _mem_type;                                                 //!< Is set according to the _affinity, if GPU, is set to CL, otherwise host
    std::shared_ptr<MetaDataReader> _meta_data_reader = nullptr;
    std::shared_ptr<MetaDataGraph> _meta_data_graph = nullptr;
    std::shared_ptr<RandomBBoxCrop_MetaDataReader> _randombboxcrop_meta_data_reader = nullptr;
    bool _first_run = true;
    bool _processing;                                                             //!< Indicates if internal processing thread should keep processing or not
    const static unsigned SAMPLE_SIZE = sizeof(unsigned char);
    int _remaining_count;                                                         //!< Keeps the count of remaining tensors yet to be processed for the user,
    bool _loop;                                                                   //!< Indicates if user wants to indefinitely loops through tensors or not
    size_t _prefetch_queue_depth;
    bool _output_routine_finished_processing = false;
    bool _is_random_bbox_crop = false;
    std::vector<std::vector<size_t>> _sequence_start_framenum_vec;                //!< Stores the starting frame number of the sequences.
    std::vector<std::vector<std::vector<float>>> _sequence_frame_timestamps_vec;  //!< Stores the timestamps of the frames in a sequences.
    size_t _sequence_batch_size = 0;                                              //!< Indicates the _user_batch_size when sequence reader outputs are required
    bool _is_sequence_reader_output = false;                                      //!< Set to true if Sequence Reader is invoked.
    // box encoder variables
    bool _is_box_encoder = false;                                                 // bool variable to set the box encoder
    std::vector<float> _anchors;                                                  // Anchors to be used for encoding, as the array of floats is in the ltrb format of size 8732x4
    size_t _num_anchors;                                                          // number of bbox anchors
    float _criteria = 0.5;                                                        // Threshold IoU for matching bounding boxes with anchors. The value needs to be between 0 and 1.
    float _scale;                                                                 // Rescales the box and anchor values before the offset is calculated (for example, to return to the absolute values).
    bool _offset;                                                                 // Returns normalized offsets ((encoded_bboxes*scale - anchors*scale) - mean) / stds in EncodedBBoxes that use std and the mean and scale arguments if offset="True"
    std::vector<float> _means, _stds;                                             //_means:  [x y w h] mean values for normalization _stds: [x y w h] standard deviations for offset normalization.
    bool _augmentation_metanode = false;
    bool _external_source_eos = false;     // If last batch, _external_source_eos will true
    bool _external_source_reader = false;  // Set to true if external source reader on
    // box IoU matcher variables
    bool _is_box_iou_matcher = false;                                             // bool variable to set the box iou matcher
    BoxIouMatcherInfo _iou_matcher_info;
#if ENABLE_HIP
    BoxEncoderGpu *_box_encoder_gpu = nullptr;
#endif
    TimingDbg _rb_block_if_empty_time, _rb_block_if_full_time;
};

template <typename T>
std::shared_ptr<T> MasterGraph::add_node(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto node = std::make_shared<T>(inputs, outputs);
    _nodes.push_back(node);

    for (auto &input : inputs) {
        if (_tensor_map.find(input) == _tensor_map.end())
            THROW("Input tensor is invalid, cannot be found among output of previously created nodes")

        auto parent_node = _tensor_map.find(input)->second;
        parent_node->add_next(node);
        node->add_previous(parent_node);
    }

    for (auto &output : outputs)
        _tensor_map.insert(std::make_pair(output, node));

    return node;
}

template <typename T, typename M>
std::shared_ptr<T> MasterGraph::meta_add_node(std::shared_ptr<M> node) {
    auto meta_node = std::make_shared<T>();
    _meta_data_graph->_meta_nodes.push_back(meta_node);
    meta_node->_node = node;
    meta_node->_batch_size = _user_batch_size;
    _augmentation_metanode = true;
    return meta_node;
}

/*
 * Explicit specialization for ImageLoaderNode
 */
template <>
inline std::shared_ptr<ImageLoaderNode> MasterGraph::add_node(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (_loader_module)
        THROW("A loader already exists, cannot have more than one loader")
#if ENABLE_HIP || ENABLE_OPENCL
    auto node = std::make_shared<ImageLoaderNode>(outputs[0], (void *)_device.resources());
#else
    auto node = std::make_shared<ImageLoaderNode>(outputs[0], nullptr);
#endif
    _loader_module = node->get_loader_module();
    _loader_module->set_prefetch_queue_depth(_prefetch_queue_depth);
    _root_nodes.push_back(node);
    for (auto &output : outputs)
        _tensor_map.insert(std::make_pair(output, node));

    return node;
}

template <>
inline std::shared_ptr<ImageLoaderSingleShardNode> MasterGraph::add_node(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (_loader_module)
        THROW("A loader already exists, cannot have more than one loader")
#if ENABLE_HIP || ENABLE_OPENCL
    auto node = std::make_shared<ImageLoaderSingleShardNode>(outputs[0], (void *)_device.resources());
#else
    auto node = std::make_shared<ImageLoaderSingleShardNode>(outputs[0], nullptr);
#endif
    _loader_module = node->get_loader_module();
    _loader_module->set_prefetch_queue_depth(_prefetch_queue_depth);
    _root_nodes.push_back(node);
    for (auto &output : outputs)
        _tensor_map.insert(std::make_pair(output, node));

    return node;
}

template <>
inline std::shared_ptr<FusedJpegCropNode> MasterGraph::add_node(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (_loader_module)
        THROW("A loader already exists, cannot have more than one loader")
#if ENABLE_HIP || ENABLE_OPENCL
    auto node = std::make_shared<FusedJpegCropNode>(outputs[0], (void *)_device.resources());
#else
    auto node = std::make_shared<FusedJpegCropNode>(outputs[0], nullptr);
#endif
    _loader_module = node->get_loader_module();
    _loader_module->set_prefetch_queue_depth(_prefetch_queue_depth);
    _loader_module->set_random_bbox_data_reader(_randombboxcrop_meta_data_reader);
    _root_nodes.push_back(node);
    for (auto &output : outputs)
        _tensor_map.insert(std::make_pair(output, node));

    return node;
}

template <>
inline std::shared_ptr<FusedJpegCropSingleShardNode> MasterGraph::add_node(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (_loader_module)
        THROW("A loader already exists, cannot have more than one loader")
#if ENABLE_HIP || ENABLE_OPENCL
    auto node = std::make_shared<FusedJpegCropSingleShardNode>(outputs[0], (void *)_device.resources());
#else
    auto node = std::make_shared<FusedJpegCropSingleShardNode>(outputs[0], nullptr);
#endif
    _loader_module = node->get_loader_module();
    _loader_module->set_prefetch_queue_depth(_prefetch_queue_depth);
    _loader_module->set_random_bbox_data_reader(_randombboxcrop_meta_data_reader);
    _root_nodes.push_back(node);
    for (auto &output : outputs)
        _tensor_map.insert(std::make_pair(output, node));

    return node;
}

/*
 * Explicit specialization for Cifar10LoaderNode
 */
template <>
inline std::shared_ptr<Cifar10LoaderNode> MasterGraph::add_node(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (_loader_module)
        THROW("A loader already exists, cannot have more than one loader")
#if ENABLE_HIP || ENABLE_OPENCL
    auto node = std::make_shared<Cifar10LoaderNode>(outputs[0], (void *)_device.resources());
#else
    auto node = std::make_shared<Cifar10LoaderNode>(outputs[0], nullptr);
#endif
    _loader_module = node->get_loader_module();
    _loader_module->set_prefetch_queue_depth(_prefetch_queue_depth);
    _root_nodes.push_back(node);
    for (auto &output : outputs)
        _tensor_map.insert(std::make_pair(output, node));

    return node;
}

#ifdef ROCAL_VIDEO
/*
 * Explicit specialization for VideoLoaderNode
 */
template <>
inline std::shared_ptr<VideoLoaderNode> MasterGraph::add_node(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (_loader_module)
        THROW("A loader already exists, cannot have more than one loader")
#if ENABLE_HIP || ENABLE_OPENCL
    auto node = std::make_shared<VideoLoaderNode>(outputs[0], (void *)_device.resources());
#else
    auto node = std::make_shared<VideoLoaderNode>(outputs[0], nullptr);
#endif
    _loader_module = node->get_loader_module();
    _loader_module->set_prefetch_queue_depth(_prefetch_queue_depth);
    _root_nodes.push_back(node);
    for (auto &output : outputs)
        _tensor_map.insert(std::make_pair(output, node));

    return node;
}

template <>
inline std::shared_ptr<VideoLoaderSingleShardNode> MasterGraph::add_node(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (_loader_module)
        THROW("A loader already exists, cannot have more than one loader")
#if ENABLE_HIP || ENABLE_OPENCL
    auto node = std::make_shared<VideoLoaderSingleShardNode>(outputs[0], (void *)_device.resources());
#else
    auto node = std::make_shared<VideoLoaderSingleShardNode>(outputs[0], nullptr);
#endif
    _loader_module = node->get_loader_module();
    _loader_module->set_prefetch_queue_depth(_prefetch_queue_depth);
    _root_nodes.push_back(node);
    for (auto &output : outputs)
        _tensor_map.insert(std::make_pair(output, node));

    return node;
}
#endif

#ifdef ROCAL_AUDIO
/*
 * Explicit specialization for AudioLoaderNode
 */
template<> inline std::shared_ptr<AudioLoaderNode> MasterGraph::add_node(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if(_loader_module)
        THROW("A loader already exists, cannot have more than one loader")
#if ENABLE_HIP || ENABLE_OPENCL
    auto node = std::make_shared<AudioLoaderNode>(outputs[0], (void *)_device.resources());
#else
    auto node = std::make_shared<AudioLoaderNode>(outputs[0], nullptr);
#endif
    _loader_module = node->GetLoaderModule();
    _loader_module->set_prefetch_queue_depth(_prefetch_queue_depth);
    _root_nodes.push_back(node);
    for(auto& output: outputs)
        _tensor_map.insert(make_pair(output, node));

    return node;
}

template<> inline std::shared_ptr<AudioLoaderSingleShardNode> MasterGraph::add_node(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if(_loader_module)
        THROW("A loader already exists, cannot have more than one loader")
#if ENABLE_HIP || ENABLE_OPENCL
    auto node = std::make_shared<AudioLoaderSingleShardNode>(outputs[0], (void *)_device.resources());
#else
    auto node = std::make_shared<AudioLoaderSingleShardNode>(outputs[0], nullptr);
#endif
    _loader_module = node->GetLoaderModule();
    _loader_module->set_prefetch_queue_depth(_prefetch_queue_depth);
    _root_nodes.push_back(node);
    for(auto& output: outputs)
        _tensor_map.insert(make_pair(output, node));

    return node;
}
#endif

/*
 * Explicit specialization for NumpyLoaderNode
 */
template <>
inline std::shared_ptr<NumpyLoaderNode> MasterGraph::add_node(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (_loader_module)
        THROW("A loader already exists, cannot have more than one loader")
#if ENABLE_HIP || ENABLE_OPENCL
    auto node = std::make_shared<NumpyLoaderNode>(outputs[0], (void *)_device.resources());
#else
    auto node = std::make_shared<NumpyLoaderNode>(outputs[0], nullptr);
#endif
    _loader_module = node->get_loader_module();
    _loader_module->set_prefetch_queue_depth(_prefetch_queue_depth);
    _root_nodes.push_back(node);
    for (auto &output : outputs)
        _tensor_map.insert(std::make_pair(output, node));

    return node;
}

template <>
inline std::shared_ptr<NumpyLoaderSingleShardNode> MasterGraph::add_node(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    if (_loader_module)
        THROW("A loader already exists, cannot have more than one loader")
#if ENABLE_HIP || ENABLE_OPENCL
    auto node = std::make_shared<NumpyLoaderSingleShardNode>(outputs[0], (void *)_device.resources());
#else
    auto node = std::make_shared<NumpyLoaderSingleShardNode>(outputs[0], nullptr);
#endif
    _loader_module = node->get_loader_module();
    _loader_module->set_prefetch_queue_depth(_prefetch_queue_depth);
    _root_nodes.push_back(node);
    for (auto &output : outputs)
        _tensor_map.insert(std::make_pair(output, node));

    return node;
}
