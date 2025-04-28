/*
Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include <assert.h>
#include "pipeline/commons.h"
#include "pipeline/context.h"
#include "loaders/image_source_evaluator.h"
#include "loaders/numpy_source_evaluator.h"
#include "loaders/image/node_cifar10_loader.h"
#include "augmentations/node_copy.h"
#include "loaders/image/node_fused_jpeg_crop.h"
#include "loaders/image/node_fused_jpeg_crop_single_shard.h"
#include "loaders/image/node_image_loader.h"
#include "loaders/image/node_image_loader_single_shard.h"
#include "loaders/image/node_numpy_loader.h"
#include "loaders/image/node_numpy_loader_single_shard.h"
#ifdef ROCAL_AUDIO
#include "loaders/audio/audio_source_evaluator.h"
#include "loaders/audio/node_audio_loader.h"
#include "loaders/audio/node_audio_loader_single_shard.h"
#include "augmentations/audio_augmentations/node_downmix.h"
#endif
#ifdef ROCAL_VIDEO
#include "loaders/video/node_video_loader.h"
#include "loaders/video/node_video_loader_single_shard.h"
#endif
#include "augmentations/geometry_augmentations/node_resize.h"
#include "rocal_api.h"

#ifdef ROCAL_AUDIO
std::tuple<unsigned, unsigned>
evaluate_audio_data_set(StorageType storage_type, DecoderType decoder_type,
                        const std::string& source_path, const std::string& file_list_path, std::shared_ptr<MetaDataReader> meta_data_reader) {
    AudioSourceEvaluator source_evaluator;
    auto reader_config = ReaderConfig(storage_type, source_path);
    reader_config.set_file_list_path(file_list_path);
    reader_config.set_meta_data_reader(meta_data_reader);
    if (source_evaluator.Create(reader_config, DecoderConfig(decoder_type)) != AudioSourceEvaluatorStatus::OK)
        THROW("Initializing file source input evaluator failed")
    auto max_samples = source_evaluator.GetMaxSamples();
    auto max_channels = source_evaluator.GetMaxChannels();
    if (max_samples == 0 || max_channels == 0)
        THROW("Cannot find size of the audio files or files cannot be accessed")
    LOG("Maximum input audio dimension [ " + TOSTR(max_samples) + " x " + TOSTR(max_channels) + " ] for audio's in " + source_path)
    return std::make_tuple(max_samples, max_channels);
}
#endif

std::tuple<unsigned, unsigned>
evaluate_image_data_set(RocalImageSizeEvaluationPolicy decode_size_policy, StorageType storage_type,
                        DecoderType decoder_type, const std::string& source_path, const std::string& json_path) {
    auto translate_image_size_policy = [](RocalImageSizeEvaluationPolicy decode_size_policy) {
        switch (decode_size_policy) {
            case ROCAL_USE_MAX_SIZE:
            case ROCAL_USE_MAX_SIZE_RESTRICTED:
                return MaxSizeEvaluationPolicy::MAXIMUM_FOUND_SIZE;
            case ROCAL_USE_MOST_FREQUENT_SIZE:
                return MaxSizeEvaluationPolicy::MOST_FREQUENT_SIZE;
            default:
                return MaxSizeEvaluationPolicy::MAXIMUM_FOUND_SIZE;
        }
    };

    ImageSourceEvaluator source_evaluator;
    source_evaluator.set_size_evaluation_policy(translate_image_size_policy(decode_size_policy));
    if (source_evaluator.create(ReaderConfig(storage_type, source_path, json_path), DecoderConfig(decoder_type)) != ImageSourceEvaluatorStatus::OK)
        THROW("Initializing file source input evaluator failed ")
    auto max_width = source_evaluator.max_width();
    auto max_height = source_evaluator.max_height();
    if (max_width == 0 || max_height == 0)
        THROW("Cannot find size of the images or images cannot be accessed")

    LOG("Maximum input image dimension [ " + TOSTR(max_width) + " x " + TOSTR(max_height) + " ] for images in " + source_path)
    return std::make_tuple(max_width, max_height);
};

std::tuple<std::vector<size_t>, RocalTensorDataType>
evaluate_numpy_data_set(StorageType storage_type, const std::string& source_path, const std::vector<std::string>& files) {
    NumpySourceEvaluator source_evaluator;
    auto reader_cfg = ReaderConfig(storage_type, source_path);
    if (!files.empty())
        reader_cfg.set_files_list(files);
    source_evaluator.create(reader_cfg);
    auto max_dims = source_evaluator.max_numpy_dims();
    auto data_type = source_evaluator.get_numpy_dtype();
    return std::make_tuple(max_dims, data_type);
};

auto convert_color_format = [](RocalImageColor color_format, size_t n, size_t h, size_t w) {
    switch (color_format) {
        case ROCAL_COLOR_RGB24: {
            std::vector<size_t> dimensions = {n, h, w, 3u};
            return std::make_tuple(RocalColorFormat::RGB24, RocalTensorlayout::NHWC, dimensions, 3u);
        }
        case ROCAL_COLOR_BGR24: {
            std::vector<size_t> dimensions = {n, h, w, 3u};
            return std::make_tuple(RocalColorFormat::BGR24, RocalTensorlayout::NHWC, dimensions, 3u);
        }
        case ROCAL_COLOR_U8: {
            std::vector<size_t> dimensions = {n, 1u, h, w};
            return std::make_tuple(RocalColorFormat::U8, RocalTensorlayout::NCHW, dimensions, 1u);
        }
        case ROCAL_COLOR_RGB_PLANAR: {
            std::vector<size_t> dimensions = {n, 3u, h, w};
            return std::make_tuple(RocalColorFormat::RGB_PLANAR, RocalTensorlayout::NCHW, dimensions, 3u);
        }
        default:
            THROW("Unsupported Image type" + TOSTR(color_format))
    }
};

auto convert_color_format_sequence = [](RocalImageColor color_format, size_t n, size_t h, size_t w, size_t f) {
    switch (color_format) {
        case ROCAL_COLOR_RGB24: {
            std::vector<size_t> dimensions = {n, f, h, w, 3u};
            return std::make_tuple(RocalColorFormat::RGB24, RocalTensorlayout::NFHWC, dimensions, 3u);
        }
        case ROCAL_COLOR_BGR24: {
            std::vector<size_t> dimensions = {n, f, h, w, 3u};
            return std::make_tuple(RocalColorFormat::BGR24, RocalTensorlayout::NFHWC, dimensions, 3u);
        }
        case ROCAL_COLOR_U8: {
            std::vector<size_t> dimensions = {n, f, 1u, h, w};
            return std::make_tuple(RocalColorFormat::U8, RocalTensorlayout::NFCHW, dimensions, 1u);
        }
        case ROCAL_COLOR_RGB_PLANAR: {
            std::vector<size_t> dimensions = {n, f, 3u, h, w};
            return std::make_tuple(RocalColorFormat::RGB_PLANAR, RocalTensorlayout::NFCHW, dimensions, 3u);
        }
        default:
            THROW("Unsupported Image type" + TOSTR(color_format))
    }
};

auto convert_decoder_mode = [](RocalDecodeDevice decode_mode) {
    switch (decode_mode) {
        case ROCAL_HW_DECODE:
            return DecodeMode::HW_VAAPI;

        case ROCAL_SW_DECODE:
            return DecodeMode::CPU;
        default:

            THROW("Unsupported decoder mode" + TOSTR(decode_mode))
    }
};

auto convert_video_decoder_type = [](RocalDecoderType decoder_type) {
    switch (decoder_type) {
        case ROCAL_DECODER_VIDEO_FFMPEG_SW:
            return DecoderType::FFMPEG_SW_DECODE;
        case ROCAL_DECODER_VIDEO_FFMPEG_HW:
            return DecoderType::FFMPEG_HW_DECODE;
        case ROCAL_DECODER_VIDEO_ROCDECODE:
            return DecoderType::ROCDEC_VIDEO_DECODE;
        default:
            THROW("Unsupported video decoder type" + TOSTR(decoder_type))
    }
};

auto convert_last_batch_policy = [](RocalLastBatchPolicy last_batch_policy) {
    switch (last_batch_policy) {
        case ROCAL_LAST_BATCH_FILL:
            return RocalBatchPolicy::FILL;
        case ROCAL_LAST_BATCH_PARTIAL:
            return RocalBatchPolicy::PARTIAL;
        case ROCAL_LAST_BATCH_DROP:
            return RocalBatchPolicy::DROP;
        default:
            THROW("Unsupported Last Batch Policy Mode" + TOSTR(last_batch_policy))
    }
};

RocalTensor ROCAL_API_CALL
rocalJpegFileSourceSingleShard(
    RocalContext p_context,
    const char* source_path,
    RocalImageColor rocal_color_format,
    unsigned shard_id,
    unsigned shard_count,
    bool is_output,
    bool shuffle,
    bool loop,
    RocalImageSizeEvaluationPolicy decode_size_policy,
    unsigned max_width,
    unsigned max_height,
    RocalDecoderType dec_type,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);
        DecoderType decType = DecoderType::TURBO_JPEG;  // default
        if (dec_type == ROCAL_DECODER_OPENCV) decType = DecoderType::OPENCV_DEC;
        if (dec_type == ROCAL_DECODER_HW_JPEG) decType = DecoderType::HW_JPEG_DEC;
        if (dec_type == ROCAL_DECODER_ROCJPEG) decType = DecoderType::ROCJPEG_DEC;

        if (shard_count < 1)
            THROW("Shard count should be bigger than 0")

        if (shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if (use_input_dimension && (max_width == 0 || max_height == 0)) {
            THROW("Invalid input max width and height");
        } else {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension ? std::make_tuple(max_width, max_height) : evaluate_image_data_set(decode_size_policy, StorageType::FILE_SYSTEM, DecoderType::TURBO_JPEG, source_path, "");
        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format(rocal_color_format, context->user_batch_size(), height, width);
        INFO("Internal buffer size width = " + TOSTR(width) + " height = " + TOSTR(height) + " depth = " + TOSTR(num_of_planes))

        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);
        output = context->master_graph->create_loader_output_tensor(info);
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(shard_count);
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);
        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count, cpu_num_threads, source_path, "", StorageType::FILE_SYSTEM, decType, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), decoder_keep_original, sharding_info);
        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalJpegFileSource(
    RocalContext p_context,
    const char* source_path,
    RocalImageColor rocal_color_format,
    unsigned internal_shard_count,
    bool is_output,
    bool shuffle,
    bool loop,
    RocalImageSizeEvaluationPolicy decode_size_policy,
    unsigned max_width,
    unsigned max_height,
    RocalDecoderType dec_type,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);
        DecoderType decType = DecoderType::TURBO_JPEG;  // default
        if (dec_type == ROCAL_DECODER_OPENCV) decType = DecoderType::OPENCV_DEC;
        if (dec_type == ROCAL_DECODER_HW_JPEG) decType = DecoderType::HW_JPEG_DEC;
        if (dec_type == ROCAL_DECODER_ROCJPEG) decType = DecoderType::ROCJPEG_DEC;

        if (internal_shard_count < 1)
            THROW("Shard count should be bigger than 0")

        if (use_input_dimension && (max_width == 0 || max_height == 0)) {
            THROW("Invalid input max width and height");
        } else {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension ? std::make_tuple(max_width, max_height) : evaluate_image_data_set(decode_size_policy, StorageType::FILE_SYSTEM, DecoderType::TURBO_JPEG, source_path, "");
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);
        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format(rocal_color_format, context->user_batch_size(), height, width);
        INFO("Internal buffer size width = " + TOSTR(width) + " height = " + TOSTR(height) + " depth = " + TOSTR(num_of_planes))

        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);
        output = context->master_graph->create_loader_output_tensor(info);
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(1);

        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count, cpu_num_threads, source_path, "", std::map<std::string, std::string>(), StorageType::FILE_SYSTEM, decType, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), decoder_keep_original, sharding_info);
        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalSequenceReader(
    RocalContext p_context,
    const char* source_path,
    RocalImageColor rocal_color_format,
    unsigned internal_shard_count,
    unsigned sequence_length,
    bool is_output,
    bool shuffle,
    bool loop,
    unsigned step,
    unsigned stride,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    if (p_context == nullptr) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    try {
        if (sequence_length == 0)
            THROW("Sequence length passed should be bigger than 0")
        // Set sequence batch size and batch ratio in master graph as it varies according to sequence length
        context->master_graph->set_sequence_reader_output();
        context->master_graph->set_sequence_batch_size(sequence_length);
        bool decoder_keep_original = true;

        // This has been introduced to support variable width and height video frames in future.
        RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MAX_SIZE_RESTRICTED;

        if (internal_shard_count < 1)
            THROW("Shard count should be bigger than 0")

        // Set default step and stride values if 0 is passed
        step = (step == 0) ? 1 : step;
        stride = (stride == 0) ? 1 : stride;

        // FILE_SYSTEM is used here only to evaluate the width and height of the frames.
        auto [width, height] = evaluate_image_data_set(decode_size_policy, StorageType::FILE_SYSTEM, DecoderType::TURBO_JPEG, source_path, "");
        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format_sequence(rocal_color_format, context->user_batch_size(), height, width, sequence_length);
        INFO("Internal buffer size width = " + TOSTR(width) + " height = " + TOSTR(height) + " depth = " + TOSTR(num_of_planes))

        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8);
        info.set_color_format(color_format);
        info.set_tensor_layout(tensor_layout);
        info.set_sequence_batch_size(sequence_length);
        info.set_max_shape();

        output = context->master_graph->create_loader_output_tensor(info);
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(1);
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);

        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count, cpu_num_threads, source_path, "", std::map<std::string, std::string>(), StorageType::SEQUENCE_FILE_SYSTEM, DecoderType::TURBO_JPEG, shuffle, loop, context->master_graph->sequence_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), decoder_keep_original, sharding_info, "", sequence_length, step, stride);
        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalSequenceReaderSingleShard(
    RocalContext p_context,
    const char* source_path,
    RocalImageColor rocal_color_format,
    unsigned shard_id,
    unsigned shard_count,
    unsigned sequence_length,
    bool is_output,
    bool shuffle,
    bool loop,
    unsigned step,
    unsigned stride,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    if (p_context == nullptr) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    try {
        if (sequence_length == 0)
            THROW("Sequence length passed should be bigger than 0")
        // Set sequence batch size and batch ratio in master graph as it varies according to sequence length
        context->master_graph->set_sequence_reader_output();
        context->master_graph->set_sequence_batch_size(sequence_length);
        bool decoder_keep_original = true;

        // This has been introduced to support variable width and height video frames in future.
        RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MAX_SIZE_RESTRICTED;

        if (shard_count < 1)
            THROW("Shard count should be bigger than 0")

        if (shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        // Set default step and stride values if 0 is passed
        step = (step == 0) ? 1 : step;
        stride = (stride == 0) ? 1 : stride;

        // FILE_SYSTEM is used here only to evaluate the width and height of the frames.
        auto [width, height] = evaluate_image_data_set(decode_size_policy, StorageType::FILE_SYSTEM, DecoderType::TURBO_JPEG, source_path, "");
        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format_sequence(rocal_color_format, context->user_batch_size(), height, width, sequence_length);
        INFO("Internal buffer size width = " + TOSTR(width) + " height = " + TOSTR(height) + " depth = " + TOSTR(num_of_planes))

        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8);
        info.set_color_format(color_format);
        info.set_tensor_layout(tensor_layout);
        info.set_sequence_batch_size(sequence_length);
        info.set_max_shape();
        output = context->master_graph->create_loader_output_tensor(info);
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(shard_count);
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);
        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count, cpu_num_threads, source_path, "", StorageType::SEQUENCE_FILE_SYSTEM, DecoderType::TURBO_JPEG, shuffle, loop, context->master_graph->sequence_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), decoder_keep_original, sharding_info, std::map<std::string, std::string>(), sequence_length, step, stride);
        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalJpegCaffe2LMDBRecordSource(
    RocalContext p_context,
    const char* source_path,
    RocalImageColor rocal_color_format,
    unsigned internal_shard_count,
    bool is_output,
    bool shuffle,
    bool loop,
    RocalImageSizeEvaluationPolicy decode_size_policy,
    unsigned max_width,
    unsigned max_height,
    RocalDecoderType dec_type,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);
        DecoderType decType = DecoderType::TURBO_JPEG;  // default
        if (dec_type == ROCAL_DECODER_OPENCV) decType = DecoderType::OPENCV_DEC;
        if (dec_type == ROCAL_DECODER_HW_JPEG) decType = DecoderType::HW_JPEG_DEC;
        if (dec_type == ROCAL_DECODER_ROCJPEG) decType = DecoderType::ROCJPEG_DEC;

        if (internal_shard_count < 1)
            THROW("internal shard count should be bigger than 0")

        if (use_input_dimension && (max_width == 0 || max_height == 0)) {
            THROW("Invalid input max width and height");
        } else {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension ? std::make_tuple(max_width, max_height) : evaluate_image_data_set(decode_size_policy, StorageType::CAFFE2_LMDB_RECORD, DecoderType::TURBO_JPEG, source_path, "");
        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format(rocal_color_format, context->user_batch_size(), height, width);
        INFO("Internal buffer size width = " + TOSTR(width) + " height = " + TOSTR(height) + " depth = " + TOSTR(num_of_planes))

        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);
        output = context->master_graph->create_loader_output_tensor(info);
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(1);
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);
        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count, cpu_num_threads, source_path, "", std::map<std::string, std::string>(), StorageType::CAFFE2_LMDB_RECORD, decType, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), decoder_keep_original, sharding_info);
        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalJpegCaffe2LMDBRecordSourceSingleShard(
    RocalContext p_context,
    const char* source_path,
    RocalImageColor rocal_color_format,
    unsigned shard_id,
    unsigned shard_count,
    bool is_output,
    bool shuffle,
    bool loop,
    RocalImageSizeEvaluationPolicy decode_size_policy,
    unsigned max_width,
    unsigned max_height,
    RocalDecoderType dec_type,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);
        DecoderType decType = DecoderType::TURBO_JPEG;  // default
        if (dec_type == ROCAL_DECODER_OPENCV) decType = DecoderType::OPENCV_DEC;
        if (dec_type == ROCAL_DECODER_HW_JPEG) decType = DecoderType::HW_JPEG_DEC;
        if (dec_type == ROCAL_DECODER_ROCJPEG) decType = DecoderType::ROCJPEG_DEC;

        if (shard_count < 1)
            THROW("Shard count should be bigger than 0")

        if (shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if (use_input_dimension && (max_width == 0 || max_height == 0)) {
            THROW("Invalid input max width and height");
        } else {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension ? std::make_tuple(max_width, max_height) : evaluate_image_data_set(decode_size_policy, StorageType::CAFFE2_LMDB_RECORD, DecoderType::TURBO_JPEG, source_path, "");
        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format(rocal_color_format, context->user_batch_size(), height, width);
        INFO("Internal buffer size width = " + TOSTR(width) + " height = " + TOSTR(height) + " depth = " + TOSTR(num_of_planes))
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);

        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);
        output = context->master_graph->create_loader_output_tensor(info);
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(shard_count);

        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count, cpu_num_threads, source_path, "", StorageType::CAFFE2_LMDB_RECORD, decType, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), decoder_keep_original, sharding_info);
        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalJpegCaffeLMDBRecordSource(
    RocalContext p_context,
    const char* source_path,
    RocalImageColor rocal_color_format,
    unsigned internal_shard_count,
    bool is_output,
    bool shuffle,
    bool loop,
    RocalImageSizeEvaluationPolicy decode_size_policy,
    unsigned max_width,
    unsigned max_height,
    RocalDecoderType dec_type,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);
        DecoderType decType = DecoderType::TURBO_JPEG;  // default
        if (dec_type == ROCAL_DECODER_OPENCV) decType = DecoderType::OPENCV_DEC;
        if (dec_type == ROCAL_DECODER_HW_JPEG) decType = DecoderType::HW_JPEG_DEC;
        if (dec_type == ROCAL_DECODER_ROCJPEG) decType = DecoderType::ROCJPEG_DEC;

        if (internal_shard_count < 1)
            THROW("internal shard count should be bigger than 0")

        if (use_input_dimension && (max_width == 0 || max_height == 0)) {
            THROW("Invalid input max width and height");
        } else {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension ? std::make_tuple(max_width, max_height) : evaluate_image_data_set(decode_size_policy, StorageType::CAFFE_LMDB_RECORD, DecoderType::TURBO_JPEG, source_path, "");
        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format(rocal_color_format, context->user_batch_size(), height, width);
        INFO("Internal buffer size width = " + TOSTR(width) + " height = " + TOSTR(height) + " depth = " + TOSTR(num_of_planes))
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);

        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);
        output = context->master_graph->create_loader_output_tensor(info);
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(1);

        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count, cpu_num_threads, source_path, "", std::map<std::string, std::string>(), StorageType::CAFFE_LMDB_RECORD, decType, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), decoder_keep_original, sharding_info);

        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalJpegCaffeLMDBRecordSourceSingleShard(
    RocalContext p_context,
    const char* source_path,
    RocalImageColor rocal_color_format,
    unsigned shard_id,
    unsigned shard_count,
    bool is_output,
    bool shuffle,
    bool loop,
    RocalImageSizeEvaluationPolicy decode_size_policy,
    unsigned max_width,
    unsigned max_height,
    RocalDecoderType dec_type,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);
        DecoderType decType = DecoderType::TURBO_JPEG;  // default
        if (dec_type == ROCAL_DECODER_OPENCV) decType = DecoderType::OPENCV_DEC;
        if (dec_type == ROCAL_DECODER_HW_JPEG) decType = DecoderType::HW_JPEG_DEC;
        if (dec_type == ROCAL_DECODER_ROCJPEG) decType = DecoderType::ROCJPEG_DEC;

        if (shard_count < 1)
            THROW("Shard count should be bigger than 0")

        if (shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if (use_input_dimension && (max_width == 0 || max_height == 0)) {
            THROW("Invalid input max width and height");
        } else {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension ? std::make_tuple(max_width, max_height) : evaluate_image_data_set(decode_size_policy, StorageType::CAFFE_LMDB_RECORD, DecoderType::TURBO_JPEG, source_path, "");
        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format(rocal_color_format, context->user_batch_size(), height, width);
        INFO("Internal buffer size width = " + TOSTR(width) + " height = " + TOSTR(height) + " depth = " + TOSTR(num_of_planes))
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);

        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);
        output = context->master_graph->create_loader_output_tensor(info);
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(shard_count);

        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count, cpu_num_threads, source_path, "", StorageType::CAFFE_LMDB_RECORD, decType, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), decoder_keep_original, sharding_info);
        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalJpegCaffeLMDBRecordSourcePartialSingleShard(
    RocalContext p_context,
    const char* source_path,
    RocalImageColor rocal_color_format,
    unsigned shard_id,
    unsigned shard_count,
    bool is_output,
    std::vector<float>& area_factor,
    std::vector<float>& aspect_ratio,
    unsigned num_attempts,
    bool shuffle,
    bool loop,
    RocalImageSizeEvaluationPolicy decode_size_policy,
    unsigned max_width,
    unsigned max_height,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);

        if (shard_count < 1)
            THROW("Shard count should be bigger than 0")

        if (shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if (use_input_dimension && (max_width == 0 || max_height == 0)) {
            THROW("Invalid input max width and height");
        } else {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension ? std::make_tuple(max_width, max_height) : evaluate_image_data_set(decode_size_policy, StorageType::CAFFE_LMDB_RECORD, DecoderType::FUSED_TURBO_JPEG, source_path, "");
        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format(rocal_color_format, context->user_batch_size(), height, width);
        INFO("Internal buffer size width = " + TOSTR(width) + " height = " + TOSTR(height) + " depth = " + TOSTR(num_of_planes))
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);

        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);
        output = context->master_graph->create_loader_output_tensor(info);
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(shard_count);

        context->master_graph->add_node<FusedJpegCropSingleShardNode>({}, {output})->init(shard_id, shard_count, cpu_num_threads, source_path, "", StorageType::CAFFE_LMDB_RECORD, DecoderType::FUSED_TURBO_JPEG, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), num_attempts, area_factor, aspect_ratio, sharding_info);

        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalJpegCaffe2LMDBRecordSourcePartialSingleShard(
    RocalContext p_context,
    const char* source_path,
    RocalImageColor rocal_color_format,
    unsigned shard_id,
    unsigned shard_count,
    bool is_output,
    std::vector<float>& area_factor,
    std::vector<float>& aspect_ratio,
    unsigned num_attempts,
    bool shuffle,
    bool loop,
    RocalImageSizeEvaluationPolicy decode_size_policy,
    unsigned max_width,
    unsigned max_height,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);

        if (shard_count < 1)
            THROW("Shard count should be bigger than 0")

        if (shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if (use_input_dimension && (max_width == 0 || max_height == 0)) {
            THROW("Invalid input max width and height");
        } else {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension ? std::make_tuple(max_width, max_height) : evaluate_image_data_set(decode_size_policy, StorageType::CAFFE2_LMDB_RECORD, DecoderType::FUSED_TURBO_JPEG, source_path, "");
        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format(rocal_color_format, context->user_batch_size(), height, width);
        INFO("Internal buffer size width = " + TOSTR(width) + " height = " + TOSTR(height) + " depth = " + TOSTR(num_of_planes))
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);

        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);
        output = context->master_graph->create_loader_output_tensor(info);
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(shard_count);

        context->master_graph->add_node<FusedJpegCropSingleShardNode>({}, {output})->init(shard_id, shard_count, cpu_num_threads, source_path, "", StorageType::CAFFE2_LMDB_RECORD, DecoderType::FUSED_TURBO_JPEG, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), num_attempts, area_factor, aspect_ratio, sharding_info);
        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalMXNetRecordSource(
    RocalContext p_context,
    const char* source_path,
    RocalImageColor rocal_color_format,
    unsigned shard_id,
    unsigned shard_count,
    bool is_output,
    std::vector<float>& area_factor,
    std::vector<float>& aspect_ratio,
    unsigned num_attempts,
    bool shuffle,
    bool loop,
    RocalImageSizeEvaluationPolicy decode_size_policy,
    unsigned max_width,
    unsigned max_height,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);

        if (shard_count < 1)
            THROW("Shard count should be bigger than 0")

        if (shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if (use_input_dimension && (max_width == 0 || max_height == 0)) {
            THROW("Invalid input max width and height");
        } else {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension ? std::make_tuple(max_width, max_height) : evaluate_image_data_set(decode_size_policy, StorageType::CAFFE_LMDB_RECORD, DecoderType::FUSED_TURBO_JPEG, source_path, "");
        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format(rocal_color_format, context->user_batch_size(), height, width);
        INFO("Internal buffer size width = " + TOSTR(width) + " height = " + TOSTR(height) + " depth = " + TOSTR(num_of_planes))
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);

        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);
        output = context->master_graph->create_loader_output_tensor(info);
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(shard_count);

        context->master_graph->add_node<FusedJpegCropSingleShardNode>({}, {output})->init(shard_id, shard_count, cpu_num_threads, source_path, "", StorageType::CAFFE_LMDB_RECORD, DecoderType::FUSED_TURBO_JPEG, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), num_attempts, area_factor, aspect_ratio, sharding_info);

        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalMXNetRecordSource(
    RocalContext p_context,
    const char* source_path,
    RocalImageColor rocal_color_format,
    unsigned internal_shard_count,
    bool is_output,
    bool shuffle,
    bool loop,
    RocalImageSizeEvaluationPolicy decode_size_policy,
    unsigned max_width,
    unsigned max_height,
    RocalDecoderType dec_type,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    if (p_context == nullptr) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    try {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);
        DecoderType decType = DecoderType::TURBO_JPEG;  // default
        if (dec_type == ROCAL_DECODER_OPENCV) decType = DecoderType::OPENCV_DEC;
        if (dec_type == ROCAL_DECODER_HW_JPEG) decType = DecoderType::HW_JPEG_DEC;
        if (dec_type == ROCAL_DECODER_ROCJPEG) decType = DecoderType::ROCJPEG_DEC;

        if (internal_shard_count < 1)
            THROW("internal shard count should be bigger than 0")

        if (use_input_dimension && (max_width == 0 || max_height == 0)) {
            THROW("Invalid input max width and height");
        } else {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension ? std::make_tuple(max_width, max_height) : evaluate_image_data_set(decode_size_policy, StorageType::MXNET_RECORDIO, DecoderType::TURBO_JPEG, source_path, "");
        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format(rocal_color_format, context->user_batch_size(), height, width);
        INFO("Internal buffer size width = " + TOSTR(width) + " height = " + TOSTR(height) + " depth = " + TOSTR(num_of_planes))
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);

        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);
        output = context->master_graph->create_loader_output_tensor(info);
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(1);

        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count, cpu_num_threads, source_path, "", std::map<std::string, std::string>(), StorageType::MXNET_RECORDIO, decType, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), decoder_keep_original, sharding_info);

        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalMXNetRecordSourceSingleShard(
    RocalContext p_context,
    const char* source_path,
    RocalImageColor rocal_color_format,
    unsigned shard_id,
    unsigned shard_count,
    bool is_output,
    bool shuffle,
    bool loop,
    RocalImageSizeEvaluationPolicy decode_size_policy,
    unsigned max_width,
    unsigned max_height,
    RocalDecoderType dec_type,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    if (p_context == nullptr) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    try {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);
        DecoderType decType = DecoderType::TURBO_JPEG;  // default
        if (dec_type == ROCAL_DECODER_OPENCV) decType = DecoderType::OPENCV_DEC;
        if (dec_type == ROCAL_DECODER_HW_JPEG) decType = DecoderType::HW_JPEG_DEC;
        if (dec_type == ROCAL_DECODER_ROCJPEG) decType = DecoderType::ROCJPEG_DEC;

        if (shard_count < 1)
            THROW("Shard count should be bigger than 0")

        if (shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if (use_input_dimension && (max_width == 0 || max_height == 0)) {
            THROW("Invalid input max width and height");
        } else {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension ? std::make_tuple(max_width, max_height) : evaluate_image_data_set(decode_size_policy, StorageType::MXNET_RECORDIO, DecoderType::TURBO_JPEG, source_path, "");
        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format(rocal_color_format, context->user_batch_size(), height, width);
        INFO("Internal buffer size width = " + TOSTR(width) + " height = " + TOSTR(height) + " depth = " + TOSTR(num_of_planes))
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);

        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);
        output = context->master_graph->create_loader_output_tensor(info);
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(shard_count);

        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count, cpu_num_threads, source_path, "", StorageType::MXNET_RECORDIO, decType, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), decoder_keep_original, sharding_info);
        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalJpegCOCOFileSource(
    RocalContext p_context,
    const char* source_path,
    const char* json_path,
    RocalImageColor rocal_color_format,
    unsigned internal_shard_count,
    bool is_output,
    bool shuffle,
    bool loop,
    RocalImageSizeEvaluationPolicy decode_size_policy,
    unsigned max_width,
    unsigned max_height,
    RocalDecoderType dec_type,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);
        DecoderType decType = DecoderType::TURBO_JPEG;  // default
        if (dec_type == ROCAL_DECODER_OPENCV) decType = DecoderType::OPENCV_DEC;
        if (dec_type == ROCAL_DECODER_HW_JPEG) decType = DecoderType::HW_JPEG_DEC;
        if (dec_type == ROCAL_DECODER_ROCJPEG) decType = DecoderType::ROCJPEG_DEC;

        if (internal_shard_count < 1)
            THROW("Shard count should be bigger than 0")

        if (use_input_dimension && (max_width == 0 || max_height == 0)) {
            THROW("Invalid input max width and height");
        } else {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension ? std::make_tuple(max_width, max_height) : evaluate_image_data_set(decode_size_policy, StorageType::COCO_FILE_SYSTEM, DecoderType::TURBO_JPEG, source_path, json_path);

        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format(rocal_color_format, context->user_batch_size(), height, width);
        INFO("Internal buffer size width = " + TOSTR(width) + " height = " + TOSTR(height) + " depth = " + TOSTR(num_of_planes))
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);

        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);
        output = context->master_graph->create_loader_output_tensor(info);
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(1);

        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count, cpu_num_threads, source_path, json_path, std::map<std::string, std::string>(), StorageType::COCO_FILE_SYSTEM, decType, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), decoder_keep_original, sharding_info);

        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalJpegCOCOFileSourceSingleShard(
    RocalContext p_context,
    const char* source_path,
    const char* json_path,
    RocalImageColor rocal_color_format,
    unsigned shard_id,
    unsigned shard_count,
    bool is_output,
    bool shuffle,
    bool loop,
    RocalImageSizeEvaluationPolicy decode_size_policy,
    unsigned max_width,
    unsigned max_height,
    RocalDecoderType dec_type,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);
        DecoderType decType = DecoderType::TURBO_JPEG;  // default
        if (dec_type == ROCAL_DECODER_OPENCV) decType = DecoderType::OPENCV_DEC;
        if (dec_type == ROCAL_DECODER_HW_JPEG) decType = DecoderType::HW_JPEG_DEC;
        if (dec_type == ROCAL_DECODER_ROCJPEG) decType = DecoderType::ROCJPEG_DEC;

        if (shard_count < 1)
            THROW("Shard count should be bigger than 0")

        if (shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if (use_input_dimension && (max_width == 0 || max_height == 0)) {
            THROW("Invalid input max width and height");
        } else {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension ? std::make_tuple(max_width, max_height) : evaluate_image_data_set(decode_size_policy, StorageType::COCO_FILE_SYSTEM, DecoderType::TURBO_JPEG, source_path, json_path);
        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format(rocal_color_format, context->user_batch_size(), height, width);
        INFO("Internal buffer size width = " + TOSTR(width) + " height = " + TOSTR(height) + " depth = " + TOSTR(num_of_planes))
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);

        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);
        output = context->master_graph->create_loader_output_tensor(info);
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(shard_count);

        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count, cpu_num_threads, source_path, json_path, StorageType::COCO_FILE_SYSTEM, decType, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), decoder_keep_original, sharding_info);
        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalFusedJpegCrop(
    RocalContext p_context,
    const char* source_path,
    RocalImageColor rocal_color_format,
    unsigned internal_shard_count,
    bool is_output,
    std::vector<float>& area_factor,
    std::vector<float>& aspect_ratio,
    unsigned num_attempts,
    bool shuffle,
    bool loop,
    RocalImageSizeEvaluationPolicy decode_size_policy,
    unsigned max_width,
    unsigned max_height,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);

        if (internal_shard_count < 1)
            THROW("Shard count should be bigger than 0")

        if (use_input_dimension && (max_width == 0 || max_height == 0)) {
            THROW("Invalid input max width and height");
        } else {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension ? std::make_tuple(max_width, max_height) : evaluate_image_data_set(decode_size_policy, StorageType::FILE_SYSTEM, DecoderType::FUSED_TURBO_JPEG, source_path, "");

        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format(rocal_color_format, context->user_batch_size(), height, width);
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);

        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);
        output = context->master_graph->create_loader_output_tensor(info);
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(1);
        context->master_graph->add_node<FusedJpegCropNode>({}, {output})->init(internal_shard_count, cpu_num_threads, source_path, "", StorageType::FILE_SYSTEM, DecoderType::FUSED_TURBO_JPEG, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), num_attempts, area_factor, aspect_ratio, sharding_info);
        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalJpegCOCOFileSourcePartial(
    RocalContext p_context,
    const char* source_path,
    const char* json_path,
    RocalImageColor rocal_color_format,
    unsigned internal_shard_count,
    bool is_output,
    std::vector<float>& area_factor,
    std::vector<float>& aspect_ratio,
    unsigned num_attempts,
    bool shuffle,
    bool loop,
    RocalImageSizeEvaluationPolicy decode_size_policy,
    unsigned max_width,
    unsigned max_height,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);

        if (internal_shard_count < 1)
            THROW("Shard count should be bigger than 0")

        if (use_input_dimension && (max_width == 0 || max_height == 0)) {
            THROW("Invalid input max width and height");
        } else {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension ? std::make_tuple(max_width, max_height) : evaluate_image_data_set(decode_size_policy, StorageType::COCO_FILE_SYSTEM, DecoderType::FUSED_TURBO_JPEG, source_path, json_path);

        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format(rocal_color_format, context->user_batch_size(), height, width);
        INFO("Internal buffer size width = " + TOSTR(width) + " height = " + TOSTR(height) + " depth = " + TOSTR(num_of_planes))
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);

        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);
        output = context->master_graph->create_loader_output_tensor(info);
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(1);

        context->master_graph->add_node<FusedJpegCropNode>({}, {output})->init(internal_shard_count, cpu_num_threads, source_path, json_path, StorageType::COCO_FILE_SYSTEM, DecoderType::FUSED_TURBO_JPEG, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), num_attempts, area_factor, aspect_ratio, sharding_info);

        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalJpegCOCOFileSourcePartialSingleShard(
    RocalContext p_context,
    const char* source_path,
    const char* json_path,
    RocalImageColor rocal_color_format,
    unsigned shard_id,
    unsigned shard_count,
    bool is_output,
    std::vector<float>& area_factor,
    std::vector<float>& aspect_ratio,
    unsigned num_attempts,
    bool shuffle,
    bool loop,
    RocalImageSizeEvaluationPolicy decode_size_policy,
    unsigned max_width,
    unsigned max_height,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);

        if (shard_count < 1)
            THROW("Shard count should be bigger than 0")

        if (shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if (use_input_dimension && (max_width == 0 || max_height == 0)) {
            THROW("Invalid input max width and height");
        } else {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension ? std::make_tuple(max_width, max_height) : evaluate_image_data_set(decode_size_policy, StorageType::COCO_FILE_SYSTEM, DecoderType::FUSED_TURBO_JPEG, source_path, json_path);

        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format(rocal_color_format, context->user_batch_size(), height, width);
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);

        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);
        output = context->master_graph->create_loader_output_tensor(info);
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(shard_count);

        context->master_graph->add_node<FusedJpegCropSingleShardNode>({}, {output})->init(shard_id, shard_count, cpu_num_threads, source_path, json_path, StorageType::COCO_FILE_SYSTEM, DecoderType::FUSED_TURBO_JPEG, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), num_attempts, area_factor, aspect_ratio, sharding_info);

        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalJpegTFRecordSource(
    RocalContext p_context,
    const char* source_path,
    RocalImageColor rocal_color_format,
    unsigned internal_shard_count,
    bool is_output,
    const char* user_key_for_encoded,
    const char* user_key_for_filename,
    bool shuffle,
    bool loop,
    RocalImageSizeEvaluationPolicy decode_size_policy,
    unsigned max_width,
    unsigned max_height,
    RocalDecoderType dec_type,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try {
        std::string user_key_for_encoded_str(user_key_for_encoded);
        std::string user_key_for_filename_str(user_key_for_filename);

        std::map<std::string, std::string> feature_key_map = {
            {"image/encoded", user_key_for_encoded_str},
            {"image/filename", user_key_for_filename_str},
        };
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        DecoderType decType = DecoderType::TURBO_JPEG;  // default
        if (dec_type == ROCAL_DECODER_OPENCV) decType = DecoderType::OPENCV_DEC;
        if (dec_type == ROCAL_DECODER_HW_JPEG) decType = DecoderType::HW_JPEG_DEC;
        if (dec_type == ROCAL_DECODER_ROCJPEG) decType = DecoderType::ROCJPEG_DEC;

        if (internal_shard_count < 1)
            THROW("internal shard count should be bigger than 0")

        if (use_input_dimension && (max_width == 0 || max_height == 0)) {
            THROW("Invalid input max width and height");
        } else {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension ? std::make_tuple(max_width, max_height) : evaluate_image_data_set(decode_size_policy, StorageType::TF_RECORD, DecoderType::TURBO_JPEG, source_path, "");
        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format(rocal_color_format, context->user_batch_size(), height, width);
        INFO("Internal buffer size width = " + TOSTR(width) + " height = " + TOSTR(height) + " depth = " + TOSTR(num_of_planes))
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);

        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);
        output = context->master_graph->create_loader_output_tensor(info);
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(1);

        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count, cpu_num_threads, source_path, "", feature_key_map, StorageType::TF_RECORD, decType, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), false, sharding_info);
        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalJpegTFRecordSourceSingleShard(
    RocalContext p_context,
    const char* source_path,
    RocalImageColor rocal_color_format,
    unsigned shard_id,
    unsigned shard_count,
    bool is_output,
    bool shuffle,
    bool loop,
    RocalImageSizeEvaluationPolicy decode_size_policy,
    unsigned max_width,
    unsigned max_height,
    RocalDecoderType dec_type,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);
        DecoderType decType = DecoderType::TURBO_JPEG;  // default
        if (dec_type == ROCAL_DECODER_OPENCV) decType = DecoderType::OPENCV_DEC;
        if (dec_type == ROCAL_DECODER_HW_JPEG) decType = DecoderType::HW_JPEG_DEC;
        if (dec_type == ROCAL_DECODER_ROCJPEG) decType = DecoderType::ROCJPEG_DEC;

        if (shard_count < 1)
            THROW("Shard count should be bigger than 0")

        if (shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if (use_input_dimension && (max_width == 0 || max_height == 0)) {
            THROW("Invalid input max width and height");
        } else {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension ? std::make_tuple(max_width, max_height) : evaluate_image_data_set(decode_size_policy, StorageType::TF_RECORD, DecoderType::TURBO_JPEG, source_path, "");
        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format(rocal_color_format, context->user_batch_size(), height, width);
        INFO("Internal buffer size width = " + TOSTR(width) + " height = " + TOSTR(height) + " depth = " + TOSTR(num_of_planes))
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);

        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);
        output = context->master_graph->create_loader_output_tensor(info);
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(shard_count);

        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count, cpu_num_threads, source_path, "", StorageType::TF_RECORD, decType, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), decoder_keep_original, sharding_info);
        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalRawTFRecordSource(
    RocalContext p_context,
    const char* source_path,
    const char* user_key_for_encoded_str,
    const char* user_key_for_filename_str,
    RocalImageColor rocal_color_format,
    bool is_output,
    bool shuffle,
    bool loop,
    unsigned out_width,
    unsigned out_height,
    const char* record_name_prefix,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    if (p_context == nullptr) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }

    auto context = static_cast<Context*>(p_context);
    try {
        unsigned internal_shard_count = 1;
        std::map<std::string, std::string> feature_key_map = {
            {"image/encoded", user_key_for_encoded_str},
            {"image/filename", user_key_for_filename_str},
        };

        if (out_width == 0 || out_height == 0) {
            THROW("Invalid output width and height");
        } else {
            LOG("User input size " + TOSTR(out_width) + " x " + TOSTR(out_height))
        }

        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format(rocal_color_format, context->user_batch_size(), out_height, out_width);
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);

        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);
        output = context->master_graph->create_loader_output_tensor(info);
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(1);

        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(internal_shard_count, cpu_num_threads, source_path, "", feature_key_map, StorageType::TF_RECORD, DecoderType::SKIP_DECODE, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), false, sharding_info, record_name_prefix, 0, 0, 0);
        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalRawTFRecordSourceSingleShard(
    RocalContext p_context,
    const char* source_path,
    RocalImageColor rocal_color_format,
    unsigned shard_id,
    unsigned shard_count,
    bool is_output,
    bool shuffle,
    bool loop,
    unsigned out_width,
    unsigned out_height,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try {
        if (shard_count < 1)
            THROW("Shard count should be bigger than 0")

        if (shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if ((out_width == 0 || out_height == 0)) {
            THROW("Invalid input max width and height");
        } else {
            LOG("User input size " + TOSTR(out_width) + " x " + TOSTR(out_height))
        }

        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format(rocal_color_format, context->user_batch_size(), out_height, out_width);
        INFO("Internal buffer size width = " + TOSTR(out_width) + " height = " + TOSTR(out_height) + " depth = " + TOSTR(num_of_planes))
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);
        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);
        output = context->master_graph->create_loader_output_tensor(info);
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(shard_count);

        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count, cpu_num_threads, source_path, "", StorageType::TF_RECORD, DecoderType::SKIP_DECODE, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), false, sharding_info);
        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalFusedJpegCropSingleShard(
    RocalContext p_context,
    const char* source_path,
    RocalImageColor rocal_color_format,
    unsigned shard_id,
    unsigned shard_count,
    bool is_output,
    std::vector<float>& area_factor,
    std::vector<float>& aspect_ratio,
    unsigned num_attempts,
    bool shuffle,
    bool loop,
    RocalImageSizeEvaluationPolicy decode_size_policy,
    unsigned max_width,
    unsigned max_height,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try {
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);

        if (shard_count < 1)
            THROW("Shard count should be bigger than 0")

        if (shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        if (use_input_dimension && (max_width == 0 || max_height == 0)) {
            THROW("Invalid input max width and height");
        } else {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = use_input_dimension ? std::make_tuple(max_width, max_height) : evaluate_image_data_set(decode_size_policy, StorageType::FILE_SYSTEM, DecoderType::FUSED_TURBO_JPEG, source_path, "");

        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format(rocal_color_format, context->user_batch_size(), height, width);
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);
        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);
        output = context->master_graph->create_loader_output_tensor(info);
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(shard_count);
        context->master_graph->add_node<FusedJpegCropSingleShardNode>({}, {output})->init(shard_id, shard_count, cpu_num_threads, source_path, "", StorageType::FILE_SYSTEM, DecoderType::FUSED_TURBO_JPEG, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), num_attempts, area_factor, aspect_ratio, sharding_info);
        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalVideoFileSource(
    RocalContext p_context,
    const char* source_path,
    RocalImageColor rocal_color_format,
    RocalDecodeDevice rocal_decode_device,
    unsigned internal_shard_count,
    unsigned sequence_length,
    bool shuffle,
    bool is_output,
    bool loop,
    RocalDecoderType rocal_decoder_type,
    unsigned step,
    unsigned stride,
    bool file_list_frame_num,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    if (p_context == nullptr) {
        ERR("Invalid ROCAL context or invalid input image")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    try {
#ifdef ROCAL_VIDEO
        if (sequence_length == 0)
            THROW("Sequence length passed should be bigger than 0")

        // Set default step and stride values if 0 is passed
        step = (step == 0) ? sequence_length : step;
        stride = (stride == 0) ? 1 : stride;

        VideoProperties video_prop;
        DecoderType decoder_type = convert_video_decoder_type(rocal_decoder_type);
        find_video_properties(video_prop, source_path, file_list_frame_num);
        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format_sequence(rocal_color_format, context->user_batch_size(),
                                                                                                video_prop.height, video_prop.width, sequence_length);
        auto decoder_mode = convert_decoder_mode(rocal_decode_device);
        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);

        output = context->master_graph->create_loader_output_tensor(info);

        context->master_graph->add_node<VideoLoaderNode>({}, {output})->init(internal_shard_count, source_path, StorageType::VIDEO_FILE_SYSTEM, decoder_type, decoder_mode, sequence_length, step, stride, video_prop, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type());
        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }
#else
        THROW("Video decoder is not enabled since ffmpeg is not present")
#endif
    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalNumpyFileSource(
    RocalContext p_context,
    const char* source_path,
    unsigned shard_count,
    RocalTensorLayout output_layout,
    std::vector<std::string> files,
    bool is_output,
    bool shuffle,
    bool loop,
    unsigned seed,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try {
        auto [max_dimensions, tensor_data_type] = evaluate_numpy_data_set(StorageType::NUMPY_DATA, source_path, files);

        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        std::vector<size_t> dims(max_dimensions.size() + 1);
        dims[0] = context->user_batch_size();
        for (uint i = 0; i < max_dimensions.size(); i++)
            dims[i + 1] = max_dimensions[i];
        auto info = TensorInfo(std::vector<size_t>(std::move(dims)),
                               context->master_graph->mem_type(),
                               tensor_data_type, op_tensor_layout);
        output = context->master_graph->create_loader_output_tensor(info);

        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);
        context->master_graph->add_node<NumpyLoaderNode>({}, {output})->init(shard_count, source_path, files, StorageType::NUMPY_DATA, DecoderType::SKIP_DECODE, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), seed, sharding_info);
        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalNumpyFileSourceSingleShard(
    RocalContext p_context,
    const char* source_path,
    RocalTensorLayout output_layout,
    std::vector<std::string> files,
    bool is_output,
    bool shuffle,
    bool loop,
    unsigned shard_id,
    unsigned shard_count,
    unsigned seed,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try {
        if (shard_count < 1)
            THROW("Shard count should be bigger than 0")

        if (shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        auto [max_dimensions, tensor_data_type] = evaluate_numpy_data_set(StorageType::NUMPY_DATA, source_path, files);

        RocalTensorlayout op_tensor_layout = static_cast<RocalTensorlayout>(output_layout);
        std::vector<size_t> dims(max_dimensions.size() + 1);
        dims[0] = context->user_batch_size();
        for (uint i = 0; i < max_dimensions.size(); i++)
            dims[i + 1] = max_dimensions[i];
        auto info = TensorInfo(std::vector<size_t>(std::move(dims)),
                               context->master_graph->mem_type(),
                               tensor_data_type, op_tensor_layout);
        output = context->master_graph->create_loader_output_tensor(info);

        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);
        context->master_graph->add_node<NumpyLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count, source_path, files, StorageType::NUMPY_DATA, DecoderType::SKIP_DECODE, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), seed, sharding_info);
        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor  ROCAL_API_CALL
rocalVideoFileSourceSingleShard(
    RocalContext p_context,
    const char* source_path,
    RocalImageColor rocal_color_format,
    RocalDecodeDevice rocal_decode_device,
    unsigned shard_id,
    unsigned shard_count,
    unsigned sequence_length,
    bool shuffle,
    bool is_output,
    bool loop,
    RocalDecoderType rocal_decoder_type,
    unsigned step,
    unsigned stride,
    bool file_list_frame_num,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    if (p_context == nullptr) {
        ERR("Invalid ROCAL context")
        return output;
    }
    auto context = static_cast<Context*>(p_context);
    try {
#ifdef ROCAL_VIDEO
        if (sequence_length == 0)
            THROW("Sequence length passed should be bigger than 0")

        if (shard_count < 1)
            THROW("Shard count should be bigger than 0")

        if (shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        // Set default step and stride values if 0 is passed
        step = (step == 0) ? sequence_length : step;
        stride = (stride == 0) ? 1 : stride;

        VideoProperties video_prop;
        DecoderType decoder_type = convert_video_decoder_type(rocal_decoder_type);
        find_video_properties(video_prop, source_path, file_list_frame_num);
        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format_sequence(rocal_color_format, context->user_batch_size(),
                                                                                                video_prop.height, video_prop.width, sequence_length);
        auto decoder_mode = convert_decoder_mode(rocal_decode_device);
        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);

        output = context->master_graph->create_loader_output_tensor(info);

        context->master_graph->add_node<VideoLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count, source_path, StorageType::VIDEO_FILE_SYSTEM, decoder_type, decoder_mode, sequence_length, step, stride, video_prop, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type());
        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }
#else
        THROW("Video decoder is not enabled since ffmpeg is not present")
#endif
    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalVideoFileResize(
    RocalContext p_context,
    const char* source_path,
    RocalImageColor rocal_color_format,
    RocalDecodeDevice rocal_decode_device,
    unsigned internal_shard_count,
    unsigned sequence_length,
    unsigned dest_width,
    unsigned dest_height,
    bool shuffle,
    bool is_output,
    bool loop,
    RocalDecoderType rocal_decoder_type,
    unsigned step,
    unsigned stride,
    bool file_list_frame_num,
    RocalResizeScalingMode scaling_mode,
    std::vector<unsigned> max_size,
    unsigned resize_shorter,
    unsigned resize_longer,
    RocalResizeInterpolationType interpolation_type,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* resize_output = nullptr;
    if (p_context == nullptr) {
        ERR("Invalid ROCAL context or invalid input image")
        return resize_output;
    }

    auto context = static_cast<Context*>(p_context);
    try {
#ifdef ROCAL_VIDEO
        if (sequence_length == 0)
            THROW("Sequence length passed should be bigger than 0")

        // Set default step and stride values if 0 is passed
        step = (step == 0) ? sequence_length : step;
        stride = (stride == 0) ? 1 : stride;

        VideoProperties video_prop;
        DecoderType decoder_type = convert_video_decoder_type(rocal_decoder_type);
        find_video_properties(video_prop, source_path, file_list_frame_num);
        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format_sequence(rocal_color_format, context->user_batch_size(),
                                                                                                video_prop.height, video_prop.width, sequence_length);
        auto decoder_mode = convert_decoder_mode(rocal_decode_device);
        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);

        Tensor* output = context->master_graph->create_loader_output_tensor(info);
        context->master_graph->add_node<VideoLoaderNode>({}, {output})->init(internal_shard_count, source_path, StorageType::VIDEO_FILE_SYSTEM, decoder_type, decoder_mode, sequence_length, step, stride, video_prop, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type());
        context->master_graph->set_loop(loop);

        if (dest_width != video_prop.width && dest_height != video_prop.height) {
            if ((dest_width | dest_height | resize_longer | resize_shorter) == 0)
                THROW("Atleast one size 'dest_width' or 'dest_height' or 'resize_shorter' or 'resize_longer' must be specified")
            if ((dest_width | dest_height) && (resize_longer | resize_shorter))
                THROW("Only one method of specifying size can be used \ndest_width and/or dest_height\nresize_shorter\nresize_longer")
            if (resize_longer && resize_shorter)
                THROW("'resize_longer' and 'resize_shorter' cannot be passed together. They are mutually exclusive.")

            unsigned out_width, out_height;
            RocalResizeScalingMode resize_scaling_mode;

            // Change the scaling mode if resize_shorter or resize_longer is specified
            if (resize_shorter) {
                resize_scaling_mode = RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_SMALLER;
                out_width = out_height = resize_shorter;
            } else if (resize_longer) {
                resize_scaling_mode = RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_LARGER;
                out_width = out_height = resize_longer;
            } else {
                resize_scaling_mode = scaling_mode;
                out_width = dest_width;
                out_height = dest_height;
            }

            std::vector<unsigned> maximum_size;
            if (max_size.size()) {
                if (max_size.size() == 1) {
                    maximum_size = {max_size[0], max_size[0]};
                } else if (max_size.size() == 2) {
                    maximum_size = {max_size[0], max_size[1]};  // {width, height}
                } else {
                    THROW("The length of max_size vector exceeds the image dimension.")
                }
            }

            // Determine the max width and height to be set to the output info
            unsigned max_out_width, max_out_height;
            if (maximum_size.size() && maximum_size[0] != 0 && maximum_size[1] != 0) {
                // If max_size is passed by the user, the resized images cannot exceed the max size,
                max_out_width = maximum_size[0];
                max_out_height = maximum_size[1];
            } else {
                // compute the output info width and height wrt the scaling modes and roi passed
                if (resize_scaling_mode == ROCAL_SCALING_MODE_STRETCH) {
                    max_out_width = out_width ? out_width : info.max_shape()[0];
                    max_out_height = out_height ? out_height : info.max_shape()[1];
                } else if (resize_scaling_mode == ROCAL_SCALING_MODE_NOT_SMALLER) {
                    max_out_width = (out_width ? out_width : out_height) * MAX_ASPECT_RATIO;
                    max_out_height = (out_height ? out_height : out_width) * MAX_ASPECT_RATIO;
                } else {
                    max_out_width = out_width ? out_width : out_height * MAX_ASPECT_RATIO;
                    max_out_height = out_height ? out_height : out_width * MAX_ASPECT_RATIO;
                }
                if (maximum_size.size() == 2) {
                    max_out_width = maximum_size[0] ? maximum_size[0] : max_out_width;
                    max_out_height = maximum_size[1] ? maximum_size[1] : max_out_height;
                }
            }

            // set the width and height in the output info
            // For the resize node, user can create an image with a different width and height
            TensorInfo output_info = info;
            std::vector<size_t> out_dims = {context->user_batch_size(), sequence_length, max_out_height,
                                            max_out_width, static_cast<unsigned>(num_of_planes)};
            output_info.set_dims(out_dims);

            resize_output = context->master_graph->create_tensor(output_info, false);

            // For the nodes that user provides the output size the dimension of all the images after this node will be fixed and equal to that size
            resize_output->reset_tensor_roi();

            std::shared_ptr<ResizeNode> resize_node = context->master_graph->add_node<ResizeNode>({output}, {resize_output});
            resize_node->init(out_width, out_height, resize_scaling_mode, maximum_size, interpolation_type);

            if (is_output) {
                auto actual_output = context->master_graph->create_tensor(output_info, is_output);
                context->master_graph->add_node<CopyNode>({resize_output}, {actual_output});
            }
        } else {
            if (is_output) {
                auto actual_output = context->master_graph->create_tensor(info, is_output);
                context->master_graph->add_node<CopyNode>({output}, {actual_output});
            }
        }
#else
        THROW("Video decoder is not enabled since ffmpeg is not present")
#endif
    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return resize_output;
}

RocalTensor ROCAL_API_CALL
rocalVideoFileResizeSingleShard(
    RocalContext p_context,
    const char* source_path,
    RocalImageColor rocal_color_format,
    RocalDecodeDevice rocal_decode_device,
    unsigned shard_id,
    unsigned shard_count,
    unsigned sequence_length,
    unsigned dest_width,
    unsigned dest_height,
    bool shuffle,
    bool is_output,
    bool loop,
    RocalDecoderType rocal_decoder_type,
    unsigned step,
    unsigned stride,
    bool file_list_frame_num,
    RocalResizeScalingMode scaling_mode,
    std::vector<unsigned> max_size,
    unsigned resize_shorter,
    unsigned resize_longer,
    RocalResizeInterpolationType interpolation_type,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* resize_output = nullptr;
    if (p_context == nullptr) {
        ERR("Invalid ROCAL context or invalid input image")
        return resize_output;
    }

    auto context = static_cast<Context*>(p_context);
    try {
#ifdef ROCAL_VIDEO
        if (sequence_length == 0)
            THROW("Sequence length passed should be bigger than 0")

        if (shard_count < 1)
            THROW("Shard count should be bigger than 0")

        if (shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")

        // Set default step and stride values if 0 is passed
        step = (step == 0) ? sequence_length : step;
        stride = (stride == 0) ? 1 : stride;

        VideoProperties video_prop;
        DecoderType decoder_type = convert_video_decoder_type(rocal_decoder_type);
        find_video_properties(video_prop, source_path, file_list_frame_num);
        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format_sequence(rocal_color_format, context->user_batch_size(),
                                                                                                video_prop.height, video_prop.width, sequence_length);
        auto decoder_mode = convert_decoder_mode(rocal_decode_device);
        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);
        Tensor* output = context->master_graph->create_loader_output_tensor(info);
        context->master_graph->add_node<VideoLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count, source_path, StorageType::VIDEO_FILE_SYSTEM, decoder_type, decoder_mode, sequence_length, step, stride, video_prop, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type());
        context->master_graph->set_loop(loop);

        if (dest_width != video_prop.width && dest_height != video_prop.height) {
            if ((dest_width | dest_height | resize_longer | resize_shorter) == 0)
                THROW("Atleast one size 'dest_width' or 'dest_height' or 'resize_shorter' or 'resize_longer' must be specified")
            if ((dest_width | dest_height) && (resize_longer | resize_shorter))
                THROW("Only one method of specifying size can be used \ndest_width and/or dest_height\nresize_shorter\nresize_longer")
            if (resize_longer && resize_shorter)
                THROW("'resize_longer' and 'resize_shorter' cannot be passed together. They are mutually exclusive.")

            unsigned out_width, out_height;
            RocalResizeScalingMode resize_scaling_mode;

            // Change the scaling mode if resize_shorter or resize_longer is specified
            if (resize_shorter) {
                resize_scaling_mode = RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_SMALLER;
                out_width = out_height = resize_shorter;
            } else if (resize_longer) {
                resize_scaling_mode = RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_LARGER;
                out_width = out_height = resize_longer;
            } else {
                resize_scaling_mode = scaling_mode;
                out_width = dest_width;
                out_height = dest_height;
            }

            std::vector<unsigned> maximum_size;
            if (max_size.size()) {
                if (max_size.size() == 1) {
                    maximum_size = {max_size[0], max_size[0]};
                } else if (max_size.size() == 2) {
                    maximum_size = {max_size[0], max_size[1]};  // {width, height}
                } else {
                    THROW("The length of max_size vector exceeds the image dimension.")
                }
            }

            // Determine the max width and height to be set to the output info
            unsigned max_out_width, max_out_height;
            if (maximum_size.size() && maximum_size[0] != 0 && maximum_size[1] != 0) {
                // If max_size is passed by the user, the resized images cannot exceed the max size,
                max_out_width = maximum_size[0];
                max_out_height = maximum_size[1];
            } else {
                // compute the output info width and height wrt the scaling modes and roi passed
                if (resize_scaling_mode == ROCAL_SCALING_MODE_STRETCH) {
                    max_out_width = out_width ? out_width : info.max_shape()[0];
                    max_out_height = out_height ? out_height : info.max_shape()[1];
                } else if (resize_scaling_mode == ROCAL_SCALING_MODE_NOT_SMALLER) {
                    max_out_width = (out_width ? out_width : out_height) * MAX_ASPECT_RATIO;
                    max_out_height = (out_height ? out_height : out_width) * MAX_ASPECT_RATIO;
                } else {
                    max_out_width = out_width ? out_width : out_height * MAX_ASPECT_RATIO;
                    max_out_height = out_height ? out_height : out_width * MAX_ASPECT_RATIO;
                }
                if (maximum_size.size() == 2) {
                    max_out_width = maximum_size[0] ? maximum_size[0] : max_out_width;
                    max_out_height = maximum_size[1] ? maximum_size[1] : max_out_height;
                }
            }

            // set the width and height in the output info
            // For the resize node, user can create an image with a different width and height
            TensorInfo output_info = info;
            std::vector<size_t> out_dims = {context->user_batch_size(), sequence_length, max_out_height,
                                            max_out_width, static_cast<unsigned>(num_of_planes)};
            output_info.set_dims(out_dims);

            resize_output = context->master_graph->create_tensor(output_info, false);
            // For the nodes that user provides the output size the dimension of all the images after this node will be fixed and equal to that size
            resize_output->reset_tensor_roi();

            std::shared_ptr<ResizeNode> resize_node = context->master_graph->add_node<ResizeNode>({output}, {resize_output});
            resize_node->init(out_width, out_height, resize_scaling_mode, maximum_size, interpolation_type);

            if (is_output) {
                auto actual_output = context->master_graph->create_tensor(output_info, is_output);
                context->master_graph->add_node<CopyNode>({resize_output}, {actual_output});
            }
        } else {
            if (is_output) {
                auto actual_output = context->master_graph->create_tensor(info, is_output);
                context->master_graph->add_node<CopyNode>({output}, {actual_output});
            }
        }
#else
        THROW("Video decoder is not enabled since ffmpeg is not present")
#endif
    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return resize_output;
}

// loader for CFAR10 raw data: Can be used for other raw data loaders as well
RocalTensor ROCAL_API_CALL
rocalRawCIFAR10Source(
    RocalContext p_context,
    const char* source_path,
    RocalImageColor rocal_color_format,
    bool is_output,
    unsigned out_width,
    unsigned out_height,
    const char* filename_prefix,
    bool loop) {
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try {
        if (out_width == 0 || out_height == 0) {
            THROW("Invalid video input width and height");
        } else {
            LOG("User input size " + TOSTR(out_width) + " x " + TOSTR(out_height));
        }

        auto [width, height] = std::make_tuple(out_width, out_height);
        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format(rocal_color_format, context->user_batch_size(), height, width);
        INFO("Internal buffer size width = " + TOSTR(width) + " height = " + TOSTR(height) + " depth = " + TOSTR(num_of_planes))

        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);
        output = context->master_graph->create_loader_output_tensor(info);

        context->master_graph->add_node<Cifar10LoaderNode>({}, {output})->init(source_path, "", StorageType::UNCOMPRESSED_BINARY_DATA, loop, context->user_batch_size(), context->master_graph->mem_type(), filename_prefix);
        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalJpegExternalFileSource(
    RocalContext p_context,
    RocalImageColor rocal_color_format,
    bool is_output,
    bool shuffle,
    bool loop,
    RocalImageSizeEvaluationPolicy decode_size_policy,
    unsigned max_width,
    unsigned max_height,
    RocalDecoderType dec_type,
    RocalExternalSourceMode external_source_mode,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try {
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);
        DecoderType decType = DecoderType::TURBO_JPEG;  // default
        if (dec_type == ROCAL_DECODER_OPENCV) decType = DecoderType::OPENCV_DEC;
        if ((decode_size_policy == ROCAL_USE_MAX_SIZE) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED))
            THROW("use_max_size is not supported in external source reader");

        // user need to specify this
        if (max_width == 0 || max_height == 0) {
            THROW("Invalid input max width and height");
        } else {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }

        auto [width, height] = std::make_tuple(max_width, max_height);
        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format(rocal_color_format, context->user_batch_size(), height, width);
        INFO("Internal buffer size width = " + TOSTR(width) + " height = " + TOSTR(height) + " depth = " + TOSTR(num_of_planes))
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);
        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);
        output = context->master_graph->create_loader_output_tensor(info);
        context->master_graph->set_external_source_reader_flag();

        unsigned shard_count = 1;  // Hardcoding the shard count to 1 for now.
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(shard_count);
        context->master_graph->add_node<ImageLoaderNode>({}, {output})->init(shard_count, cpu_num_threads, "", "", std::map<std::string, std::string>(), StorageType::EXTERNAL_FILE_SOURCE, decType, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), decoder_keep_original, sharding_info, "", 0, 0, 0, ExternalSourceFileMode(external_source_mode));
        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }

    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalAudioFileSourceSingleShard(
    RocalContext p_context,
    const char* source_path,
    const char* source_file_list_path,
    unsigned shard_id,
    unsigned shard_count,
    bool is_output,
    bool shuffle,
    bool loop,
    bool downmix,
    RocalImageSizeEvaluationPolicy decode_size_policy,
    unsigned max_decoded_samples,
    unsigned max_decoded_channels, 
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try {
#ifdef ROCAL_AUDIO
        if (shard_count < 1)
            THROW("Shard count should be bigger than 0")
        if (shard_id >= shard_count)
            THROW("Shard id should be smaller than shard count")
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);

        if (use_input_dimension && (max_decoded_samples == 0 || max_decoded_channels == 0)) {
            THROW("Invalid input max width and height");
        } else {
            LOG("User input size " + TOSTR(max_decoded_samples) + " x " + TOSTR(max_decoded_channels))
        }
        auto [max_sample_length, max_channels] = use_input_dimension ? std::make_tuple(max_decoded_samples, max_decoded_channels) : evaluate_audio_data_set(StorageType::FILE_SYSTEM, DecoderType::AUDIO_SOFTWARE_DECODE, source_path, source_file_list_path, context->master_graph->meta_data_reader());
        INFO("Internal buffer size for audio samples = " + TOSTR(max_sample_length) + " and channels = " + TOSTR(max_channels))
        RocalTensorDataType tensor_data_type = RocalTensorDataType::FP32;
        std::vector<size_t> dims = {context->user_batch_size(), max_sample_length, max_channels};
        auto info = TensorInfo(std::vector<size_t>(std::move(dims)),
                               context->master_graph->mem_type(),
                               tensor_data_type,
                               RocalTensorlayout::NHW);
        output = context->master_graph->create_loader_output_tensor(info);
        output->reset_audio_sample_rate();
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(shard_count);
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);
        context->master_graph->add_node<AudioLoaderSingleShardNode>({}, {output})->Init(shard_id, shard_count, cpu_num_threads, source_path, source_file_list_path, StorageType::FILE_SYSTEM, DecoderType::AUDIO_SOFTWARE_DECODE, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), sharding_info);
        context->master_graph->set_loop(loop);
        if (downmix && (max_channels > 1)) {
            TensorInfo output_info = info;
            std::vector<size_t> output_dims = {context->user_batch_size(), info.dims()[1], 1};
            output_info.set_dims(output_dims);
            auto downmixed_output = context->master_graph->create_tensor(output_info, false);
            std::shared_ptr<DownmixNode> downmix_node = context->master_graph->add_node<DownmixNode>({output}, {downmixed_output});

            if (is_output) {
                auto actual_output = context->master_graph->create_tensor(output_info, is_output);
                context->master_graph->add_node<CopyNode>({downmixed_output}, {actual_output});
            }
            return downmixed_output;
        } else if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }
#else
        THROW("Audio decoder is not enabled since sndfile is not present")
#endif
    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalAudioFileSource(
    RocalContext p_context,
    const char* source_path,
    const char* source_file_list_path,
    unsigned shard_count,
    bool is_output,
    bool shuffle,
    bool loop,
    bool downmix,
    RocalImageSizeEvaluationPolicy decode_size_policy,
    unsigned max_decoded_samples,
    unsigned max_decoded_channels,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try {
#ifdef ROCAL_AUDIO
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);

        if (use_input_dimension && (max_decoded_samples == 0 || max_decoded_channels == 0)) {
            THROW("Invalid input max width and height");
        } else {
            LOG("User input size " + TOSTR(max_decoded_samples) + " x " + TOSTR(max_decoded_channels))
        }
        auto [max_sample_length, max_channels] = use_input_dimension ? std::make_tuple(max_decoded_samples, max_decoded_channels) : evaluate_audio_data_set(StorageType::FILE_SYSTEM, DecoderType::AUDIO_SOFTWARE_DECODE, source_path, source_file_list_path, context->master_graph->meta_data_reader());
        INFO("Internal buffer size for audio samples = " + TOSTR(max_sample_length) + " and channels = " + TOSTR(max_channels))
        RocalTensorDataType tensor_data_type = RocalTensorDataType::FP32;
        std::vector<size_t> dims = {context->user_batch_size(), max_sample_length, max_channels};
        auto info = TensorInfo(std::vector<size_t>(std::move(dims)),
                               context->master_graph->mem_type(),
                               tensor_data_type,
                               RocalTensorlayout::NHW);
        output = context->master_graph->create_loader_output_tensor(info);
        output->reset_audio_sample_rate();

        if (shard_count < 1)
            THROW("internal shard count should be bigger than 0")
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(shard_count);
        context->master_graph->add_node<AudioLoaderNode>({}, {output})->Init(shard_count, cpu_num_threads, source_path, source_file_list_path, StorageType::FILE_SYSTEM, DecoderType::AUDIO_SOFTWARE_DECODE, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), sharding_info);
        context->master_graph->set_loop(loop);
        if (downmix && (max_channels > 1)) {
            TensorInfo output_info = info;
            std::vector<size_t> output_dims = {context->user_batch_size(), info.dims()[1], 1};
            output_info.set_dims(output_dims);
            auto downmixed_output = context->master_graph->create_tensor(output_info, false);
            std::shared_ptr<DownmixNode> downmix_node = context->master_graph->add_node<DownmixNode>({output}, {downmixed_output});
            if (is_output) {
                auto actual_output = context->master_graph->create_tensor(output_info, is_output);
                context->master_graph->add_node<CopyNode>({downmixed_output}, {actual_output});
            }
            return downmixed_output;
        } else if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }
#else
        THROW("Audio decoder is not enabled since sndfile is not present")
#endif
    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalTensor ROCAL_API_CALL
rocalWebDatasetSourceSingleShard(
    RocalContext p_context,
    const char* source_path,
    const char* index_path,
    RocalImageColor rocal_color_format,
    unsigned shard_id,
    unsigned shard_count,
    bool is_output,
    bool shuffle,
    bool loop,
    RocalImageSizeEvaluationPolicy decode_size_policy,
    unsigned max_width,
    unsigned max_height,
    RocalDecoderType dec_type,
    RocalShardingInfo rocal_sharding_info) {
    Tensor* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    try {
#ifdef ENABLE_WDS
        bool use_input_dimension = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE) || (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED);
        bool decoder_keep_original = (decode_size_policy == ROCAL_USE_USER_GIVEN_SIZE_RESTRICTED) || (decode_size_policy == ROCAL_USE_MAX_SIZE_RESTRICTED);
        DecoderType decType = DecoderType::TURBO_JPEG;  // default
        if (dec_type == ROCAL_DECODER_OPENCV) {
            decType = DecoderType::OPENCV_DEC;
        } else if (dec_type == ROCAL_DECODER_HW_JPEG) {
            decType = DecoderType::HW_JPEG_DEC;
        }

        if (shard_count < 1) {
            THROW("Shard count should be bigger than 0");
        } else if (shard_id >= shard_count) {
            THROW("Shard id should be smaller than shard count");
        }

        if (use_input_dimension && (max_width == 0 || max_height == 0)) {
            THROW("Invalid input max width and height");
        } else {
            LOG("User input size " + TOSTR(max_width) + " x " + TOSTR(max_height))
        }
        auto [width, height] = evaluate_image_data_set(decode_size_policy, StorageType::WEBDATASET_RECORDS, decType, source_path, index_path);
        auto [color_format, tensor_layout, dims, num_of_planes] = convert_color_format(rocal_color_format, context->user_batch_size(), height, width);
        INFO("Internal buffer size width = " + TOSTR(width) + " height = " + TOSTR(height) + " depth = " + TOSTR(num_of_planes))

        auto info = TensorInfo(std::move(dims),
                               context->master_graph->mem_type(),
                               RocalTensorDataType::UINT8,
                               tensor_layout,
                               color_format);
        output = context->master_graph->create_loader_output_tensor(info);
        auto cpu_num_threads = context->master_graph->calculate_cpu_num_threads(shard_count);
        ShardingInfo sharding_info(convert_last_batch_policy(rocal_sharding_info.last_batch_policy), rocal_sharding_info.pad_last_batch_repeated, rocal_sharding_info.stick_to_shard, rocal_sharding_info.shard_size);
        context->master_graph->add_node<ImageLoaderSingleShardNode>({}, {output})->init(shard_id, shard_count, cpu_num_threads, source_path, "", StorageType::WEBDATASET_RECORDS, decType, shuffle, loop, context->user_batch_size(), context->master_graph->mem_type(), context->master_graph->meta_data_reader(), decoder_keep_original, sharding_info, 
                                                                                        std::map<std::string, std::string>(), 0, 0, 0, ExternalSourceFileMode::NONE, index_path);
        context->master_graph->set_loop(loop);

        if (is_output) {
            auto actual_output = context->master_graph->create_tensor(info, is_output);
            context->master_graph->add_node<CopyNode>({output}, {actual_output});
        }
#else
        THROW("Webdataset reader is not enabled since libtar is not present")
#endif
    } catch (const std::exception& e) {
        context->capture_error(e.what());
        std::cerr << e.what() << '\n';
    }
    return output;
}

RocalStatus ROCAL_API_CALL
rocalResetLoaders(RocalContext p_context) {
    auto context = static_cast<Context*>(p_context);
    try {
        context->master_graph->reset();
    } catch (const std::exception& e) {
        context->capture_error(e.what());
        ERR(e.what())
        return ROCAL_RUNTIME_ERROR;
    }
    return ROCAL_OK;
}
