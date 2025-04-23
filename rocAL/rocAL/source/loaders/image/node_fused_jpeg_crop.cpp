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

#include "loaders/image/node_fused_jpeg_crop.h"

#include "pipeline/exception.h"

FusedJpegCropNode::FusedJpegCropNode(Tensor *output, void *device_resources) : Node({}, {output}) {
    _loader_module = std::make_shared<ImageLoaderSharded>(device_resources);
}

void FusedJpegCropNode::init(unsigned internal_shard_count, unsigned cpu_num_threads, const std::string &source_path, const std::string &json_path, StorageType storage_type,
                             DecoderType decoder_type, bool shuffle, bool loop, size_t load_batch_count, RocalMemType mem_type, std::shared_ptr<MetaDataReader> meta_data_reader,
                             unsigned num_attempts, std::vector<float> &random_area, std::vector<float> &random_aspect_ratio, const ShardingInfo& sharding_info) {
    if (!_loader_module)
        THROW("ERROR: loader module is not set for FusedJpegCropNode, cannot initialize")
    if (internal_shard_count < 1)
        THROW("Shard count should be greater than or equal to one")
    _loader_module->set_output(_outputs[0]);
    // Set reader and decoder config accordingly for the FusedJpegCropNode
    auto reader_cfg = ReaderConfig(storage_type, source_path, json_path, std::map<std::string, std::string>(), shuffle, loop);
    reader_cfg.set_shard_count(internal_shard_count);
    reader_cfg.set_cpu_num_threads(cpu_num_threads);
    reader_cfg.set_batch_count(load_batch_count);
    reader_cfg.set_meta_data_reader(meta_data_reader);
    reader_cfg.set_sharding_info(sharding_info);
    auto decoder_cfg = DecoderConfig(decoder_type);

    decoder_cfg.set_random_area(random_area);
    decoder_cfg.set_random_aspect_ratio(random_aspect_ratio);
    decoder_cfg.set_num_attempts(num_attempts);
    decoder_cfg.set_seed(ParameterFactory::instance()->get_seed());
    _loader_module->initialize(reader_cfg, decoder_cfg,
                               mem_type,
                               _batch_size);
    _loader_module->start_loading();
}

std::shared_ptr<LoaderModule> FusedJpegCropNode::get_loader_module() {
    if (!_loader_module)
        WRN("FusedJpegCropNode's loader module is null, not initialized")
    return _loader_module;
}

FusedJpegCropNode::~FusedJpegCropNode() {
    _loader_module = nullptr;
}
