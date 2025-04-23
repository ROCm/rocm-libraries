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

#include "loaders/image/node_numpy_loader.h"

#include "pipeline/exception.h"

NumpyLoaderNode::NumpyLoaderNode(Tensor *output, void *device_resources) : Node({}, {output}) {
    _loader_module = std::make_shared<NumpyLoaderSharded>(device_resources);
}

void NumpyLoaderNode::init(unsigned internal_shard_count, const std::string &source_path, const std::vector<std::string> &files, StorageType storage_type, DecoderType decoder_type, bool shuffle, bool loop,
                           size_t load_batch_count, RocalMemType mem_type, unsigned seed, const ShardingInfo& sharding_info) {
    if (!_loader_module)
        THROW("ERROR: loader module is not set for NumpyLoaderNode, cannot initialize")
    if (internal_shard_count < 1)
        THROW("Shard count should be greater than or equal to one")
    _loader_module->set_output(_outputs[0]);
    // Set reader and decoder config accordingly for the NumpyLoaderNode
    auto reader_cfg = ReaderConfig(storage_type, source_path, "", std::map<std::string, std::string>(), shuffle, loop);
    reader_cfg.set_shard_count(internal_shard_count);
    reader_cfg.set_batch_count(load_batch_count);
    reader_cfg.set_sharding_info(sharding_info);
    reader_cfg.set_files_list(files);
    reader_cfg.set_seed(seed);
    _loader_module->initialize(reader_cfg, DecoderConfig(DecoderType::SKIP_DECODE), mem_type, _batch_size);
    _loader_module->start_loading();
}

std::shared_ptr<LoaderModule> NumpyLoaderNode::get_loader_module() {
    if (!_loader_module)
        WRN("NumpyLoaderNode's loader module is null, not initialized")
    return _loader_module;
}

NumpyLoaderNode::~NumpyLoaderNode() {
    _loader_module = nullptr;
}
