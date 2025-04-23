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

#include "loaders/image/node_cifar10_loader.h"

#include "pipeline/exception.h"

Cifar10LoaderNode::Cifar10LoaderNode(Tensor *output, void *device_resources) : Node({}, {output}) {
    _loader_module = std::make_shared<CIFAR10DataLoader>(device_resources);
}

void Cifar10LoaderNode::init(const std::string &source_path, const std::string &json_path, StorageType storage_type,
                             bool loop, size_t load_batch_count, RocalMemType mem_type, const std::string &file_prefix) {
    if (!_loader_module)
        THROW("ERROR: loader module is not set for Cifar10LoaderNode, cannot initialize")
    _loader_module->set_output(_outputs[0]);
    // Set reader and decoder config accordingly for the Cifar10LoaderNode
    auto reader_cfg = ReaderConfig(storage_type, source_path, json_path, std::map<std::string, std::string>(), loop);
    reader_cfg.set_batch_count(load_batch_count);
    reader_cfg.set_file_prefix(file_prefix);
    // DecoderConfig will be ignored in loader. Just passing it for api match
    _loader_module->initialize(reader_cfg, DecoderConfig(DecoderType::TURBO_JPEG),
                               mem_type, _batch_size);
    _loader_module->start_loading();
}

std::shared_ptr<LoaderModule> Cifar10LoaderNode::get_loader_module() {
    if (!_loader_module)
        WRN("Cifar10LoaderNode's loader module is null, not initialized")
    return _loader_module;
}

Cifar10LoaderNode::~Cifar10LoaderNode() {
    _loader_module = nullptr;
}