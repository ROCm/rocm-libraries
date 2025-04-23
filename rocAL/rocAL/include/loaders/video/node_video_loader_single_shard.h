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
#include <tuple>

#include "pipeline/graph.h"
#include "pipeline/node.h"
#include "video_loader_sharded.h"

#ifdef ROCAL_VIDEO
class VideoLoaderSingleShardNode : public Node {
   public:
    VideoLoaderSingleShardNode(Tensor *output, void *device_resources);
    ~VideoLoaderSingleShardNode() override;

    /// \param user_shard_count shard count from user
    /// \param  user_shard_id shard id from user
    /// \param source_path Defines the path that includes the video dataset
    /// \param load_batch_count Defines the quantum count of the sequences to be loaded. It's usually equal to the user's batch size.
    /// The loader will repeat sequences if necessary to be able to have sequences in multiples of the load_batch_count,
    /// for example if there are 10 sequences in the dataset and load_batch_count is 3, the loader repeats 2 sequences as if there are 12 sequences available.
    void init(unsigned shard_id, unsigned shard_count, const std::string &source_path, StorageType storage_type, DecoderType decoder_type, DecodeMode decoder_mode,
              unsigned sequence_length, unsigned step, unsigned stride, VideoProperties &video_prop, bool shuffle, bool loop, size_t load_batch_count, RocalMemType mem_type);

    std::shared_ptr<LoaderModule> get_loader_module();

   protected:
    void create_node() override{};
    void update_node() override{};

   private:
    DecodeMode _decode_mode = DecodeMode::CPU;
    std::shared_ptr<VideoLoader> _loader_module = nullptr;
};
#endif
