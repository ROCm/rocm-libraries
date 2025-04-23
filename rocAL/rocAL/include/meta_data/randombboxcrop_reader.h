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
#include "pipeline/graph.h"
#include <vx_ext_rpp.h>

#include <map>
#include <random>

#include "meta_data/caffe2_meta_data_reader_detection.h"
#include "meta_data/caffe_meta_data_reader_detection.h"
#include "meta_data/coco_meta_data_reader.h"
#include "pipeline/commons.h"
#include "meta_data/meta_data_reader.h"
#include "parameters/parameter_crop_factory.h"
#include "parameters/parameter_factory.h"
#include "meta_data/randombboxcrop_meta_data_reader.h"
#include "meta_data/tf_meta_data_reader_detection.h"

// todo:: move this to common header
template <typename T = std::mt19937, std::size_t state_size = T::state_size>
class SeededRNG {
    /*
     * @param batch_size How many RNGs to store
     * @param state_size How many seed are used to initialize one RNG. Used to lower probablity of
     * collisions between seeds used to initialize RNGs in different operators.
     */
   public:
    SeededRNG(int batch_size = 128) {
        std::random_device source;
        _batch_size = batch_size;
        std::size_t _random_data_size = state_size * batch_size;
        std::vector<std::random_device::result_type> random_data(_random_data_size);
        std::generate(random_data.begin(), random_data.end(), std::ref(source));
        _rngs.reserve(batch_size);
        for (int i = 0; i < (int)(_batch_size * state_size); i += state_size) {
            std::seed_seq seeds(std::begin(random_data) + i, std::begin(random_data) + i + state_size);
            _rngs.emplace_back(T(seeds));
        }
    }

    /**
     * Returns engine corresponding to given sample ID
     */
    T &operator[](int sample) noexcept {
        return _rngs[sample % _batch_size];
    }

   private:
    std::vector<T> _rngs;
    int _batch_size;
};

class RandomBBoxCropReader : public RandomBBoxCrop_MetaDataReader {
   public:
    void init(const RandomBBoxCrop_MetaDataConfig &cfg, std::shared_ptr<CropCordBatch> meta_data_batch) override;
    void lookup(const std::vector<std::string> &image_names) override;
    std::vector<std::vector<float>> get_batch_crop_coords(const std::vector<std::string> &image_names) override;
    void read_all() override;
    void release() override;
    void print_map_contents();
    void update_meta_data();
    std::shared_ptr<CropCordBatch> get_output() override { return _output; }
    bool is_entire_iou() { return _entire_iou; }
    void set_meta_data(std::shared_ptr<MetaDataReader> meta_data_reader) override;
    pCropCord get_crop_cord(const std::string &image_names) override;
    RandomBBoxCropReader();

   private:
    std::shared_ptr<MetaDataReader> _meta_data_reader = nullptr;
    std::map<std::string, std::shared_ptr<MetaData>> _meta_bbox_map_content;
    bool _all_boxes_overlap;
    bool _no_crop;
    bool _has_shape;
    int _num_of_attempts = 20;
    int _total_num_of_attempts = 0;
    bool _entire_iou = false;
    FloatParam *crop_area_factor = NULL;
    FloatParam *crop_aspect_ratio = NULL;
    int _user_batch_size;
    int64_t _seed;
    void add(std::string image_name, BoundingBoxCord bbox);
    std::vector<std::vector<float>> _crop_coords;
    bool exists(const std::string &image_name);
    std::map<std::string, std::shared_ptr<CropCord>> _map_content;
    std::map<std::string, std::shared_ptr<CropCord>>::iterator _itr;
    std::shared_ptr<Graph> _graph = nullptr;
    std::shared_ptr<CropCordBatch> _output;
    SeededRNG<std::mt19937, 4> _rngs;  // setting the state_size to 4 for 4 random parameters.
    size_t _sample_cnt;
};
