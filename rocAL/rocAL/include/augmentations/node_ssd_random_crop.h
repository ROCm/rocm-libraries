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
#include "augmentations/geometry_augmentations/node_crop.h"
#include "parameters/parameter_crop_factory.h"
#include "parameters/parameter_factory.h"

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

class SSDRandomCropNode : public CropNode {
   public:
    SSDRandomCropNode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs);
    SSDRandomCropNode() = delete;
    void init(FloatParam *crop_area_factor, FloatParam *crop_aspect_ratio, FloatParam *x_drift, FloatParam *y_drift, int num_of_attempts);
    unsigned int get_dst_width() { return _outputs[0]->info().max_shape()[0]; }
    unsigned int get_dst_height() { return _outputs[0]->info().max_shape()[1]; }
    std::shared_ptr<RocalRandomCropParam> get_crop_param() { return _crop_param; }
    float get_threshold() { return _threshold; }
    std::vector<std::pair<float, float>> get_iou_range() { return _iou_range; }
    bool is_entire_iou() { return _entire_iou; }
    void set_meta_data_batch() {}

   protected:
    void create_node() override;
    void update_node() override;

   private:
    std::shared_ptr<RocalRandomCropParam> _meta_crop_param;
    std::vector<uint> _x1_val, _y1_val, _crop_width_val, _crop_height_val;
    size_t _dest_width;
    size_t _dest_height;
    float _threshold = 0.05;
    std::vector<std::pair<float, float>> _iou_range;
    int _num_of_attempts = 20;
    bool _entire_iou = false;
    std::shared_ptr<RocalRandomCropParam> _crop_param;
    SeededRNG<std::mt19937, 4> _rngs;  // setting the state_size to 4 for 4 random parameters.
};
