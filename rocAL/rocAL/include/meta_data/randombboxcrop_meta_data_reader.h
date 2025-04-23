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

#include <memory>
#include <string>

#include "pipeline/graph.h"
#include "meta_data/meta_data_reader.h"
#include "parameters/parameter_factory.h"
#include "meta_data/randombboxcrop_meta_data.h"
#include "readers/image/image_reader.h"

enum class RandomBBoxCrop_MetaDataReaderType {
    RandomBBoxCropReader = 0,
};
enum class RandomBBoxCrop_MetaDataType {
    BoundingBox
};

struct RandomBBoxCrop_MetaDataConfig {
   private:
    RandomBBoxCrop_MetaDataType _type;
    RandomBBoxCrop_MetaDataReaderType _reader_type;
    bool _all_boxes_overlap;
    bool _no_crop;
    FloatParam* _aspect_ratio;
    bool _has_shape;
    int _crop_width;
    int _crop_height;
    int _num_attempts;
    FloatParam* _scaling;
    int _total_num_attempts;
    int64_t _seed;

   public:
    RandomBBoxCrop_MetaDataConfig(const RandomBBoxCrop_MetaDataType& type, const RandomBBoxCrop_MetaDataReaderType& reader_type, const bool& all_boxes_overlap,
                                  const bool& no_crop, FloatParam* aspect_ratio, const bool& has_shape, const int& crop_width, const int& crop_height, const int& num_attempts,
                                  FloatParam* scaling, const int& total_num_attempts, const int64_t& seed) : _type(type), _reader_type(reader_type), _all_boxes_overlap(all_boxes_overlap), _no_crop(no_crop), _aspect_ratio(aspect_ratio), _has_shape(has_shape), _crop_width(crop_width), _crop_height(crop_height), _num_attempts(num_attempts), _scaling(scaling), _total_num_attempts(total_num_attempts), _seed(seed) {}
    RandomBBoxCrop_MetaDataConfig() = delete;
    RandomBBoxCrop_MetaDataType type() const { return _type; }
    RandomBBoxCrop_MetaDataReaderType reader_type() const { return _reader_type; }
    bool all_boxes_overlap() const { return _all_boxes_overlap; }
    bool no_crop() const { return _no_crop; }
    bool has_shape() const { return _has_shape; }
    int crop_width() const { return _crop_width; }
    int crop_height() const { return _crop_height; }
    FloatParam* aspect_ratio() const { return _aspect_ratio; }
    FloatParam* scaling() const { return _scaling; }
    int num_attempts() const { return _num_attempts; }
    int total_num_attempts() const { return _total_num_attempts; }
    int seed() const { return _seed; }
};

class RandomBBoxCrop_MetaDataReader {
   public:
    enum class Status {
        OK = 0
    };
    virtual ~RandomBBoxCrop_MetaDataReader() = default;
    virtual void init(const RandomBBoxCrop_MetaDataConfig& cfg, std::shared_ptr<CropCordBatch> meta_data_batch) = 0;
    virtual void read_all() = 0;                                                                                     // Reads all the meta data information
    virtual void lookup(const std::vector<std::string>& image_names) = 0;                                            // finds meta_data info associated with given names and fills the output
    virtual std::vector<std::vector<float>> get_batch_crop_coords(const std::vector<std::string>& image_names) = 0;  // returns the crop coords for a batch
    virtual void release() = 0;                                                                                      // Deletes the loaded information
    virtual void set_meta_data(std::shared_ptr<MetaDataReader> meta_data_reader) = 0;
    virtual std::shared_ptr<CropCordBatch> get_output() = 0;
    virtual pCropCord get_crop_cord(const std::string& image_names) = 0;
};
