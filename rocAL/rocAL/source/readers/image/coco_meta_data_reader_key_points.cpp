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

#include "meta_data/coco_meta_data_reader_key_points.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <utility>

#include "meta_data/lookahead_parser.h"

using namespace std;

void COCOMetaDataReaderKeyPoints::init(const MetaDataConfig &cfg, pMetaDataBatch meta_data_batch) {
    _path = cfg.path();
    _output = meta_data_batch;
    _out_img_width = cfg.out_img_width();
    _out_img_height = cfg.out_img_height();
}

bool COCOMetaDataReaderKeyPoints::exists(const std::string &image_name) {
    return _map_content.find(image_name) != _map_content.end();
}

void COCOMetaDataReaderKeyPoints::lookup(const std::vector<std::string> &image_names) {
    if (image_names.empty()) {
        WRN("No image names passed")
        return;
    }
    if (image_names.size() != (unsigned)_output->size())
        _output->resize(image_names.size());

    JointsDataBatch joints_data_batch;
    for (unsigned i = 0; i < image_names.size(); i++) {
        auto image_name = image_names[i];
        auto it = _map_content.find(image_name);
        if (_map_content.end() == it)
            THROW("ERROR: Given name not present in the map" + image_name);
        const JointsData *joints_data;
        joints_data = &(it->second->get_joints_data());
        joints_data_batch.image_id_batch.push_back(joints_data->image_id);
        joints_data_batch.annotation_id_batch.push_back(joints_data->annotation_id);
        joints_data_batch.image_path_batch.push_back(joints_data->image_path);
        joints_data_batch.center_batch.push_back({joints_data->center[0], joints_data->center[1]});
        joints_data_batch.scale_batch.push_back({joints_data->scale[0], joints_data->scale[1]});
        joints_data_batch.joints_batch.push_back(joints_data->joints);
        joints_data_batch.joints_visibility_batch.push_back(joints_data->joints_visibility);
        joints_data_batch.score_batch.push_back(joints_data->score);
        joints_data_batch.rotation_batch.push_back(joints_data->rotation);
    }
    _output->get_joints_data_batch() = joints_data_batch;
}

void COCOMetaDataReaderKeyPoints::add(std::string image_id, ImgSize image_size, JointsData *joints_data) {
    pMetaDataKeyPoint info = std::make_shared<KeyPoint>(image_size, joints_data);
    _map_content.insert(pair<std::string, std::shared_ptr<KeyPoint>>(image_id, info));
}

void COCOMetaDataReaderKeyPoints::print_map_contents() {
    JointsData joints_data;
    for (auto &elem : _map_content) {
        std::cout << "\nName :\t " << elem.first << std::endl;
        joints_data = elem.second->get_joints_data();
        std::cout << "ImageID: " << joints_data.image_id << std::endl;
        std::cout << "AnnotationID: " << joints_data.annotation_id << std::endl;
        std::cout << "ImagePath: " << joints_data.image_path << std::endl;
        std::cout << "center (x,y) : " << joints_data.center[0] << " " << joints_data.center[1] << std::endl;
        std::cout << "scale (w,h) : " << joints_data.scale[0] << " " << joints_data.scale[1] << std::endl;
        for (unsigned int i = 0; i < NUMBER_OF_JOINTS; i++) {
            std::cout << " x : " << joints_data.joints[i][0] << " , y : " << joints_data.joints[i][1] << " , v : " << joints_data.joints_visibility[i][0] << std::endl;
        }
        std::cout << "Score: " << joints_data.score << std::endl;
        std::cout << "Rotation: " << joints_data.rotation << std::endl;
    }
}

void COCOMetaDataReaderKeyPoints::read_all(const std::string &path) {
    _coco_metadata_read_time.start();  // Debug timing
    std::ifstream f;
    f.open(path, std::ifstream::in | std::ios::binary);
    if (f.fail())
        THROW("ERROR: Given annotations file not present " + path);
    f.ignore(std::numeric_limits<std::streamsize>::max());
    auto file_size = f.gcount();
    f.clear();             //  Since ignore will have set eof.
    if (file_size == 0) {  // If file is empty return
        f.close();
        THROW("ERROR: Given annotations file not valid " + path);
    }
    std::unique_ptr<char, std::function<void(char *)>> buff(
        new char[file_size + 1],
        [](char *data) { delete[] data; });
    f.seekg(0, std::ios::beg);
    buff.get()[file_size] = '\0';
    f.read(buff.get(), file_size);
    f.close();

    LookaheadParser parser(buff.get());

    ImgSizes img_sizes;
    JointsData joints_data;

    ImgSize img_size;
    float box_center[2], box_scale[2];
    float score = 1.0;
    float rotation = 0.0;
    float aspect_ratio = ((float)_out_img_width / _out_img_height);
    float inverse_aspect_ratio = 1 / aspect_ratio;
    float inverse_pixel_std = 1 / ((float)PIXEL_STD);

    RAPIDJSON_ASSERT(parser.PeekType() == kObjectType);
    parser.EnterObject();
    while (const char *key = parser.NextObjectKey()) {
        if (0 == std::strcmp(key, "images")) {
            RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
            parser.EnterArray();
            while (parser.NextArrayValue()) {
                string image_name;
                if (parser.PeekType() != kObjectType) {
                    continue;
                }
                parser.EnterObject();
                while (const char *internal_key = parser.NextObjectKey()) {
                    if (0 == std::strcmp(internal_key, "width")) {
                        img_size.w = parser.GetInt();
                    } else if (0 == std::strcmp(internal_key, "height")) {
                        img_size.h = parser.GetInt();
                    } else if (0 == std::strcmp(internal_key, "file_name")) {
                        image_name = parser.GetString();
                    } else {
                        parser.SkipValue();
                    }
                }
                _map_img_sizes.insert(pair<std::string, ImgSize>(image_name, img_size));
                img_size = {};
            }
        } else if (0 == std::strcmp(key, "categories")) {
            RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
            parser.EnterArray();
            int id = 1, continuous_idx = 1;
            while (parser.NextArrayValue()) {
                if (parser.PeekType() != kObjectType) {
                    continue;
                }
                parser.EnterObject();
                while (const char *internal_key = parser.NextObjectKey()) {
                    if (0 == std::strcmp(internal_key, "id")) {
                        id = parser.GetInt();
                    } else {
                        parser.SkipValue();
                    }
                }
                _label_info.insert(std::make_pair(id, continuous_idx));
                continuous_idx++;
            }
        } else if (0 == std::strcmp(key, "annotations")) {
            RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
            parser.EnterArray();
            while (parser.NextArrayValue()) {
                int id = 1, label = 0, is_crowd = 0;
                float joint_sum = 0.0, area = 0.0;
                long int ann_id = 0;
                std::array<float, NUMBER_OF_JOINTS * 3> keypoint{};
                if (parser.PeekType() != kObjectType) {
                    continue;
                }
                parser.EnterObject();
                while (const char *internal_key = parser.NextObjectKey()) {
                    if (0 == std::strcmp(internal_key, "image_id")) {
                        id = parser.GetInt();
                    } else if (0 == std::strcmp(internal_key, "category_id")) {
                        label = parser.GetInt();
                    } else if (0 == std::strcmp(internal_key, "id")) {
                        ann_id = parser.GetDouble();
                    } else if (0 == std::strcmp(internal_key, "is_crowd")) {
                        is_crowd = parser.GetInt();
                    } else if (0 == std::strcmp(internal_key, "area")) {
                        area = parser.GetDouble();
                    } else if (0 == std::strcmp(internal_key, "bbox")) {
                        RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
                        parser.EnterArray();

                        box_center[0] = parser.NextArrayValue() * parser.GetDouble();
                        box_center[1] = parser.NextArrayValue() * parser.GetDouble();
                        box_scale[0] = parser.NextArrayValue() * parser.GetDouble();
                        box_scale[1] = parser.NextArrayValue() * parser.GetDouble();

                        // Move to next section
                        parser.NextArrayValue();
                    } else if (0 == std::strcmp(internal_key, "keypoints")) {
                        RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
                        parser.EnterArray();
                        int i = 0;
                        while (parser.NextArrayValue()) {
                            keypoint[i] = parser.GetDouble();
                            joint_sum += keypoint[i];
                            ++i;
                        }
                    } else {
                        parser.SkipValue();
                    }
                }
                char buffer[13];
                sprintf(buffer, "%012d", id);
                string str(buffer);
                std::string file_name = str + ".jpg";
                auto it = _map_img_sizes.find(file_name);
                ImgSize image_size = it->second;  // Normalizing the co-ordinates & convert to "ltrb" format

                // Ignore annotations if
                // label is not person (label !=1)
                // joint_sum <= 0
                // is_crowd==1
                if (label != 1 || joint_sum <= 0 || is_crowd == 1) {
                    std::fill(std::begin(box_center), std::end(box_center), 0.0);
                    std::fill(std::begin(box_scale), std::end(box_scale), 0.0);
                    continue;
                }

                // Validate bbox values
                float x1, y1, x2, y2;
                x1 = std::max(box_center[0], 0.0f);
                y1 = std::max(box_center[1], 0.0f);
                float box_w = std::max(box_scale[0] - 1, 0.0f);
                float box_h = std::max(box_scale[1] - 1, 0.0f);
                x2 = std::min((float)image_size.w - 1, x1 + box_w);
                y2 = std::min((float)image_size.h - 1, y1 + box_h);

                // check area
                if (area > 0 && x2 >= x1 && y2 >= y1) {
                    box_center[0] = x1;
                    box_center[1] = y1;
                    box_scale[0] = x2 - x1;
                    box_scale[1] = y2 - y1;
                }

                // Convert from xywh to center,scale
                box_center[0] += (0.5 * box_scale[0]);
                box_center[1] += (0.5 * box_scale[1]);

                if (box_scale[0] > aspect_ratio * box_scale[1]) {
                    box_scale[1] = box_scale[0] * inverse_aspect_ratio * inverse_pixel_std;
                    box_scale[0] = box_scale[0] * inverse_pixel_std;
                } else if (box_scale[0] < aspect_ratio * box_scale[1]) {
                    box_scale[0] = box_scale[1] * aspect_ratio * inverse_pixel_std;
                    box_scale[1] = box_scale[1] * inverse_pixel_std;
                }

                if (box_center[0] != -1) {
                    box_scale[0] = SCALE_CONSTANT_CS * box_scale[0];
                    box_scale[1] = SCALE_CONSTANT_CS * box_scale[1];
                }

                // Convert raw keypoint values to Joints,Joint Visibilities - Clip the visibilities to range [0,1]
                std::vector<std::vector<float>> joints(NUMBER_OF_JOINTS), joints_visibility(NUMBER_OF_JOINTS);
                unsigned int j = 0;
                for (unsigned int i = 0; i < NUMBER_OF_JOINTS; i++) {
                    joints[i].push_back(keypoint[j]);
                    joints[i].push_back(keypoint[j + 1]);
                    if (keypoint[j + 2] > 1.0) {
                        keypoint[j + 2] = 1.0;
                    }
                    joints_visibility[i].push_back(keypoint[j + 2]);
                    joints_visibility[i].push_back(keypoint[j + 2]);
                    j = j + 3;
                }

                // Add values to joints_data structure
                joints_data.annotation_id = ann_id;
                joints_data.image_id = id;
                joints_data.image_path = file_name;
                memcpy(joints_data.center, &box_center, sizeof(box_center));
                memcpy(joints_data.scale, &box_scale, sizeof(box_scale));
                joints_data.joints = joints;
                joints_data.joints_visibility = joints_visibility;
                joints_data.score = score;
                joints_data.rotation = rotation;

                add(file_name, image_size, &joints_data);
                joints_data = {};
                std::fill(std::begin(box_center), std::end(box_center), 0.0);
                std::fill(std::begin(box_scale), std::end(box_scale), 0.0);
                joints = {};
                joints_visibility = {};
            }
        } else {
            parser.SkipValue();
        }
    }
    _coco_metadata_read_time.end();  // Debug timing
    // print_map_contents();
    // std::cout << "coco read time in sec: " << _coco_metadata_read_time.get_timing() / 1000 << std::endl;
}

void COCOMetaDataReaderKeyPoints::release(std::string image_name) {
    if (!exists(image_name)) {
        WRN("ERROR: Given name not present in the map" + image_name);
        return;
    }
    _map_content.erase(image_name);
}

void COCOMetaDataReaderKeyPoints::release() {
    _map_content.clear();
    _map_img_sizes.clear();
}

COCOMetaDataReaderKeyPoints::COCOMetaDataReaderKeyPoints() : _coco_metadata_read_time("coco meta read time", DBG_TIMING) {
}
