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
#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pipeline/commons.h"

// Defined constants since needed in reader and meta nodes for Pose Estimation
#define NUMBER_OF_JOINTS 17
#define NUMBER_OF_JOINTS_HALFBODY 8
#define PIXEL_STD 200
#define SCALE_CONSTANT_CS 1.25
#define SCALE_CONSTANT_HALF_BODY 1.5
typedef struct BoundingBoxCord_ {
    float l;
    float t;
    float r;
    float b;
    BoundingBoxCord_() {}
    BoundingBoxCord_(float l_, float t_, float r_, float b_) : l(l_), t(t_), r(r_), b(b_) {}        // constructor
    BoundingBoxCord_(const BoundingBoxCord_& cord) : l(cord.l), t(cord.t), r(cord.r), b(cord.b) {}  // copy constructor
} BoundingBoxCord;

typedef std::vector<BoundingBoxCord> BoundingBoxCords;
typedef std::vector<int> Labels;

typedef std::shared_ptr<std::vector<uint8_t>> AsciiComponent; // Shared pointer to a vector that can store each component of the samples in a tar file in ASCII format.
typedef std::vector<AsciiComponent> AsciiValues; // A vector that can store all the components of the samples in a tar file in ASCII format.

typedef struct {
    int w;
    int h;
} ImgSize;
typedef std::vector<ImgSize> ImgSizes;

typedef std::vector<float> MaskCords;
typedef std::vector<int> ImageIDBatch, AnnotationIDBatch;
typedef std::vector<std::string> ImagePathBatch;
typedef std::vector<float> Joint, JointVisibility, ScoreBatch, RotationBatch;
typedef std::vector<std::vector<float>> Joints, JointsVisibility, CenterBatch, ScaleBatch;
typedef std::vector<std::vector<std::vector<float>>> JointsBatch, JointsVisibilityBatch;

enum class MetaDataType {
    Label,
    BoundingBox,
    PolygonMask,
    KeyPoints,
    AsciiValue
};

enum class BoundingBoxType {
    XYWH = 0,
    LTRB
};

typedef struct
{
    int image_id;
    int annotation_id;
    std::string image_path;
    float center[2];
    float scale[2];
    Joints joints;
    JointsVisibility joints_visibility;
    float score;
    float rotation;
} JointsData;

typedef struct
{
    ImageIDBatch image_id_batch;
    AnnotationIDBatch annotation_id_batch;
    ImagePathBatch image_path_batch;
    CenterBatch center_batch;
    ScaleBatch scale_batch;
    JointsBatch joints_batch;
    JointsVisibilityBatch joints_visibility_batch;
    ScoreBatch score_batch;
    RotationBatch rotation_batch;
} JointsDataBatch;

typedef class MetaDataInfo {
   public:
    int img_id = -1;
    std::string img_name = "";
    ImgSize img_size = {};
    ImgSize img_roi_size = {};
} MetaDataInfo;

class MetaData {
   public:
    virtual ~MetaData() = default;
    virtual std::vector<int>& get_labels() = 0;
    virtual AsciiValues& get_ascii_values() = 0;
    virtual void set_labels(Labels label_ids) = 0;
    virtual void set_ascii_values(AsciiValues ascii_values) = 0;
    virtual BoundingBoxCords& get_bb_cords() = 0;
    virtual void set_bb_cords(BoundingBoxCords bb_cords) = 0;
    virtual std::vector<int>& get_polygon_count() = 0;
    virtual std::vector<std::vector<int>>& get_vertices_count() = 0;
    virtual MaskCords& get_mask_cords() = 0;
    virtual void set_mask_cords(MaskCords mask_cords) = 0;
    virtual void set_polygon_counts(std::vector<int> polygon_count) = 0;
    virtual void set_vertices_counts(std::vector<std::vector<int>> vertices_count) = 0;
    virtual JointsData& get_joints_data() = 0;
    virtual void set_joints_data(JointsData* joints_data) = 0;
    ImgSize& get_img_size() { return _info.img_size; }
    ImgSize& get_img_roi_size() { return _info.img_roi_size; }
    std::string& get_image_name() { return _info.img_name; }
    int& get_image_id() { return _info.img_id; }
    void set_img_size(ImgSize img_size) { _info.img_size = std::move(img_size); }
    void set_img_roi_size(ImgSize img_roi_size) { _info.img_roi_size = std::move(img_roi_size); }
    void set_img_id(int img_id) { _info.img_id = img_id; }
    void set_img_name(std::string img_name) { _info.img_name = img_name; }
    void set_metadata_info(MetaDataInfo info) { _info = std::move(info); }

   protected:
    MetaDataInfo _info;
};

class AsciiValue : public MetaData {
   public:
    AsciiValue() = default;
    AsciiValue(AsciiValues ascii_values) {
        _ascii_values = std::move(ascii_values);
    }
    void set_ascii_values(AsciiValues ascii_values) override { _ascii_values = std::move(ascii_values); }
    AsciiValues& get_ascii_values() override { return _ascii_values; }
    // Not Implemnented
    void set_labels(Labels label_ids) override { THROW("Not Implemented") }
    std::vector<int>& get_labels() override {  THROW("Not Implemented") }
    BoundingBoxCords& get_bb_cords() override { THROW("Not Implemented") }
    void set_bb_cords(BoundingBoxCords bb_cords) override{THROW("Not Implemented")} 
    std::vector<int>& get_polygon_count() override{THROW("Not Implemented")} 
    std::vector<std::vector<int>>& get_vertices_count() override{THROW("Not Implemented")} 
    MaskCords& get_mask_cords() override { THROW("Not Implemented") }
    void set_mask_cords(MaskCords mask_cords) override { THROW("Not Implemented") }
    void set_polygon_counts(std::vector<int> polygon_count) override { THROW("Not Implemented") }
    void set_vertices_counts(std::vector<std::vector<int>> vertices_count) override{THROW("Not Implemented")} 
    JointsData& get_joints_data() override { THROW("Not Implemented") }
    void set_joints_data(JointsData* joints_data) override { THROW("Not Implemented") }

   protected:
    AsciiValues _ascii_values = {};  // For storing ASCII data only
};

class Label : public MetaData {
   public:
    Label(int label) { _label_ids = {label}; }
    Label() { _label_ids = {-1}; }
    ~Label() = default;
    std::vector<int>& get_labels() override { return _label_ids; }
    void set_labels(Labels label_ids) override { _label_ids = std::move(label_ids); }
    BoundingBoxCords& get_bb_cords() override { THROW("Not Implemented") }
    void set_bb_cords(BoundingBoxCords bb_cords) override{THROW("Not Implemented")} 
    std::vector<int>& get_polygon_count() override{THROW("Not Implemented")} 
    std::vector<std::vector<int>>& get_vertices_count() override{THROW("Not Implemented")} 
    MaskCords& get_mask_cords() override { THROW("Not Implemented") }
    void set_mask_cords(MaskCords mask_cords) override { THROW("Not Implemented") }
    void set_polygon_counts(std::vector<int> polygon_count) override { THROW("Not Implemented") }
    void set_vertices_counts(std::vector<std::vector<int>> vertices_count) override{THROW("Not Implemented")} 
    JointsData& get_joints_data() override { THROW("Not Implemented") }
    void set_joints_data(JointsData* joints_data) override { THROW("Not Implemented") }
    void set_ascii_values(AsciiValues ascii_values) override { THROW("Not Implemented") }
    AsciiValues& get_ascii_values() override { THROW("Not Implemented") }

   protected:
    Labels _label_ids = {};  // For label use only
};

class BoundingBox : public Label {
   public:
    BoundingBox() = default;
    ~BoundingBox() = default;
    BoundingBox(BoundingBoxCords bb_cords, Labels bb_label_ids, ImgSize img_size = ImgSize{0, 0}, int img_id = 0) {
        _bb_cords = std::move(bb_cords);
        _label_ids = std::move(bb_label_ids);
        _info.img_size = std::move(img_size);
        _info.img_id = img_id;
    }
    BoundingBoxCords& get_bb_cords() override { return _bb_cords; }
    void set_bb_cords(BoundingBoxCords bb_cords) override { _bb_cords = std::move(bb_cords); }

   protected:
    BoundingBoxCords _bb_cords = {};  // For bb use
};

struct PolygonMask : public BoundingBox {
   public:
    PolygonMask(BoundingBoxCords bb_cords, Labels bb_label_ids, ImgSize img_size, MaskCords mask_cords, std::vector<int> polygon_count, std::vector<std::vector<int>> vertices_count, int img_id = 0) {
        _bb_cords = std::move(bb_cords);
        _label_ids = std::move(bb_label_ids);
        _info.img_size = std::move(img_size);
        _mask_cords = std::move(mask_cords);
        _polygon_count = std::move(polygon_count);
        _vertices_count = std::move(vertices_count);
        _info.img_id = img_id;
    }
    ~PolygonMask() = default;
    std::vector<int>& get_polygon_count() override { return _polygon_count; }
    std::vector<std::vector<int>>& get_vertices_count() override { return _vertices_count; }
    MaskCords& get_mask_cords() override { return _mask_cords; }
    void set_mask_cords(MaskCords mask_cords) override { _mask_cords = std::move(mask_cords); }
    void set_polygon_counts(std::vector<int> polygon_count) override { _polygon_count = std::move(polygon_count); }
    void set_vertices_counts(std::vector<std::vector<int>> vertices_count) override { _vertices_count = std::move(vertices_count); }

   protected:
    MaskCords _mask_cords = {};
    std::vector<int> _polygon_count = {};
    std::vector<std::vector<int>> _vertices_count = {};
};

class KeyPoint : public BoundingBox {
   public:
    KeyPoint() = default;
    ~KeyPoint() = default;
    KeyPoint(ImgSize img_size, JointsData* joints_data) {
        _info.img_size = std::move(img_size);
        _joints_data = std::move(*joints_data);
    }
    void set_joints_data(JointsData* joints_data) override { _joints_data = std::move(*joints_data); }
    JointsData& get_joints_data() override { return _joints_data; }

   protected:
    JointsData _joints_data = {};
};

class MetaDataInfoBatch {
   public:
    std::vector<int> img_ids = {};
    std::vector<std::string> img_names = {};
    std::vector<ImgSize> img_sizes = {};
    std::vector<ImgSize> img_roi_sizes = {};
    void clear() {
        img_ids.clear();
        img_names.clear();
        img_sizes.clear();
        img_roi_sizes.clear();
    }
    void resize(int batch_size) {
        img_ids.resize(batch_size);
        img_names.resize(batch_size);
        img_sizes.resize(batch_size);
        img_roi_sizes.resize(batch_size);
    }
    void insert(MetaDataInfoBatch& other) {
        img_sizes.insert(img_sizes.end(), other.img_sizes.begin(), other.img_sizes.end());
        img_ids.insert(img_ids.end(), other.img_ids.begin(), other.img_ids.end());
        img_names.insert(img_names.end(), other.img_names.begin(), other.img_names.end());
        img_roi_sizes.insert(img_roi_sizes.end(), other.img_roi_sizes.begin(), other.img_roi_sizes.end());
    }
};

class MetaDataBatch {
   public:
    virtual ~MetaDataBatch() = default;
    virtual void clear() = 0;
    virtual void resize(int batch_size) = 0;
    virtual int size() = 0;
    virtual void copy_data(std::vector<void*> buffer) = 0;
    virtual std::vector<size_t>& get_buffer_size() = 0;
    virtual MetaDataBatch& operator+=(MetaDataBatch& other) = 0;
    MetaDataBatch* concatenate(MetaDataBatch* other) {
        *this += *other;
        return this;
    }
    virtual std::shared_ptr<MetaDataBatch> clone(bool copy_contents = true) = 0;
    virtual int mask_size() = 0;
    virtual std::vector<Labels>& get_labels_batch() = 0;
    virtual std::vector<AsciiValues>& get_ascii_values_batch() = 0;
    virtual std::vector<BoundingBoxCords>& get_bb_cords_batch() = 0;
    virtual void set_xywh_bbox() = 0;
    virtual std::vector<MaskCords>& get_mask_cords_batch() = 0;
    virtual std::vector<std::vector<int>>& get_mask_polygons_count_batch() = 0;
    virtual std::vector<std::vector<std::vector<int>>>& get_mask_vertices_count_batch() = 0;
    virtual JointsDataBatch& get_joints_data_batch() = 0;
    std::vector<int>& get_image_id_batch() { return _info_batch.img_ids; }
    std::vector<std::string>& get_image_names_batch() { return _info_batch.img_names; }
    ImgSizes& get_img_sizes_batch() { return _info_batch.img_sizes; }
    ImgSizes& get_img_roi_sizes_batch() { return _info_batch.img_roi_sizes; }
    MetaDataInfoBatch& get_info_batch() { return _info_batch; }
    void set_metadata_type(MetaDataType metadata_type) { _type = metadata_type; }
    MetaDataType get_metadata_type() { return _type; }

   protected:
    MetaDataInfoBatch _info_batch;
    MetaDataType _type;
};

class AsciiValueBatch : public MetaDataBatch {
   public:
    void clear() override {
        for (auto value : _ascii_values) {
            value.clear();
        }
        _info_batch.clear();
        _ascii_values.clear();
        _buffer_size.clear();
    }
    MetaDataBatch& operator+=(MetaDataBatch& other) override {
        _ascii_values.insert(_ascii_values.end(), other.get_ascii_values_batch().begin(), other.get_ascii_values_batch().end());
        _info_batch.insert(other.get_info_batch());
        return *this;
    }
    void resize(int batch_size) override {
        _ascii_values.resize(batch_size);
        _info_batch.resize(batch_size);
    }
    int size() override {
        return _ascii_values.size();
    }
    std::shared_ptr<MetaDataBatch> clone(bool copy_contents) override {
        if (copy_contents) {
            return std::make_shared<AsciiValueBatch>(*this);
        } else {
            std::shared_ptr<MetaDataBatch> ascii_value_batch_instance = std::make_shared<AsciiValueBatch>();
            ascii_value_batch_instance->resize(this->size());
            ascii_value_batch_instance->get_info_batch() = this->get_info_batch();
            return ascii_value_batch_instance;
        }
    }
    explicit AsciiValueBatch(std::vector<AsciiValues>& ascii_values) {
        _ascii_values = std::move(ascii_values);
    }
    AsciiValueBatch() = default;
    void copy_data(std::vector<void*> buffer) override {
        if (buffer.size() < 1)
            THROW("The buffers are insufficient")

        for (unsigned component = 0; component < _ascii_values[0].size(); component++) {
            if (buffer[component]) {
                auto ascii_values_buffer = static_cast<uint8_t*>(buffer[component]);
                for (unsigned i = 0; i < _ascii_values.size(); i++) {
                    if(_ascii_values[i][component]) {
                        AsciiValues sample = _ascii_values[i];
                        memcpy(ascii_values_buffer, sample[component]->data(), sample[component]->size() * sizeof(uint8_t));
                        ascii_values_buffer += sample[component]->size();
                    }
                }
            }
        }
    }

    std::vector<size_t>& get_buffer_size() override {
        _buffer_size.clear();
        for (unsigned component = 0; component < _ascii_values[0].size(); component++) {
            size_t size = 0;
            for (unsigned i = 0; i < _ascii_values.size(); i++) {
                if(_ascii_values[i][component])
                    size += _ascii_values[i][component]->size();
            }
            _buffer_size.emplace_back(size * sizeof(uint8_t));
        }
        return _buffer_size;
    }
    std::vector<AsciiValues>& get_ascii_values_batch() override { return _ascii_values; }

    // Not implemented
    std::vector<Labels>& get_labels_batch() override { THROW("Not Implemented") }
    int mask_size() override{THROW("Not Implemented")} 
    std::vector<BoundingBoxCords>& get_bb_cords_batch() override { THROW("Not Implemented") }
    void set_xywh_bbox() override{THROW("Not Implemented")} 
    std::vector<MaskCords>& get_mask_cords_batch() override{THROW("Not Implemented")} 
    std::vector<std::vector<int>>& get_mask_polygons_count_batch() override{THROW("Not Implemented")} 
    std::vector<std::vector<std::vector<int>>>& get_mask_vertices_count_batch() override{THROW("Not Implemented")} 
    JointsDataBatch& get_joints_data_batch() override { THROW("Not Implemented") }

   protected:
    std::vector<AsciiValues> _ascii_values = {};
    std::vector<size_t> _buffer_size;
};

class LabelBatch : public MetaDataBatch {
   public:
    void clear() override {
        for (auto label : _label_ids) {
            label.clear();
        }
        _info_batch.clear();
        _label_ids.clear();
        _buffer_size.clear();
    }
    MetaDataBatch& operator+=(MetaDataBatch& other) override {
        _label_ids.insert(_label_ids.end(), other.get_labels_batch().begin(), other.get_labels_batch().end());
        _info_batch.insert(other.get_info_batch());
        return *this;
    }
    void resize(int batch_size) override {
        _label_ids.resize(batch_size);
        _info_batch.resize(batch_size);
    }
    int size() override {
        return _label_ids.size();
    }
    std::shared_ptr<MetaDataBatch> clone(bool copy_contents) override {
        if (copy_contents) {
            return std::make_shared<LabelBatch>(*this);  // Copy the entire metadata batch with all the metadata values and info
        } else {
            std::shared_ptr<MetaDataBatch> label_batch_instance = std::make_shared<LabelBatch>();
            label_batch_instance->resize(this->size());
            label_batch_instance->get_info_batch() = this->get_info_batch();  // Copy only info to newly created instance excluding the metadata values
            return label_batch_instance;
        }
    }
    explicit LabelBatch(std::vector<Labels>& labels) {
        _label_ids = std::move(labels);
    }
    LabelBatch() = default;
    void copy_data(std::vector<void*> buffer) override {
        if (buffer.size() < 1)
            THROW("The buffers are insufficient")  // TODO -change
        auto labels_buffer = (int*)buffer[0];
        for (unsigned i = 0; i < _label_ids.size(); i++) {
            memcpy(labels_buffer, _label_ids[i].data(), _label_ids[i].size() * sizeof(int));
            labels_buffer += _label_ids[i].size();
        }
    }
    std::vector<size_t>& get_buffer_size() override {
        _buffer_size.clear();
        size_t size = 0;
        for (auto label : _label_ids)
            size += label.size();
        _buffer_size.emplace_back(size * sizeof(int));
        return _buffer_size;
    }
    std::vector<Labels>& get_labels_batch() override { return _label_ids; }
    std::vector<AsciiValues>& get_ascii_values_batch() override {THROW("Not Implemented")}
    int mask_size() override{THROW("Not Implemented")} std::vector<BoundingBoxCords>& get_bb_cords_batch() override { THROW("Not Implemented") }
    void set_xywh_bbox() override{THROW("Not Implemented")} std::vector<MaskCords>& get_mask_cords_batch() override{THROW("Not Implemented")} std::vector<std::vector<int>>& get_mask_polygons_count_batch() override{THROW("Not Implemented")} std::vector<std::vector<std::vector<int>>>& get_mask_vertices_count_batch() override{THROW("Not Implemented")} JointsDataBatch& get_joints_data_batch() override { THROW("Not Implemented") }

   protected:
    std::vector<Labels> _label_ids = {};
    std::vector<size_t> _buffer_size;
};

class BoundingBoxBatch : public LabelBatch {
   public:
    void clear() override {
        _bb_cords.clear();
        _label_ids.clear();
        _info_batch.clear();
        _buffer_size.clear();
    }
    MetaDataBatch& operator+=(MetaDataBatch& other) override {
        _bb_cords.insert(_bb_cords.end(), other.get_bb_cords_batch().begin(), other.get_bb_cords_batch().end());
        _label_ids.insert(_label_ids.end(), other.get_labels_batch().begin(), other.get_labels_batch().end());
        _info_batch.insert(other.get_info_batch());
        return *this;
    }
    void resize(int batch_size) override {
        _bb_cords.resize(batch_size);
        _label_ids.resize(batch_size);
        _info_batch.resize(batch_size);
    }
    int size() override {
        return _bb_cords.size();
    }
    std::shared_ptr<MetaDataBatch> clone(bool copy_contents) override {
        if (copy_contents) {
            return std::make_shared<BoundingBoxBatch>(*this);  // Copy the entire metadata batch with all the metadata values and info
        } else {
            std::shared_ptr<MetaDataBatch> bbox_batch_instance = std::make_shared<BoundingBoxBatch>();
            bbox_batch_instance->resize(this->size());
            bbox_batch_instance->get_info_batch() = this->get_info_batch();  // Copy only info to newly created instance excluding the metadata values
            return bbox_batch_instance;
        }
    }
    void convert_ltrb_to_xywh(BoundingBoxCords& ltrb_bbox_list) {
        for (unsigned i = 0; i < ltrb_bbox_list.size(); i++) {
            auto& bbox = ltrb_bbox_list[i];
            // Change the values in place
            bbox.r = bbox.r - bbox.l;
            bbox.b = bbox.b - bbox.t;
        }
    }
    void copy_data(std::vector<void*> buffer) override {
        if (buffer.size() < 2)
            THROW("The buffers are insufficient")  // TODO -change
        int* labels_buffer = (int*)buffer[0];
        float* bbox_buffer = (float*)buffer[1];
        for (unsigned i = 0; i < _label_ids.size(); i++) {
            memcpy(labels_buffer, _label_ids[i].data(), _label_ids[i].size() * sizeof(int));
            if (_bbox_output_type == BoundingBoxType::XYWH) convert_ltrb_to_xywh(_bb_cords[i]);
            memcpy(bbox_buffer, _bb_cords[i].data(), _label_ids[i].size() * 4 * sizeof(float));
            labels_buffer += _label_ids[i].size();
            bbox_buffer += (_label_ids[i].size() * 4);
        }
    }
    std::vector<size_t>& get_buffer_size() override {
        _buffer_size.clear();
        size_t size = 0;
        for (auto label : _label_ids)
            size += label.size();
        _buffer_size.emplace_back(size * sizeof(int));
        _buffer_size.emplace_back(size * 4 * sizeof(float));
        return _buffer_size;
    }
    std::vector<BoundingBoxCords>& get_bb_cords_batch() override { return _bb_cords; }
    void set_xywh_bbox() override { _bbox_output_type = BoundingBoxType::XYWH; }

   protected:
    std::vector<BoundingBoxCords> _bb_cords = {};
    BoundingBoxType _bbox_output_type = BoundingBoxType::LTRB;
};

struct PolygonMaskBatch : public BoundingBoxBatch {
   public:
    void clear() override {
        _bb_cords.clear();
        _label_ids.clear();
        _info_batch.clear();
        _mask_cords.clear();
        _polygon_counts.clear();
        _vertices_counts.clear();
        _buffer_size.clear();
    }
    MetaDataBatch& operator+=(MetaDataBatch& other) override {
        _bb_cords.insert(_bb_cords.end(), other.get_bb_cords_batch().begin(), other.get_bb_cords_batch().end());
        _label_ids.insert(_label_ids.end(), other.get_labels_batch().begin(), other.get_labels_batch().end());
        _info_batch.insert(other.get_info_batch());
        _mask_cords.insert(_mask_cords.end(), other.get_mask_cords_batch().begin(), other.get_mask_cords_batch().end());
        _polygon_counts.insert(_polygon_counts.end(), other.get_mask_polygons_count_batch().begin(), other.get_mask_polygons_count_batch().end());
        _vertices_counts.insert(_vertices_counts.end(), other.get_mask_vertices_count_batch().begin(), other.get_mask_vertices_count_batch().end());
        return *this;
    }
    void resize(int batch_size) override {
        _bb_cords.resize(batch_size);
        _label_ids.resize(batch_size);
        _info_batch.resize(batch_size);
        _mask_cords.resize(batch_size);
        _polygon_counts.resize(batch_size);
        _vertices_counts.resize(batch_size);
    }
    std::vector<MaskCords>& get_mask_cords_batch() override { return _mask_cords; }
    std::vector<std::vector<int>>& get_mask_polygons_count_batch() override { return _polygon_counts; }
    std::vector<std::vector<std::vector<int>>>& get_mask_vertices_count_batch() override { return _vertices_counts; }
    int mask_size() override { return _mask_cords.size(); }
    std::shared_ptr<MetaDataBatch> clone(bool copy_contents) override {
        if (copy_contents) {
            return std::make_shared<PolygonMaskBatch>(*this);  // Copy the entire metadata batch with all the metadata values and info
        } else {
            std::shared_ptr<MetaDataBatch> mask_batch_instance = std::make_shared<PolygonMaskBatch>();
            mask_batch_instance->resize(this->size());
            mask_batch_instance->get_info_batch() = this->get_info_batch();  // Copy only info to newly created instance excluding the metadata values
            return mask_batch_instance;
        }
    }
    void copy_data(std::vector<void*> buffer) override {
        if (buffer.size() < 2)
            THROW("The buffers are insufficient")  // TODO -change
        int* labels_buffer = (int*)buffer[0];
        float* bbox_buffer = (float*)buffer[1];
        float* mask_buffer = (float*)buffer[2];
        for (unsigned i = 0; i < _label_ids.size(); i++) {
            mempcpy(labels_buffer, _label_ids[i].data(), _label_ids[i].size() * sizeof(int));
            if (_bbox_output_type == BoundingBoxType::XYWH) convert_ltrb_to_xywh(_bb_cords[i]);
            memcpy(bbox_buffer, _bb_cords[i].data(), _label_ids[i].size() * 4 * sizeof(float));
            memcpy(mask_buffer, _mask_cords[i].data(), _mask_cords[i].size() * sizeof(float));
            labels_buffer += _label_ids[i].size();
            bbox_buffer += (_label_ids[i].size() * 4);
            mask_buffer += _mask_cords[i].size();
        }
    }
    std::vector<size_t>& get_buffer_size() override {
        _buffer_size.clear();
        size_t size = 0;
        for (auto label : _label_ids)
            size += label.size();
        _buffer_size.emplace_back(size * sizeof(int));
        _buffer_size.emplace_back(size * 4 * sizeof(float));
        size = 0;
        for (auto mask : _mask_cords)
            size += mask.size();
        _buffer_size.emplace_back(size * sizeof(float));
        return _buffer_size;
    }

   protected:
    std::vector<MaskCords> _mask_cords = {};
    std::vector<std::vector<int>> _polygon_counts = {};
    std::vector<std::vector<std::vector<int>>> _vertices_counts = {};
};

class KeyPointBatch : public BoundingBoxBatch {
   public:
    void clear() override {
        _info_batch.clear();
        _joints_data = {};
        _bb_cords.clear();
        _label_ids.clear();
    }
    MetaDataBatch& operator+=(MetaDataBatch& other) override {
        _joints_data.image_id_batch.insert(_joints_data.image_id_batch.end(), other.get_joints_data_batch().image_id_batch.begin(), other.get_joints_data_batch().image_id_batch.end());
        _joints_data.annotation_id_batch.insert(_joints_data.annotation_id_batch.end(), other.get_joints_data_batch().annotation_id_batch.begin(), other.get_joints_data_batch().annotation_id_batch.end());
        _joints_data.center_batch.insert(_joints_data.center_batch.end(), other.get_joints_data_batch().center_batch.begin(), other.get_joints_data_batch().center_batch.end());
        _joints_data.scale_batch.insert(_joints_data.scale_batch.end(), other.get_joints_data_batch().scale_batch.begin(), other.get_joints_data_batch().scale_batch.end());
        _joints_data.joints_batch.insert(_joints_data.joints_batch.end(), other.get_joints_data_batch().joints_batch.begin(), other.get_joints_data_batch().joints_batch.end());
        _joints_data.joints_visibility_batch.insert(_joints_data.joints_visibility_batch.end(), other.get_joints_data_batch().joints_visibility_batch.begin(), other.get_joints_data_batch().joints_visibility_batch.end());
        _joints_data.score_batch.insert(_joints_data.score_batch.end(), other.get_joints_data_batch().score_batch.begin(), other.get_joints_data_batch().score_batch.end());
        _joints_data.rotation_batch.insert(_joints_data.rotation_batch.end(), other.get_joints_data_batch().rotation_batch.begin(), other.get_joints_data_batch().rotation_batch.end());
        _info_batch.insert(other.get_info_batch());
        return *this;
    }
    void resize(int batch_size) override {
        _joints_data.image_id_batch.resize(batch_size);
        _joints_data.annotation_id_batch.resize(batch_size);
        _joints_data.center_batch.resize(batch_size);
        _joints_data.scale_batch.resize(batch_size);
        _joints_data.joints_batch.resize(batch_size);
        _joints_data.joints_visibility_batch.resize(batch_size);
        _joints_data.score_batch.resize(batch_size);
        _joints_data.rotation_batch.resize(batch_size);
        _info_batch.resize(batch_size);
        _bb_cords.resize(batch_size);
        _label_ids.resize(batch_size);
    }
    int size() override {
        return _joints_data.image_id_batch.size();
    }
    std::shared_ptr<MetaDataBatch> clone(bool copy_contents) override {
        if (copy_contents) {
            return std::make_shared<KeyPointBatch>(*this);  // Copy the entire metadata batch with all the metadata values and info
        } else {
            std::shared_ptr<MetaDataBatch> joints_batch_instance = std::make_shared<KeyPointBatch>();
            joints_batch_instance->resize(this->size());
            joints_batch_instance->get_info_batch() = this->get_info_batch();  // Copy only info to newly created instance excluding the metadata values
            return joints_batch_instance;
        }
    }
    JointsDataBatch& get_joints_data_batch() override { return _joints_data; }
    void copy_data(std::vector<void*> buffer) override {}
    std::vector<size_t>& get_buffer_size() override { return _buffer_size; }

   protected:
    JointsDataBatch _joints_data = {};
};

using ImageNameBatch = std::vector<std::string>;
using pMetaData = std::shared_ptr<Label>;
using pMetaDataAscii = std::shared_ptr<AsciiValue>;
using pMetaDataBox = std::shared_ptr<BoundingBox>;
using pMetaDataPolygonMask = std::shared_ptr<PolygonMask>;
using pMetaDataKeyPoint = std::shared_ptr<KeyPoint>;
using pMetaDataBatch = std::shared_ptr<MetaDataBatch>;
