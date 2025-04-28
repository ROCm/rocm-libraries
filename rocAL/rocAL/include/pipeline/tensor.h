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

#pragma once

#include <VX/vx.h>
#include <VX/vx_types.h>

#include <array>
#include <cstring>
#include <memory>
#include <queue>
#include <vector>
#if ENABLE_HIP
#include "device/device_manager_hip.h"
#include "hip/hip_runtime.h"
#else
#include "device/device_manager.h"
#endif
#include "pipeline/commons.h"
#include "rocal_api_tensor.h"

/*! \brief Converts Rocal Memory type to OpenVX memory type
 *
 * @param mem input Rocal type
 * @return the OpenVX type associated with input argument
 */
vx_enum vx_mem_type(RocalMemType mem);

/*! \brief Returns the size of the data type
 *
 * @param RocalTensorDataType input data type
 * @return the OpenVX data type size associated with input argument
 */
vx_uint64 tensor_data_size(RocalTensorDataType data_type);

/*! \brief Allocated memory for given size
 *
 * @param void * The ptr for which memory is allocated
 * @param size_t size of the buffer
 * @param RocalMemType For HIP memType pinned memory is allocated
 *        else HOST memory is allocated
 */
void allocate_host_or_pinned_mem(void** ptr, size_t size, RocalMemType mem_type);

struct Roi {
    unsigned* get_ptr() { return _roi_ptr.get(); }
    Roi2DCords* get_2D_roi() {
        if (_roi_no_of_dims != 2)
            THROW("ROI has more than 2 dimensions. Cannot return Roi2DCords")
        return reinterpret_cast<Roi2DCords*>(_roi_ptr.get());
    }
    void set_ptr(unsigned* ptr, RocalMemType mem_type, unsigned batch_size, unsigned no_of_dims = 0) {
        if (!_roi_no_of_dims) _roi_no_of_dims = no_of_dims;
        _roi_buffer_size = batch_size * _roi_no_of_dims * 2 * sizeof(unsigned); // 2 denotes, one coordinate each for begin and end
        _roi_buf = ptr;
        if (mem_type == RocalMemType::HIP) {
#if ENABLE_HIP
            _roi_ptr.reset(_roi_buf, hipHostFree);
#endif
        } else {
            _roi_ptr.reset(_roi_buf, free);
        }
    }
    void reset_ptr(unsigned* ptr) {
        auto deleter = [&](unsigned* ptr) {};  // Empty destructor used, since memory is handled by the pipeline
        _roi_ptr.reset(ptr, deleter);
    }
    void copy(void* roi_buffer) {
        if (_roi_ptr.get() != nullptr && roi_buffer != nullptr) {
            memcpy(roi_buffer, (const void*)_roi_ptr.get(), _roi_buffer_size);
        } else {
            WRN("ROI data is not available for the tensor")
        }
    }
    unsigned no_of_dims() { return _roi_no_of_dims; }
    size_t roi_buffer_size() { return _roi_buffer_size; }
    RoiCords& operator[](const int i) {
        int idx = i * _roi_no_of_dims * 2;
        _roi_coords.begin = (_roi_buf + idx);
        _roi_coords.end = (_roi_buf + idx + _roi_no_of_dims);
        return _roi_coords;
    }

   private:
    unsigned* _roi_buf = nullptr;
    std::shared_ptr<unsigned> _roi_ptr;
    unsigned _roi_no_of_dims = 0;
    RoiCords _roi_coords;
    size_t _roi_buffer_size = 0;
};

/*! \brief Holds the information about a Tensor */
class TensorInfo {
   public:
    friend class Tensor;
    enum class Type {
        UNKNOWN = -1,
        REGULAR = 0,
        VIRTUAL = 1,
        HANDLE = 2
    };

    // Default constructor
    /*! initializes memory type to host and dimension as empty vector*/
    TensorInfo();

    //! Initializer constructor with only fields common to all types (Image/ Video / Audio)
    TensorInfo(std::vector<size_t> dims, RocalMemType mem_type,
               RocalTensorDataType data_type);
    TensorInfo(std::vector<size_t> dims, RocalMemType mem_type,
               RocalTensorDataType data_type, RocalTensorlayout layout,
               RocalColorFormat color_format = RocalColorFormat::U8);

    // Setting properties required for Image / Video
    void set_roi_type(RocalROIType roi_type) { _roi_type = roi_type; }
    void set_data_type(RocalTensorDataType data_type) {
        if (_data_type == data_type)
            return;
        _data_type = data_type;
        _data_type_size = tensor_data_size(_data_type);
        modify_strides();
        _data_size = _strides[0] * _dims[0];
    }
    void get_modified_dims_from_layout(RocalTensorlayout input_layout, RocalTensorlayout output_layout, std::vector<size_t>& new_dims) {
        std::vector<size_t> dims_mapping;
        if (input_layout == RocalTensorlayout::NHWC && output_layout == RocalTensorlayout::NCHW) {
            dims_mapping = {0, 3, 1, 2};
        } else if (input_layout == RocalTensorlayout::NCHW && output_layout == RocalTensorlayout::NHWC) {
            dims_mapping = {0, 2, 3, 1};
        } else if (input_layout == RocalTensorlayout::NFHWC && output_layout == RocalTensorlayout::NFCHW) {
            dims_mapping = {0, 1, 4, 2, 3};
        } else if (input_layout == RocalTensorlayout::NFCHW && output_layout == RocalTensorlayout::NFHWC) {
            dims_mapping = {0, 1, 3, 4, 2};
        } else {
            THROW("Invalid layout conversion")
        }
        for (unsigned i = 0; i < _num_of_dims; i++)
            new_dims[i] = _dims.at(dims_mapping[i]);
    }
    void set_max_shape() {
        if (_is_metadata) return;  // For metadata tensors max shape is not required
        if (_layout != RocalTensorlayout::NONE) {
            if (!_max_shape.size()) _max_shape.resize(2);  // Since 2 values will be stored in the vector
            if (_layout == RocalTensorlayout::NHW || _layout == RocalTensorlayout::NFT || _layout == RocalTensorlayout::NTF) {   // For Audio/2D layouts
                _max_shape[0] = _dims.at(1);
                _max_shape[1] = _dims.at(2);
            } else {            // For Image layouts
                _is_image = true;
                if (_layout == RocalTensorlayout::NHWC) {
                    _max_shape[0] = _dims.at(2);
                    _max_shape[1] = _dims.at(1);
                    _channels = _dims.at(3);
                } else if (_layout == RocalTensorlayout::NCHW) {
                    _max_shape[0] = _dims.at(3);
                    _max_shape[1] = _dims.at(2);
                    _channels = _dims.at(1);
                } else if (_layout == RocalTensorlayout::NFHWC) {
                    _max_shape[0] = _dims.at(3);
                    _max_shape[1] = _dims.at(2);
                    _channels = _dims.at(4);
                } else if (_layout == RocalTensorlayout::NFCHW) {
                    _max_shape[0] = _dims.at(4);
                    _max_shape[1] = _dims.at(3);
                    _channels = _dims.at(2);
                }
            }
        } else {                                                             // For other tensors
            if (!_max_shape.size()) _max_shape.resize(_num_of_dims - 1, 0);  // Since 2 values will be stored in the vector
            _max_shape.assign(_dims.begin() + 1, _dims.end());
        }
        reset_tensor_roi_buffers();
    }
    void set_tensor_layout(RocalTensorlayout layout) {
        if (layout == RocalTensorlayout::NONE) return;
        if (_layout != layout && _layout != RocalTensorlayout::NONE && (_num_of_dims > 3)) {  // If layout input and current layout's are different modify dims accordingly
            std::vector<size_t> new_dims(_num_of_dims, 0);
            get_modified_dims_from_layout(_layout, layout, new_dims);
            _dims = new_dims;
            modify_strides();
        }
        _layout = layout;
    }
    void set_dims(std::vector<size_t>& new_dims) {
        if (_num_of_dims != new_dims.size())
            THROW("The size of number of dimensions does not match with the dimensions of existing tensor")
        _dims = new_dims;
        modify_strides();
        _data_size = _strides[0] * _dims[0];
        set_max_shape();
    }
    void modify_dims_width_and_height(RocalTensorlayout layout, size_t width, size_t height) {
        switch (_layout) {
            case RocalTensorlayout::NHWC: {
                _max_shape[1] = _dims[1] = height;
                _max_shape[0] = _dims[2] = width;
                break;
            }
            case RocalTensorlayout::NCHW:
            case RocalTensorlayout::NFHWC: {
                _max_shape[1] = _dims[2] = height;
                _max_shape[0] = _dims[3] = width;
                break;
            }
            case RocalTensorlayout::NFCHW: {
                _max_shape[1] = _dims[3] = height;
                _max_shape[0] = _dims[4] = width;
                break;
            }
            default: {
                THROW("Invalid layout type specified")
            }
        }
        modify_strides();
        _data_size = _strides[0] * _dims[0];  // Modify data size wrt latest width and height
        set_tensor_layout(layout);            // Modify the layout and dims based on the layout input
        reset_tensor_roi_buffers();           // Reset ROI buffers to reflect the modified width and height
    }
    void modify_strides() {
        _strides[_num_of_dims - 1] = _data_type_size;
        for (int i = _num_of_dims - 2; i >= 0; i--) {
            _strides[i] = _strides[i + 1] * _dims[i + 1];
        }
    }
    void set_color_format(RocalColorFormat color_format) {
        _color_format = color_format;
    }
    // Introduce for SequenceReader, as batch size is different in case of sequence reader
    void set_sequence_batch_size(unsigned sequence_length) {
        _batch_size *= sequence_length;
    }
    size_t get_channels() const { return _channels; }
    unsigned num_of_dims() const { return _num_of_dims; }
    unsigned batch_size() const { return _batch_size; }
    uint64_t data_size() const { return _data_size; }
    std::vector<size_t> max_shape() const { return _max_shape; }
    std::vector<size_t> dims() const { return _dims; }
    std::vector<size_t> strides() const { return _strides; }
    RocalMemType mem_type() const { return _mem_type; }
    RocalROIType roi_type() const { return _roi_type; }
    RocalTensorDataType data_type() const { return _data_type; }
    RocalTensorlayout layout() const { return _layout; }
    Roi roi() const { return _roi; }
    RocalColorFormat color_format() const { return _color_format; }
    Type type() const { return _type; }
    uint64_t data_type_size() {
        _data_type_size = tensor_data_size(_data_type);
        return _data_type_size;
    }
    bool is_image() const { return _is_image; }
    void set_metadata() { _is_metadata = true; }
    bool is_metadata() const { return _is_metadata; }
    void set_roi_ptr(unsigned* roi_ptr) { _roi.reset_ptr(roi_ptr); }
    void copy_roi(void* roi_buffer) { _roi.copy(roi_buffer); }
    std::shared_ptr<std::vector<float>> get_sample_rates() const { return _sample_rates; }  //!< The number of samples of audio carried per second

   private:
    Type _type = Type::UNKNOWN;                                  //!< tensor type, whether is virtual tensor, created from handle or is a regular tensor
    unsigned _num_of_dims;                                       //!< denotes the number of dimensions in the tensor
    std::vector<size_t> _dims;                                   //!< denotes the dimensions of the tensor
    std::vector<size_t> _strides;                                //!< stores the stride for each dimension in the tensor
    unsigned _batch_size;                                        //!< the batch size
    RocalMemType _mem_type;                                      //!< memory type, currently either OpenCL or Host
    RocalROIType _roi_type = RocalROIType::XYWH;                 //!< ROI type, currently either XYWH or LTRB
    RocalTensorDataType _data_type = RocalTensorDataType::FP32;  //!< tensor data type
    RocalTensorlayout _layout = RocalTensorlayout::NONE;         //!< layout of the tensor
    RocalColorFormat _color_format;                              //!< color format of the image
    Roi _roi;
    uint64_t _data_type_size = tensor_data_size(_data_type);
    uint64_t _data_size = 0;
    std::vector<size_t> _max_shape;  //!< stores the the width and height dimensions in the tensor
    void reset_tensor_roi_buffers();
    void reallocate_tensor_sample_rate_buffers(); //!< Reallocating the sample_rate buffer
    bool _is_image = false;
    bool _is_metadata = false;
    size_t _channels = 3;  //!< stores the channel dimensions in the tensor
    std::shared_ptr<std::vector<float>> _sample_rates;  //!< Stores the sample rates for the audio
};

bool operator==(const TensorInfo& rhs, const TensorInfo& lhs);
/*! \brief Holds an OpenVX tensor and it's info
 * Keeps the information about the tensor that can be queried using OVX API as
 * well, but for simplicity and ease of use, they are kept in separate fields
 */
class Tensor : public rocalTensor {
   public:
    int swap_handle(void* handle);
    const TensorInfo& info() { return _info; }
    //! Default constructor
    Tensor() = delete;
    void* buffer() override { return _mem_handle; }
    vx_tensor handle() { return _vx_handle; }
    vx_context context() { return _context; }
    void set_mem_handle(void* buffer) override {
        if (buffer)
            _mem_handle = buffer;
        else {
            THROW("Invalid buffer pointer passed")
        }
    }
    void reset_mem_handle() { _mem_handle = nullptr; }
#if ENABLE_OPENCL
    unsigned copy_data(cl_command_queue queue, unsigned char* user_buffer, bool sync);
    unsigned copy_data(cl_command_queue queue, cl_mem user_buffer, bool sync);
#elif ENABLE_HIP
    unsigned copy_data(hipStream_t stream, void* host_memory, bool sync);
#endif
    unsigned copy_data(void* user_buffer, RocalOutputMemType external_mem_type) override;
    //! Copying the output buffer with specified max_cols and max_rows values for the 2D buffer of size batch_size
    unsigned copy_data(void* user_buffer, uint x_offset, uint y_offset, uint max_rows, uint max_cols) override;
    //! Default destructor
    /*! Releases the OpenVX Tensor object */
    ~Tensor();

    //! Constructor accepting the tensor information as input
    explicit Tensor(const TensorInfo& tensor_info);
    int create(vx_context context);
    void create_roi_tensor_from_handle(void** handle);
    void update_tensor_roi(const std::vector<uint32_t>& width, const std::vector<uint32_t>& height);
    void update_tensor_roi(const std::vector<std::vector<uint32_t>>& shape);
    void reset_tensor_roi() { _info.reset_tensor_roi_buffers(); }
    void reset_audio_sample_rate() { _info.reallocate_tensor_sample_rate_buffers(); }
    void update_audio_tensor_sample_rate(const std::vector<float>& sample_rate);
    void set_roi(unsigned* roi_ptr) { _info.set_roi_ptr(roi_ptr); }
    void copy_roi(void* roi_buffer) override { _info.copy_roi(roi_buffer); }
    size_t get_roi_dims_size() override { return _info.roi().no_of_dims(); }
    vx_tensor get_roi_tensor() { return _vx_roi_handle; }
    // create_from_handle() no internal memory allocation is done here since
    // tensor's handle should be swapped with external buffers before usage
    int create_from_handle(vx_context context);
    int create_virtual(vx_context context, vx_graph graph);
    bool is_handle_set() { return (_vx_handle != 0); }
    void set_dims(std::vector<size_t> dims) override { _info.set_dims(dims); }
    unsigned num_of_dims() override { return _info.num_of_dims(); }
    unsigned batch_size() override { return _info.batch_size(); }
    std::vector<size_t> dims() override { return _info.dims(); }
    std::vector<size_t> strides() override { return _info.strides(); }
    RocalTensorLayout layout() override { return (RocalTensorLayout)_info.layout(); }
    void set_tensor_layout(RocalTensorLayout layout) override { _info.set_tensor_layout((RocalTensorlayout)layout); }
    RocalTensorOutputType data_type() override { return (RocalTensorOutputType)_info.data_type(); }
    RocalOutputMemType mem_type() override { return (_info.mem_type() == RocalMemType::HOST ? ROCAL_MEMCPY_HOST : ROCAL_MEMCPY_GPU); }
    size_t data_size() override { return _info.data_size(); }
    RocalROICordsType roi_type() override { return (RocalROICordsType)_info.roi_type(); }
    std::vector<size_t> shape() override { return _info.max_shape(); }
    RocalImageColor color_format() const { return (RocalImageColor)_info.color_format(); }
    RocalTensorBackend backend() override {
        return (_info.mem_type() == RocalMemType::HOST ? ROCAL_CPU : ROCAL_GPU);
    }
    uint64_t data_type_size() override { return _info.data_type_size(); }

   private:
    vx_tensor _vx_handle = nullptr;  //!< The OpenVX tensor
    void* _mem_handle = nullptr;     //!< Pointer to the tensor's internal buffer (opencl or host)
    TensorInfo _info;                //!< The structure holding the info related to the stored OpenVX tensor
    vx_context _context = nullptr;
    vx_tensor _vx_roi_handle = nullptr;  //!< The OpenVX tensor for ROI
};

/*! \brief Contains a list of rocalTensors */
class TensorList : public rocalTensorList {
   public:
    uint64_t size() override { return _tensor_list.size(); }
    bool empty() { return _tensor_list.empty(); }
    Tensor* front() { return _tensor_list.front(); }
    void push_back(Tensor* tensor) {
        _tensor_list.emplace_back(tensor);
        _tensor_data_size.emplace_back(tensor->info().data_size());
        _tensor_roi_size.emplace_back(tensor->info().roi().roi_buffer_size());
    }
    std::vector<uint64_t>& data_size() { return _tensor_data_size; }
    std::vector<uint64_t>& roi_size() { return _tensor_roi_size; }
    void release() {
        for (auto& tensor : _tensor_list) delete tensor;
    }
    Tensor* operator[](size_t index) { return _tensor_list[index]; }
    Tensor* at(size_t index) override { return _tensor_list[index]; }
    void operator=(TensorList& other) {
        for (unsigned idx = 0; idx < other.size(); idx++) {
            auto* new_tensor = new Tensor(other[idx]->info());
            if (new_tensor->create_from_handle(other[idx]->context()) != 0)
                THROW("Cannot create the tensor from handle")
            this->push_back(new_tensor);
        }
    }

   private:
    std::vector<Tensor*> _tensor_list;
    std::vector<uint64_t> _tensor_data_size;
    std::vector<uint64_t> _tensor_roi_size;
};

/*! \brief Contains a list of rocalTensorList */
class TensorListVector : public rocalListOfTensorList {
   public:
    uint64_t size() override { return _tensor_list_vector.size(); }
    bool empty() { return _tensor_list_vector.empty(); }
    TensorList* front() { return _tensor_list_vector.front(); }
    void push_back(TensorList* tensor_list) {
        _tensor_list_vector.push_back(tensor_list);
    }
    void emplace_back(TensorList* tensor_list) {
        _tensor_list_vector.emplace_back(tensor_list);
    }
    void release() {
        for (auto& tensor_list : _tensor_list_vector) tensor_list->release();
    }
    TensorList* operator[](size_t index) { return _tensor_list_vector[index]; }
    TensorList* at(size_t index) override { return _tensor_list_vector[index]; }
   private:
    std::vector<TensorList*> _tensor_list_vector;
};
