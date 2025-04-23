
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

#include <cstdio>
#if ENABLE_OPENCL
#include <CL/cl.h>
#endif
#include <vx_ext_amd.h>

#include <cstring>
#include <stdexcept>
#include "pipeline/commons.h"
#include "pipeline/tensor.h"

vx_enum vx_mem_type(RocalMemType mem) {
    switch (mem) {
        case RocalMemType::OCL:
            return VX_MEMORY_TYPE_OPENCL;
        case RocalMemType::HOST:
            return VX_MEMORY_TYPE_HOST;
        case RocalMemType::HIP:
            return VX_MEMORY_TYPE_HIP;
        default:
            throw std::runtime_error("Memory type not valid");
    }
}

vx_size tensor_data_size(RocalTensorDataType data_type) {
    switch (data_type) {
        case RocalTensorDataType::FP32:
            return sizeof(vx_float32);
        case RocalTensorDataType::FP16:
#if defined(AMD_FP16_SUPPORT)
            return sizeof(vx_float16);
#else
            THROW("FLOAT16 type tensor not supported")
            return 0;
#endif
        case RocalTensorDataType::UINT8:
            return sizeof(vx_uint8);
        case RocalTensorDataType::UINT32:
            return sizeof(vx_uint32);
        case RocalTensorDataType::INT32:
            return sizeof(vx_int32);
        case RocalTensorDataType::INT16:
            return sizeof(vx_int16);
        default:
            throw std::runtime_error("tensor data_type not valid");
    }
}

//! Converts the Rocal data_type to OpenVX
vx_enum interpret_tensor_data_type(RocalTensorDataType data_type) {
    switch (data_type) {
        case RocalTensorDataType::FP32:
            return VX_TYPE_FLOAT32;
        case RocalTensorDataType::FP16:
            return VX_TYPE_FLOAT16;
        case RocalTensorDataType::UINT8:
            return VX_TYPE_UINT8;
        case RocalTensorDataType::UINT32:
            return VX_TYPE_UINT32;
        case RocalTensorDataType::INT32:
            return VX_TYPE_INT32;
        case RocalTensorDataType::INT16:
            return VX_TYPE_INT16;
        default:
            THROW("Unsupported Tensor type " + TOSTR(data_type))
    }
}

void allocate_host_or_pinned_mem(void **ptr, size_t size, RocalMemType mem_type) {
    if (mem_type == RocalMemType::HIP) {
#if ENABLE_HIP
        hipError_t err = hipHostMalloc((void **)ptr, size, hipHostMallocDefault);
        if (err != hipSuccess || !*ptr)
            THROW("hipHostMalloc of size " + TOSTR(size) + " failed " + TOSTR(err))
        err = hipMemset((void *)*ptr, 0, size);
        if (err != hipSuccess)
            THROW("hipMemset of size " + TOSTR(size) + " failed " + TOSTR(err))
#endif
    } else {
        *ptr = (void *)malloc(size);
        memset((void *)*ptr, 0, size);
    }
}

bool operator==(const TensorInfo &rhs, const TensorInfo &lhs) {
    return (rhs.dims() == lhs.dims() &&
            rhs.mem_type() == lhs.mem_type() &&
            rhs.data_type() == lhs.data_type() &&
            rhs.color_format() == lhs.color_format() &&
            rhs.layout() == lhs.layout());
}

void TensorInfo::reset_tensor_roi_buffers() {
    unsigned *roi_buf;
    auto roi_no_of_dims = _is_image ? 2 : (_num_of_dims - 1);
    auto roi_size = (_layout == RocalTensorlayout::NFCHW || _layout == RocalTensorlayout::NFHWC) ? _dims[0] * _dims[1] : _batch_size;  // For Sequences pre allocating the ROI to N * F to replicate in OpenVX extensions
    allocate_host_or_pinned_mem((void **)&roi_buf, roi_size * roi_no_of_dims * 2 * sizeof(unsigned), _mem_type);                       // 2 denotes, one coordinate each for begin and end
    _roi.set_ptr(roi_buf, _mem_type, roi_size, roi_no_of_dims);
    if (_is_image) {
        Roi2DCords *roi = _roi.get_2D_roi();
        for (unsigned i = 0; i < _batch_size; i++) {
            roi[i].xywh.w = _max_shape.at(0);
            roi[i].xywh.h = _max_shape.at(1);
        }
    } else {
        for (unsigned i = 0; i < _batch_size; i++) {
            unsigned *tensor_shape = _roi[i].end;
            for (unsigned j = 0; j < _max_shape.size(); j++)
                tensor_shape[j] = _max_shape[j];
        }
    }
}

void TensorInfo::reallocate_tensor_sample_rate_buffers() {
    if (_is_image)
        THROW("Sample rate not available for Image data")
    _sample_rates = std::make_shared<std::vector<float>>(_batch_size);
}

TensorInfo::TensorInfo()
    : _type(Type::UNKNOWN),
      _num_of_dims(0),
      _dims({}),
      _mem_type(RocalMemType::HOST),
      _data_type(RocalTensorDataType::FP32) {}

TensorInfo::TensorInfo(std::vector<size_t> dims,
                       RocalMemType mem_type,
                       RocalTensorDataType data_type)
    : _type(Type::UNKNOWN),
      _dims(dims),
      _mem_type(mem_type),
      _data_type(data_type) {
    _batch_size = dims.at(0);
    _num_of_dims = dims.size();
    _strides.resize(_num_of_dims);
    _strides[_num_of_dims - 1] = tensor_data_size(data_type);
    for (int i = _num_of_dims - 2; i >= 0; i--) {
        _strides[i] = _strides[i + 1] * dims[i + 1];
    }
    _data_size = _strides[0] * dims[0];

    if (_num_of_dims <= 3) _is_image = false;
}

TensorInfo::TensorInfo(std::vector<size_t> dims,
                       RocalMemType mem_type,
                       RocalTensorDataType data_type,
                       RocalTensorlayout layout,
                       RocalColorFormat color_format)
    : _type(Type::UNKNOWN),
      _dims(dims),
      _mem_type(mem_type),
      _data_type(data_type),
      _layout(layout),
      _color_format(color_format) {
    _batch_size = dims.at(0);
    _num_of_dims = dims.size();
    _strides.resize(_num_of_dims);
    _strides[_num_of_dims - 1] = tensor_data_size(data_type);
    for (int i = _num_of_dims - 2; i >= 0; i--) {
        _strides[i] = _strides[i + 1] * dims[i + 1];
    }
    _data_size = _strides[0] * dims[0];

    if (_num_of_dims <= 3) _is_image = false;
    set_max_shape();
}

void Tensor::update_tensor_roi(const std::vector<uint32_t> &width,
                               const std::vector<uint32_t> &height) {
    if (_info.is_image()) {
        auto max_shape = _info.max_shape();
        unsigned max_width = max_shape.at(0);
        unsigned max_height = max_shape.at(1);
        Roi2DCords *roi = _info.roi().get_2D_roi();

        if (width.size() != height.size())
            THROW("Batch size of Tensor height and width info does not match")

        if (width.size() != info().batch_size())
            THROW("The batch size of actual Tensor height and width different from Tensor batch size " + TOSTR(width.size()) + " != " + TOSTR(info().batch_size()))

        for (unsigned i = 0; i < info().batch_size(); i++) {
            if (width[i] > max_width) {
                WRN("Given ROI width is larger than buffer width for tensor[" + TOSTR(i) + "] " + TOSTR(width[i]) + " > " + TOSTR(max_width))
                roi[i].xywh.w = max_width;
            } else {
                roi[i].xywh.w = width[i];
            }
            if (height[i] > max_height) {
                WRN("Given ROI height is larger than buffer height for tensor[" + TOSTR(i) + "] " + TOSTR(height[i]) + " > " + TOSTR(max_height))
                roi[i].xywh.h = max_height;
            } else {
                roi[i].xywh.h = height[i];
            }
        }
    } else if (!_info.is_metadata()) {  // Audio - Data
        auto max_dims = _info.max_shape();
        unsigned max_samples = max_dims.at(0);
        unsigned max_channels = max_dims.at(1);
        auto samples = width;
        auto channels = height;

        if (samples.size() != channels.size())
            THROW("Size of tensor's samples and channels info does not match")
        if (samples.size() != info().batch_size())
            THROW("The size of actual tensor's samples and channels are different from tensor batch size " + TOSTR(samples.size()) + " != " + TOSTR(info().batch_size()))
        
        // Update the ROI with the samples and channels of the audio files
        for (unsigned i = 0; i < info().batch_size(); i++) {
            unsigned *tensor_shape = _info.roi()[i].end;
            if (samples[i] > max_samples) {
                ERR("Given ROI samples is larger than buffer samples for tensor[" + TOSTR(i) + "] " + TOSTR(samples[i]) + " > " + TOSTR(max_samples))
                tensor_shape[0] = max_samples;
            } else {
                tensor_shape[0] = samples[i];
            }
            if (channels[i] > max_channels) {
                ERR("Given ROI channels is larger than buffer channels for tensor[" + TOSTR(i) + "] " + TOSTR(channels[i]) + " > " + TOSTR(max_channels))
                tensor_shape[1] = max_channels;
            } else {
                tensor_shape[1] = channels[i];
            }
        }
    }
}

void Tensor::update_audio_tensor_sample_rate(const std::vector<float> &sample_rate) {
    if (_info.is_image()) {
        THROW("No sample rate available for Image data")
    } else if (!_info.is_metadata()) {
        for (unsigned i = 0; i < info().batch_size(); i++) {
            _info.get_sample_rates()->at(i) = sample_rate[i];
        }
    }
}

void Tensor::update_tensor_roi(const std::vector<std::vector<uint32_t>> &shape) {
    auto max_shape = _info.max_shape();
    if (shape.size() != info().batch_size())
        THROW("The batch size of actual Tensor shape different from Tensor batch size " + TOSTR(shape.size()) + " != " + TOSTR(info().batch_size()))

    for (unsigned i = 0; i < info().batch_size(); i++) {
        if (shape[i].size() != max_shape.size())
            THROW("The ROI shape must match the max_shape of the tensor")

        unsigned *tensor_shape = _info.roi()[i].end;
        for (unsigned d = 0; d < shape[i].size(); d++) {
            if (shape[i][d] > max_shape[d]) {
                WRN("Given ROI shape is larger than buffer shape for tensor[" + TOSTR(i) + "] " + TOSTR(shape[i][d]) + " > " + TOSTR(max_shape[d]))
                tensor_shape[d] = max_shape[d];
            } else {
                tensor_shape[d] = shape[i][d];
            }
        }
    }
}

Tensor::~Tensor() {
    _mem_handle = nullptr;
    if (_vx_handle) vxReleaseTensor(&_vx_handle);
    if (_vx_roi_handle) vxReleaseTensor(&_vx_roi_handle);
}

Tensor::Tensor(const TensorInfo &tensor_info)
    : _info(tensor_info) {
    _info._type = TensorInfo::Type::UNKNOWN;
    _mem_handle = nullptr;
}

int Tensor::create_virtual(vx_context context, vx_graph graph) {
    if (_vx_handle) {
        WRN("Tensor object create method is already called ")
        return -1;
    }

    _context = context;
    _vx_handle = vxCreateVirtualTensor(graph, _info.num_of_dims(), _info.dims().data(), interpret_tensor_data_type(_info.data_type()), 0);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_vx_handle)) != VX_SUCCESS)
        THROW("Error: vxCreateVirtualTensor(input:[" + TOSTR(_info.max_shape().at(0)) + "W" + TOSTR(_info.max_shape().at(1)) + "H" + "]): failed " + TOSTR(status))

    _info._type = TensorInfo::Type::VIRTUAL;
    void *roi_handle = reinterpret_cast<void *>(_info.roi().get_ptr());
    create_roi_tensor_from_handle(&roi_handle);  // Create ROI tensor from handle
    return 0;
}

int Tensor::create_from_handle(vx_context context) {
    if (_vx_handle) {
        WRN("Tensor object create method is already called ")
        return -1;
    }

    _context = context;
    vx_enum tensor_data_type = interpret_tensor_data_type(_info.data_type());
    unsigned num_of_dims = _info.num_of_dims();
    std::vector<vx_size> stride(num_of_dims);
    void *ptr[1] = {nullptr};

    stride[0] = tensor_data_size(_info.data_type());
    for (unsigned i = 1; i < num_of_dims; i++)
        stride[i] = stride[i - 1] * _info.dims().at(i - 1);

    _vx_handle = vxCreateTensorFromHandle(_context, _info.num_of_dims(), _info.dims().data(), tensor_data_type, 0, stride.data(), ptr, vx_mem_type(_info._mem_type));
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_vx_handle)) != VX_SUCCESS)
        THROW("Error: vxCreateTensorFromHandle(input: failed " + TOSTR(status))
    _info._type = TensorInfo::Type::HANDLE;
    void *roi_handle = reinterpret_cast<void *>(_info.roi().get_ptr());
    create_roi_tensor_from_handle(&roi_handle);  // Create ROI tensor from handle
    return 0;
}

int Tensor::create(vx_context context) {
    if (_vx_handle) {
        WRN("Tensor object create method is already called ")
        return -1;
    }

    _context = context;
    vx_status status;
    vx_enum tensor_data_type = interpret_tensor_data_type(_info.data_type());
    _vx_handle = vxCreateTensor(context, _info.num_of_dims(), _info.dims().data(), tensor_data_type, 0);
    if ((status = vxGetStatus((vx_reference)_vx_handle)) != VX_SUCCESS)
        THROW("Error: vxCreateTensor(input: failed " + TOSTR(status))
    _info._type = TensorInfo::Type::REGULAR;
    return 0;
}

void Tensor::create_roi_tensor_from_handle(void **handle) {
    if (_vx_roi_handle) {
        WRN("ROI Tensor object is already created")
        return;
    }

    if (*handle == nullptr) {
        THROW("Empty ROI handle is passed")
    }

    const vx_size num_of_dims = 2;
    vx_size stride[num_of_dims];
    std::vector<size_t> roi_dims = {_info.batch_size(), 4};
    if (_info.layout() == RocalTensorlayout::NFCHW || _info.layout() == RocalTensorlayout::NFHWC)
        roi_dims = {_info.dims()[0] * _info.dims()[1], 4};  // For Sequences pre allocating the ROI to N * F to replicate in OpenVX extensions        stride[0] = sizeof(vx_uint32);
    stride[0] = sizeof(vx_uint32);
    stride[1] = stride[0] * roi_dims[0];
    vx_enum mem_type = VX_MEMORY_TYPE_HOST;
    if (_info.mem_type() == RocalMemType::HIP)
        mem_type = VX_MEMORY_TYPE_HIP;

    _vx_roi_handle = vxCreateTensorFromHandle(_context, num_of_dims, roi_dims.data(),
                                              VX_TYPE_UINT32, 0, stride, *handle, mem_type);
    vx_status status;
    if ((status = vxGetStatus((vx_reference)_vx_roi_handle)) != VX_SUCCESS)
        THROW("Error: vxCreateTensorFromHandle(src tensor roi: failed " + TOSTR(status))
}

#if ENABLE_OPENCL
unsigned Tensor::copy_data(cl_command_queue queue, unsigned char *user_buffer, bool sync) {
    if (_info._type != TensorInfo::Type::HANDLE) return 0;

    if (_info._mem_type == RocalMemType::OCL) {
        cl_int status;
        if ((status = clEnqueueReadBuffer(
                 queue, (cl_mem)_mem_handle, sync ? (CL_TRUE) : CL_FALSE, 0,
                 _info.data_size(), user_buffer, 0, nullptr, nullptr)) != CL_SUCCESS) {
            THROW("clEnqueueReadBuffer failed: " + TOSTR(status))
        }
    } else {
        memcpy(user_buffer, _mem_handle, _info.data_size());
    }
    return 0;
}
#elif ENABLE_HIP
unsigned Tensor::copy_data(hipStream_t stream, void *host_memory, bool sync) {
    if (_info._type != TensorInfo::Type::HANDLE) return 0;

    if (_info._mem_type == RocalMemType::HIP) {
        // copy from device to host
        hipError_t status;
        if ((status = hipMemcpyDtoHAsync((void *)host_memory, _mem_handle, _info.data_size(), stream)))
            THROW("copy_data::hipMemcpyDtoH failed: " + TOSTR(status))
        if (sync) {
            if ((status = hipStreamSynchronize(stream)))
                THROW("copy_data::hipStreamSynchronize failed: " + TOSTR(status))
        }
    } else {
        memcpy(host_memory, _mem_handle, _info.data_size());
    }
    return 0;
}
#endif

unsigned Tensor::copy_data(void *user_buffer, RocalOutputMemType external_mem_type) {
    if (_mem_handle == nullptr) return 0;
    if (external_mem_type == RocalOutputMemType::ROCAL_MEMCPY_GPU) {
#if ENABLE_HIP
        if (_info._mem_type == RocalMemType::HIP) {
            // copy from device to device
            hipError_t status;
            if ((status = hipMemcpyDtoD((void *)user_buffer, _mem_handle, _info.data_size())))
                THROW("copy_data::hipMemcpyDtoD failed: " + hipGetErrorName(status))
        } else if (_info._mem_type == RocalMemType::HOST) {
            // copy from host to device
            hipError_t status;
            if ((status = hipMemcpyHtoD((void *)user_buffer, _mem_handle, _info.data_size())))
                THROW("copy_data::hipMemcpyHtoD failed: " + TOSTR(status))
        }
#else
        THROW("copy_data failed as HIP is not supported")
#endif
    } else if (external_mem_type == RocalOutputMemType::ROCAL_MEMCPY_HOST) {
        if (_info._mem_type == RocalMemType::HIP) {
#if ENABLE_HIP
            // copy from device to host
            hipError_t status;
            if ((status = hipMemcpyDtoH((void *)user_buffer, _mem_handle, _info.data_size())))
                THROW("copy_data::hipMemcpyDtoH failed: " + TOSTR(status))
#else
            THROW("copy_data failed as HIP is not supported")
#endif
        } else if (_info._mem_type == RocalMemType::HOST) {
            memcpy(user_buffer, _mem_handle, _info.data_size());
        }
    } else {
        THROW("copy_data requested mem type not supported")
    }
    return 0;
}

unsigned Tensor::copy_data(void *user_buffer, uint x_offset, uint y_offset, uint roi_width, uint roi_height) {
    if (_mem_handle == nullptr) return 0;
    // TODO : Handle this case for HIP buffer
    auto max_shape_rows = _info.max_shape().at(1);
    auto dtype_size = _info.data_type_size();
    auto num_of_bytes_max_rows = max_shape_rows * dtype_size;
    auto src_stride = (_info.max_shape().at(0) * num_of_bytes_max_rows);
    auto num_of_bytes_rows = roi_width * dtype_size;
    auto dst_stride = (roi_height * num_of_bytes_rows);

    for (uint i = 0; i < _info._batch_size; i++) {
        auto temp_src_ptr = static_cast<unsigned char *>(_mem_handle) + i * src_stride + _info.max_shape().at(0) * y_offset + x_offset;
        auto temp_dst_ptr = static_cast<unsigned char *>(user_buffer) + i * dst_stride;
        for (uint height = 0; height < roi_height; height++) {
            memcpy(temp_dst_ptr, temp_src_ptr, num_of_bytes_rows);
            temp_src_ptr += num_of_bytes_max_rows;
            temp_dst_ptr += num_of_bytes_rows;
        }
    }
    return 0;
}

int Tensor::swap_handle(void *handle) {
    vx_status status;
    if ((status = vxSwapTensorHandle(_vx_handle, handle, nullptr)) != VX_SUCCESS) {
        ERR("Swap handles failed for tensor" + TOSTR(status));
        return -1;
    }

    // Updating the buffer pointer as well,
    // user might want to copy directly using it
    _mem_handle = handle;
    return 0;
}
