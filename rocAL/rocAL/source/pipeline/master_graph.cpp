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
#if ENABLE_OPENCL
#include <CL/cl.h>
#endif
#include <vx_ext_amd.h>
#include <VX/vx_types.h>
#include <cstring>
#include <sched.h>
#include <half/half.hpp>
#include "pipeline/master_graph.h"
#include "parameters/parameter_factory.h"
#include "device/ocl_setup.h"
#include "pipeline/log.h"
#include "meta_data/meta_data_reader_factory.h"
#include "meta_data/meta_data_graph_factory.h"
#include "meta_data/randombboxcrop_meta_data_reader_factory.h"
#include "augmentations/node_copy.h"

using half_float::half;

#if ENABLE_HIP
#include <rocal_hip_kernels.h>
#endif

static void VX_CALLBACK log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char *string) {
    size_t len = strnlen(string, MAX_STRING_LENGTH);
    if (len > 0) {
        printf("%s", string);
        if (string[len - 1] != '\n')
            printf("\n");
        fflush(stdout);
    }
}

auto get_ago_affinity_info = [](RocalAffinity rocal_affinity,
                                int cpu_id,
                                int gpu_id) {
    AgoTargetAffinityInfo affinity;
    switch (rocal_affinity) {
        case RocalAffinity::GPU:
            affinity.device_type = AGO_TARGET_AFFINITY_GPU;
            affinity.device_info = (gpu_id >= 0 && gpu_id <= 9) ? gpu_id : 0;
            affinity.group = affinity.reserved = 0;
            break;
        case RocalAffinity::CPU:
            affinity.device_type = AGO_TARGET_AFFINITY_CPU;
            affinity.device_info = (cpu_id >= 0 && cpu_id <= 9) ? cpu_id : 0;
            affinity.group = affinity.reserved = 0;
            break;
        default:
            throw std::invalid_argument("Unsupported affinity");
    }
    return affinity;
};

// Function to append ImageNameBatch
ImageNameBatch &operator+=(ImageNameBatch &dest, const ImageNameBatch &src) {
    dest.insert(dest.end(), src.cbegin(), src.cend());
    return dest;
}

// Function to append vectors
std::vector<size_t> &operator+=(std::vector<size_t> &dest, const std::vector<size_t> &src) {
    dest.insert(dest.end(), src.cbegin(), src.cend());
    return dest;
}

// Function to append vector of vectors
std::vector<std::vector<float>> &operator+=(std::vector<std::vector<float>> &dest, const std::vector<std::vector<float>> &src) {
    dest.insert(dest.end(), src.cbegin(), src.cend());
    return dest;
}

MasterGraph::~MasterGraph() {
    release();
}

MasterGraph::MasterGraph(size_t batch_size, RocalAffinity affinity, size_t cpu_thread_count, int gpu_id, size_t prefetch_queue_depth, RocalTensorDataType output_tensor_data_type) : _ring_buffer(prefetch_queue_depth),
                                                                                                                                                                                     _graph(nullptr),
                                                                                                                                                                                     _affinity(affinity),
                                                                                                                                                                                     _cpu_num_threads(cpu_thread_count),
                                                                                                                                                                                     _gpu_id(gpu_id),
                                                                                                                                                                                     _convert_time("Conversion Time", DBG_TIMING),
                                                                                                                                                                                     _process_time("Process Time", DBG_TIMING),
                                                                                                                                                                                     _bencode_time("BoxEncoder Time", DBG_TIMING),
                                                                                                                                                                                     _user_batch_size(batch_size),
#if ENABLE_HIP
                                                                                                                                                                                     _mem_type((_affinity == RocalAffinity::GPU) ? RocalMemType::HIP : RocalMemType::HOST),
#elif ENABLE_OPENCL
                                                                                                                                                                                     _mem_type((_affinity == RocalAffinity::GPU) ? RocalMemType::OCL : RocalMemType::HOST),
#else
                                                                                                                                                                                     _mem_type(RocalMemType::HOST),
#endif
                                                                                                                                                                                     _first_run(true),
                                                                                                                                                                                     _processing(false),
                                                                                                                                                                                     _prefetch_queue_depth(prefetch_queue_depth),
#if ENABLE_HIP
                                                                                                                                                                                     _box_encoder_gpu(nullptr),
#endif
                                                                                                                                                                                     _rb_block_if_empty_time("Ring Buffer Block IF Empty Time"),
                                                                                                                                                                                     _rb_block_if_full_time("Ring Buffer Block IF Full Time") {
    try {
        vx_status status;
        vxRegisterLogCallback(NULL, log_callback, vx_false_e);
        _context = vxCreateContext();
        vxRegisterLogCallback(_context, log_callback, vx_false_e);
        auto vx_affinity = get_ago_affinity_info(_affinity, 0, gpu_id);
        if ((status = vxGetStatus((vx_reference)_context)) != VX_SUCCESS)
            THROW("vxCreateContext failed" + TOSTR(status))

        if (affinity == RocalAffinity::GPU) {
#if ENABLE_OPENCL
            if (_mem_type == RocalMemType::OCL) {
                cl_context _cl_context = nullptr;
                cl_device_id _cl_device_id = nullptr;
                get_device_and_context(gpu_id, &_cl_context, &_cl_device_id, CL_DEVICE_TYPE_GPU);
                if ((status = vxSetContextAttribute(_context,
                                                    VX_CONTEXT_ATTRIBUTE_AMD_OPENCL_CONTEXT,
                                                    &_cl_context, sizeof(cl_context)) != VX_SUCCESS))
                    THROW("vxSetContextAttribute for CL_CONTEXT failed " + TOSTR(status))
            }
#elif ENABLE_HIP
            if (_mem_type == RocalMemType::HIP) {
                hipError_t err = hipInit(0);
                if (err != hipSuccess) {
                    THROW("ERROR: hipInit(0) => %d (failed)" + TOSTR(err));
                }
                // initialize HIP device for rocAL
                int hip_num_devices = -1;
                err = hipGetDeviceCount(&hip_num_devices);
                if (err != hipSuccess) {
                    THROW("ERROR: hipGetDeviceCount() => %d (failed)" + TOSTR(err));
                }
                // set the device for context if specified.
                if (gpu_id < hip_num_devices) {
                    int hipDevice = gpu_id;
                    if ((status = vxSetContextAttribute(_context,
                                                        VX_CONTEXT_ATTRIBUTE_AMD_HIP_DEVICE,
                                                        &hipDevice, sizeof(hipDevice)) != VX_SUCCESS))
                        THROW("vxSetContextAttribute for hipDevice(%d) failed " + TOSTR(hipDevice) + TOSTR(status))
                } else
                    THROW("ERROR: HIP Device(%d) out of range" + TOSTR(gpu_id));
            }
#endif
        }

        // Setting attribute to run on CPU or GPU should be called before load kernel module
        if ((status = vxSetContextAttribute(_context,
                                            VX_CONTEXT_ATTRIBUTE_AMD_AFFINITY,
                                            &vx_affinity,
                                            sizeof(vx_affinity))) != VX_SUCCESS)
            THROW("vxSetContextAttribute for AMD_AFFINITY failed " + TOSTR(status))

        // loading OpenVX RPP modules
        if ((status = vxLoadKernels(_context, "vx_rpp")) != VX_SUCCESS)
            THROW("Cannot load vx_rpp extension (vx_rpp), vxLoadKernels failed " + TOSTR(status))
        else
            LOG("vx_rpp module loaded successfully")
        if (_affinity == RocalAffinity::GPU) {
#if ENABLE_HIP
            _device.init_hip(_context);
#elif ENABLE_OPENCL
            _device.init_ocl(_context);
#endif
        }
        ParameterFactory::instance()->set_seed(0);  // Setting default seed for ParameterFactory instance. User can set the seed manually by calling rocalSetSeed(seed_value)
    } catch (const std::exception &e) {
        release();
        throw;
    }
}

MasterGraph::Status
MasterGraph::run() {
    if (!_processing)  // The user should not call the run function before the build() is called or while reset() is happening
        return MasterGraph::Status::NOT_RUNNING;

    if (no_more_processed_data()) {
        return MasterGraph::Status::NO_MORE_DATA;
    }

    _rb_block_if_empty_time.start();
    _ring_buffer.block_if_empty();  // wait here if the user thread (caller of this function) is faster in consuming the processed images compare to th output routine in producing them
    _rb_block_if_empty_time.end();

    if (_first_run) {
        // calling run pops the processed images that have been used by user, when user calls run() for the first time
        // they've not used anything yet, so we don't pop a batch from the _ring_buffer
        _first_run = false;
    } else {
        _ring_buffer.pop();  // Pop previously used output images and metadata from the ring buffer
    }

    // If the last batch of processed imaged has been just popped from the ring_buffer it means user has previously consumed all the processed images.
    // User should check using the IsEmpty() API and not call run() or copy() API when there is no more data. run() will return MasterGraph::Status::NO_MORE_DATA flag to notify it.
    if (no_more_processed_data()) {
        return MasterGraph::Status::NO_MORE_DATA;
    }

    decrease_image_count();

    return MasterGraph::Status::OK;
}

void MasterGraph::decrease_image_count() {
    if (!_loop)
        _remaining_count -= (_is_sequence_reader_output ? _sequence_batch_size : _user_batch_size);
}

size_t
MasterGraph::calculate_cpu_num_threads(size_t shard_count) {
    if (_cpu_num_threads <= 0) {
        const unsigned minimum_cpu_thread_count = 2;
        const unsigned default_smt_count = 2;
        unsigned thread_count = std::thread::hardware_concurrency();
        if (thread_count < minimum_cpu_thread_count) {
            thread_count = minimum_cpu_thread_count;
            WRN("hardware_concurrency() call failed, assuming rocAL can run " + TOSTR(thread_count) + " threads")
        }
        size_t core_count = thread_count / default_smt_count;
        _cpu_num_threads = core_count / shard_count;
    }
    // Use _cpu_num_threads if user has already passed non-negative num_threads
    return _cpu_num_threads;
}

void MasterGraph::create_single_graph() {
    // Actual graph creating and calls into adding nodes to graph is deferred and is happening here to enable potential future optimizations
    _graph = std::make_shared<Graph>(_context, _affinity, 0, _cpu_num_threads, _gpu_id);
    for (auto &node : _nodes) {
        // Any tensor not yet created can be created as virtual tensor
        for (auto &tensor : node->output())
            if (tensor->info().type() == TensorInfo::Type::UNKNOWN) {
                tensor->create_virtual(_context, _graph->get());
                _internal_tensors.push_back(tensor);
            }
        node->create(_graph);
    }
    _graph->verify();
}

MasterGraph::Status
MasterGraph::build() {
    if (_internal_tensor_list.empty())
        THROW("No output tensors are there, cannot create the pipeline")

#if ENABLE_HIP || ENABLE_OPENCL
    _ring_buffer.init(_mem_type, (void *)_device.resources(), _internal_tensor_list.data_size(), _internal_tensor_list.roi_size());
#else
    _ring_buffer.init(_mem_type, nullptr, _internal_tensor_list.data_size(), _internal_tensor_list.roi_size());
#endif
    if (_is_box_encoder) _ring_buffer.initBoxEncoderMetaData(_mem_type, _user_batch_size * _num_anchors * 4 * sizeof(float), _user_batch_size * _num_anchors * sizeof(int));
    create_single_graph();
    start_processing();
    return Status::OK;
}

Tensor *
MasterGraph::create_loader_output_tensor(const TensorInfo &info) {
    /*
     *   NOTE: Output tensor for a source node needs to be created as a regular (non-virtual) tensor
     */
    auto output = new Tensor(info);
    if (output->create_from_handle(_context) != 0)
        THROW("Creating output tensor for loader failed");

    _internal_tensors.push_back(output);

    return output;
}

Tensor *
MasterGraph::create_tensor(const TensorInfo &info, bool is_output) {
    auto *output = new Tensor(info);
    // if the tensor is not an output tensor, the tensor creation is deferred and later it'll be created as a virtual tensor
    if (is_output) {
        if (output->create_from_handle(_context) != 0)
            THROW("Cannot create the tensor from handle")
        _internal_tensor_list.push_back(output);
        _output_tensor_list.push_back(new Tensor(info));  // Creating a replica of the output tensor to be returned to the user
    }

    return output;
}

void MasterGraph::set_output(Tensor *output_tensor) {
    if (output_tensor->is_handle_set() == false) {
        if (output_tensor->create_from_handle(_context) != 0)
            THROW("Cannot create the tensor from handle")

        _internal_tensor_list.push_back(output_tensor);
        _output_tensor_list.push_back(new Tensor(output_tensor->info()));  // Creating a replica of the output tensor to be returned to the user
    } else {
        // Decoder case only
        auto actual_output = create_tensor(output_tensor->info(), true);
        add_node<CopyNode>({output_tensor}, {actual_output});
    }
}

void MasterGraph::release() {
    LOG("MasterGraph release ...")
    stop_processing();
    _nodes.clear();
    _root_nodes.clear();
    _meta_data_nodes.clear();
    _tensor_map.clear();
    _ring_buffer.release_gpu_res();
    // shut_down loader:: required for releasing any allocated resourses
    _loader_module->shut_down();
    // release output buffer if allocated
    if (_output_tensor_buffer != nullptr) {
#if ENABLE_OPENCL
        clReleaseMemObject((cl_mem)_output_tensor_buffer);
#elif ENABLE_HIP
        hipError_t err = hipFree(_output_tensor_buffer);
        if (err != hipSuccess) {
            THROW("MasterGraph::deallocate_output_tensor  hipFree failed " + TOSTR(err))
        }
#endif
        _output_tensor_buffer = nullptr;
    }

    // release all openvx resources.
    vx_status status;
    for (auto &tensor : _internal_tensors)
        delete tensor;                // It will call the vxReleaseTensor internally in the destructor
    _internal_tensor_list.release();  // It will call the vxReleaseTensor internally in the destructor for each tensor in the list
    _output_tensor_list.release();    // It will call the vxReleaseTensor internally in the destructor for each tensor in the list
    _metadata_output_tensor_list.release(); // It will call the vxReleaseTensor internally in the destructor for each tensor in the list of TensorList
    _bbox_encoded_output.release(); // It will call the vxReleaseTensor internally in the destructor for each tensor in the list of TensorList

    if (_graph != nullptr)
        _graph->release();
    if (_meta_data_reader != nullptr)
        _meta_data_reader->release();

    _augmented_meta_data = nullptr;
    _meta_data_graph = nullptr;
    _meta_data_reader = nullptr;
    if (_context && (status = vxReleaseContext(&_context)) != VX_SUCCESS)
        LOG("Failed to call vxReleaseContext " + TOSTR(status))
}

MasterGraph::Status
MasterGraph::update_node_parameters() {
    // Randomize random parameters
    ParameterFactory::instance()->renew_parameters();

    // Apply renewed parameters to VX parameters used in augmentation
    for (auto &node : _nodes)
        node->update_parameters();

    return Status::OK;
}

size_t
MasterGraph::augmentation_branch_count() {
    return _output_tensor_list.size();
}

RocalColorFormat
MasterGraph::output_color_format() {
    return _output_tensor_list[0]->info().color_format();
}

size_t
MasterGraph::output_width() {
    return _output_tensor_list[0]->info().max_shape()[0];
}

size_t
MasterGraph::output_height() {
    return _output_tensor_list[0]->info().max_shape()[1];
}

void MasterGraph::sequence_start_frame_number(std::vector<size_t> &sequence_start_framenum) {
    sequence_start_framenum = _sequence_start_framenum_vec.back();
    _sequence_start_framenum_vec.pop_back();
}

void MasterGraph::sequence_frame_timestamps(std::vector<std::vector<float>> &sequence_frame_timestamp) {
    sequence_frame_timestamp = _sequence_frame_timestamps_vec.back();
    _sequence_frame_timestamps_vec.pop_back();
}

MasterGraph::Status
MasterGraph::reset() {
    // stop the internal processing thread so that the
    _processing = false;
    _ring_buffer.unblock_writer();
    if (_output_thread.joinable())
        _output_thread.join();
    _ring_buffer.reset();
    _sequence_start_framenum_vec.clear();
    _sequence_frame_timestamps_vec.clear();
    // clearing meta ring buffer
    // if random_bbox meta reader is used: read again to get different crops
    if (_randombboxcrop_meta_data_reader != nullptr)
        _randombboxcrop_meta_data_reader->release();
    // resetting loader module to start from the beginning of the media and clear it's internal state/buffers
    _loader_module->reset();
    // restart processing of the images
    _first_run = true;
    _output_routine_finished_processing = false;
    start_processing();
    return Status::OK;
}

size_t
MasterGraph::remaining_count() {
    if (!_external_source_eos && _external_source_reader)
        return _user_batch_size;
    return (_remaining_count >= 0) ? _remaining_count : 0;
}

RocalMemType
MasterGraph::mem_type() {
    return _mem_type;
}

size_t
MasterGraph::last_batch_padded_size() {
    return _loader_module->last_batch_padded_size();
}

Timing
MasterGraph::timing() {
    Timing t = _loader_module->timing();
    t.process_time += _process_time.get_timing();
    t.copy_to_output += _convert_time.get_timing();
    t.bb_process_time += _bencode_time.get_timing();
    return t;
}

#define CHECK_CL_CALL_RET(x)                                                                \
    {                                                                                       \
        cl_int ret;                                                                         \
        ret = x;                                                                            \
        if (ret != CL_SUCCESS) THROW("ocl call failed " + STR(#x) + " error " + TOSTR(ret)) \
    }

MasterGraph::Status
MasterGraph::to_tensor(void *out_ptr, RocalTensorlayout format, float multiplier0, float multiplier1,
                       float multiplier2, float offset0, float offset1, float offset2, bool reverse_channels, RocalTensorDataType output_data_type, RocalOutputMemType output_mem_type, uint max_roi_height, uint max_roi_width) {
    if (no_more_processed_data())
        return MasterGraph::Status::NO_MORE_DATA;

    if (_output_tensor_list.size() != 1)
        THROW("Cannot copy, Multiple output tensors present in the list")

    auto output_tensor_info = _output_tensor_list[0]->info();
    if (output_tensor_info.data_type() != RocalTensorDataType::UINT8)
        THROW("The output tensor is not of UINT8 type")

    if (output_tensor_info.color_format() == RocalColorFormat::RGB_PLANAR)
        return MasterGraph::copy_out_tensor_planar(out_ptr, format, multiplier0, multiplier1, multiplier2, offset0, offset1, offset2, reverse_channels, output_data_type);

    _convert_time.start();
    // Copies to the output context given by the user
    auto dims = output_tensor_info.dims();
    unsigned int n = dims[0];
    const size_t c = dims[3];
    const size_t h = dims[1];
    const size_t w = dims[2];
    const size_t single_output_tensor_size = output_tensor_info.data_size();
    if ((max_roi_height == 0) || (max_roi_width == 0)) {
        max_roi_height = h;
        max_roi_width = w;
    }

#if ENABLE_OPENCL
    if (output_tensor_info.mem_type() == RocalMemType::OCL) {
        if (output_data_type == RocalTensorDataType::FP16)
            THROW("FP16 tensor output for GPU affinity is not implemented")
        // OCL device memory
        cl_int status, ret;

        size_t global_work_size = output_tensor_info.data_size();  // Sample size
        size_t local_work_size = 256;

        // TODO: Use the runKernel function instead

        auto kernel_name = (format == RocalTensorlayout::NHWC) ? "copyInt8ToNHWC" : "copyInt8ToNCHW";
        cl_kernel kernel = _device["utility"][kernel_name];
        auto queue = _device.resources()->cmd_queue;
        unsigned dest_buf_offset = 0;
        auto output_buffers = _ring_buffer.get_read_buffers().first;

        if (_output_tensor_buffer == nullptr) {
            size_t size = output_tensor_info.data_size() * sizeof(cl_float);
            cl_mem clImgFloat = clCreateBuffer(_device.resources()->context,
                                               CL_MEM_READ_WRITE,
                                               size,
                                               nullptr, &ret);
            if (!clImgFloat || ret != CL_SUCCESS)
                THROW("clCreateBuffer of size " + TOSTR(size) + " failed " + TOSTR(ret))

            _output_tensor_buffer = clImgFloat;
        }

        for (auto &&out_tensor : output_buffers) {
            int argIdx = 0;
            unsigned reverse_chnl = reverse_channels ? 1 : 0;
            auto img_buffer = out_tensor;
            CHECK_CL_CALL_RET(clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), (void *)&(img_buffer)))
            CHECK_CL_CALL_RET(clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), (void *)&_output_tensor_buffer))
            CHECK_CL_CALL_RET(clSetKernelArg(kernel, argIdx++, sizeof(cl_uint), (void *)&dest_buf_offset))
            CHECK_CL_CALL_RET(clSetKernelArg(kernel, argIdx++, sizeof(cl_uint), (void *)&w))
            CHECK_CL_CALL_RET(clSetKernelArg(kernel, argIdx++, sizeof(cl_uint), (void *)&h))
            CHECK_CL_CALL_RET(clSetKernelArg(kernel, argIdx++, sizeof(cl_uint), (void *)&c))
            CHECK_CL_CALL_RET(clSetKernelArg(kernel, argIdx++, sizeof(cl_float), (void *)&multiplier0))
            CHECK_CL_CALL_RET(clSetKernelArg(kernel, argIdx++, sizeof(cl_float), (void *)&multiplier1))
            CHECK_CL_CALL_RET(clSetKernelArg(kernel, argIdx++, sizeof(cl_float), (void *)&multiplier2))
            CHECK_CL_CALL_RET(clSetKernelArg(kernel, argIdx++, sizeof(cl_float), (void *)&offset0))
            CHECK_CL_CALL_RET(clSetKernelArg(kernel, argIdx++, sizeof(cl_float), (void *)&offset1))
            CHECK_CL_CALL_RET(clSetKernelArg(kernel, argIdx++, sizeof(cl_float), (void *)&offset2))
            CHECK_CL_CALL_RET(clSetKernelArg(kernel, argIdx++, sizeof(cl_uint), (void *)&reverse_chnl))

            if ((status = clEnqueueNDRangeKernel(queue,
                                                 kernel,
                                                 1,
                                                 nullptr,
                                                 &global_work_size,
                                                 &local_work_size,
                                                 0, nullptr, nullptr)) != CL_SUCCESS)
                THROW("clEnqueueNDRangeKernel failed on kernel " + STR(kernel_name) + " error " + TOSTR(status))
            dest_buf_offset += single_output_tensor_size;
        }

        int read_size = single_output_tensor_size * _output_tensor_list.size() * sizeof(cl_float);
        if ((status = clEnqueueReadBuffer(queue,
                                          (cl_mem)_output_tensor_buffer,
                                          CL_TRUE,
                                          0,
                                          read_size,
                                          out_ptr,
                                          0, nullptr, nullptr)) != CL_SUCCESS)
            THROW("clEnqueueReadBuffer failed: " + TOSTR(status))
    }
#elif ENABLE_HIP
    if (output_tensor_info.mem_type() == RocalMemType::HIP) {
        unsigned int fp16 = (output_data_type == RocalTensorDataType::FP16);

        auto output_buffers = _ring_buffer.get_read_buffers().first;
        unsigned dest_buf_offset = 0;
        // copy hip buffer to out_ptr
        // todo:: add callback routing to exchange memory pointer to avoid extra copy
        for (auto &&out_tensor : output_buffers) {
            auto img_buffer = out_tensor;
            if (format == RocalTensorlayout::NHWC) {
                HipExecCopyInt8ToNHWC(_device.resources()->hip_stream, (const void *)img_buffer, out_ptr, dest_buf_offset, n, c, h, w,
                                      multiplier0, multiplier1, multiplier2, offset0, offset1, offset2, reverse_channels, fp16, max_roi_height, max_roi_width);

            } else {
                HipExecCopyInt8ToNCHW(_device.resources()->hip_stream, (const void *)img_buffer, out_ptr, dest_buf_offset, n, c, h, w,
                                      multiplier0, multiplier1, multiplier2, offset0, offset1, offset2, reverse_channels, fp16, max_roi_height, max_roi_width);
            }
            dest_buf_offset += single_output_tensor_size;
        }
    }
    if ((output_tensor_info.mem_type() == RocalMemType::HOST)) {
        if (output_mem_type == RocalOutputMemType::ROCAL_MEMCPY_GPU) {
            unsigned int fp16 = (output_data_type == RocalTensorDataType::FP16);

            auto output_buffers = _ring_buffer.get_read_buffers().first;
            unsigned dest_buf_offset = 0;

            if (_output_tensor_buffer == nullptr) {
                size_t size = output_tensor_info.data_size() * (output_data_type == RocalTensorDataType::FP32 ? sizeof(float) : sizeof(half));
                hipError_t status = hipMalloc(&_output_tensor_buffer, size);
                if ((status != hipSuccess) || !_output_tensor_buffer)
                    THROW("ROCAL::hipMalloc of size " + TOSTR(size) + " failed " + TOSTR(status))
            }

            // copy hip buffer to out_ptr
            // todo:: add callback routing to exchange memory pointer to avoid extra copy
            for (auto &&out_tensor : output_buffers) {
                auto img_buffer = out_tensor;
                auto return_status = hipMemcpyHtoDAsync(_output_tensor_buffer, (void *)img_buffer, sizeof(unsigned char) * n * c * h * w, _device.resources()->hip_stream);
                if (return_status != hipSuccess) {
                    THROW("hipMemcpy failed with status " + TOSTR(return_status))
                }
                // sync to finish copy
                if (hipStreamSynchronize(_device.resources()->hip_stream) != hipSuccess)
                    THROW("hipStreamSynchronize failed for hipMemcpy ")

                if (format == RocalTensorlayout::NHWC) {
                    HipExecCopyInt8ToNHWC(_device.resources()->hip_stream, (const void *)_output_tensor_buffer, out_ptr, dest_buf_offset, n, c, h, w,
                                          multiplier0, multiplier1, multiplier2, offset0, offset1, offset2, reverse_channels, fp16, max_roi_height, max_roi_width);

                } else {
                    HipExecCopyInt8ToNCHW(_device.resources()->hip_stream, (const void *)_output_tensor_buffer, out_ptr, dest_buf_offset, n, c, h, w,
                                          multiplier0, multiplier1, multiplier2, offset0, offset1, offset2, reverse_channels, fp16, max_roi_height, max_roi_width);
                }
                dest_buf_offset += single_output_tensor_size;
            }
        }
    }
#endif
    if ((output_tensor_info.mem_type() == RocalMemType::HOST)) {
        if (output_mem_type == RocalOutputMemType::ROCAL_MEMCPY_HOST) {
            float multiplier[3] = {multiplier0, multiplier1, multiplier2};
            float offset[3] = {offset0, offset1, offset2};
            size_t dest_buf_offset_start = 0;

            auto output_buffers = _ring_buffer.get_read_buffers().first;
            auto num_threads = _cpu_num_threads * 2;
            for (auto &&out_tensor : output_buffers) {
                unsigned int single_tensor_size = w * c * h;
                unsigned int channel_size = max_roi_width * max_roi_height;
                unsigned int output_single_tensor_size = max_roi_height * max_roi_width * c;
                unsigned int input_width_stride = w * c;
#pragma omp parallel for num_threads(num_threads)
                for (unsigned int batch_count = 0; batch_count < n; batch_count++) {
                    size_t dest_buf_offset = dest_buf_offset_start + output_single_tensor_size * batch_count;
                    auto in_buffer = (unsigned char *)out_tensor + single_tensor_size * batch_count;

                    if (format == RocalTensorlayout::NHWC) {
                        if (output_data_type == RocalTensorDataType::FP32) {
                            float *output_tensor_32 = static_cast<float *>(out_ptr);
                            for (unsigned channel_idx = 0; channel_idx < c; channel_idx++) {
                                for (unsigned i = 0; i < channel_size; i++)
                                    output_tensor_32[dest_buf_offset + channel_idx + i * c] =
                                        offset[channel_idx] + multiplier[channel_idx] *
                                                                  (reverse_channels ? static_cast<float>(in_buffer[i * c + c - channel_idx - 1])
                                                                                    : static_cast<float>(in_buffer[i * c + channel_idx]));
                            }
                        } else if (output_data_type == RocalTensorDataType::FP16) {
                            half *output_tensor_16 = static_cast<half *>(out_ptr);
                            for (unsigned channel_idx = 0; channel_idx < c; channel_idx++) {
                                for (unsigned i = 0; i < channel_size; i++)
                                    output_tensor_16[dest_buf_offset + channel_idx + i * c] =
                                        offset[channel_idx] + multiplier[channel_idx] *
                                                                  (reverse_channels ? (half)(in_buffer[i * c + c - channel_idx - 1])
                                                                                    : (half)(in_buffer[i * c + channel_idx]));
                            }
                        }
                    }
                    if (format == RocalTensorlayout::NCHW) {
                        if (output_data_type == RocalTensorDataType::FP32) {
                            float *output_tensor_32 = static_cast<float *>(out_ptr);
                            if (c != 3) {
                                for (unsigned i = 0; i < channel_size; i++)
                                    output_tensor_32[dest_buf_offset + i] = offset[0] + multiplier[0] * static_cast<float>(in_buffer[c * i]);
                            } else {
#if (ENABLE_SIMD && __AVX2__)
                                float *B_buf = output_tensor_32 + dest_buf_offset;
                                float *G_buf = B_buf + channel_size;
                                float *R_buf = G_buf + channel_size;

                                __m256i mask_B, mask_G, mask_R;
                                if (reverse_channels) {
                                    mask_B = avx_pkdMaskR;
                                    mask_G = avx_pkdMaskG;
                                    mask_R = avx_pkdMaskB;
                                } else {
                                    mask_R = avx_pkdMaskR;
                                    mask_G = avx_pkdMaskG;
                                    mask_B = avx_pkdMaskB;
                                }
                                __m256 pmul0 = _mm256_set1_ps(multiplier0);
                                __m256 pmul1 = _mm256_set1_ps(multiplier1);
                                __m256 pmul2 = _mm256_set1_ps(multiplier2);
                                __m256 padd0 = _mm256_set1_ps(offset0);
                                __m256 padd1 = _mm256_set1_ps(offset1);
                                __m256 padd2 = _mm256_set1_ps(offset2);
                                uint alignedLength = (max_roi_width & ~7);  // multiple of 8

                                __m256 fR, fG, fB;
                                for (uint row = 0; row < max_roi_height; row++) {
                                    unsigned char *in_buffer_row = reinterpret_cast<unsigned char *>(in_buffer) + (row * input_width_stride);
                                    uint col = 0;
                                    for (; col < alignedLength; col += 8) {
                                        __m256i pix0 = _mm256_loadu_si256((const __m256i *)in_buffer_row);
                                        pix0 = _mm256_permutevar8x32_epi32(pix0, _mm256_setr_epi32(0, 1, 2, 3, 3, 4, 5, 6));
                                        fB = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(pix0, mask_R));
                                        fG = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(pix0, mask_G));
                                        fR = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(pix0, mask_B));
                                        fB = _mm256_fmadd_ps(fB, pmul0, padd0);
                                        fG = _mm256_fmadd_ps(fG, pmul1, padd1);
                                        fR = _mm256_fmadd_ps(fR, pmul2, padd2);
                                        _mm256_storeu_ps(B_buf, fB);
                                        _mm256_storeu_ps(G_buf, fG);
                                        _mm256_storeu_ps(R_buf, fR);
                                        B_buf += 8;
                                        G_buf += 8;
                                        R_buf += 8;
                                        in_buffer_row += 24;
                                    }
                                    for (; col < max_roi_width; col++, in_buffer_row += 3) {
                                        *B_buf++ = (in_buffer_row[0] * multiplier0) + offset0;
                                        *G_buf++ = (in_buffer_row[1] * multiplier1) + offset1;
                                        *R_buf++ = (in_buffer_row[2] * multiplier2) + offset1;
                                    }
                                }
#else
                                for (unsigned channel_idx = 0; channel_idx < c; channel_idx++) {
                                    for (unsigned i = 0; i < channel_size; i++)
                                        output_tensor_32[dest_buf_offset + channel_idx * channel_size + i] =
                                            offset[channel_idx] + multiplier[channel_idx] * (reverse_channels ? static_cast<float>(in_buffer[(c * i + c - channel_idx - 1)]) : static_cast<float>(in_buffer[(c * i + channel_idx)]));
                                }
#endif
                            }
                        } else if (output_data_type == RocalTensorDataType::FP16) {
                            half *output_tensor_16 = static_cast<half *>(out_ptr);
                            if (c != 3) {
                                for (unsigned i = 0; i < channel_size; i++)
                                    output_tensor_16[dest_buf_offset + i] = offset[0] + multiplier[0] * (half)in_buffer[c * i];
                            } else {
#if (ENABLE_SIMD && __AVX2__)
                                half *B_buf_16 = output_tensor_16 + dest_buf_offset;
                                half *G_buf_16 = B_buf_16 + channel_size;
                                half *R_buf_16 = G_buf_16 + channel_size;

                                __m256i mask_B, mask_G, mask_R;
                                if (reverse_channels) {
                                    mask_B = avx_pkdMaskR;
                                    mask_G = avx_pkdMaskG;
                                    mask_R = avx_pkdMaskB;
                                } else {
                                    mask_R = avx_pkdMaskR;
                                    mask_G = avx_pkdMaskG;
                                    mask_B = avx_pkdMaskB;
                                }
                                __m256 pmul0 = _mm256_set1_ps(multiplier0);
                                __m256 pmul1 = _mm256_set1_ps(multiplier1);
                                __m256 pmul2 = _mm256_set1_ps(multiplier2);
                                __m256 padd0 = _mm256_set1_ps(offset0);
                                __m256 padd1 = _mm256_set1_ps(offset1);
                                __m256 padd2 = _mm256_set1_ps(offset2);
                                uint alignedLength = (max_roi_width & ~7);  // multiple of 8

                                __m256 fR, fG, fB;
                                __m128i tempR, tempG, tempB;
                                for (uint row = 0; row < max_roi_height; row++) {
                                    unsigned char *in_buffer_row = reinterpret_cast<unsigned char *>(in_buffer) + (row * input_width_stride);
                                    uint col = 0;
                                    for (; col < alignedLength; col += 8) {
                                        __m256i pix0 = _mm256_loadu_si256((const __m256i *)in_buffer_row);
                                        pix0 = _mm256_permutevar8x32_epi32(pix0, _mm256_setr_epi32(0, 1, 2, 3, 3, 4, 5, 6));
                                        fB = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(pix0, mask_R));
                                        fG = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(pix0, mask_G));
                                        fR = _mm256_cvtepi32_ps(_mm256_shuffle_epi8(pix0, mask_B));
                                        fB = _mm256_fmadd_ps(fB, pmul0, padd0);
                                        fG = _mm256_fmadd_ps(fG, pmul1, padd1);
                                        fR = _mm256_fmadd_ps(fR, pmul2, padd2);
                                        tempB = _mm256_cvtps_ph(fB, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
                                        tempG = _mm256_cvtps_ph(fG, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
                                        tempR = _mm256_cvtps_ph(fR, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
                                        _mm_storeu_si128((__m128i *)B_buf_16, tempB);
                                        _mm_storeu_si128((__m128i *)G_buf_16, tempG);
                                        _mm_storeu_si128((__m128i *)R_buf_16, tempR);
                                        B_buf_16 += 8;
                                        G_buf_16 += 8;
                                        R_buf_16 += 8;
                                        in_buffer_row += 24;
                                    }
                                    for (; col < max_roi_width; col++, in_buffer_row += 3) {
                                        *B_buf_16++ = (half)(in_buffer_row[0] * multiplier0) + offset0;
                                        *G_buf_16++ = (half)(in_buffer_row[1] * multiplier1) + offset1;
                                        *R_buf_16++ = (half)(in_buffer_row[2] * multiplier2) + offset2;
                                    }
                                }
#else
                                for (unsigned channel_idx = 0; channel_idx < c; channel_idx++) {
                                    for (unsigned i = 0; i < channel_size; i++)
                                        output_tensor_16[dest_buf_offset + channel_idx * channel_size + i] =
                                            offset[channel_idx] + multiplier[channel_idx] *
                                                                      (reverse_channels ? (half)(in_buffer[(c * i + c - channel_idx - 1)])
                                                                                        : (half)(in_buffer[(c * i + channel_idx)]));
                                }
#endif
                            }
                        }
                    }  // NCHW or NHWC
                }      // for loop batch

                dest_buf_offset_start += single_output_tensor_size;
            }
        }
    }
    _convert_time.end();
    return Status::OK;
}

MasterGraph::Status
MasterGraph::copy_output(unsigned char *out_ptr, size_t out_size_in_bytes) {
    if (no_more_processed_data())
        return MasterGraph::Status::NO_MORE_DATA;

    if (_output_tensor_list.size() != 1)
        THROW("Cannot copy, Multiple output tensors present in the list")

    auto output_tensor_info = _output_tensor_list[0]->info();
    // Copies to the output context given by the user
    size_t size = output_tensor_info.data_size();
    if (out_size_in_bytes != (size * _output_tensor_list.size()))
        return MasterGraph::Status::INVALID_ARGUMENTS;

    _convert_time.start();

#if ENABLE_OPENCL
    if (output_tensor_info.mem_type() == RocalMemType::OCL) {
        size_t dest_buf_offset = 0;
        // NOTE: the CL_TRUE flag is only used on the last buffer read
        //  to avoid unnecessary sequence of synchronizations

        // get_read_buffers() calls block_if_empty() internally and blocks if buffers are empty until a new batch is processed
        auto output_buffers = _ring_buffer.get_read_buffers().first;
        auto out_image_idx = output_buffers.size();
        for (auto &&output_handle : output_buffers) {
            bool sync_flag = (--out_image_idx == 0) ? CL_TRUE : CL_FALSE;
            cl_int status;
            if ((status = clEnqueueReadBuffer(_device.resources()->cmd_queue,
                                              (cl_mem)output_handle,
                                              sync_flag ? (CL_TRUE) : CL_FALSE,
                                              0,
                                              size,
                                              out_ptr + dest_buf_offset,
                                              0, nullptr, nullptr)) != CL_SUCCESS)
                THROW("clEnqueueReadBuffer failed: " + TOSTR(status))
            dest_buf_offset += size;
        }
    } else {
#elif ENABLE_HIP
    if (output_tensor_info.mem_type() == RocalMemType::HIP) {
        // NOTE: the CL_TRUE flag is only used on the last buffer read call,
        //  to avoid unnecessary sequence of synchronizations

        // get_read_buffers() calls block_if_empty() internally and blocks if buffers are empty until a new batch is processed
        size_t dest_buf_offset = 0;
        auto output_buffers = _ring_buffer.get_read_buffers().first;
        for (auto &&output_handle : output_buffers) {
            hipError_t err = hipMemcpyDtoHAsync((void *)(out_ptr + dest_buf_offset), output_handle, size, _device.resources()->hip_stream);
            if (err) {
                THROW("hipMemcpyDtoHAsync failed: " + TOSTR(err))
            }
            dest_buf_offset += size;
        }
        // sync to finish copy
        if (hipStreamSynchronize(_device.resources()->hip_stream) != hipSuccess)
            THROW("hipStreamSynchronize failed for hipMemcpy ")

    } else {
#endif
        // get_read_buffer is blocking if _ring_buffer is empty, and blocks this thread till internal processing thread process a new batch and store in the _ring_buffer
        auto output_buffer = _ring_buffer.get_read_buffers().first[0];
        memcpy(out_ptr, output_buffer, size);
#if ENABLE_OPENCL || ENABLE_HIP
    }
#endif
    _convert_time.end();
    return Status::OK;
}

TensorList *
MasterGraph::get_output_tensors() {
    auto read_buffers = _ring_buffer.get_read_buffers();
    auto output_ptr = read_buffers.first;
    auto roi_ptr = read_buffers.second;
    for (unsigned i = 0; i < _internal_tensor_list.size(); i++) {
        _output_tensor_list[i]->set_mem_handle(output_ptr[i]);
        _output_tensor_list[i]->set_roi(roi_ptr[i]);
    }
    return &_output_tensor_list;
}

void MasterGraph::output_routine() {
    INFO("Output routine started with " + TOSTR(_remaining_count) + " to load");
    try {
        while (_processing) {
            if (_loader_module->remaining_count() < (_is_sequence_reader_output ? _sequence_batch_size : _user_batch_size)) {
                // If the internal process routine ,output_routine(), has finished processing all the images, and last
                // processed images stored in the _ring_buffer will be consumed by the user when it calls the run() func
                notify_user_thread();
                // the following call is required in case the ring buffer is waiting for more data to be loaded and there is no more data to process.
                _ring_buffer.release_if_empty();
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            _rb_block_if_full_time.start();
            // _ring_buffer.get_write_buffers() is blocking and blocks here until user uses processed image by calling run() and frees space in the ring_buffer
            auto write_buffers = _ring_buffer.get_write_buffers();
            auto write_output_buffers = write_buffers.first;
            _rb_block_if_full_time.end();

            // Swap handles on the input tensor, so that new tensor is loaded to be processed
            auto load_ret = _loader_module->load_next();
            if (load_ret != LoaderModuleStatus::OK)
                THROW("Loader module failed to load next batch of images, status " + TOSTR(load_ret))
            if (!_processing)
                break;
            auto full_batch_data_names = _loader_module->get_id();
            auto decode_data_info = _loader_module->get_decode_data_info();
            auto crop_image_info = _loader_module->get_crop_image_info();

            if (full_batch_data_names.size() != _user_batch_size)
                WRN("Master Graph: Names count does not equal batch_size" + TOSTR(full_batch_data_names.size()))

            // meta_data lookup is done before _meta_data_graph->process() is called to have the new meta_data ready for processing
            if (_meta_data_reader)
                _meta_data_reader->lookup(full_batch_data_names);

            if (!_processing)
                break;

            // Swap handles on the output tensor, so that new processed tensor will be written to the a new buffer
            for (size_t idx = 0; idx < _internal_tensor_list.size(); idx++)
                _internal_tensor_list[idx]->swap_handle(write_output_buffers[idx]);

            if (!_processing)
                break;

            for (auto node : _nodes) {
                if (node->_is_ssd) {
                    node->set_meta_data(_augmented_meta_data);
                }
            }

            update_node_parameters();
            pMetaDataBatch output_meta_data = nullptr;
            if (_augmented_meta_data) {
                output_meta_data = _augmented_meta_data->clone(!_augmentation_metanode);  // copy the data if metadata is not processed by the nodes, else create an empty instance
                if (_meta_data_graph) {
                    if (_is_random_bbox_crop) {
                        _meta_data_graph->update_random_bbox_meta_data(_augmented_meta_data, output_meta_data, decode_data_info, crop_image_info);
                    } else {
                        _meta_data_graph->update_meta_data(_augmented_meta_data, decode_data_info);
                    }
                    _meta_data_graph->process(_augmented_meta_data, output_meta_data);
                }
            }
            _process_time.start();
            _graph->process();
            _process_time.end();

            auto write_roi_buffers = write_buffers.second;   // Obtain ROI buffers from ring buffer
            for (size_t idx = 0; idx < _internal_tensor_list.size(); idx++)
                _internal_tensor_list[idx]->copy_roi(write_roi_buffers[idx]);   // Copy ROI from internal tensor's buffer to ring buffer
            _bencode_time.start();
            if (_is_box_encoder) {
                auto bbox_encode_write_buffers = _ring_buffer.get_box_encode_write_buffers();
#if ENABLE_HIP
                if (_mem_type == RocalMemType::HIP) {
                    // get bbox encoder read buffers
                    if (_box_encoder_gpu) _box_encoder_gpu->Run(output_meta_data, (float *)bbox_encode_write_buffers.first, (int *)bbox_encode_write_buffers.second);
                } else
#endif
                    _meta_data_graph->update_box_encoder_meta_data(&_anchors, output_meta_data, _criteria, _offset, _scale, _means, _stds, (float *)bbox_encode_write_buffers.first, (int *)bbox_encode_write_buffers.second);
            }
            if (_is_box_iou_matcher) {
                int *matches_write_buffer = reinterpret_cast<int *>(_ring_buffer.get_meta_write_buffers()[2]);
                _meta_data_graph->update_box_iou_matcher(_iou_matcher_info, matches_write_buffer, output_meta_data);
            }
            _bencode_time.end();
#ifdef ROCAL_VIDEO
            _sequence_start_framenum_vec.insert(_sequence_start_framenum_vec.begin(), _loader_module->get_sequence_start_frame_number());
            _sequence_frame_timestamps_vec.insert(_sequence_frame_timestamps_vec.begin(), _loader_module->get_sequence_frame_timestamps());
#endif
            _ring_buffer.set_meta_data(full_batch_data_names, output_meta_data);
            _ring_buffer.push();  // The data and metadata is now stored in output the ring_buffer, increases it's level by 1
        }
    } catch (const std::exception &e) {
        ERR("Exception thrown in the process routine: " + STR(e.what()) + STR("\n"));
        _processing = false;
        _ring_buffer.release_all_blocked_calls();
    }
}

void MasterGraph::start_processing() {
    _processing = true;
    _remaining_count = _loader_module->remaining_count();
    _output_thread = std::thread(&MasterGraph::output_routine, this);
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
#else
//  Changing thread scheduling policy and it's priority does not help on latest Ubuntu builds
//  and needs tweaking the Linux security settings , can be turned on for experimentation
#if 0
    struct sched_param params;
    params.sched_priority = sched_get_priority_max(SCHED_FIFO);
    auto thread = _output_thread.native_handle();
    auto ret = pthread_setschedparam(thread, SCHED_FIFO, &params);
    if (ret != 0)
        WRN("Unsuccessful in setting thread realtime priority for process thread err = "+STR(std::strerror(ret)))
#endif
#endif
}

void MasterGraph::stop_processing() {
    _processing = false;
    _ring_buffer.unblock_reader();
    _ring_buffer.unblock_writer();
    if (_output_thread.joinable())
        _output_thread.join();
}

TensorListVector* MasterGraph::create_coco_meta_data_reader(const char *source_path, bool is_output, MetaDataReaderType reader_type, MetaDataType metadata_type, bool ltrb_bbox, bool is_box_encoder, bool avoid_class_remapping, bool aspect_ratio_grouping, bool is_box_iou_matcher, float sigma, unsigned pose_output_width, unsigned pose_output_height) {
    if (_meta_data_reader)
        THROW("A metadata reader has already been created")
    if (_augmented_meta_data)
        THROW("Metadata output already defined, there can only be a single output for metadata augmentation");

    MetaDataConfig config(metadata_type, reader_type, source_path, std::map<std::string, std::string>(), std::string());
    config.set_avoid_class_remapping(avoid_class_remapping);
    config.set_aspect_ratio_grouping(aspect_ratio_grouping);
    config.set_out_img_width(pose_output_width);
    config.set_out_img_height(pose_output_height);
    _meta_data_graph = create_meta_data_graph(config);
    _meta_data_reader = create_meta_data_reader(config, _augmented_meta_data);
    _meta_data_reader->read_all(source_path);
    if (!ltrb_bbox) _augmented_meta_data->set_xywh_bbox();
    std::vector<size_t> dims;
    size_t max_objects = static_cast<size_t>(is_box_encoder ? MAX_SSD_ANCHORS : MAX_OBJECTS);
    dims = {max_objects};
    auto default_labels_info = TensorInfo(std::move(dims), _mem_type, RocalTensorDataType::INT32);  // Create default labels Info
    default_labels_info.set_metadata();
    _meta_data_buffer_size.emplace_back(_user_batch_size * default_labels_info.data_size());

    dims = {max_objects, BBOX_COUNT};
    auto default_bbox_info = TensorInfo(std::move(dims), _mem_type, RocalTensorDataType::FP32);  // Create default Bbox Info
    default_bbox_info.set_metadata();
    _meta_data_buffer_size.emplace_back(_user_batch_size * default_bbox_info.data_size());

    TensorInfo default_matches_info;
    TensorInfo default_mask_info;
    if (metadata_type == MetaDataType::PolygonMask) {
        dims = {MAX_MASK_BUFFER, 1};
        default_mask_info = TensorInfo(std::move(dims), _mem_type, RocalTensorDataType::FP32);  // Create default mask Info
        default_mask_info.set_metadata();
        _meta_data_buffer_size.emplace_back(_user_batch_size * default_mask_info.data_size());
    }
    if (is_box_iou_matcher) {
        _is_box_iou_matcher = true;
        dims = {MAX_RETINANET_ANCHORS};
        default_matches_info = TensorInfo(std::move(dims), _mem_type, RocalTensorDataType::INT32);  // Create default matches info
        default_matches_info.set_metadata();
        _meta_data_buffer_size.emplace_back(_user_batch_size * default_matches_info.data_size());
    }

    for (unsigned i = 0; i < _user_batch_size; i++)  // Create rocALTensorList for each metadata
    {
        auto labels_info = default_labels_info;
        auto bbox_info = default_bbox_info;
        _labels_tensor_list.push_back(new Tensor(labels_info));
        _bbox_tensor_list.push_back(new Tensor(bbox_info));
        if (metadata_type == MetaDataType::PolygonMask) {
            auto mask_info = default_mask_info;
            _mask_tensor_list.push_back(new Tensor(mask_info));
        }
        if(is_box_iou_matcher) {
            auto matches_info = default_matches_info;
            _matches_tensor_list.push_back(new Tensor(matches_info));
        }
    }
    _ring_buffer.init_metadata(RocalMemType::HOST, _meta_data_buffer_size);
    _metadata_output_tensor_list.emplace_back(&_labels_tensor_list);
    _metadata_output_tensor_list.emplace_back(&_bbox_tensor_list);
    if (metadata_type == MetaDataType::PolygonMask)
        _metadata_output_tensor_list.emplace_back(&_mask_tensor_list);
    if(is_box_iou_matcher)
        _metadata_output_tensor_list.emplace_back(&_matches_tensor_list);

    return &_metadata_output_tensor_list;
}

TensorListVector* MasterGraph::create_tf_record_meta_data_reader(const char *source_path, MetaDataReaderType reader_type, MetaDataType label_type, std::map<std::string, std::string> feature_key_map) {
    if (_meta_data_reader)
        THROW("A metadata reader has already been created")
    if (_augmented_meta_data)
        THROW("Metadata can only have a single output")

    MetaDataConfig config(label_type, reader_type, source_path, feature_key_map);
    _meta_data_graph = create_meta_data_graph(config);
    _meta_data_reader = create_meta_data_reader(config, _augmented_meta_data);
    _meta_data_reader->read_all(source_path);

    if (reader_type == MetaDataReaderType::TF_META_DATA_READER) {
        std::vector<size_t> dims = {1};
        auto default_labels_info = TensorInfo(std::move(dims), _mem_type, RocalTensorDataType::INT32);  // Create default labels Info
        default_labels_info.set_metadata();
        _meta_data_buffer_size.emplace_back(_user_batch_size * sizeof(vx_int32));

        for (unsigned i = 0; i < _user_batch_size; i++) {
            auto info = default_labels_info;
            auto tensor = new Tensor(info);
            _labels_tensor_list.push_back(tensor);
        }
        _metadata_output_tensor_list.emplace_back(&_labels_tensor_list);
    } else if (reader_type == MetaDataReaderType::TF_DETECTION_META_DATA_READER) {
        std::vector<size_t> dims = {MAX_OBJECTS};
        auto default_labels_info = TensorInfo(std::move(dims), _mem_type, RocalTensorDataType::INT32);  // Create default labels Info
        default_labels_info.set_metadata();
        _meta_data_buffer_size.emplace_back(_user_batch_size * default_labels_info.data_size());

        dims = {MAX_OBJECTS, BBOX_COUNT};
        auto default_bbox_info = TensorInfo(std::move(dims), _mem_type, RocalTensorDataType::FP32);  // Create default Bbox Info
        default_bbox_info.set_metadata();
        _meta_data_buffer_size.emplace_back(_user_batch_size * default_bbox_info.data_size());

        for (unsigned i = 0; i < _user_batch_size; i++) {
            auto labels_info = default_labels_info;
            auto bbox_info = default_bbox_info;
            _labels_tensor_list.push_back(new Tensor(labels_info));
            _bbox_tensor_list.push_back(new Tensor(bbox_info));
        }
        _metadata_output_tensor_list.emplace_back(&_labels_tensor_list);
        _metadata_output_tensor_list.emplace_back(&_bbox_tensor_list);
    }

    _ring_buffer.init_metadata(RocalMemType::HOST, _meta_data_buffer_size);

    return &_metadata_output_tensor_list;
}

TensorListVector* MasterGraph::create_label_reader(const char *source_path, MetaDataReaderType reader_type) {
    if (_meta_data_reader)
        THROW("A metadata reader has already been created")
    if (_augmented_meta_data)
        THROW("Metadata can only have a single output")

    MetaDataConfig config(MetaDataType::Label, reader_type, source_path);
    _meta_data_reader = create_meta_data_reader(config, _augmented_meta_data);
    _meta_data_reader->read_all(source_path);

    std::vector<size_t> dims = {1};
    auto default_labels_info = TensorInfo(std::move(dims), _mem_type, RocalTensorDataType::INT32);  // Create default labels Info
    default_labels_info.set_metadata();
    _meta_data_buffer_size.emplace_back(_user_batch_size * sizeof(vx_int32));

    for (unsigned i = 0; i < _user_batch_size; i++) {
        auto info = default_labels_info;
        _labels_tensor_list.push_back(new Tensor(info));
    }
    _ring_buffer.init_metadata(RocalMemType::HOST, _meta_data_buffer_size);
    _metadata_output_tensor_list.emplace_back(&_labels_tensor_list);

    return &_metadata_output_tensor_list;
}

TensorListVector* MasterGraph::create_webdataset_reader(
    const char *source_path, const char *index_path,
    std::vector<std::set<std::string>> extensions,
    MetaDataReaderType reader_type, MissingComponentsBehaviour missing_component_behaviour) {
    if (_meta_data_reader)
        THROW("A metadata reader has already been created")
    if (_augmented_meta_data)
        THROW("Metadata can only have a single output")

    bool generate_index = (index_path[0] == '\0') ? true : false;
    if (generate_index)
        std::cerr << "Index file is not provided, it may take some time to infer it from the tar file";

    _ascii_tensor_list.resize(extensions[0].size() - 1);
    MetaDataConfig config(MetaDataType::AsciiValue, reader_type, source_path,
                          std::map<std::string, std::string>(), std::string(),
                          0, 0, 0, index_path, missing_component_behaviour,
                          extensions);
    _meta_data_reader = create_meta_data_reader(config, _augmented_meta_data);
    _meta_data_reader->read_all(source_path);
    std::vector<size_t> dims;
    dims = {MAX_ASCII_BUFFER};
    auto default_ascii_values_info = TensorInfo(std::move(dims), _mem_type,RocalTensorDataType::UINT8); // Create default ascii values Info
    default_ascii_values_info.set_metadata();
    for (uint ext_count = 0; ext_count < _ascii_tensor_list.size(); ext_count++) {
        for (unsigned i = 0; i < _user_batch_size; i++) {
            _meta_data_buffer_size.emplace_back(_user_batch_size * default_ascii_values_info.data_size());
            auto info = default_ascii_values_info;
            auto tensor = new Tensor(info);
            _ascii_tensor_list[ext_count].push_back(tensor);
        }
        _metadata_output_tensor_list.emplace_back(&_ascii_tensor_list[ext_count]);
        _webdataset_output_tensor_list.emplace_back(&_ascii_tensor_list[ext_count]);
    }

    _ring_buffer.init_metadata(RocalMemType::HOST, _meta_data_buffer_size);

    return &_metadata_output_tensor_list;
}

TensorListVector* MasterGraph::create_video_label_reader(const char *source_path, MetaDataReaderType reader_type, unsigned sequence_length, unsigned frame_step, unsigned frame_stride, bool file_list_frame_num) {
    if (_meta_data_reader)
        THROW("A metadata reader has already been created")
    if (_augmented_meta_data)
        THROW("Metadata can only have a single output")

    MetaDataConfig config(MetaDataType::Label, reader_type, source_path, std::map<std::string, std::string>(), std::string(), sequence_length, frame_step, frame_stride);
    _meta_data_reader = create_meta_data_reader(config, _augmented_meta_data);

    if (!file_list_frame_num) {
        _meta_data_reader->set_timestamp_mode();
    }

    std::vector<size_t> dims = {1};
    auto default_labels_info = TensorInfo(std::move(dims), _mem_type, RocalTensorDataType::INT32);  // Create default labels Info
    default_labels_info.set_metadata();
    _meta_data_buffer_size.emplace_back(_user_batch_size * sizeof(vx_int32));

    for (unsigned i = 0; i < _user_batch_size; i++) {
        auto info = default_labels_info;
        auto tensor = new Tensor(info);
        _labels_tensor_list.push_back(tensor);
    }
    _ring_buffer.init_metadata(RocalMemType::HOST, _meta_data_buffer_size);
    _meta_data_reader->read_all(source_path);
    _metadata_output_tensor_list.emplace_back(&_labels_tensor_list);

    return &_metadata_output_tensor_list;
}

TensorListVector* MasterGraph::create_mxnet_label_reader(const char *source_path, bool is_output) {
    if (_meta_data_reader)
        THROW("A metadata reader has already been created")
    if (_augmented_meta_data)
        THROW("Metadata output already defined, there can only be a single output for metadata augmentation")

    MetaDataConfig config(MetaDataType::Label, MetaDataReaderType::MXNET_META_DATA_READER, source_path);
    _meta_data_graph = create_meta_data_graph(config);
    _meta_data_reader = create_meta_data_reader(config, _augmented_meta_data);
    _meta_data_reader->read_all(source_path);
    std::vector<size_t> dims = {1};
    auto default_labels_info = TensorInfo(std::move(dims), _mem_type, RocalTensorDataType::INT32);  // Create default labels Info
    default_labels_info.set_metadata();
    _meta_data_buffer_size.emplace_back(_user_batch_size * sizeof(vx_int32));

    for (unsigned i = 0; i < _user_batch_size; i++) {
        auto info = default_labels_info;
        auto tensor = new Tensor(info);
        _labels_tensor_list.push_back(tensor);
    }
    _metadata_output_tensor_list.emplace_back(&_labels_tensor_list);
    _ring_buffer.init_metadata(RocalMemType::HOST, _meta_data_buffer_size);

    return &_metadata_output_tensor_list;
}

void MasterGraph::create_randombboxcrop_reader(RandomBBoxCrop_MetaDataReaderType reader_type, RandomBBoxCrop_MetaDataType label_type, bool all_boxes_overlap, bool no_crop, FloatParam *aspect_ratio, bool has_shape, int crop_width, int crop_height, int num_attempts, FloatParam *scaling, int total_num_attempts, int64_t seed) {
    if (_randombboxcrop_meta_data_reader)
        THROW("A metadata reader has already been created")
    if (_random_bbox_crop_cords_data)
        THROW("Metadata can only have a single output")
    _is_random_bbox_crop = true;
    RandomBBoxCrop_MetaDataConfig config(label_type, reader_type, all_boxes_overlap, no_crop, aspect_ratio, has_shape, crop_width, crop_height, num_attempts, scaling, total_num_attempts, seed);
    _randombboxcrop_meta_data_reader = create_meta_data_reader(config, _random_bbox_crop_cords_data);
    _randombboxcrop_meta_data_reader->set_meta_data(_meta_data_reader);
}

void MasterGraph::box_encoder(std::vector<float> &anchors, float criteria, const std::vector<float> &means, const std::vector<float> &stds, bool offset, float scale) {
    _is_box_encoder = true;
    _num_anchors = anchors.size() / 4;
    std::vector<float> inv_stds = {(float)(1. / stds[0]), (float)(1. / stds[1]), (float)(1. / stds[2]), (float)(1. / stds[3])};

#if ENABLE_HIP
    // Intialize gpu box encoder if _mem_type is HIP
    if (_mem_type == RocalMemType::HIP) {
        _box_encoder_gpu = new BoxEncoderGpu(_user_batch_size, anchors, criteria, means, inv_stds, offset, scale, _device.resources()->hip_stream, _device.resources()->dev_prop.canMapHostMemory);
        return;
    }
#endif
    _offset = offset;
    _anchors = anchors;
    _scale = scale;
    _means = means;
    _stds = stds;
}

TensorListVector* MasterGraph::create_caffe2_lmdb_record_meta_data_reader(const char *source_path, MetaDataReaderType reader_type, MetaDataType label_type) {
    if (_meta_data_reader)
        THROW("A metadata reader has already been created")
    if (_augmented_meta_data)
        THROW("Metadata output already defined, there can only be a single output for metadata augmentation")

    MetaDataConfig config(label_type, reader_type, source_path);
    _meta_data_graph = create_meta_data_graph(config);
    _meta_data_reader = create_meta_data_reader(config, _augmented_meta_data);
    _meta_data_reader->read_all(source_path);
    if (reader_type == MetaDataReaderType::CAFFE2_META_DATA_READER) {
        std::vector<size_t> dims = {1};
        auto default_labels_info = TensorInfo(std::move(dims), _mem_type, RocalTensorDataType::INT32);  // Create default labels Info
        default_labels_info.set_metadata();
        _meta_data_buffer_size.emplace_back(_user_batch_size * sizeof(vx_int32));

        for (unsigned i = 0; i < _user_batch_size; i++) {
            auto info = default_labels_info;
            auto tensor = new Tensor(info);
            _labels_tensor_list.push_back(tensor);
        }
        _metadata_output_tensor_list.emplace_back(&_labels_tensor_list);
    } else if (reader_type == MetaDataReaderType::CAFFE2_DETECTION_META_DATA_READER) {
        std::vector<size_t> dims = {MAX_OBJECTS};
        auto default_labels_info = TensorInfo(std::move(dims), _mem_type, RocalTensorDataType::INT32);  // Create default labels Info
        default_labels_info.set_metadata();
        _meta_data_buffer_size.emplace_back(_user_batch_size * default_labels_info.data_size());

        dims = {MAX_OBJECTS, BBOX_COUNT};
        auto default_bbox_info = TensorInfo(std::move(dims), _mem_type, RocalTensorDataType::FP32);  // Create default Bbox Info
        default_bbox_info.set_metadata();
        _meta_data_buffer_size.emplace_back(_user_batch_size * default_bbox_info.data_size());

        for (unsigned i = 0; i < _user_batch_size; i++) {
            auto labels_info = default_labels_info;
            auto bbox_info = default_bbox_info;
            _labels_tensor_list.push_back(new Tensor(labels_info));
            _bbox_tensor_list.push_back(new Tensor(bbox_info));
        }
        _metadata_output_tensor_list.emplace_back(&_labels_tensor_list);
        _metadata_output_tensor_list.emplace_back(&_bbox_tensor_list);
    }

    _ring_buffer.init_metadata(RocalMemType::HOST, _meta_data_buffer_size);

    return &_metadata_output_tensor_list;
}

TensorListVector* MasterGraph::create_caffe_lmdb_record_meta_data_reader(const char *source_path, MetaDataReaderType reader_type, MetaDataType label_type) {
    if (_meta_data_reader)
        THROW("A metadata reader has already been created")
    if (_augmented_meta_data)
        THROW("Metadata output already defined, there can only be a single output for metadata augmentation")

    MetaDataConfig config(label_type, reader_type, source_path);
    _meta_data_graph = create_meta_data_graph(config);
    _meta_data_reader = create_meta_data_reader(config, _augmented_meta_data);
    _meta_data_reader->read_all(source_path);
    if (reader_type == MetaDataReaderType::CAFFE_META_DATA_READER) {
        std::vector<size_t> dims = {1};
        auto default_labels_info = TensorInfo(std::move(dims), _mem_type, RocalTensorDataType::INT32);  // Create default labels Info
        default_labels_info.set_metadata();
        _meta_data_buffer_size.emplace_back(_user_batch_size * sizeof(vx_int32));

        for (unsigned i = 0; i < _user_batch_size; i++) {
            auto info = default_labels_info;
            auto tensor = new Tensor(info);
            _labels_tensor_list.push_back(tensor);
        }
        _metadata_output_tensor_list.emplace_back(&_labels_tensor_list);
    } else if (reader_type == MetaDataReaderType::CAFFE_DETECTION_META_DATA_READER) {
        std::vector<size_t> dims = {MAX_OBJECTS};
        auto default_labels_info = TensorInfo(std::move(dims), _mem_type, RocalTensorDataType::INT32);  // Create default labels Info
        default_labels_info.set_metadata();
        _meta_data_buffer_size.emplace_back(_user_batch_size * default_labels_info.data_size());

        dims = {MAX_OBJECTS, BBOX_COUNT};
        auto default_bbox_info = TensorInfo(std::move(dims), _mem_type, RocalTensorDataType::FP32);  // Create default Bbox Info
        default_bbox_info.set_metadata();
        _meta_data_buffer_size.emplace_back(_user_batch_size * default_bbox_info.data_size());

        for (unsigned i = 0; i < _user_batch_size; i++) {
            auto labels_info = default_labels_info;
            auto bbox_info = default_bbox_info;
            _labels_tensor_list.push_back(new Tensor(labels_info));
            _bbox_tensor_list.push_back(new Tensor(bbox_info));
        }
        _metadata_output_tensor_list.emplace_back(&_labels_tensor_list);
        _metadata_output_tensor_list.emplace_back(&_bbox_tensor_list);
    }

    _ring_buffer.init_metadata(RocalMemType::HOST, _meta_data_buffer_size);

    return &_metadata_output_tensor_list;
}

TensorListVector* MasterGraph::create_cifar10_label_reader(const char *source_path, const char *file_prefix) {
    if (_meta_data_reader)
        THROW("A metadata reader has already been created")
    if (_augmented_meta_data)
        THROW("Metadata can only have a single output")

    MetaDataConfig config(MetaDataType::Label, MetaDataReaderType::CIFAR10_META_DATA_READER, source_path, std::map<std::string, std::string>(), file_prefix);
    _meta_data_reader = create_meta_data_reader(config, _augmented_meta_data);
    _meta_data_reader->read_all(source_path);
    std::vector<size_t> dims = {1};
    auto default_labels_info = TensorInfo(std::move(dims), _mem_type, RocalTensorDataType::INT32);  // Create default labels Info
    default_labels_info.set_metadata();
    _meta_data_buffer_size.emplace_back(_user_batch_size * sizeof(vx_int32));

    for (unsigned i = 0; i < _user_batch_size; i++) {
        auto info = default_labels_info;
        auto tensor = new Tensor(info);
        _labels_tensor_list.push_back(tensor);
    }
    _metadata_output_tensor_list.emplace_back(&_labels_tensor_list);
    _ring_buffer.init_metadata(RocalMemType::HOST, _meta_data_buffer_size);

    return &_metadata_output_tensor_list;
}

const std::pair<ImageNameBatch, pMetaDataBatch> &MasterGraph::meta_data() {
    if (_ring_buffer.level() == 0)
        THROW("No meta data has been loaded")
    return _ring_buffer.get_meta_data();
}

void MasterGraph::box_iou_matcher(std::vector<float> &anchors,
                                  float high_threshold, float low_threshold,
                                  bool allow_low_quality_matches) {
    if (!_is_box_iou_matcher)
        THROW("Box IOU matcher variable not set cannot return matched idx")
    _anchors = anchors;                     // Uses existing _anchors variable used for box encoder
    _iou_matcher_info.anchors = &_anchors;
    _iou_matcher_info.high_threshold = high_threshold;
    _iou_matcher_info.low_threshold = low_threshold;
    _iou_matcher_info.allow_low_quality_matches = allow_low_quality_matches;
}

size_t MasterGraph::bounding_box_batch_count(pMetaDataBatch meta_data_batch) {
    size_t size = 0;
    for (unsigned i = 0; i < _user_batch_size; i++)
        size += _is_box_encoder ? _num_anchors : meta_data_batch->get_labels_batch()[i].size();

    return size;
}

TensorList *MasterGraph::labels_meta_data() {
    if (_external_source_reader)
        return &_labels_tensor_list;
    if (_ring_buffer.level() == 0)
        THROW("No meta data has been loaded")
    auto meta_data_buffers = (unsigned char *)_ring_buffer.get_meta_read_buffers()[0];  // Get labels buffer from ring buffer
    auto labels = _ring_buffer.get_meta_data().second->get_labels_batch();
    for (unsigned i = 0; i < _labels_tensor_list.size(); i++) {
        _labels_tensor_list[i]->set_dims({labels[i].size()});
        _labels_tensor_list[i]->set_mem_handle((void *)meta_data_buffers);
        meta_data_buffers += _labels_tensor_list[i]->info().data_size();
    }
    return &_labels_tensor_list;
}

TensorListVector *MasterGraph::ascii_values_meta_data() {
    if (_external_source_reader) {
        return &_webdataset_output_tensor_list;
    }
    if (_ring_buffer.level() == 0)
        THROW("No meta data has been loaded")

    for (uint ext = 0; ext < _ascii_tensor_list.size(); ext++) {
        auto meta_data_buffers = (uint8_t *)_ring_buffer.get_meta_read_buffers()[ext]; // Get ASCII buffer from ring buffer
        auto ascii_values = _ring_buffer.get_meta_data().second->get_ascii_values_batch();
        for (unsigned i = 0; i < _ascii_tensor_list[ext].size(); i++) {
            if (ascii_values[i][ext]) {
                _ascii_tensor_list[ext][i]->set_dims({ascii_values[i][ext]->size()});
                _ascii_tensor_list[ext][i]->set_mem_handle((void *)meta_data_buffers);
                meta_data_buffers += _ascii_tensor_list[ext][i]->info().data_size();
            } else { // To Handle Empty Case of Missing Behaviour Component
                _ascii_tensor_list[ext][i]->set_dims({0});
                _ascii_tensor_list[ext][i]->reset_mem_handle();
                meta_data_buffers += _ascii_tensor_list[ext][i]->info().data_size();
            }
        }
    }
    return &_webdataset_output_tensor_list;
}

TensorList *MasterGraph::bbox_meta_data() {
    if (_ring_buffer.level() == 0)
        THROW("No meta data has been loaded")
    auto meta_data_buffers = (unsigned char *)_ring_buffer.get_meta_read_buffers()[1];  // Get bbox buffer from ring buffer
    auto bbox_cords = _ring_buffer.get_meta_data().second->get_bb_cords_batch();
    for (unsigned i = 0; i < _bbox_tensor_list.size(); i++) {
        _bbox_tensor_list[i]->set_dims({bbox_cords[i].size(), 4});
        _bbox_tensor_list[i]->set_mem_handle((void *)meta_data_buffers);
        meta_data_buffers += _bbox_tensor_list[i]->info().data_size();
    }

    return &_bbox_tensor_list;
}

TensorList *MasterGraph::mask_meta_data() {
    if (_ring_buffer.level() == 0)
        THROW("No meta data has been loaded")
    auto meta_data_buffers = (unsigned char *)_ring_buffer.get_meta_read_buffers()[2];  // Get mask buffer from ring buffer
    auto mask_cords = _ring_buffer.get_meta_data().second->get_mask_cords_batch();
    for (unsigned i = 0; i < _mask_tensor_list.size(); i++) {
        _mask_tensor_list[i]->set_dims({mask_cords[i].size(), 1});
        _mask_tensor_list[i]->set_mem_handle((void *)meta_data_buffers);
        meta_data_buffers += _mask_tensor_list[i]->info().data_size();
    }

    return &_mask_tensor_list;
}

TensorList *MasterGraph::matched_index_meta_data() {
    if (_ring_buffer.level() == 0)
        THROW("No meta data has been loaded")
    auto meta_data_buffers = reinterpret_cast<unsigned char *>(_ring_buffer.get_meta_read_buffers()[2]);  // Get matches buffer from ring buffer
    for (unsigned i = 0; i < _matches_tensor_list.size(); i++) {
        _matches_tensor_list[i]->set_mem_handle(reinterpret_cast<void *>(meta_data_buffers));
        meta_data_buffers += _matches_tensor_list[i]->info().data_size();
    }
    return &_matches_tensor_list;
}

void MasterGraph::notify_user_thread() {
    if (_output_routine_finished_processing)
        return;
    LOG("Output routine finished processing all images, no more image to be processed")
    _output_routine_finished_processing = true;
}

bool MasterGraph::no_more_processed_data() {
    return (_output_routine_finished_processing && _ring_buffer.empty());
}

MasterGraph::Status
MasterGraph::copy_out_tensor_planar(void *out_ptr, RocalTensorlayout format, float multiplier0, float multiplier1,
                                    float multiplier2, float offset0, float offset1, float offset2, bool reverse_channels, RocalTensorDataType output_data_type) {
    if (no_more_processed_data())
        return MasterGraph::Status::NO_MORE_DATA;

    _convert_time.start();
    // Copies to the output context given by the user, each image is copied separate for planar
    auto output_tensor_info = _output_tensor_list[0]->info();
    auto dims = output_tensor_info.dims();
    const size_t w = dims[2];
    const size_t h = dims[1];
    const size_t c = dims[3];
    const size_t n = dims[0];

    const size_t single_output_tensor_size = output_tensor_info.data_size();

    if (output_tensor_info.mem_type() == RocalMemType::OCL || output_tensor_info.mem_type() == RocalMemType::HIP) {
        THROW("copy_out_tensor_planar for GPU affinity is not implemented")
    } else if (output_tensor_info.mem_type() == RocalMemType::HOST) {
        float multiplier[3] = {multiplier0, multiplier1, multiplier2};
        float offset[3] = {offset0, offset1, offset2};
        size_t dest_buf_offset = 0;

        auto output_buffers = _ring_buffer.get_read_buffers().first;

        for (auto &&out_tensor : output_buffers) {
            for (unsigned batch = 0; batch < n; batch++) {
                const size_t batch_offset = w * h * c * batch;
                auto channel_size = w * h;
                auto in_buffer = (unsigned char *)out_tensor + batch_offset;
                if (format == RocalTensorlayout::NHWC) {
                    if (output_data_type == RocalTensorDataType::FP32) {
                        float *output_tensor_32 = static_cast<float *>(out_ptr) + batch_offset;
                        for (unsigned channel_idx = 0; channel_idx < c; channel_idx++)
                            for (unsigned i = 0; i < channel_size; i++)
                                output_tensor_32[dest_buf_offset + channel_idx + i * c] =
                                    offset[channel_idx] + multiplier[channel_idx] *
                                                              (reverse_channels ? static_cast<float>(in_buffer[i +
                                                                                                               (c - channel_idx -
                                                                                                                1) *
                                                                                                                   channel_size])
                                                                                : static_cast<float>(in_buffer[i + channel_idx *
                                                                                                                       channel_size]));
                    } else if (output_data_type == RocalTensorDataType::FP16) {
                        half *output_tensor_16 = static_cast<half *>(out_ptr) + batch_offset;
                        for (unsigned channel_idx = 0; channel_idx < c; channel_idx++)
                            for (unsigned i = 0; i < channel_size; i++)
                                output_tensor_16[dest_buf_offset + channel_idx + i * c] =
                                    offset[channel_idx] + multiplier[channel_idx] *
                                                              (reverse_channels ? static_cast<half>(in_buffer[(c - channel_idx - 1) * channel_size + i])
                                                                                : static_cast<half>(in_buffer[channel_idx * channel_size + i]));
                    }
                }
                if (format == RocalTensorlayout::NCHW) {
                    if (output_data_type == RocalTensorDataType::FP32) {
                        float *output_tensor_32 = static_cast<float *>(out_ptr) + batch_offset;
                        // output_tensor_32 += batch_offset;
                        if (c != 3) {
                            for (unsigned channel_idx = 0; channel_idx < c; channel_idx++)
                                for (unsigned i = 0; i < channel_size; i++)
                                    output_tensor_32[dest_buf_offset + channel_idx * channel_size + i] =
                                        offset[channel_idx] + multiplier[channel_idx] *
                                                                  (reverse_channels ? (float)(in_buffer[(c - channel_idx - 1) * channel_size + i])
                                                                                    : (float)(in_buffer[channel_idx * channel_size + i]));
                        } else {
#if (ENABLE_SIMD && __AVX2__)

                            float *B_buf = output_tensor_32 + dest_buf_offset;
                            float *G_buf = B_buf + channel_size;
                            float *R_buf = G_buf + channel_size;
                            unsigned char *in_buffer_R = in_buffer;
                            unsigned char *in_buffer_G = in_buffer + channel_size;
                            unsigned char *in_buffer_B = in_buffer_G + channel_size;

                            __m256 pmul0 = _mm256_set1_ps(multiplier0);
                            __m256 pmul1 = _mm256_set1_ps(multiplier1);
                            __m256 pmul2 = _mm256_set1_ps(multiplier2);
                            __m256 padd0 = _mm256_set1_ps(offset0);
                            __m256 padd1 = _mm256_set1_ps(offset1);
                            __m256 padd2 = _mm256_set1_ps(offset2);
                            unsigned int alignedLength = (channel_size & ~7);  // multiple of 8
                            unsigned int i = 0;

                            __m256 fR, fG, fB;
                            for (; i < alignedLength; i += 8) {
                                __m128i pixR, pixG, pixB;
                                if (reverse_channels) {
                                    pixB = _mm_loadl_epi64((const __m128i *)in_buffer_R);
                                    pixG = _mm_loadl_epi64((const __m128i *)in_buffer_G);
                                    pixR = _mm_loadl_epi64((const __m128i *)in_buffer_B);
                                } else {
                                    pixR = _mm_loadl_epi64((const __m128i *)in_buffer_R);
                                    pixG = _mm_loadl_epi64((const __m128i *)in_buffer_G);
                                    pixB = _mm_loadl_epi64((const __m128i *)in_buffer_B);
                                }
                                fB = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(pixR));
                                fG = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(pixG));
                                fR = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(pixB));
                                fB = _mm256_mul_ps(fB, pmul0);
                                fG = _mm256_mul_ps(fG, pmul1);
                                fR = _mm256_mul_ps(fR, pmul2);
                                fB = _mm256_add_ps(fB, padd0);
                                fG = _mm256_add_ps(fG, padd1);
                                fR = _mm256_add_ps(fR, padd2);
                                _mm256_storeu_ps(B_buf, fB);
                                _mm256_storeu_ps(G_buf, fG);
                                _mm256_storeu_ps(R_buf, fR);
                                B_buf += 8;
                                G_buf += 8;
                                R_buf += 8;
                                in_buffer_R += 8, in_buffer_G += 8, in_buffer_B += 8;
                            }
                            for (; i < channel_size; i++) {
                                *B_buf++ = (*in_buffer_R++ * multiplier0) + offset0;
                                *G_buf++ = (*in_buffer_G++ * multiplier1) + offset1;
                                *R_buf++ = (*in_buffer_B++ * multiplier2) + offset1;
                            }

#else
                                for (unsigned channel_idx = 0; channel_idx < c; channel_idx++)
                                    for (unsigned i = 0; i < channel_size; i++)
                                        output_tensor_32[dest_buf_offset + channel_idx * channel_size + i] =
                                            offset[channel_idx] + multiplier[channel_idx] * (reverse_channels ? (float)(in_buffer[i + (c - channel_idx - 1) * channel_size]) : (float)(in_buffer[i + channel_idx * channel_size]));
#endif
                        }
                    } else if (output_data_type == RocalTensorDataType::FP16) {
                        half *output_tensor_16 = static_cast<half *>(out_ptr) + batch_offset;
                        for (unsigned channel_idx = 0; channel_idx < c; channel_idx++)
                            for (unsigned i = 0; i < channel_size; i++)
                                output_tensor_16[dest_buf_offset + channel_idx * channel_size + i] =
                                    offset[channel_idx] + multiplier[channel_idx] *
                                                              (reverse_channels ? (half)(in_buffer[i +
                                                                                                   (c - channel_idx -
                                                                                                    1) *
                                                                                                       channel_size])
                                                                                : (half)(in_buffer[i + channel_idx *
                                                                                                           channel_size]));
                    }
                }
            }
            dest_buf_offset += single_output_tensor_size;
        }
    }
    _convert_time.end();
    return Status::OK;
}

TensorListVector*
MasterGraph::get_bbox_encoded_buffers(size_t num_encoded_boxes) {
    if (_is_box_encoder) {
        if (num_encoded_boxes != _user_batch_size * _num_anchors) {
            THROW("num_encoded_boxes is not correct");
        }
        auto encoded_boxes_and_lables = _ring_buffer.get_box_encode_read_buffers();
        unsigned char *boxes_buf_ptr = (unsigned char *)encoded_boxes_and_lables.first;
        unsigned char *labels_buf_ptr = (unsigned char *)encoded_boxes_and_lables.second;
        auto labels = _ring_buffer.get_meta_data().second->get_labels_batch();

        if (_bbox_tensor_list.size() != _labels_tensor_list.size())
            THROW("The number of tensors between bbox and bbox_labels do not match")
        for (unsigned i = 0; i < _bbox_tensor_list.size(); i++) {
            _labels_tensor_list[i]->set_dims({labels[i].size()});
            _bbox_tensor_list[i]->set_dims({labels[i].size(), 4});
            _labels_tensor_list[i]->set_mem_handle((void *)labels_buf_ptr);
            _bbox_tensor_list[i]->set_mem_handle((void *)boxes_buf_ptr);
            labels_buf_ptr += _labels_tensor_list[i]->info().data_size();
            boxes_buf_ptr += _bbox_tensor_list[i]->info().data_size();
        }
        _bbox_encoded_output.emplace_back(&_labels_tensor_list);
        _bbox_encoded_output.emplace_back(&_bbox_tensor_list);
    }
    return &_bbox_encoded_output;
}

void MasterGraph::feed_external_input(const std::vector<std::string>& input_images_names, bool is_labels, const std::vector<unsigned char *>& input_buffer,
                                      const std::vector<ROIxywh>& roi_xywh, unsigned int max_width, unsigned int max_height, unsigned int channels,
                                      ExternalSourceFileMode mode, RocalTensorlayout layout, bool eos) {
    _external_source_eos = eos;
    _loader_module->feed_external_input(input_images_names, input_buffer, roi_xywh, max_width, max_height, channels, mode, eos);

    if (is_labels) {
        if (_labels_tensor_list.size() == 0) {  // Labels tensor list is initialized only once for the pipeline
            std::vector<size_t> dims = {1};
            auto default_labels_info = TensorInfo(std::move(dims), _mem_type, RocalTensorDataType::INT32);
            default_labels_info.set_metadata();

            for (unsigned i = 0; i < _user_batch_size; i++) {
                auto info = default_labels_info;
                _labels_tensor_list.push_back(new Tensor(info));
            }
            _metadata_output_tensor_list.emplace_back(&_labels_tensor_list);
        }
    }
}
