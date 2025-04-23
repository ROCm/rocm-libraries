# Copyright (c) 2018 - 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##
# @file pipeline.py
#
# @brief File containing the Pipeline class containing the pybind API functions

import rocal_pybind as b
import amd.rocal.types as types
import numpy as np
import ctypes
import functools
import inspect


class Pipeline(object):

    """!Pipeline class internally calls RocalCreate which returns context which will have all
    the info set by the user.

    @param batch_size (int, optional, default = -1)                                                       Batch size of the pipeline. Negative values for this parameter are invalid - the default value may only be used with serialized pipeline (the value stored in serialized pipeline is used instead).
    @param num_threads (int, optional, default = -1)                                                      Number of CPU threads used by the pipeline. Negative values for this parameter are invalid - the default value may only be used with serialized pipeline (the value stored in serialized pipeline is used instead).
    @param device_id (int, optional, default = 0)                                                         Id of GPU used by the pipeline. Negative values for this parameter are invalid
    @param seed (int, optional, default = -1)                                                             Seed used for random number generation. Leaving the default value for this parameter results in random seed.
    @param exec_pipelined (bool, optional, default = True)                                                Whether to execute the pipeline in a way that enables overlapping CPU and GPU computation, typically resultingin faster execution speed, but larger memory consumption.
    @param prefetch_queue_depth (int or {"cpu_size": int, "gpu_size": int}, optional, default = 2)        Depth of the executor pipeline. Deeper pipeline makes ROCAL more resistant to uneven execution time of each batch, but it also consumes more memory for internal buffers. Specifying a dict: ``{ "cpu_size": x, "gpu_size": y }`` instead of an integer will cause the pipeline to use separated queues executor, with buffer queue size `x` for cpu stage and `y` for mixed and gpu stages. It is not supported when both `exec_async` and `exec_pipelined` are set to `False`. Executor will buffer cpu and gpu stages separatelly, and will fill the buffer queues when the first :meth:`amd.rocal.pipeline.Pipeline.run` is issued.
    @param exec_async (bool, optional, default = True)                                                    Whether to execute the pipeline asynchronously. his makes :meth:`amd.rocal.pipeline.Pipeline.run` method run asynchronously with respect to the calling Python thread.
    @param bytes_per_sample  (int, optional, default = 0)                                                 A hint for ROCAL for how much memory to use for its tensors.
    @param rocal_cpu (bool, optional, default = False)                                                    Whether to use CPU or GPU for the pipeline
    @param max_streams (int, optional, default = -1)                                                      Limit the number of HIP streams used by the executor. Value of -1 does not impose a limit. This parameter is currently unused (and behavior of unrestricted number of streams is assumed).
    @param default_cuda_stream_priority (int, optional, default = 0)                                      HIP stream priority used by ROCAL. See `cudaStreamCreateWithPriority` in HIP documentation
    @param tensor_layout (int, optional, default = 0)                                                     Tensor layout used for the augmentations
    @param reverse_channels (int, optional, default = 0)                                                  Whether to reverse channels for the output tensors
    @param mean (int, optional, default = 0)                                                              Mean value used for the image normalization
    @param std (int, optional, default = 0)                                                               Standard deviation value used for the image normalization
    @param tensor_dtype (int, optional, default = 0)                                                      Tensor datatype used for the pipeline
    @param output_memory_type (int, optional, default = 0)                                                Output memory type used for the output tensors
    """
    '''.
    Args: batch_size
          rocal_cpu
          gpu_id (default 0)
          cpu_threads (default 1)
    This returns a context'''
    _handle = None
    _current_pipeline = None

    def __init__(self, batch_size=-1, num_threads=0, device_id=0, seed=1,
                 exec_pipelined=True, prefetch_queue_depth=2,
                 exec_async=True, bytes_per_sample=0,
                 rocal_cpu=False, max_streams=-1, default_cuda_stream_priority=0, tensor_layout=types.NCHW, reverse_channels=False, mean=None, std=None, tensor_dtype=types.FLOAT, output_memory_type=None): 
        if (rocal_cpu):
            self._handle = b.rocalCreate(
                batch_size, types.CPU, device_id, num_threads, prefetch_queue_depth, tensor_dtype)
        else:
            self._handle = b.rocalCreate(
                batch_size, types.GPU, device_id, num_threads, prefetch_queue_depth, tensor_dtype)

        if (b.getStatus(self._handle) == types.OK):
            print("Pipeline has been created succesfully")
        else:
            raise Exception("Failed creating the pipeline")
        self._check_ops = ["CropMirrorNormalize"]
        self._check_crop_ops = ["Resize"]
        self._check_ops_decoder = [
            "ImageDecoder", "ImageDecoderSlice", "ImageDecoderRandomCrop", "ImageDecoderRaw"]
        self._check_ops_reader = ["labelReader", "TFRecordReaderClassification", "TFRecordReaderDetection",
                                  "COCOReader", "Caffe2Reader", "Caffe2ReaderDetection", "CaffeReader", "CaffeReaderDetection", "NumpyReader"]
        self._batch_size = batch_size
        self._num_threads = num_threads
        self._device_id = device_id
        self._output_memory_type = output_memory_type if output_memory_type else (
            types.HOST_MEMORY if rocal_cpu else types.DEVICE_MEMORY)
        self._seed = seed
        self._exec_pipelined = exec_pipelined
        self._prefetch_queue_depth = prefetch_queue_depth
        self._exec_async = exec_async
        self._bytes_per_sample = bytes_per_sample
        self._rocal_cpu = rocal_cpu
        self._max_streams = max_streams
        self._default_cuda_stream_priority = default_cuda_stream_priority
        self._tensor_layout = tensor_layout
        self._tensor_dtype = tensor_dtype
        self._multiplier = list(map(lambda x: 1/x, std)
                                ) if std else [1.0, 1.0, 1.0]
        self._offset = list(map(lambda x, y: -(x/y), mean, std)
                            ) if mean and std else [0.0, 0.0, 0.0]
        self._reverse_channels = reverse_channels
        self._img_h = None
        self._img_w = None
        self._shuffle = None
        self._name = None
        self._anchors = None
        self._box_encoder = None
        self._box_iou_matcher = None
        self._encode_tensor = None
        self._num_classes = None
        self._one_hot_encoding = False
        self._castLabels = False
        self._current_pipeline = None
        self._reader = None
        self._define_graph_set = False
        self.set_seed(self._seed)
        self._is_external_source_operator = False
        self._external_source = None
        self._external_source_mode = None
        self._last_batch_policy = None

    def build(self):
        """!Build the pipeline using rocalVerify call
        """
        status = b.rocalVerify(self._handle)
        if (status != types.OK):
            print("Verify graph failed")
            exit(0)
        return self

    def rocal_run(self):
        """! Run the pipeline using rocalRun call
        """
        status = b.rocalRun(self._handle)
        return status

    def define_graph(self):
        """!This function is defined by the user to construct the
        graph of operations for their pipeline.
        It returns a list of outputs created by calling ROCAL Operators."""
        print("define_graph is deprecated")
        raise NotImplementedError

    def get_handle(self):
        return self._handle

    def copyToExternalTensor(self, array,  multiplier, offset, reverse_channels, tensor_format, tensor_dtype, max_roi_height=0, max_roi_width=0):
        b.rocalToTensor(self._handle, ctypes.c_void_p(array.data_ptr()), tensor_format, tensor_dtype,
                        multiplier[0], multiplier[1], multiplier[2], offset[0], offset[1], offset[2], (1 if reverse_channels else 0), self._output_memory_type, max_roi_height, max_roi_width)

    def get_one_hot_encoded_labels(self, array_ptr, dest_device_type):
            b.getOneHotEncodedLabels(self._handle, array_ptr, self._num_classes, dest_device_type)

    def set_outputs(self, *output_list):
        b.setOutputs(self._handle, len(output_list), output_list)

    def __enter__(self):
        Pipeline._current_pipeline = self
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def set_seed(self, seed=0):
        return b.setSeed(seed)

    @classmethod
    def create_int_param(self, value=1):
        return b.createIntParameter(value)

    @classmethod
    def create_float_param(self, value=1):
        return b.createFloatParameter(value)

    @classmethod
    def update_int_param(self, value=1, param=1):
        b.updateIntParameter(value, param)

    @classmethod
    def update_float_param(self, value=1, param=1):
        b.updateFloatParameter(value, param)

    @classmethod
    def get_int_value(self, param):
        return b.getIntValue(param)

    @classmethod
    def get_float_value(self, param):
        return b.getFloatValue(param)

    def get_image_name(self, array_len):
        return b.getImageName(self._handle, array_len)

    def get_image_id(self, array):
        b.getImageId(self._handle, array)

    def get_bounding_box_count(self):
        return b.getBoundingBoxCount(self._handle)

    def get_bounding_box_labels(self):
        return b.getBoundingBoxLabels(self._handle)

    def get_bounding_box_cords(self):
        return b.getBoundingBoxCords(self._handle)
    
    def get_ascii_datas(self):
        return b.getAsciiDatas(self._handle)

    def get_mask_count(self, array):
        return b.getMaskCount(self._handle, array)

    def get_mask_coordinates(self, array_count, array):
        return b.getMaskCoordinates(self._handle, array_count, array)
    
    def get_image_labels(self):
        return b.getImageLabels(self._handle)

    def copy_encoded_boxes_and_lables(self, bbox_array, label_array):
        b.rocalCopyEncodedBoxesAndLables(self._handle, bbox_array, label_array)

    def get_encoded_boxes_and_lables(self, batch_size, num_anchors):
        return b.rocalGetEncodedBoxesAndLables(self._handle, batch_size, num_anchors)

    def get_img_sizes(self, array):
        return b.getImgSizes(self._handle, array)

    def get_roi_img_sizes(self, array):
        return b.getROIImgSizes(self._handle, array)
    
    def get_image_name_length(self, idx):
        return b.getImageNameLen(self._handle, idx)

    def get_remaining_images(self):
        return b.getRemainingImages(self._handle)

    def rocal_release(self):
        return b.rocalRelease(self._handle)

    def rocal_reset_loaders(self):
        return b.rocalResetLoaders(self._handle)

    def is_empty(self):
        return b.isEmpty(self._handle)

    def timing_info(self):
        return b.getTimingInfo(self._handle)

    def get_matched_indices(self):
        return b.getMatchedIndices(self._handle)

    def get_output_tensors(self):
        return b.getOutputTensors(self._handle)
    
    def get_last_batch_padded_size(self):
        return b.getLastBatchPaddedSize(self._handle)

    def run(self):
        """
        It raises StopIteration if data set reached its end.
        return:
        :return:
        A list of `rocalTensorList` objects for respective pipeline outputs.
        """
        try:
            if self.get_remaining_images() > 0:
                self.rocal_run()
                return b.getOutputTensors(self._handle)
        except:
            raise StopIteration


def _discriminate_args(func, **func_kwargs):
    """!Split args on those applicable to Pipeline constructor and the decorated function."""
    func_argspec = inspect.getfullargspec(func)
    ctor_argspec = inspect.getfullargspec(Pipeline.__init__)

    if 'debug' not in func_argspec.args and 'debug' not in func_argspec.kwonlyargs:
        func_kwargs.pop('debug', False)

    ctor_args = {}
    fn_args = {}

    if func_argspec.varkw is not None:
        raise TypeError(
            f"Using variadic keyword argument `**{func_argspec.varkw}` in a  "
            f"graph-defining function is not allowed.")

    for farg in func_kwargs.items():
        is_ctor_arg = farg[0] in ctor_argspec.args or farg[0] in ctor_argspec.kwonlyargs
        is_fn_arg = farg[0] in func_argspec.args or farg[0] in func_argspec.kwonlyargs
        if is_fn_arg:
            fn_args[farg[0]] = farg[1]
            if is_ctor_arg:
                print(
                    "Warning: the argument `{farg[0]}` shadows a Pipeline constructor "
                    "argument of the same name.")
        elif is_ctor_arg:
            ctor_args[farg[0]] = farg[1]
        else:
            assert False, f"This shouldn't happen. Please double-check the `{farg[0]}` argument"

    return ctor_args, fn_args


def pipeline_def(fn=None, **pipeline_kwargs):
    """!
    Decorator that converts a graph definition function into a rocAL pipeline factory.

    A graph definition function is a function that returns intended pipeline outputs.
    You can decorate this function with ``@pipeline_def``::

        @pipeline_def
        def my_pipe(flip_vertical, flip_horizontal):
            ''' Creates a rocAL pipeline, which returns flipped and original images '''
            data, _ = fn.readers.file(file_root=images_dir)
            img = fn.decoders.image(data, device="mixed")
            flipped = fn.flip(img, horizontal=flip_horizontal, vertical=flip_vertical)
            return flipped, img

    The decorated function returns a rocAL Pipeline object::

        pipe = my_pipe(True, False)
        # pipe.build()  # the pipeline is not configured properly yet

    A pipeline requires additional parameters such as batch size, number of worker threads,
    GPU device id and so on (see :meth:`amd.rocal.Pipeline()` for a
    complete list of pipeline parameters).
    These parameters can be supplied as additional keyword arguments,
    passed to the decorated function::

        pipe = my_pipe(True, False, batch_size=32, num_threads=1, device_id=0)
        pipe.build()  # the pipeline is properly configured, we can build it now

    The outputs from the original function became the outputs of the Pipeline::

        flipped, img = pipe.run()

    When some of the pipeline parameters are fixed, they can be specified by name in the decorator::

        @pipeline_def(batch_size=42, num_threads=3)
        def my_pipe(flip_vertical, flip_horizontal):
            ...

    Any Pipeline constructor parameter passed later when calling the decorated function will
    override the decorator-defined params::

        @pipeline_def(batch_size=32, num_threads=3)
        def my_pipe():
            data = fn.external_source(source=my_generator)
            return data

        pipe = my_pipe(batch_size=128)  # batch_size=128 overrides batch_size=32

    .. warning::

        The arguments of the function being decorated can shadow pipeline constructor arguments -
        in which case there's no way to alter their values.

    .. note::

        Using ``**kwargs`` (variadic keyword arguments) in graph-defining function is not allowed.
        They may result in unwanted, silent hijacking of some arguments of the same name by
        Pipeline constructor. Code written this way would cease to work with future versions of rocAL
        when new parameters are added to the Pipeline constructor.

    To access any pipeline arguments within the body of a ``@pipeline_def`` function, the function
    :meth:`amd.rocal.Pipeline.current()` can be used:: ( note: this is not supported yet)

        @pipeline_def()
        def my_pipe():
            pipe = Pipeline.current()
            batch_size = pipe.batch_size
            num_threads = pipe.num_threads
            ...

        pipe = my_pipe(batch_size=42, num_threads=3)
        ...
    """

    def actual_decorator(func):

        @functools.wraps(func)
        def create_pipeline(*args, **kwargs):
            ctor_args, fn_kwargs = _discriminate_args(func, **kwargs)
            # Merge and overwrite dict
            pipe = Pipeline(**{**pipeline_kwargs, **ctor_args})
            with pipe:
                pipe_outputs = func(*args, **fn_kwargs)
                if isinstance(pipe_outputs, tuple):
                    po = pipe_outputs
                elif pipe_outputs is None:
                    po = ()
                else:
                    po = (pipe_outputs, )
                pipe.set_outputs(*po)
            return pipe

        # Add `is_pipeline_def` attribute to the function marked as `@pipeline_def`
        create_pipeline._is_pipeline_def = True
        return create_pipeline

    return actual_decorator(fn) if fn else actual_decorator
