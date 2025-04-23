# Copyright (c) 2018 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
# @file generic.py
#
# @brief File containing iterators for generic use case

import numpy as np
import rocal_pybind as b
import amd.rocal.types as types
import ctypes

class ROCALGenericIterator(object):
    """!Iterator for processing data

        @param pipeline            The rocAL pipeline to use for processing data.
        @param tensor_layout       The layout of the output tensors
        @param reverse_channels    Whether to reverse the order of color channels.
        @param multiplier          Multiplier values for color normalization.
        @param offset              Offset values for color normalization.
        @param tensor_dtype        Data type of the output tensors.
        @param display             Whether to display images during processing
        @param device              The device to use for processing
        @param device_id           The ID of the device to use
    """

    def __init__(self, pipeline, tensor_layout=types.NCHW, reverse_channels=False, multiplier=[1.0, 1.0, 1.0], offset=[0.0, 0.0, 0.0], tensor_dtype=types.FLOAT, display=False, device="cpu", device_id=0):
        self.loader = pipeline
        self.tensor_format = tensor_layout
        self.multiplier = multiplier
        self.offset = offset
        self.device = device
        self.device_id = device_id
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.display = display
        self.batch_size = pipeline._batch_size
        if self.loader._name is None:
            self.loader._name = self.loader._reader
        self.labels_size = ((self.batch_size * self.loader._num_classes)
                            if self.loader._one_hot_encoding else self.batch_size)
        self.output_list = self.dimensions = self.dtype = None
        self.labels_tensor = None
        self.iterator_length = b.getRemainingImages(
            self.loader._handle) // self.batch_size  # iteration length
        if self.loader._is_external_source_operator:
            self.eos = False
            self.index = 0
            self.num_batches = self.loader._external_source.n // self.batch_size if self.loader._external_source.n % self.batch_size == 0 else (
                self.loader._external_source.n // self.batch_size + 1)
        else:
            self.num_batches = None

    def next(self):
        return self.__next__()

    def __next__(self):
        if (self.loader._is_external_source_operator):
            if (self.index + 1) == self.num_batches:
                self.eos = True
            if (self.index + 1) <= self.num_batches:
                data_loader_source = next(self.loader._external_source)
                # Extract all data from the source
                images_list = data_loader_source[0] if (self.loader._external_source_mode == types.EXTSOURCE_FNAME) else []
                input_buffer = data_loader_source[0] if (self.loader._external_source_mode != types.EXTSOURCE_FNAME) else []
                labels_data = data_loader_source[1] if (len(data_loader_source) > 1) else None
                roi_height = data_loader_source[2] if (len(data_loader_source) > 2) else []
                roi_width = data_loader_source[3] if (len(data_loader_source) > 3) else []
                ROIxywh_list = []
                for i in range(self.batch_size):
                    ROIxywh = b.ROIxywh()
                    ROIxywh.x =  0
                    ROIxywh.y =  0
                    ROIxywh.w = roi_width[i] if len(roi_width) > 0 else 0
                    ROIxywh.h = roi_height[i] if len(roi_height) > 0 else 0
                    ROIxywh_list.append(ROIxywh)
                if (len(data_loader_source) == 6 and self.loader._external_source_mode == types.EXTSOURCE_RAW_UNCOMPRESSED):
                    decoded_height = data_loader_source[4]
                    decoded_width = data_loader_source[5]
                else:
                    decoded_height = self.loader._external_source_user_given_height
                    decoded_width = self.loader._external_source_user_given_width

                kwargs_pybind = {
                    "handle": self.loader._handle,
                    "source_input_images": images_list,
                    "labels": labels_data,
                    "input_batch_buffer": input_buffer,
                    "roi_xywh": ROIxywh_list,
                    "decoded_width": decoded_width,
                    "decoded_height": decoded_height,
                    "channels": 3,
                    "external_source_mode": self.loader._external_source_mode,
                    "rocal_tensor_layout": types.NCHW,
                    "eos": self.eos}
                b.externalSourceFeedInput(*(kwargs_pybind.values()))
            self.index = self.index + 1
        if self.loader.rocal_run() != 0:
            raise StopIteration
        self.output_tensor_list = self.loader.get_output_tensors()

        if self.output_list is None:  # Checking if output_list is empty and initializing the buffers
            self.output_list = []
            for i in range(len(self.output_tensor_list)):
                self.dimensions = self.output_tensor_list[i].dimensions()
                self.dtype = self.output_tensor_list[i].dtype()
                self.output = np.empty(self.dimensions, dtype=self.dtype)
                # returned as numpy always - no ROCM CuPy support available
                self.output_tensor_list[i].copy_data(self.output)
                self.output_list.append(self.output)
        else:
            for i in range(len(self.output_tensor_list)):
                self.output_tensor_list[i].copy_data(self.output_list[i])
        if (self.loader._is_external_source_operator):
            self.labels = self.loader.get_image_labels()
            self.labels_tensor = self.labels.astype(dtype=np.int_)
            return self.output_list, self.labels_tensor

        if self.loader._name == "labelReader":
            if self.loader._one_hot_encoding == True:
                self.labels = np.empty(self.labels_size, dtype="int32")
                self.loader.get_one_hot_encoded_labels(
                        self.labels.ctypes.data, self.loader._output_memory_type)
                self.labels_tensor = self.labels.reshape(
                    -1, self.batch_size, self.loader._num_classes)
            else:
                if self.display:
                    for output in self.output_tensor_list:
                        for i in range(self.batch_size):
                            draw_patches(output[i], i)
                self.labels = self.loader.get_image_labels()
                self.labels_tensor = self.labels.astype(dtype=np.int_)

        return self.output_list, self.labels_tensor

    def reset(self):
        b.rocalResetLoaders(self.loader._handle)

    def __iter__(self):
        return self

    def __len__(self):
        return self.iterator_length

    def __del__(self):
        b.rocalRelease(self.loader._handle)


class ROCALClassificationIterator(ROCALGenericIterator):
    """!ROCAL iterator for classification tasks for generic use case. It returns 2 outputs
    (data and label) in the form of numpy Tensor.

    Calling

    .. code-block:: python

       ROCALClassificationIterator(pipelines, size)

    is equivalent to calling

    .. code-block:: python

       ROCALGenericIterator(pipelines, ["data", "label"], size)

    Please keep in mind that Tensors returned by the iterator are
    still owned by ROCAL. They are valid till the next iterator call.
    If the content needs to be preserved please copy it to another tensor.

    @param pipelines(list of amd.rocal.pipeline.Pipeline)         List of pipelines to use
    @param size (int)                                             Number of samples in the epoch (Usually the size of the dataset).
    @param auto_reset (bool, optional, default = False)           Whether the iterator resets itself for the next epoch or it requires reset() to be called separately.
    @param fill_last_batch (bool, optional, default = True        Whether to fill the last batch with data up to 'self.batch_size'. The iterator would return the first integer multiple of self._num_gpus * self.batch_size entries which exceeds 'size'. Setting this flag to False will cause the iterator to return exactly 'size' entries.
    @param dynamic_shape (bool, optional, default = False)        Whether the shape of the output of the ROCAL pipeline can change during execution. If True, the numpy tensor will be resized accordingly if the shape of ROCAL returned tensors changes during execution. If False, the iterator will fail in case of change.
    @param last_batch_padded (bool, optional, default = False)    Whether the last batch provided by ROCAL is padded with the last sample or it just wraps up. In the conjunction with `fill_last_batch` it tells if the iterator returning last batch with data only partially filled with data from the current epoch is dropping padding samples or samples from the next epoch. If set to False next epoch will end sooner as data from it was consumed but dropped. If set to True next epoch would be the same length as the first one.
    @param display (bool, optional, default = False)              Whether the images should be saved as png files for display.
    @param device (str, optional, default = "cpu")                Whether to use CPU or GPU backend for the dataloader.
    @param device_id (int, optional, default = 0)                 Device ID of the GPU being used for the training loader.

    Example
    -------
    With the data set [1,2,3,4,5,6,7] and the batch size 2:
    fill_last_batch = False, last_batch_padded = True  -> last batch = [7], next iteration will return [1, 2]
    fill_last_batch = False, last_batch_padded = False -> last batch = [7], next iteration will return [2, 3]
    fill_last_batch = True, last_batch_padded = True   -> last batch = [7, 7], next iteration will return [1, 2]
    fill_last_batch = True, last_batch_padded = False  -> last batch = [7, 1], next iteration will return [2, 3]
    """

    def __init__(self,
                 pipelines,
                 size=0,
                 auto_reset=False,
                 fill_last_batch=True,
                 dynamic_shape=False,
                 last_batch_padded=False,
                 display=False,
                 device="cpu",
                 device_id=0):
        pipe = pipelines
        super(ROCALClassificationIterator, self).__init__(pipe, tensor_layout=pipe._tensor_layout, tensor_dtype=pipe._tensor_dtype,
                                                          multiplier=pipe._multiplier, offset=pipe._offset, display=display, device=device, device_id=device_id)


class ROCALNumpyIterator(object):
    def __init__(self, pipeline):
        self.loader = pipeline
        self.output_list = None
        self.batch_size = self.loader._batch_size
        self.len = b.getRemainingImages(self.loader._handle)
        self.last_batch_policy = self.loader._last_batch_policy
        self.last_batch_size = None

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.loader.rocal_run() != 0:
            raise StopIteration
        self.output_tensor_list = self.loader.get_output_tensors()

        if self.output_list is None:  # Checking if output_list is empty and initializing the buffers
            self.output_list = []
            for i in range(len(self.output_tensor_list)):
                dimensions = self.output_tensor_list[i].dimensions()
                dtype = self.output_tensor_list[i].dtype()
                output = np.empty(dimensions, dtype=dtype)
                # returned as numpy always - no ROCM CuPy support available
                self.output_tensor_list[i].copy_data(output)
                self.output_list.append(output)
        else:
            for i in range(len(self.output_tensor_list)):
                self.output_tensor_list[i].copy_data(self.output_list[i])
        # Check if last batch policy is partial and only return the valid images in last batch
        if (self.last_batch_policy is (types.LAST_BATCH_PARTIAL)) and b.getRemainingImages(self.loader._handle) < self.batch_size:
            if (self.last_batch_size is None):
                self.last_batch_size = self.batch_size - \
                    b.getLastBatchPaddedSize(self.loader._handle)
            return [inner_list[0:self.last_batch_size, :] for inner_list in self.output_list]
        return self.output_list

    def reset(self):
        b.rocalResetLoaders(self.loader._handle)

    def __iter__(self):
        return self

    def __len__(self):
        return self.len // self.batch_size

    def __del__(self):
        b.rocalRelease(self.loader._handle)

def draw_patches(img, idx):
    # image is expected as an array
    """!Writes images to disk as a PNG file.

        @param img    The input image as an array.
        @param idx    Index used for naming the output file.
    """
    import cv2
    img = img.cpu()
    image = img.detach().numpy()
    image = image.transpose([1, 2, 0])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.UMat(image).get()
    cv2.imwrite(str(idx)+"_"+"train"+".png", image)
