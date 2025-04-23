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
# @file tf.py
# @brief File containing iterators to be used with TF trainings

import numpy as np
import ctypes
import rocal_pybind as b
import amd.rocal.types as types
import tensorflow as tf

class ROCALGenericImageIterator(object):
    """!Generic iterator for rocAL pipelines that process images

        @param pipeline: The rocAL pipeline to use for processing data.
    """

    def __init__(self, pipeline):
        self.loader = pipeline
        self.output_list = None
        self.bs = pipeline._batch_size

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.loader.rocal_run() != 0:
            raise StopIteration
        self.output_tensor_list = self.loader.get_output_tensors()

        if self.output_list is None:
            # Output list used to store pipeline outputs - can support multiple augmentation outputs
            self.output_list = []
            for i in range(len(self.output_tensor_list)):
                self.dimensions = self.output_tensor_list[i].dimensions()
                self.dtype = self.output_tensor_list[i].dtype()
                self.output = np.empty(self.dimensions, dtype=self.dtype)

                self.output_tensor_list[i].copy_data(self.output)
                self.output_list.append(self.output)
        else:
            for i in range(len(self.output_tensor_list)):
                self.output_tensor_list[i].copy_data(self.output_list[i])
        return self.output_list

    def reset(self):
        b.rocalResetLoaders(self.loader._handle)

    def __iter__(self):
        return self

    def __del__(self):
        b.rocalRelease(self.loader._handle)


class ROCALGenericIteratorDetection(object):
    """!Iterator for processing data

        @param pipeline            The rocAL pipeline to use for processing data.
        @param tensor_layout       The layout of the output tensors
        @param reverse_channels    Whether to reverse the order of color channels.
        @param multiplier          Multiplier values for color normalization.
        @param offset              Offset values for color normalization.
        @param tensor_dtype        Data type of the output tensors.
        @param device              The device to use for processing
        @param device_id           The ID of the device to use
    """

    def __init__(self, pipeline, tensor_layout=types.NCHW, reverse_channels=False, multiplier=None, offset=None, tensor_dtype=types.FLOAT, device=None, device_id=0):
        self.loader = pipeline
        self.tensor_format = tensor_layout
        self.multiplier = multiplier or [1.0, 1.0, 1.0]
        self.offset = offset or [0.0, 0.0, 0.0]
        self.device = device
        self.device_id = device_id
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.bs = pipeline._batch_size
        self.output_list = self.dimensions = self.dtype = None
        if self.loader._name is None:
            self.loader._name = self.loader._reader
        self.iterator_length = b.getRemainingImages(self.loader._handle)

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.loader.rocal_run() != 0:
            raise StopIteration
        self.output_tensor_list = self.loader.get_output_tensors()
        # Output list used to store pipeline outputs - can support multiple augmentation outputs
        self.output_list = []
        for i in range(len(self.output_tensor_list)):
            # returns tf tensor on gpu/cpu 
            self.output = tf.experimental.dlpack.from_dlpack(self.output_tensor_list[i].__dlpack__(self.device_id))
            self.output_list.append(self.output)

        if self.loader._name == "TFRecordReaderDetection":
            self.bbox_list = []
            self.label_list = []
            # 1D labels array in a batch
            self.labels = self.loader.get_bounding_box_labels()
            # 1D bboxes array in a batch
            self.bboxes = self.loader.get_bounding_box_cords()
            # 1D Image sizes array of image in a batch
            self.img_size = np.zeros((self.bs * 2), dtype="int32")
            self.num_bboxes_list = []
            self.loader.get_img_sizes(self.img_size)
            for i in range(self.bs):
                self.label_2d_numpy = np.reshape(
                    self.labels[i], (-1, 1)).tolist()
                self.bb_2d_numpy = np.reshape(self.bboxes[i], (-1, 4)).tolist()
                self.num_bboxes_list.append(len(self.bboxes[i]))
                self.label_list.append(self.label_2d_numpy)
                self.bbox_list.append(self.bb_2d_numpy)

            self.target = self.bbox_list
            self.target1 = self.label_list
            max_cols = max([len(row)
                           for batch in self.target for row in batch])
            # max_rows = max([len(batch) for batch in self.target])
            max_rows = 100
            bb_padded = [
                batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in self.target]
            bb_padded_1 = [row + [0] * (max_cols - len(row))
                           for batch in bb_padded for row in batch]
            arr = np.asarray(bb_padded_1)
            self.res = np.reshape(arr, (-1, max_rows, max_cols))
            max_cols = max([len(row)
                           for batch in self.target1 for row in batch])
            # max_rows = max([len(batch) for batch in self.target1])
            max_rows = 100
            lab_padded = [
                batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in self.target1]
            lab_padded_1 = [row + [0] * (max_cols - len(row))
                            for batch in lab_padded for row in batch]
            labarr = np.asarray(lab_padded_1)
            self.l = np.reshape(labarr, (-1, max_rows, max_cols))
            self.num_bboxes_arr = np.array(self.num_bboxes_list)

            return self.output_list, self.res, self.l, self.num_bboxes_arr
        elif (self.loader._name == "TFRecordReaderClassification"):
            if (self.loader._one_hot_encoding == True):
                if self.device == "cpu":
                    self.labels = np.zeros(
                        (self.bs) * (self.loader._num_classes), dtype="int32")
                    self.loader.get_one_hot_encoded_labels(
                        self.labels.ctypes.data, self.loader._output_memory_type)
                    self.labels = np.reshape(
                        self.labels, (-1, self.bs, self.loader._num_classes))
                else:
                    # TODO: one hot labels in dlpack? no support yet
                    self.labels = np.zeros((self.bs) * (self.loader._num_classes), dtype="int32")
                    self.loader.get_one_hot_encoded_labels(self.labels, self.loader._output_memory_type)
                    self.labels = np.reshape(self.labels, (-1, self.bs, self.loader._num_classes))
            else:
                self.labels = self.loader.get_image_labels()

            return self.output_list, self.labels

    def reset(self):
        b.rocalResetLoaders(self.loader._handle)

    def __iter__(self):
        return self

    def __del__(self):
        b.rocalRelease(self.loader._handle)

    def __len__(self):
        return self.iterator_length

class ROCALIterator(ROCALGenericIteratorDetection):
    """!ROCAL iterator for detection and classification tasks for TF reader. It returns 2 or 3 outputs
    (data and label) or (data , bbox , labels) in the form of TF tensors.
    Calling
    .. code-block:: python
       ROCALIterator(pipelines, size)
    is equivalent to calling
    .. code-block:: python
       ROCALGenericIteratorDetection(pipelines, ["data", "label"], size)

        @param pipelines            The rocAL pipelines to use for processing data.
        @param size                 The size of the iterator.
        @param auto_reset           Whether to automatically reset the iterator after an epoch.
        @param fill_last_batch      Whether to fill the last batch with repeated data to match the batch size.
        @param dynamic_shape        Whether the iterator supports dynamic shapes.
        @param last_batch_padded    Whether the last batch should be padded to match the batch size.


    """

    def __init__(self,
                 pipelines,
                 size=0,
                 auto_reset=False,
                 fill_last_batch=True,
                 dynamic_shape=False,
                 last_batch_padded=False,
                 device="cpu",
                 device_id=0):
        pipe = pipelines
        super(ROCALIterator, self).__init__(pipe, tensor_layout=pipe._tensor_layout, tensor_dtype=pipe._tensor_dtype,
                                            multiplier=pipe._multiplier, offset=pipe._offset, device=device, device_id=device_id)


class ROCAL_iterator(ROCALGenericImageIterator):
    """! ROCAL iterator for processing images for TF reader. It returns outputs in the form of tf tensors.

        @param pipelines            The rocAL pipelines to use for processing data.
        @param size                 The size of the iterator.
        @param auto_reset           Whether to automatically reset the iterator after an epoch.
        @param fill_last_batch      Whether to fill the last batch with repeated data to match the batch size.
        @param dynamic_shape        Whether the iterator supports dynamic shapes.
        @param last_batch_padded    Whether the last batch should be padded to match the batch size.
    """

    def __init__(self,
                 pipelines,
                 size=0,
                 auto_reset=False,
                 fill_last_batch=True,
                 dynamic_shape=False,
                 last_batch_padded=False):
        pipe = pipelines
        super(ROCAL_iterator, self).__init__(pipe)
