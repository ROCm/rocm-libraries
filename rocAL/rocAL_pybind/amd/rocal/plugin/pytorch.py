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
# @file pytorch.py
# @brief File containing iterators to be used with pytorch trainings

import torch
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

    def __init__(self, pipeline, tensor_layout=types.NCHW, reverse_channels=False, multiplier=[1.0, 1.0, 1.0], offset=[0.0, 0.0, 0.0], tensor_dtype=types.FLOAT, device="cpu", device_id=0, display=False):
        self.loader = pipeline
        self.tensor_format = tensor_layout
        self.multiplier = multiplier
        self.offset = offset
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.device = device
        self.device_id = device_id
        self.batch_size = self.loader._batch_size
        self.labels_size = ((self.batch_size * self.loader._num_classes)
                            if self.loader._one_hot_encoding else self.batch_size)
        self.output_list = None
        self.output_memory_type = self.loader._output_memory_type
        self.iterator_length = b.getRemainingImages(self.loader._handle)
        self.display = display
        self.batch_size = pipeline._batch_size
        if self.loader._is_external_source_operator:
            self.eos = False
            self.index = 0
            self.num_batches = self.loader._external_source.n // self.batch_size if self.loader._external_source.n % self.batch_size == 0 else (
                self.loader._external_source.n // self.batch_size + 1)
        else:
            self.num_batches = None
        if self.loader._name is None:
            self.loader._name = self.loader._reader
        self.last_batch_policy = self.loader._last_batch_policy
        self.last_batch_size = None

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
                    "roi_width": roi_width,
                    "roi_height": roi_height,
                    "decoded_width": decoded_width,
                    "decoded_height": decoded_height,
                    "channels": 3,
                    "external_source_mode": self.loader._external_source_mode,
                    "rocal_tensor_layout": types.NCHW,
                    "eos": self.eos}
                b.externalSourceFeedInput(*(kwargs_pybind.values()))
            self.index = self.index + 1
        if self.loader.rocal_run() != 0:
            b.rocalResetLoaders(self.loader._handle)
            raise StopIteration
        else:
            self.output_tensor_list = self.loader.get_output_tensors()

        if self.output_list is None:
            # Output list used to store pipeline outputs - can support multiple augmentation outputs
            self.output_list = []
            for i in range(len(self.output_tensor_list)):
                dimensions = self.output_tensor_list[i].dimensions()
                if self.device == "cpu":
                    torch_dtype = self.output_tensor_list[i].dtype()
                    output = torch.empty(
                        dimensions, dtype=getattr(torch, torch_dtype))
                    self.labels_tensor = torch.empty(
                        self.labels_size, dtype=getattr(torch, "int32"))
                else:
                    torch_gpu_device = torch.device('cuda', self.device_id)
                    torch_dtype = self.output_tensor_list[i].dtype()
                    output = torch.empty(dimensions, dtype=getattr(
                        torch, torch_dtype), device=torch_gpu_device)
                    self.labels_tensor = torch.empty(self.labels_size, dtype=getattr(
                        torch, "int32"), device=torch_gpu_device)

                self.output_tensor_list[i].copy_data(ctypes.c_void_p(
                    output.data_ptr()), self.output_memory_type)
                self.output_list.append(output)
        else:
            for i in range(len(self.output_tensor_list)):
                self.output_tensor_list[i].copy_data(ctypes.c_void_p(
                    self.output_list[i].data_ptr()), self.output_memory_type)

        if ((self.loader._name == "Caffe2ReaderDetection") or (self.loader._name == "CaffeReaderDetection")):
            self.bbox_list = []  # Empty list for bboxes
            self.labels_list = []  # Empty list of labels

            # 1D labels array in a batch
            self.labels = self.loader.get_bounding_box_labels()
            # 1D bboxes array in a batch
            self.bboxes = self.loader.get_bounding_box_cords()
            # Image sizes of a batch
            self.img_size = np.zeros((self.batch_size * 2), dtype="int32")
            self.loader.get_img_sizes(self.img_size)

            for i in range(self.batch_size):
                self.label_2d_numpy = np.reshape(
                    self.labels[i], (-1, 1)).tolist()
                self.bb_2d_numpy = np.reshape(self.bboxes[i], (-1, 4)).tolist()

                self.labels_list.append(self.label_2d_numpy)
                self.bbox_list.append(self.bb_2d_numpy)

                if self.display:
                    for output in self.output_list:
                        img = output
                        draw_patches(img[i], i, self.bb_2d_numpy)

            max_cols = max([len(row)
                           for batch in self.bbox_list for row in batch])
            max_rows = max([len(batch) for batch in self.bbox_list])
            self.bb_padded = [
                batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in self.bbox_list]
            self.bb_padded = torch.FloatTensor(
                [row + [0] * (max_cols - len(row)) for batch in self.bb_padded for row in batch])
            self.bb_padded = self.bb_padded.view(-1, max_rows, max_cols)

            max_cols1 = max([len(row)
                            for batch in self.labels_list for row in batch])
            max_rows1 = max([len(batch) for batch in self.labels_list])
            self.labels_padded = [
                batch + [[0] * (max_cols1)] * (max_rows1 - len(batch)) for batch in self.labels_list]
            self.labels_padded = torch.LongTensor(
                [row + [0] * (max_cols1 - len(row)) for batch in self.labels_padded for row in batch])
            self.labels_padded = self.labels_padded.view(
                -1, max_rows1, max_cols1)

            # Check if last batch policy is partial and only return the valid images in last batch
            if (self.last_batch_policy is (types.LAST_BATCH_PARTIAL)) and b.getRemainingImages(self.loader._handle) <= 0:
                if (self.last_batch_size is None):
                    self.last_batch_size = self.batch_size - \
                        b.getLastBatchPaddedSize(self.loader._handle)
                return [inner_list[0:self.last_batch_size, :] for inner_list in self.output_list], self.bb_padded[0:self.last_batch_size], self.labels_padded[0:self.last_batch_size]
            else:
                return self.output_list, self.bb_padded, self.labels_padded

        elif self.loader._is_external_source_operator:
            self.labels = self.loader.get_image_labels()
            self.labels_tensor = self.labels_tensor.copy_(
                torch.from_numpy(self.labels)).long()
            return self.output_list, self.labels_tensor
        else:
            if self.loader._one_hot_encoding:
                self.loader.get_one_hot_encoded_labels(
                    self.labels_tensor.data_ptr(), self.loader._output_memory_type)
                self.labels_tensor = self.labels_tensor.reshape(
                    -1, self.batch_size, self.loader._num_classes)
            else:
                if self.display:
                    for i in range(self.batch_size):
                        img = (self.output_list[0])
                        draw_patches(img[i], i, [])
                if (self.loader._name == "WebDataset"):
                    self.ascii_outputs = self.loader.get_ascii_datas()
                else:
                    self.labels = self.loader.get_image_labels()
                    self.labels_tensor = self.labels_tensor.copy_(
                        torch.from_numpy(self.labels)).long()

            # Check if last batch policy is partial and only return the valid images in last batch
            if (self.last_batch_policy is (types.LAST_BATCH_PARTIAL)) and b.getRemainingImages(self.loader._handle) < self.batch_size:
                if (self.last_batch_size is None):
                    self.last_batch_size = self.batch_size - \
                        b.getLastBatchPaddedSize(self.loader._handle)
                if (self.loader._name == "WebDataset"):
                    return [inner_list[0:self.last_batch_size, :] for inner_list in self.output_list], [inner_list[0:self.last_batch_size] for inner_list in self.ascii_outputs]
                else:
                    return [inner_list[0:self.last_batch_size, :] for inner_list in self.output_list], self.labels_tensor[0:self.last_batch_size]
            else:
                return self.output_list, self.ascii_outputs if (self.loader._name == "WebDataset") else self.labels_tensor

    def reset(self):
        b.rocalResetLoaders(self.loader._handle)

    def __iter__(self):
        return self

    def __len__(self):
        return self.iterator_length // self.batch_size

    def __del__(self):
        b.rocalRelease(self.loader._handle)


class ROCALClassificationIterator(ROCALGenericIterator):
    """!rocAL iterator for classification tasks for PyTorch. It returns 2 outputs
    (data and label) in the form of PyTorch's Tensors.

    Calling

    .. code-block:: python

       ROCALClassificationIterator(pipelines, size)

    is equivalent to calling

    .. code-block:: python

       ROCALGenericIterator(pipelines, ["data", "label"], size)

    Please keep in mind that Tensors returned by the iterator are
    still owned by rocAL. They are valid till the next iterator call.
    If the content needs to be preserved please copy it to another tensor.

    pipelines (list of amd.rocal.pipeline.Pipeline)       List of pipelines to use
    size (int)                                            Number of samples in the epoch (Usually the size of the dataset).
    auto_reset (bool, optional, default = False)          Whether the iterator resets itself for the next epoch or it requires reset() to be called separately.
    fill_last_batch (bool, optional, default = True)      Whether to fill the last batch with data up to 'self.batch_size'. The iterator would return the first integer multiple of self._num_gpus * self.batch_size entries which exceeds 'size'. Setting this flag to False will cause the iterator to return exactly 'size' entries.
    dynamic_shape (bool, optional, default = False)       Whether the shape of the output of the rocAL pipeline can change during execution. If True, the pytorch tensor will be resized accordingly if the shape of rocAL returned tensors changes during execution. If False, the iterator will fail in case of change.
    last_batch_padded (bool, optional, default = False)   Whether the last batch provided by rocAL is padded with the last sample or it just wraps up. In the conjunction with fill_last_batch it tells if the iterator returning last batch with data only partially filled with data from the current epoch is dropping padding samples or samples from the next epoch. If set to False next epoch will end sooner as data from it was consumed but dropped. If set to True next epoch would be the same length as the first one.

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


class ROCALAudioIterator(object):
    """! rocAL iterator for audio tasks for PyTorch
    The Tensors that are returned by the iterator will be owned by rocAL and would be valid until next iteration.
        @param pipeline            The rocAL pipeline to use for processing data.
        @param tensor_dtype        Data type of the output tensors.
        @param size                Number of samples in the epoch (Usually the size of the dataset).
        @param auto_reset          Whether the iterator resets itself for the next epoch or it requires reset() to be called separately.
        @param device              The device to use for processing - CPU / GPU
        @param device_id           The ID of the device to use
    """

    def __init__(self, pipeline, tensor_dtype=types.FLOAT, size=-1, auto_reset=False, device="cpu", device_id=0):
        self.loader = pipeline
        self.device = device
        self.device_id = device_id
        self.output = None
        # To change the name of getRemainingImages to getRemainingSamples in upcoming PRs
        self.iterator_length = b.getRemainingImages(self.loader._handle)
        self.max_shape = None
        self.batch_size = self.loader._batch_size
        self.output_list = None
        self.labels_size = self.batch_size
        self.output_memory_type = self.loader._output_memory_type
        self.last_batch_padded_size = b.getLastBatchPaddedSize(
            self.loader._handle)
        self.last_batch_policy = self.loader._last_batch_policy
        self.shard_size = size
        self.auto_reset = auto_reset
        self.batch_count = 0

    def next(self):
        return self.__next__()

    def __next__(self):
        rocal_run = self.loader.rocal_run()
        if (rocal_run != 0 or self.shard_size > 0 and self.batch_count >= self.shard_size):
            if self.auto_reset:
                self.reset()
            raise StopIteration
        else:
            self.output_tensor_list = self.loader.get_output_tensors()
        self.batch_count += self.batch_size
        self.last_batch_size = self.batch_size - \
            b.getLastBatchPaddedSize(self.loader._handle)
        self.output_list = []
        for i in range(len(self.output_tensor_list)):
            dimensions = self.output_tensor_list[i].dimensions()
            self.num_roi_dims = self.output_tensor_list[i].roi_dims_size()
            self.num_of_rois = self.num_roi_dims * 2
            self.roi_array = np.zeros(
                self.batch_size * self.num_of_rois, dtype=np.int32)
            self.output_tensor_list[i].copy_roi(self.roi_array)
            roi = self.roi_array.reshape(self.batch_size, self.num_of_rois)
            max_x1 = np.max(roi[..., self.num_of_rois-2:self.num_of_rois-1])
            max_y1 = np.max(roi[..., self.num_of_rois-1:self.num_of_rois])
            if self.device == "cpu":
                torch_dtype = self.output_tensor_list[i].dtype()
                if (max_x1 == 0 or max_y1 == 0):
                    output = torch.empty(
                        dimensions, dtype=getattr(torch, torch_dtype))
                else:
                    output = torch.empty(
                        dimensions[0], max_x1, max_y1, dtype=getattr(torch, torch_dtype))
                self.labels_tensor = torch.empty(
                    self.labels_size, dtype=getattr(torch, torch_dtype))
            else:
                torch_gpu_device = torch.device('cuda', self.device_id)
                torch_dtype = self.output_tensor_list[i].dtype()
                if (max_x1 == 0 or max_y1 == 0):
                    output = torch.empty(dimensions, dtype=getattr(
                        torch, torch_dtype), device=torch_gpu_device)
                else:
                    output = torch.empty(
                        dimensions[0], max_x1, max_y1, dtype=getattr(torch, torch_dtype))
                self.labels_tensor = torch.empty(self.labels_size, dtype=getattr(
                    torch, torch_dtype), device=torch_gpu_device)

            if (max_x1 == 0 or max_y1 == 0):
                self.output_tensor_list[i].copy_data(ctypes.c_void_p(
                    output.data_ptr()), self.output_memory_type)
            else:
                self.output_tensor_list[i].copy_data(
                    ctypes.c_void_p(output.data_ptr()), 0, 0, max_y1, max_x1)
            self.output_list.append(output)

        self.labels = self.loader.get_image_labels()
        self.labels_tensor = self.labels_tensor.copy_(
            torch.from_numpy(self.labels)).long()

        if (self.last_batch_policy is (types.LAST_BATCH_PARTIAL)) and b.getRemainingImages(self.loader._handle) < self.batch_size:
            return [inner_list[0:self.last_batch_size, :] for inner_list in self.output_list], self.labels_tensor[0:self.last_batch_size], torch.from_numpy(self.roi_array.reshape(self.batch_size, self.num_of_rois)[..., self.num_of_rois-2:self.num_of_rois][0:self.last_batch_size, :])
        else:
            return self.output_list, self.labels_tensor, torch.from_numpy(self.roi_array.reshape(self.batch_size, self.num_of_rois)[..., self.num_of_rois-2:self.num_of_rois])

    def reset(self):
        self.batch_count = 0
        b.rocalResetLoaders(self.loader._handle)

    def __iter__(self):
        return self

    def __len__(self):
        return self.iterator_length

    def __del__(self):
        b.rocalRelease(self.loader._handle)


class ROCALNumpyIterator(object):
    def __init__(self, pipeline, device="cpu", device_id=0):
        self.loader = pipeline
        self.device = device
        self.device_id = device_id
        self.output_memory_type = self.loader._output_memory_type
        self.output_list = None
        self.batch_size = self.loader._batch_size
        self.len = b.getRemainingImages(self.loader._handle) // self.batch_size
        self.last_batch_policy = self.loader._last_batch_policy
        self.last_batch_size = None

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
                dimensions = self.output_tensor_list[i].dimensions()
                torch_dtype = self.output_tensor_list[i].dtype()
                if self.device == "cpu":
                    output = torch.empty(
                        dimensions, dtype=getattr(torch, torch_dtype))
                else:
                    torch_gpu_device = torch.device('cuda', self.device_id)
                    output = torch.empty(dimensions, dtype=getattr(
                        torch, torch_dtype), device=torch_gpu_device)

                self.output_tensor_list[i].copy_data(ctypes.c_void_p(
                    output.data_ptr()), self.output_memory_type)
                self.output_list.append(output)
        else:
            for i in range(len(self.output_tensor_list)):
                self.output_tensor_list[i].copy_data(ctypes.c_void_p(
                    self.output_list[i].data_ptr()), self.output_memory_type)
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
        return self.len

    def __del__(self):
        b.rocalRelease(self.loader._handle)

def draw_patches(img, idx, bboxes):
    """!Writes images to disk as a PNG file.

        @param img       The input image as a tensor.
        @param idx       Index used for naming the output file.
        @param bboxes    List of bounding boxes.
    """
    # image is expected as a tensor
    import cv2
    img = img.cpu()
    image = img.detach().numpy()
    image = image.transpose([1, 2, 0])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.UMat(image).get()
    cv2.imwrite(str(idx)+"_"+"train"+".png", image)
    try:
        path = "OUTPUT_IMAGES_PYTHON/PYTORCH/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
    except OSError as error:
        print(error)
    if bboxes:
        for (l, t, r, b) in bboxes:
            loc_ = [l, t, r, b]
            color = (255, 0, 0)
            thickness = 2
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.UMat(image).get()
            image = cv2.rectangle(image, (int(loc_[0]), int(loc_[1])), (int(
                (loc_[2])), int((loc_[3]))), color, thickness)
            cv2.imwrite("OUTPUT_IMAGES_PYTHON/PYTORCH/" +
                        str(idx) + "_" + "train" + ".png", image * 255)
    else:
        cv2.imwrite("OUTPUT_IMAGES_PYTHON/PYTORCH/" + str(idx) +
                    "_" + "train" + ".png", image * 255)
