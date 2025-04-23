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

import torch
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import numpy as np
from parse_config import parse_args


class ROCALVideoIterator(object):
    """
    ROCALVideoIterator for pyTorch.

    Parameters
    ----------
    pipelines : list of amd.rocal.pipeline.Pipeline
                List of pipelines to use
    size : int
           Epoch size.
    """

    def __init__(self, pipelines, tensor_layout=types.NCHW, reverse_channels=False, multiplier=None, offset=None, tensor_dtype=types.FLOAT, display=False, sequence_length=3):

        try:
            assert pipelines is not None, "Number of provided pipelines has to be at least 1"
        except Exception as ex:
            print(ex)

        self.loader = pipelines
        self.tensor_format = tensor_layout
        self.multiplier = multiplier if multiplier else [1.0, 1.0, 1.0]
        self.offset = offset if offset else [0.0, 0.0, 0.0]
        self.reverse_channels = reverse_channels
        self.tensor_dtype = tensor_dtype
        self.batch_size = self.loader._batch_size
        self.rim = self.loader.get_remaining_images()
        self.display = display
        self.iter_num = 0
        self.sequence_length = sequence_length
        print("____________REMAINING IMAGES____________:", self.rim)
        self.output = self.dimensions = self.dtype = None

    def next(self):
        return self.__next__()

    def __next__(self):
        if (self.loader.is_empty()):
            raise StopIteration

        if self.loader.rocal_run() != 0:
            raise StopIteration
        self.output_tensor_list = self.loader.get_output_tensors()
        self.iter_num += 1
        # Copy output from buffer to numpy array
        if self.output is None:
            self.dimensions = self.output_tensor_list[0].dimensions()
            self.dtype = self.output_tensor_list[0].dtype()
            self.layout = self.output_tensor_list[0].layout()
            self.output = np.empty(
                (self.dimensions[0]*self.dimensions[1], self.dimensions[2], self.dimensions[3], self.dimensions[4]), dtype=self.dtype)
        self.output_tensor_list[0].copy_data(self.output)
        img = torch.from_numpy(self.output)
        # Display Frames in a video sequence
        if self.display:
            for batch_i in range(self.batch_size):
                draw_frames(img[batch_i], batch_i, self.iter_num, self.layout)
        return img

    def reset(self):
        self.loader.rocal_reset_loaders()

    def __iter__(self):
        return self

    def __del__(self):
        self.loader.rocal_release()


def draw_frames(img, batch_idx, iter_idx, layout):
    # image is expected as a tensor, bboxes as numpy
    import cv2
    image = img.detach().numpy()
    if layout == 'NFCHW':
        image = image.transpose([1, 2, 0])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    import os
    if not os.path.exists("output_folder/video_reader"):
        os.makedirs("output_folder/video_reader")
    image = cv2.UMat(image).get()
    cv2.imwrite("output_folder/video_reader/" +
                "iter_"+str(iter_idx)+"_batch_"+str(batch_idx)+".png", image)


def main():
    # Args
    args = parse_args()
    video_path = args.video_path
    rocal_cpu = False if args.rocal_gpu else True
    batch_size = args.batch_size
    user_sequence_length = args.sequence_length
    display = args.display
    num_threads = args.num_threads
    random_seed = args.seed
    tensor_format = types.NFHWC if args.NHWC else types.NFCHW
    tensor_dtype = types.FLOAT16 if args.fp16 else types.FLOAT
    # Create Pipeline instance
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=args.local_rank, seed=random_seed, rocal_cpu=rocal_cpu,
                    tensor_layout=tensor_format, tensor_dtype=tensor_dtype)
    # Use pipeline instance to make calls to reader, decoder & augmentation's
    with pipe:
        images = fn.readers.video(file_root=video_path, sequence_length=user_sequence_length,
                                  random_shuffle=False, image_type=types.RGB)
        crop_size = (512, 960)
        output_images = fn.crop_mirror_normalize(images,
                                                 output_layout=tensor_format,
                                                 output_dtype=tensor_dtype,
                                                 crop=crop_size,
                                                 mean=[0, 0, 0],
                                                 std=[1, 1, 1])
        pipe.set_outputs(output_images)
    # Build the pipeline
    pipe.build()
    # Dataloader
    data_loader = ROCALVideoIterator(pipe, multiplier=pipe._multiplier,
                                     offset=pipe._offset, display=display, sequence_length=user_sequence_length)
    import timeit
    start = timeit.default_timer()
    # Enumerate over the Dataloader
    for epoch in range(int(args.num_epochs)):
        print("EPOCH:::::", epoch)
        for i, it in enumerate(data_loader, 0):
            if args.print_tensor:
                print("**************", i, "*******************")
                print("**************starts*******************")
                print("\n IMAGES : \n", it)
                print("**************ends*******************")
                print("**************", i, "*******************")
        data_loader.reset()
    # Your statements here
    stop = timeit.default_timer()

    print('\n Time: ', stop - start)
    print("##############################  VIDEO READER  SUCCESS  ############################")


if __name__ == '__main__':
    main()
