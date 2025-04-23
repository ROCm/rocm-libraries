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

from amd.rocal.plugin.generic import ROCALClassificationIterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import sys
import datetime
import time

def HybridTrainPipe(batch_size, num_threads, device_id, data_dir, rocal_cpu=True, prefetch_queue_depth=2):
    world_size = 1
    local_rank = 0
    resize_width = 300
    resize_height = 300
    decoder_device = 'cpu'  # hardcoding decoder_device to cpu until VCN can decode all JPEGs
    # Create Pipeline instance
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                    rocal_cpu=rocal_cpu, prefetch_queue_depth=prefetch_queue_depth)
    with pipe:
        jpegs, _ = fn.readers.file(file_root=data_dir)
        images = fn.decoders.image(jpegs, file_root=data_dir, device=decoder_device,
                                   output_type=types.RGB, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        images = fn.resize(images, resize_width=resize_width,
                           resize_height=resize_height)
        output = fn.rain(images, rain=0.5)
        pipe.set_outputs(output)
    return pipe


def main():
    if len(sys.argv) < 5:
        print('Please pass image_folder cpu/gpu batch_size prefetch_queue_depth')
        exit(0)
    image_folder_path = sys.argv[1]
    if (sys.argv[2] == "cpu"):
        rocal_cpu = True
        device= "cpu"
    else:
        rocal_cpu = False
        device = "gpu"
    batch_size = int(sys.argv[3])
    prefetch_queue_depth = int(sys.argv[4])
    num_threads = 8
    device_id = 0
    pipe = HybridTrainPipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                           data_dir=image_folder_path, rocal_cpu=rocal_cpu, prefetch_queue_depth=prefetch_queue_depth)
    pipe.build()
    imageIterator = ROCALClassificationIterator(pipe,device=device)
    start = datetime.datetime.now()
    for _ in range(0, 10):
        for _ in imageIterator:
            time.sleep(1)
        imageIterator.reset()
    end = datetime.datetime.now()
    print("Time taken (averaged over 10 runs) ", int(
        (end - start).total_seconds() * 1000) / 10, "milli seconds")

if __name__ == '__main__':
    main()
