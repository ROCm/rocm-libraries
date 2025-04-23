
import sys
from amd.rocal.pipeline import pipeline_def
from amd.rocal.plugin.generic import ROCALClassificationIterator
import amd.rocal.fn as fn
import amd.rocal.types as types
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

seed = 1549361629
image_dir = "/opt/rocm/share/rocal/test/data/images/AMD-tinyDataSet"
batch_size = 4
gpu_id = 0

def show_images(image_batch):
    columns = 4
    rows = (batch_size + 1) // (columns)
    #fig = plt.figure(figsize = (32,(32 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows*columns):
        #print('\n Display image: ', j)
        plt.subplot(gs[j])
        img = image_batch[j]
        plt.axis("off")
        plt.imshow(img)
    plt.show()


def show_pipeline_output(pipe, device):
    pipe.build()
    data_loader = ROCALClassificationIterator(pipe, device=device)
    images = next(iter(data_loader))
    show_images(images[0][0])

@pipeline_def(seed=seed)
def image_decoder_pipeline(device="cpu", path=image_dir):
    jpegs, labels = fn.readers.file(file_root=path)
    images = fn.decoders.image(jpegs, file_root=path, device=device, output_type=types.RGB, shard_id=0, num_shards=1, random_shuffle=False)
    return fn.resize(images, device=device, resize_width=300, resize_height=300)

def main():
    print ('Optional arguments: <cpu/gpu image_folder>')
    bs = batch_size
    rocal_device = "cpu"
    rocal_cpu = True
    img_folder = image_dir
    if  len(sys.argv) > 1:
      if(sys.argv[1] == "gpu"):
          rocal_device = "gpu"
          rocal_cpu = False
    if  len(sys.argv) > 2:
      img_folder = sys.argv[2]
    pipe = image_decoder_pipeline(batch_size=bs, num_threads=1, device_id=gpu_id, rocal_cpu=rocal_cpu, tensor_layout=types.NHWC, 
                                reverse_channels=True, mean = [0, 0, 0], std=[255,255,255], device=rocal_device, path=img_folder)
    show_pipeline_output(pipe, device=rocal_device)

if __name__ == '__main__':
    main()
