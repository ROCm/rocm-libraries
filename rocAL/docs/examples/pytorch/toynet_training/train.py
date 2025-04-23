################################################################################
#
# MIT License
#
# Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import sys
from amd.rocal.plugin.pytorch import ROCALClassificationIterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

def trainPipeline(data_path, batch_size, num_classes, one_hot, local_rank, world_size, num_thread, crop, rocal_cpu, fp16):
    pipe = Pipeline(batch_size=batch_size, num_threads=num_thread, device_id=local_rank, seed=local_rank+10, 
                rocal_cpu=rocal_cpu, tensor_dtype = types.FLOAT16 if fp16 else types.FLOAT, tensor_layout=types.NCHW, 
                prefetch_queue_depth = 7)
    with pipe:
        jpegs, labels = fn.readers.file(file_root=data_path, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        rocal_device = 'cpu' if rocal_cpu else 'gpu'
        # decode = fn.decoders.image(jpegs, output_type=types.RGB,file_root=data_path, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        decode = fn.decoders.image_slice(jpegs, output_type=types.RGB,
                                                    file_root=data_path, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        res = fn.resize(decode, resize_x=224, resize_y=224)
        flip_coin = fn.random.coin_flip(probability=0.5)
        cmnp = fn.crop_mirror_normalize(res, device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            mirror=flip_coin,
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        if(one_hot):
            _ = fn.one_hot(labels, num_classes)
        pipe.set_outputs(cmnp)
    print('rocal "{0}" variant'.format(rocal_device))
    return pipe

class trainLoader():
    def __init__(self, data_path, batch_size, num_thread, crop, rocal_cpu):
        super(trainLoader, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_thread = num_thread
        self.crop = crop
        self.rocal_cpu = rocal_cpu
        self.num_classes = 1000
        self.one_hot = 0.0
        self.local_rank = 0
        self.world_size = 1
        self.fp16 = True

    def get_pytorch_train_loader(self):
        print("in get_pytorch_train_loader function")   
        pipe_train = trainPipeline(self.data_path, self.batch_size, self.num_classes, self.one_hot, self.local_rank, 
                                    self.world_size, self.num_thread, self.crop, self.rocal_cpu, self.fp16)
        pipe_train.build()
        train_loader = ROCALClassificationIterator(pipe_train, device="cpu" if self.rocal_cpu else "cuda", device_id = self.local_rank)
        if self.rocal_cpu:
            return PrefetchedWrapper_rocal(train_loader, self.rocal_cpu) ,len(train_loader)
        else:
            return train_loader , len(train_loader)

class PrefetchedWrapper_rocal(object):
    def prefetched_loader(loader, rocal_cpu):

        stream = torch.cuda.Stream()
        first = True
        input = None
        target = None
        for next_input, next_target in loader:
            with torch.cuda.stream(stream):
                if rocal_cpu:
                    next_input = next_input.cuda(non_blocking=True)
                    next_target = next_target.cuda(non_blocking=True)

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __init__(self, dataloader, rocal_cpu):
        self.dataloader = dataloader
        self.epoch = 0
        self.rocal_cpu = rocal_cpu

    def reset(self):
        self.dataloader.reset()

    def __iter__(self):
        self.epoch += 1
        return PrefetchedWrapper_rocal.prefetched_loader(self.dataloader, self.rocal_cpu)

class ToyNet(nn.Module):
    def __init__(self,num_classes):
        super(ToyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 64, 3)
        self.conv4 = nn.Conv2d(64, 256, 3)
        self.fc0 = nn.Linear(256 * 11*11, 2048)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes) # Two classes only
        self.m = nn.Softmax()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 256 * 11 *11)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    if  len(sys.argv) < 4:
        print ('Please pass image_folder cpu/gpu batch_size')
        exit(0)
    if(sys.argv[2] == "cpu"):
        rocal_cpu = True
    else:
        rocal_cpu = False
    bs = int(sys.argv[3])
    nt = 1
    crop_size = 224
    device="cpu" if rocal_cpu else "cuda"
    image_path = sys.argv[1]
    dataset_train = image_path + '/train'
    num_classes = len(next(os.walk(image_path))[1])
    print("num_classes:: ",num_classes)
    
    net = ToyNet(num_classes)
    net.to(device)

    #train loader
    train_loader_obj = trainLoader(dataset_train, batch_size=bs, num_thread=nt, crop=crop_size, rocal_cpu=rocal_cpu)
    train_loader, train_loader_len = train_loader_obj.get_pytorch_train_loader()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)

    # Training loop
    for epoch in range(10):  # loop over the dataset multiple times
        print("\n epoch:: ",epoch)
        running_loss = 0.0

        for i, (inputs,labels) in enumerate(train_loader, 0):

            sys.stdout.write("\r Mini-batch " + str(i))
            # print("Images",inputs)
            # print("Labels",labels)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print_interval = 10
            if i % print_interval == (print_interval-1):
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / print_interval))
                running_loss = 0.0
        train_loader.reset()

    print('Finished Training')


if __name__ == '__main__':
    main()
