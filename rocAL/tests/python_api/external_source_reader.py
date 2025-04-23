from random import shuffle
from amd.rocal.pipeline import Pipeline
from amd.rocal.plugin.generic import ROCALClassificationIterator
import amd.rocal.fn as fn
import amd.rocal.types as types
import os
import numpy as np
import cv2
import sys
import glob


def main():
    if  len(sys.argv) < 3:
        print ('Please pass cpu/gpu batch_size')
        exit(0)
    batch_size = int(sys.argv[2])
    device = "cpu" if sys.argv[1] == "cpu" else "gpu"
    data_dir = os.environ["ROCAL_DATA_PATH"] + \
        "rocal_data/coco/coco_10_img/images/"
    try:
        path_mode0 = "output_folder/external_source_reader/mode0/"
        isExist = os.path.exists(path_mode0)
        if not isExist:
            os.makedirs(path_mode0)
    except OSError as error:
        print(error)
    try:
        path_mode1 = "output_folder/external_source_reader/mode1/"
        isExist = os.path.exists(path_mode1)
        if not isExist:
            os.makedirs(path_mode1)
    except OSError as error:
        print(error)
    try:
        path_mode2 = "output_folder/external_source_reader/mode2/"
        isExist = os.path.exists(path_mode2)
        if not isExist:
            os.makedirs(path_mode2)
    except OSError as error:
        print(error)

    def image_dump(img, idx, mode=0):
        img = img.transpose([1, 2, 0])  # NCHW
        img = (img).astype('uint8')
        if mode!=2:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite("output_folder/external_source_reader/mode" + str(mode) + "/"+
                    str(idx)+"_"+"train"+".png", img)

    ##################### MODE 0 #########################
    # Define the Data Source for all image samples - User needs to define their own source
    class ExternalInputIteratorMode0(object):
        def __init__(self, batch_size):
            self.images_dir = data_dir
            self.batch_size = batch_size
            self.files = []
            self.file_patterns = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']
            for pattern in self.file_patterns:
                self.files.extend(glob.glob(os.path.join(self.images_dir, pattern)))
            shuffle(self.files)

        def __iter__(self):
            self.i = 0
            self.n = len(self.files)
            return self

        def __next__(self):
            batch = []
            labels = []
            label = 1
            for _ in range(self.batch_size):
                jpeg_filename = self.files[self.i]
                batch.append(jpeg_filename)
                labels.append(label)
                label = label + 1
                self.i = (self.i + 1) % self.n
            labels = np.array(labels).astype('int32')
            return batch, labels

    # Mode 0
    external_input_source = ExternalInputIteratorMode0(batch_size)

    # Create the pipeline
    external_source_pipeline_mode0 = Pipeline(batch_size=batch_size, num_threads=1, device_id=0, prefetch_queue_depth=4,
                                              seed=1, rocal_cpu=True if device == "cpu" else False, tensor_layout=types.NCHW)

    with external_source_pipeline_mode0:
        jpegs, _ = fn.external_source(
            source=external_input_source, mode=types.EXTSOURCE_FNAME)
        output = fn.resize(jpegs, resize_width=300, resize_height=300,
                           output_layout=types.NCHW, output_dtype=types.UINT8)
        external_source_pipeline_mode0.set_outputs(output)

    # build the external_source_pipeline_mode0
    external_source_pipeline_mode0.build()
    # Index starting from 0
    cnt = 0
    # Dataloader
    data_loader = ROCALClassificationIterator(
        external_source_pipeline_mode0, device=device)
    for i, output_list in enumerate(data_loader, 0):
        print("**************MODE 0*******************")
        print("**************", i, "*******************")
        print("**************starts*******************")
        print("\nImages:\n", output_list)
        print("**************ends*******************")
        print("**************", i, "*******************")
        for img in output_list[0][0]:
            cnt = cnt + 1
            image_dump(img, cnt, mode=0)

    ##################### MODE 0 #########################
    
    ##################### MODE 1 #########################
        # Define the Data Source for all image samples
    class ExternalInputIteratorMode1(object):
        def __init__(self, batch_size):
            self.images_dir = data_dir
            self.batch_size = batch_size
            self.files = []
            import os
            self.file_patterns = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']
            for pattern in self.file_patterns:
                self.files.extend(glob.glob(os.path.join(self.images_dir, pattern)))

        def __iter__(self):
            self.i = 0
            self.n = len(self.files)
            self.maxWidth = None
            self.maxHeight = None
            return self

        def __next__(self):
            batch = []
            labels = []
            srcsize_height = []
            label = 1
            for x in range(self.batch_size):
                jpeg_filename = self.files[self.i]
                f = open(jpeg_filename, 'rb')
                numpy_buffer = np.frombuffer(f.read(), dtype=np.uint8)
                batch.append(numpy_buffer)
                srcsize_height.append(len(numpy_buffer))
                labels.append(label)
                label = label + 1
                self.i = (self.i + 1) % self.n
            labels = np.array(labels).astype('int32')
            return (batch, labels, srcsize_height)

# Mode 1
    eii_1 = ExternalInputIteratorMode1(batch_size)

    # Create the pipeline
    external_source_pipeline_mode1 = Pipeline(batch_size=batch_size, num_threads=1, device_id=0, prefetch_queue_depth=4,
                                              seed=1, rocal_cpu=True if device == "cpu" else False, tensor_layout=types.NCHW)

    with external_source_pipeline_mode1:
        jpegs, _ = fn.external_source(
            source=eii_1, mode=types.EXTSOURCE_RAW_COMPRESSED, max_width=2000, max_height=2000)
        output = fn.resize(jpegs, resize_width=2000, resize_height=2000,
                           output_layout=types.NCHW, output_dtype=types.UINT8)
        external_source_pipeline_mode1.set_outputs(output)

    # build the external_source_pipeline_mode1
    external_source_pipeline_mode1.build()
    # Index starting from 0
    cnt = 0
    # Dataloader
    data_loader = ROCALClassificationIterator(
        external_source_pipeline_mode1, device=device)
    for i, output_list in enumerate(data_loader, 0):
        print("**************MODE 1*******************")
        print("**************", i, "*******************")
        print("**************starts*******************")
        print("\nImages:\n", output_list)
        print("**************ends*******************")
        print("**************", i, "*******************")
        for img in output_list[0][0]:
            cnt = cnt + 1
            image_dump(img, cnt, mode=1)
    ##################### MODE 1 #########################
    
    ##################### MODE 2 #########################
        # Define the Data Source for all image samples

    class ExternalInputIteratorMode2(object):
        def __init__(self, batch_size):
            self.images_dir = data_dir
            self.batch_size = batch_size
            self.files = []
            self.maxHeight = self.maxWidth = 0
            import os
            self.file_patterns = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']
            for pattern in self.file_patterns:
                self.files.extend(glob.glob(os.path.join(self.images_dir, pattern)))
            shuffle(self.files)
            self.i = 0
            self.n = len(self.files)

            for x in range(self.n):
                jpeg_filename = self.files[x]
                label = 1
                image = cv2.imread(jpeg_filename, cv2.IMREAD_COLOR)
                # Check if the image was loaded successfully
                if image is None:
                    print("Error: Failed to load the image.")
                else:
                    # Get the height and width of the image
                    height, width = image.shape[:2]
                self.maxHeight = height if height > self.maxHeight else self.maxHeight
                self.maxWidth = width if width > self.maxWidth else self.maxWidth

        def __iter__(self):
            return self

        def __next__(self):
            batch = []
            batch_of_numpy = []
            labels = []
            label = 1
            roi_height = []
            roi_width = []
            self.out_image = np.zeros(
                (self.batch_size, self.maxHeight, self.maxWidth, 3), dtype="uint8")
            for x in range(self.batch_size):
                jpeg_filename = self.files[self.i]
                image = cv2.imread(jpeg_filename, cv2.IMREAD_COLOR)
                # Check if the image was loaded successfully
                if image is None:
                    print("Error: Failed to load the image.")
                else:
                    # Get the height and width of the image
                    height, width = image.shape[:2]
                batch.append(np.asarray(image))
                roi_height.append(height)
                roi_width.append(width)
                self.out_image[x][:roi_height[x], :roi_width[x], :] = batch[x]
                batch_of_numpy.append(self.out_image[x])
                labels.append(label)
                label = label + 1
                self.i = (self.i + 1) % self.n
            labels = np.array(labels).astype('int32')
            return (batch_of_numpy, labels, roi_height, roi_width, self.maxHeight, self.maxWidth)


# Mode 2
    eii_2 = ExternalInputIteratorMode2(batch_size)

    # Create the pipeline
    external_source_pipeline_mode2 = Pipeline(batch_size=batch_size, num_threads=1, device_id=0, prefetch_queue_depth=4,
                                              seed=1, rocal_cpu=True if device == "cpu" else False, tensor_layout=types.NCHW)

    with external_source_pipeline_mode2:
        jpegs, _ = fn.external_source(source=eii_2, mode=types.EXTSOURCE_RAW_UNCOMPRESSED,
                                      max_width=eii_2.maxWidth, max_height=eii_2.maxHeight)
        output = fn.resize(jpegs, resize_width=300, resize_height=300,
                           output_layout=types.NCHW, output_dtype=types.UINT8)
        external_source_pipeline_mode2.set_outputs(output)

    # build the external_source_pipeline_mode2
    external_source_pipeline_mode2.build()
    # Index starting from 0
    cnt = 0
    # Dataloader
    data_loader = ROCALClassificationIterator(
        external_source_pipeline_mode2, device=device)
    for i, output_list in enumerate(data_loader, 0):
        print("**************MODE 2*******************")
        print("**************", i, "*******************")
        print("**************starts*******************")
        print("\nImages:\n", output_list)
        print("**************ends*******************")
        print("**************", i, "*******************")
        for img in output_list[0][0]:
            cnt = cnt+1
            image_dump(img, cnt, mode=2)
    ##################### MODE 2 #########################
if __name__ == '__main__':
    main()
