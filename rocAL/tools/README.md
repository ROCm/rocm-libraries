# Steps to create MXNet RecordIO files using MXNet's im2rec.py script

## MXNet Installation

```shell
pip3 install mxnet
```

## Step1 : to create .lst file

```shell
python3 im2rec.py --list test Dataset_path --recursive
```

test - name of your .lst file

Dataset_path - path to the list of image folders

--recursive - If set recursively walk through subdirs and assign an unique label to images in each folder. Otherwise only include images in the root folder and give them label 0

## Step2 : to create RecordIO files

```shell
python3 im2rec.py lst_file Dataset_path
```

lst_file - *.lst file created using Step1

Dataset_path - path to the list of image folders

# Steps to create index file for webdataset tar files

```shell
python3 tar2idx.py tar_file_path
```

tar_file_path - path to the folder containing tar files
