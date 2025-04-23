import csv
import os
import shutil
import sys
import warnings
from scipy.io import loadmat
import pandas as pd
import splitfolders

path = os.getcwd()
os.system('pip install split_folders')
if not os.path.exists("dataset_flower"):
    os.mkdir("dataset_flower")
os.chdir("dataset_flower")
os.system("wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz")
os.system("tar -xvf 102flowers.tgz")

os.chdir(path)
if not os.path.exists("Flower102"):
    os.mkdir("Flower102")

os.system("wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat")
# get the path/directory
imgs = []
folder_dir = "./dataset_flower/jpg/"
for image in os.listdir(folder_dir):
    imgs.append(image)
imgs.sort()

warnings.filterwarnings("ignore")
mat_labels = loadmat("./imagelabels.mat")
label = mat_labels["labels"]
label = label[0]
for i in range(len(label)):
    label[i] = label[i] - 1

if not os.path.exists("./Flower102/map.csv"):
    os.system("touch ./Flower102/map.csv")

fields = ["images", "label"]
with open("./Flower102/map.csv", "w") as csvfile:
    csvwriter = csv.writer(csvfile)
    # writing the fields
    csvwriter.writerow(fields)
    for i in range(len(imgs)):
        row = [imgs[i], label[i]]
        csvwriter.writerow(row)

labels_map = pd.read_csv(r"./Flower102/map.csv")
train_dir = r"./dataset_flower/jpg"  # source folder
dest_folder = r"./Flower102/flower/"  # destination folder
if not os.path.exists(dest_folder):
    os.mkdir(dest_folder)

for filename, class_name in labels_map.values:
    # Create subdirectory with `class_name`
    if int(class_name) >= 50:
        continue
    else:
        if not os.path.exists(dest_folder + str(class_name)):
            os.mkdir(dest_folder + str(class_name))
        src_path = train_dir + "/" + filename
        dst_path = dest_folder + str(class_name) + "/" + filename
        try:
            shutil.copy(src_path, dst_path)
            print("sucessfull")
        except IOError as e:
            print("Unable to copy file {} to {}".format(src_path, dst_path))
        except:
            print(
                "When try copy file {} to {}, unexpected error: {}".format(
                    src_path, dst_path, sys.exc_info()
                )
            )

input_folder = "./Flower102/flower"
splitfolders.ratio(
    input_folder,
    output="./Flower102/split_data",
    seed=42,
    ratio=(0.7, 0.2, 0.1),
    group_prefix=None,
)
