import os
from PIL import Image

os.system('(mkdir data; cd data; wget https://www.ptgui.com/images/vptut/viewpointcorrection.zip)')
os.system('(cd data; unzip viewpointcorrection.zip)')
os.system('(cp data/step7.pts calibration.pts)')
Image.open("data/img_2619.jpg").save('cam00.bmp')
Image.open("data/img_2620.jpg").save('cam01.bmp')
Image.open("data/img_2621.jpg").save('cam02.bmp')
Image.open("data/img_2622.jpg").save('cam03.bmp')
Image.open("data/nadir.jpg").save('cam04.bmp')
