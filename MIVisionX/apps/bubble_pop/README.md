[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

<p align="center"><img width="50%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/OpenVX_logo.png" /></p>

# OpenVX Samples

<a href="https://www.khronos.org/openvx/" target="_blank">Khronos OpenVX&trade;</a> is an open, royalty-free standard for cross-platform acceleration of computer vision applications. OpenVX enables a performance and power-optimized computer vision processing, especially important in embedded and real-time use cases such as face, body, and gesture tracking, smart video surveillance, advanced driver assistance systems (ADAS), object and scene reconstruction, augmented reality, visual inspection, robotics and more.

In this project, we provide OpenVX sample applications to use with any conformant implementation of OpenVX.

## VX Bubble Pop Sample

In this sample we will create an OpenVX graph to run VX Bubble Pop on a live camera. This sample application uses <a href="https://en.wikipedia.org/wiki/OpenCV" target="_blank">OpenCV</a> to decode input image, draw bubbles/donuts and display the output.

 <p align="center"><img width="60%" src="https://raw.githubusercontent.com/ROCm/MIVisionX/master/docs/data/vx-pop-app.gif" /></p>

### Prerequisites

* [OpenCV](https://github.com/opencv/opencv/releases/tag/3.4.0)

* Camera

* Build and install [MIVisionX](https://github.com/ROCm/MIVisionX#build--install-mivisionx) 

### Steps to run the Bubble Pop sample

* **Step - 1:** Clone the MIVisionX project

``` 
cd ~/ && mkdir OpenVX-bubble-pop
cd OpenVX-bubble-pop/
git clone https://github.com/ROCm/MIVisionX
```

* **Step - 2:** CMake and Build the pop application

``` 
mkdir pop-build && cd pop-build
cmake ../MIVisionX/apps/bubble_pop/
make
```

* **Step - 3:** Run VX Pop application

 - **Bubbles**

 ``` 
 ./vxPop --bubble
 ```

 - **Donuts**

 ``` 
 ./vxPop --donut
 ````
