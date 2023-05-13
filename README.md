# Yolov8TensorrtRos

## Instruction for Applying Yolov8 on ROS with Tensorrt acceleration
### Get Yolov8 onnx model using [Ultralytics](https://github.com/ultralytics/ultralytics)
* Directly coverting from pt to engine may have some problems, so I choose to convert it to onnx, and then compile in C++.
* [Export the official pt model](https://docs.ultralytics.com/usage/cfg/#export)
  `yolo export model=yolov8s.pt format=onnx device=0 imgsz=640 simplify=True opset=13`  
  it seems on jetson you can also do
  `yolo mode=export model=yolov8s.pt format=onnx`, you don't need to put the device onto cuda when generating the onnx.
* My tensorrt 8.2.1 supports onnx 1.9.0, and the max opset is 14
* You may visualize the onnx model in [netron](https://netron.app/)
### Compile it in C++ node
* This part is modified based on [tensorrt-cpp-api](https://github.com/cyrusbehr/tensorrt-cpp-api/tree/main)
* What I have done it to add a postprocess of the output (Nx84x8400) of the YOLOv8 
  * Remove unnecessary classes, only keeps the 0-16
    0: person
    1: bicycle
    2: car
    3: motorcycle
    4: airplane
    5: bus
    6: train
    7: truck
    8: boat
    9: traffic light
    10: fire hydrant
    11: stop sign
    12: parking meter
    13: bench
    14: bird
    15: cat
    16: dog
  * The NMS for each class
  * Set the parameters in roslaunch
    * topics
    * conf_threshold
    * IOU_threshold
    * GPU_workspace (bytes)

## Current environment
Jetson Orin (Jetpack 5.0.2), CUDA, cuDNN, OPENCV are included


## Environment setup on Ubuntu 18.04
Using the Cuda 11.4.4, CuDNN 8.2.2, TensorRT 8.2.1, OpenCV 4.5.2
Mainly following the [Cuda Cudnn Opencv Install](https://medium.com/@pydoni/how-to-install-cuda-11-4-cudnn-8-2-opencv-4-5-on-ubuntu-20-04-65c4aa415a7b)

### First remove all the cuda and cudnn and nvidia driver
Make sure you also unclick the source of nvidia in software and updates, otherwise it will install the newest version of cuda
```
sudo apt-get --purge remove "*cublas*" "cuda*" "nsight*" "nvidia*"
sudo apt-get --purge remove "*cudnn*"
sudo apt-get --purge remove "*nvinfer*"
sudo apt-get --purge remove "*opencv*"
sudo apt-get --purge remove "*nvidia*"
sudo apt-get autoremove
sudo apt-get autoclean
```
### Install Cuda 11.4.4
Refer offical website [Cuda Toolkit Archive](https://developer.nvidia.com/cuda-11-4-4-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=deb_local)

Then add the path to .bashrc, open the .bashrc add the following lines
```
export PATH="/usr/local/cuda-11.4/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH"
```
Make sure you can run `nvcc -V` and `nvidia-smi` in terminal
### Install CuDNN 8.2.2
Download the cudnn file from [CuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)

Then cd to cudnn* folder and run the following command
```
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
sudo cp -P cuda/include/cudnn.h /usr/include
sudo cp -P cuda/lib64/libcudnn* /usr/lib/x86_64-linux-gnu/
sudo chmod a+r /usr/lib/x86_64-linux-gnu/libcudnn*
```
And Reboot

### Install TensorRT 8.2.1
Download the TensorRT 8.2.1 from [TensorRT Archive](https://developer.nvidia.com/nvidia-tensorrt-8x-download)

Then cd to TensorRT-
```
sudo dpkg -i nv-tensorrt-repo-ubuntu1804-cuda11.4-trt8.2.1.8-ga-20211117_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-ubuntu1804-cuda11.4-trt8.2.1.8-ga-20211117/7fa2af80.pub
sudo apt-get update
sudo apt-get install tensorrt
sudo apt-get install python3-libnvinfer-dev
```

### Install OpenCV 4.5.2
Install the dependencies
```
sudo apt install cmake pkg-config unzip yasm git checkinstall libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libavresample-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev libfaac-dev libmp3lame-dev libvorbis-dev libopencore-amrnb-dev libopencore-amrwb-dev
sudo apt-get install libdc1394-22 libdc1394-22-dev libxine2-dev libv4l-dev v4l-utils
cd /usr/include/linux
sudo ln -s -f ../libv4l1-videodev.h videodev.h
cd ~
sudo apt-get install libgtk-3-dev libtbb-dev libatlas-base-dev gfortran libopenblas-dev libblas-dev
sudo apt-get install libprotobuf-dev protobuf-compiler
sudo apt-get install libgoogle-glog-dev libgflags-dev
sudo apt-get install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen
```
Download the OpenCV 4.5.2 and contrib from [OpenCV Archive](https://opencv.org/releases/)

```
wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.5.2.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.5.2.zip
unzip opencv.zip
unzip opencv_contrib.zip
```

Then cd to opencv-4.5.2 and create a build folder
```
cd opencv-4.5.2
mkdir build
cd build
```
Then run the cmake, follow previous link to set the flags. For my case, the 3080ti has CUDA_ARCH_PTX to 8.6.
```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_CXX_FLAGS_RELEASE="-O3" \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=/home/minghao/Documents/Gits/CudaInstall/opencv_contrib-4.5.2/modules \
        -D BUILD_TIFF=ON \
        -D WITH_GSTREAMER=ON \
        -D WITH_TBB=ON \
        -D BUILD_TBB=ON \
        -D WITH_EIGEN=ON \
        -D WITH_V4L=ON \
        -D WITH_LIBV4L=ON \
        -D WITH_VTK=OFF \
        -D WITH_OPENGL=ON \
        -D WITH_OPENCL=ON \
        -D WITH_LAPACK=ON \
        -D BUILD_WEBP=OFF \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D INSTALL_C_EXAMPLES=OFF \
        -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
        -D PYTHON_INCLUDE_DIR=/usr/include/python3.6m \
        -D PYTHON_LIBRARY=/usr/lib/libpython3.6m.so \
        -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/lib/python3.6/site-packages/numpy/core/include \
        -D BUILD_OPENCV_PYTHON3=ON \
        -D INSTALL_PYTHON_EXAMPLES=ON \
        -D BUILD_NEW_PYTHON_SUPPORT=ON \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D BUILD_TESTS=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D WITH_CUDA=ON \
        -D ENABLE_FAST_MATH=ON \
        -D CUDA_FAST_MATH=ON \
        -D WITH_CUDNN=ON \
        -D OPENCV_DNN_CUDA=ON \
        -D CUDA_ARCH_BIN=5.3,6.0,6.1,7.0,7.5,8.6 \
        -D CUDA_ARCH_PTX=8.6 \
        -D WITH_CUBLAS=ON ..
```

cmake command for jetson orin, specify the opencl path and library
```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_CXX_FLAGS_RELEASE="-O3" \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=/home/jetson/Downloads/opencv/opencv_contrib-4.5.4/modules \
        -D OPENCL_INCLUDE_DIR=/usr/include \
        -D OPENCL_LIBRARY=/usr/lib/aarch64-linux-gnu/libOpenCL.so.1 \
        -D BUILD_TIFF=ON \
        -D WITH_GSTREAMER=ON \
        -D WITH_TBB=ON \
        -D BUILD_TBB=ON \
        -D WITH_EIGEN=ON \
        -D WITH_V4L=ON \
        -D WITH_LIBV4L=ON \
        -D WITH_VTK=OFF \
        -D WITH_OPENGL=ON \
        -D WITH_OPENCL=ON \
        -D WITH_LAPACK=ON \
        -D BUILD_WEBP=OFF \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D INSTALL_C_EXAMPLES=OFF \
        -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
        -D PYTHON_INCLUDE_DIR=/usr/include/python3.8 \
        -D PYTHON_LIBRARY=/usr/lib/libpython3.8.so \
        -D PYTHON3_NUMPY_INCLUDE_DIRS=/home/jetson/.local/lib/python3.8/site-packages/numpy/core/include \
        -D BUILD_OPENCV_PYTHON3=ON \
        -D INSTALL_PYTHON_EXAMPLES=ON \
        -D BUILD_NEW_PYTHON_SUPPORT=ON \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D BUILD_TESTS=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D WITH_CUDA=ON \
        -D ENABLE_FAST_MATH=ON \
        -D CUDA_FAST_MATH=ON \
        -D WITH_CUDNN=ON \
        -D OPENCV_DNN_CUDA=ON \
        -D CUDA_ARCH_BIN=5.3,6.0,6.1,7.0,7.5,8.7 \
        -D CUDA_ARCH_PTX=8.7 \
        -D WITH_CUBLAS=ON ..
```
Then run the make and install
```
make -j$(nproc)
sudo make install
sudo ldconfig
sudo apt-get update
pip install opencv-contrib-python
```

### Reinstall ROS packages previously removed that relies on OpenCV
```
sudo apt-get --reinstall install ros-melodic-image-transport
sudo apt-get --reinstall install ros-melodic-vision-msgs
sudo apt-get --reinstall install ros-melodic-image-publisher
sudo apt-get --reinstall install ros-melodic-image-view
```

Need some special treatment for cv_bridge and image_transport_plugins, we need to compile them from source, and link them to the OpenCV 4.5.2.
For cv_bridge:
```
cd ~/catkin_ws/src
git clone https://github.com/fizyr-forks/vision_opencv.git
cd vision_opencv
git checkout opencv4
cd ..
catkin_make
```
For ubuntu 20, you may clone the vision_opencv from official repo
```
git clone https://github.com/ros-perception/vision_opencv.git
cd vision_opencv
git checkout noetic
cd ..
catkin_make
```
For image_transport_plugins:
```
cd ~/catkin_ws/src
git clone https://github.com/ros-perception/image_transport_plugins.git
cd image_transport_plugins
git checkout melodic-devel
cd ..
catkin_make
```
add the source the corresponding devel setup.bash to the bashrc
