# yolov5-onnxruntime

C++ YOLO v5 ONNX Runtime inference code for object detection.

## Dependecies:
- OpenCV 4.x
- ONNXRuntime 1.15+
- OS: Tested on centos8 archlinux
- CUDA 11+ [Optional]


## Build
To build the project you should run the following commands, don't forget to change `ORT_INSTALL_DIR` cmake option:

```bash
mkdir build
cd build
cmake .. -DORT_INSTALL_DIR=[path_to_onnxruntime_install_dir] -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build .
```

#### cmake options:  
- ENABLE_PERF  
Enabling this will output per-image inference elapsed time, as well as generate profile_filexxx.json performance statistics, which you can visualize in [Perfetto UI](https://www.ui.perfetto.dev/) 
- ENABLE_OCL  
Enable opencl ep probing  

example:
```
cmake .. -DORT_INSTALL_DIR=[path_to_onnxruntime_src] -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_PERF=ON -DENABLE_OCL=ON
```
## Run
Before running the executable you should convert your PyTorch model to ONNX if you haven't done it yet. Check the [official tutorial](https://github.com/ultralytics/yolov5/issues/251).

`On Windows`: to run the executable you should add OpenCV and ONNX Runtime libraries to your environment path `or` put all needed libraries near the executable (onnxruntime.dll and opencv_world.dll).

Run from CLI:
```bash
./yolo_ort --model_path ../models/yolov5s.onnx --class_names ../models/coco.names --image ../images/bus.jpg --gpu
# On Windows ./yolo_ort.exe with arguments as above
```

## Demo

YOLOv5m onnx:

<p align="center">
  <a href="images/bus_result.jpg"><img src="images/bus_result.jpg" style="width:60%; height:60%;"/></a>
</p>
<p align="center">
  <a href="images/zidane_result.jpg"><img src="images/zidane_result.jpg" style="width:60%; height:60%;"/></a>
</p>


## References

- YOLO v5 repo: https://github.com/ultralytics/yolov5
- YOLOv5 Runtime Stack repo: https://github.com/zhiqwang/yolov5-rt-stack
- ONNXRuntime Inference examples: https://github.com/microsoft/onnxruntime-inference-examples
