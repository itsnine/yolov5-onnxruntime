cmake_minimum_required(VERSION 3.20.0)
project(yolo_ort)

option(ORT_INSTALL_DIR "Path to built ONNX Runtime directory." STRING)
option(ENABLE_OCL  "enable opencl ep" OFF)
message(STATUS "ORT_INSTALL_DIR: ${ORT_INSTALL_DIR}")

find_package(OpenCV REQUIRED)

include_directories("include/")

add_executable(yolo_ort
               src/main.cpp
               src/detector.cpp
               src/utils.cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_include_directories(yolo_ort PRIVATE "${ORT_INSTALL_DIR}/include/onnxruntime/")
target_compile_features(yolo_ort PRIVATE cxx_std_14)

if (ENABLE_OCL)
    target_compile_definitions(yolo_ort PUBLIC ENABLE_OCL)
endif(ENABLE_OCL)

if (UNIX)
    target_link_libraries(yolo_ort "${ORT_INSTALL_DIR}/lib/libonnxruntime.so"
        ${OpenCV_LIBS}
    )
endif(UNIX)

