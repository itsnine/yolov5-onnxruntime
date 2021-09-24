cmake_minimum_required(VERSION 3.0.0)
project(yolov5_ort)

option(ONNXRUNTIME_DIR "Path to built ONNX Runtime directory." STRING)
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

find_package(OpenCV REQUIRED)

include_directories("include/")

add_executable(yolov5_ort
               main.cpp
               src/detector.cpp
               src/utils.cpp)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_include_directories(yolov5_ort PRIVATE "${ONNXRUNTIME_DIR}/include")
# link_directories("${ONNXRUNTIME_DIR}/lib")
target_compile_features(yolov5_ort PRIVATE cxx_std_14)
target_link_libraries(yolov5_ort ${OpenCV_LIBS})

if (WIN32)
    target_link_libraries(yolov5_ort "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
endif(WIN32)

if (UNIX)
    target_link_libraries(yolov5_ort "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif(UNIX)

