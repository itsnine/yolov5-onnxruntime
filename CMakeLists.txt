cmake_minimum_required(VERSION 3.0.0)
project(onnx_detection VERSION 0.1.0)

set(OpenCV_DIR "C:/opencv/build_dynamic/x64/vc15/lib")
set(ONNXRUNTIME_DIR "C:/onnxruntime-win-x64-1.8.1")

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(onnx_detection main.cpp)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_include_directories(onnx_detection PRIVATE "${ONNXRUNTIME_DIR}/include")
# link_directories("${ONNXRUNTIME_ROOTDIR}/lib")
target_link_libraries(onnx_detection ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT} "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")


