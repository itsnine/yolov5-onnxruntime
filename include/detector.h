#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
#include <utility>

#include "utils.h"


class Yolov5Detector
{
public:
    Yolov5Detector(const std::wstring& modelPath,
                   const std::string& device,
                   const cv::Size& inputSize);
    Detection detect(cv::Mat& image, float confThreshold, float iouThreshold);

private:
    Ort::Env env{nullptr};
    Ort::SessionOptions sessionOptions{nullptr};
    Ort::Session session{nullptr};

    cv::Mat preprocessing(cv::Mat& image);
    Detection postprocessing(cv::Mat& image, std::vector<float> &outputTensorValues, float confThreshold, float iouThreshold);

    int numClasses;
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    std::vector<int64_t> outputDims;
    std::vector<int64_t> inputDims;
    cv::Size inputImageSize;

};