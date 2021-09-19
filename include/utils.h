#pragma once
#include <codecvt>
#include <fstream>
#include <opencv2/opencv.hpp>


struct Detection
{
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    Detection() = default;

    Detection(std::vector<cv::Rect> boxes,
              std::vector<float> confs,
              std::vector<int> classIds)
    {
        this->boxes = std::move(boxes);
        this->confs = std::move(confs);
        this->classIds = std::move(classIds);
    }

    size_t size()
    {
        return classIds.size();
    }
};

namespace utils
{
    std::wstring charToWstring(const char* str);
    std::vector<std::string> loadNames(const std::string& path);
    size_t vectorProduct(const std::vector<int64>& vector);
    void visualizeDetection(cv::Mat& image, Detection &detection, std::vector<std::string> classNames);
}
