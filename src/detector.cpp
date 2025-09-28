#include "detector.h"

YOLODetector::YOLODetector(const std::string& modelPath,
                           const bool& isGPU = true,
                           const bool& enPerf = false,
                           const cv::Size& inputSize = cv::Size(640, 640))
{
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    sessionOptions = Ort::SessionOptions();

    if (enPerf) {
        std::cout<<"Enable profiling\n";
        sessionOptions.EnableProfiling("profile_file_yolov5.json");
    }

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    auto rocmAvailable = std::find(availableProviders.begin(), availableProviders.end(), "ROCMExecutionProvider");
    auto oclAvailable = std::find(availableProviders.begin(), availableProviders.end(), "OpenCLExecutionProvider");
    OrtCUDAProviderOptions cudaOption;
    OrtROCMProviderOptions rocmOption;

    if (isGPU && (cudaAvailable != availableProviders.end()))
    {
        std::cout << "Inference device: CUDA GPU" << std::endl;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    else if (isGPU && (rocmAvailable != availableProviders.end()))
    {
        std::cout << "Inference device: ROCM GPU" << std::endl;
        sessionOptions.AppendExecutionProvider_ROCM(rocmOption);
    }
#ifdef ENABLE_OCL
    else if (isGPU &&(oclAvailable != availableProviders.end()))
    {
        std::cout << "Inference device: OPENCL GPU" << std::endl;
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_OpenCL(sessionOptions, 0, 0));
    }
#endif
    else
    {
        std::cout << "Inference device: CPU" << std::endl;
    }
#ifdef _WIN32
    std::wstring w_modelPath = utils::charToWstring(modelPath.c_str());
    session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
#else
    session = Ort::Session(env, modelPath.c_str(), sessionOptions);
#endif

    Ort::AllocatorWithDefaultOptions allocator;

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    this->isDynamicInputShape = false;
    // checking if width and height are dynamic
    if (inputTensorShape[2] == -1 && inputTensorShape[3] == -1)
    {
        std::cout << "Dynamic input shape" << std::endl;
        this->isDynamicInputShape = true;
    }

    for (auto shape : inputTensorShape)
        std::cout << "Input shape: " << shape << std::endl;

    // inputNames.push_back(session.GetInputName(0, allocator));
    // outputNames.push_back(session.GetOutputName(0, allocator));
    auto input_name = session.GetInputNameAllocated(0, allocator);
    inputNodeNameAllocatedStrings.push_back(std::move(input_name));
    inputNames.push_back(inputNodeNameAllocatedStrings.back().get());

    auto output_name = session.GetOutputNameAllocated(0, allocator);
    outputNodeNameAllocatedStrings.push_back(std::move(output_name));
    outputNames.push_back(outputNodeNameAllocatedStrings.back().get());

    std::cout << "Input name: " << inputNames[0] << std::endl;
    std::cout << "Output name: " << outputNames[0] << std::endl;

    this->inputImageShape = cv::Size2f(inputSize);
}

void YOLODetector::getBestClassInfo(std::vector<float>::iterator it, const int& numClasses,
                                    float& bestConf, int& bestClassId)
{
    // first 5 element are box and obj confidence
    bestClassId = 5;
    bestConf = 0;

    for (int i = 5; i < numClasses + 5; i++)
    {
        if (it[i] > bestConf)
        {
            bestConf = it[i];
            bestClassId = i - 5;
        }
    }

}

void YOLODetector::preprocessing(cv::Mat &image, float*& blob, std::vector<int64_t>& inputTensorShape)
{
    cv::Mat resizedImage, floatImage;
    cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);
    utils::letterbox(resizedImage, resizedImage, this->inputImageShape,
                     cv::Scalar(114, 114, 114), this->isDynamicInputShape,
                     false, true, 32);

    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
    blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
    cv::Size floatImageSize {floatImage.cols, floatImage.rows};

    // hwc -> chw
    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < floatImage.channels(); ++i)
    {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);
}

static float calculateIoU(const cv::Rect& rect1, const cv::Rect& rect2) {
    // 计算交集坐标
    int x1 = std::max(rect1.x, rect2.x);
    int y1 = std::max(rect1.y, rect2.y);
    int x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    int y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);

    // 计算交集面积
    int interArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);

    // 计算并集面积
    float unionArea = rect1.area() + rect2.area() - interArea;

    // 避免除以零
    return (unionArea > 0) ? (interArea / unionArea) : 0.0f;
}

// 自定义NMS实现
void NMSBoxes(
    const std::vector<cv::Rect>& bboxes,
    const std::vector<float>& scores,
    float score_threshold,
    float nms_threshold,
    std::vector<int>& indices
) {
    indices.clear();

    // 第一步：按置信度过滤边界框
    std::vector<int> idxs;
    for (int i = 0; i < scores.size(); ++i) {
        if (scores[i] >= score_threshold) {
            idxs.push_back(i);
        }
    }

    // 第二步：按置信度降序排序
    std::sort(idxs.begin(), idxs.end(), [&](int a, int b) {
        return scores[a] > scores[b];
    });

    // 第三步：NMS处理
    std::vector<bool> suppressed(idxs.size(), false);
    for (int i = 0; i < idxs.size(); ++i) {
        if (suppressed[i]) continue;

        indices.push_back(idxs[i]);  // 保留当前框

        for (int j = i + 1; j < idxs.size(); ++j) {
            if (suppressed[j]) continue;

            // 计算IoU并抑制重叠框
            float iou = calculateIoU(bboxes[idxs[i]], bboxes[idxs[j]]);
            if (iou >= nms_threshold) {
                suppressed[j] = true;
            }
        }
    }
}

std::vector<Detection> YOLODetector::postprocessing(const cv::Size& resizedImageShape,
                                                    const cv::Size& originalImageShape,
                                                    std::vector<Ort::Value>& outputTensors,
                                                    const float& confThreshold, const float& iouThreshold)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    auto* rawOutput = outputTensors[0].GetTensorData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output(rawOutput, rawOutput + count);

    // for (const int64_t& shape : outputShape)
    //     std::cout << "Output Shape: " << shape << std::endl;

    // first 5 elements are box[4] and obj confidence
    int numClasses = (int)outputShape[2] - 5;
    int elementsInBatch = (int)(outputShape[1] * outputShape[2]);

    // only for batch size = 1
    for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += outputShape[2])
    {
        float clsConf = it[4];

        if (clsConf > confThreshold)
        {
            int centerX = (int) (it[0]);
            int centerY = (int) (it[1]);
            int width = (int) (it[2]);
            int height = (int) (it[3]);
            int left = centerX - width / 2;
            int top = centerY - height / 2;

            float objConf;
            int classId;
            this->getBestClassInfo(it, numClasses, objConf, classId);

            float confidence = clsConf * objConf;

            boxes.emplace_back(left, top, width, height);
            confs.emplace_back(confidence);
            classIds.emplace_back(classId);
        }
    }

    std::vector<int> indices;
    NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);
    // cv::dnn::NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);
    // std::cout << "amount of NMS indices: " << indices.size() << std::endl;

    std::vector<Detection> detections;

    for (int idx : indices)
    {
        Detection det;
        det.box = cv::Rect(boxes[idx]);
        utils::scaleCoords(resizedImageShape, det.box, originalImageShape);

        det.conf = confs[idx];
        det.classId = classIds[idx];
        detections.emplace_back(det);
    }

    return detections;
}

std::vector<Detection> YOLODetector::detect(cv::Mat &image, const float& confThreshold = 0.4,
                                            const float& iouThreshold = 0.45)
{
    float *blob = nullptr;
    std::vector<int64_t> inputTensorShape {1, 3, -1, -1};
    this->preprocessing(image, blob, inputTensorShape);

    size_t inputTensorSize = utils::vectorProduct(inputTensorShape);

    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    std::vector<Ort::Value> inputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorSize,
            inputTensorShape.data(), inputTensorShape.size()
    ));

    std::vector<Ort::Value> outputTensors = this->session.Run(Ort::RunOptions{nullptr},
                                                              inputNames.data(),
                                                              inputTensors.data(),
                                                              1,
                                                              outputNames.data(),
                                                              1);

    cv::Size resizedShape = cv::Size((int)inputTensorShape[3], (int)inputTensorShape[2]);
    std::vector<Detection> result = this->postprocessing(resizedShape,
                                                         image.size(),
                                                         outputTensors,
                                                         confThreshold, iouThreshold);

    delete[] blob;

    return result;
}
