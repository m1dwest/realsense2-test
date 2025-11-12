#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace vision {

struct Detections {
    std::vector<int> ids;
    std::vector<float> scores;
    std::vector<cv::Rect> boxes;
};

struct DetectedObject {
    std::string class_name;
    float score;
    cv::Rect box;
};

struct Thresholds {
    float score;
    float nms;
    float objectness = 0;
};

enum class ModelKind { YOLOv5, YOLOv8 };

struct DetectorConfig {
    ModelKind model_kind;
    std::string model_path;
    std::string names_path;
    int input_w;
    int input_h;
    cv::Scalar letterbox_color;
};
}  // namespace vision
