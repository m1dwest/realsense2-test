#pragma once

#include <string>

#include <opencv2/opencv.hpp>

namespace vision {

struct Detection {
    std::string label;
    float score;
    cv::Rect box;
};

struct DetectionsRaw {
    std::vector<int> class_ids;
    std::vector<float> scores;
    std::vector<cv::Rect> boxes;
};

struct Thresholds {
    float score;
    float nms;
    float objectness = 0;
};

class Parser {
   public:
    virtual ~Parser() = default;
    Parser(std::size_t class_num, int input_w, int input_h)
        : class_num(class_num), input_w(input_w), input_h(input_h) {}

    virtual DetectionsRaw parse(const cv::Mat&, const Thresholds&) const = 0;
    virtual void validate(const cv::Mat&) const = 0;

   protected:
    std::size_t class_num = 0;
    int input_w = 0;
    int input_h = 0;
};

}  // namespace vision
