#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "../types.h"

namespace vision {

class DetectorBackend {
   public:
    virtual ~DetectorBackend() = default;

    virtual void validate(const cv::Mat& output) const = 0;
    virtual Detections parse(const cv::Mat& bgr,
                             Thresholds thresholds) const = 0;
    virtual void input(const cv::Mat&) = 0;
    virtual cv::Mat forward() = 0;

    virtual const std::vector<std::string>& class_names() const = 0;

   protected:
    static cv::dnn::Net load_model(const std::string& path);
    static std::vector<std::string> load_names(const std::string& path);
};

}  // namespace vision
