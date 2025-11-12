#pragma once

#include <memory>
#include <string>

#include "backends/backend.h"
#include "detail/letterbox.h"
#include "types.h"

namespace vision {

class Detector {
    Letterbox _letterbox;
    DetectorConfig _cfg;
    std::unique_ptr<DetectorBackend> _backend;
    std::optional<cv::Mat> _outputs;

    [[nodiscard]] cv::Mat preprocess(const cv::Mat& bgr);
    std::vector<DetectedObject> apply_nms_filter(const Detections&,
                                                 const Thresholds&) const;
    std::string name_by_class_id(std::size_t id) const;

   public:
    Detector(DetectorConfig cfg);

    void input(const cv::Mat& bgr);
    void forward();
    std::vector<DetectedObject> parse(const Thresholds&) const;
};

}  // namespace vision
