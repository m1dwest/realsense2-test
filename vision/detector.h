#pragma once

#include "detail/letterbox.h"
#include "parsers/parser.h"

namespace vision {

struct ModelRuntime {
    cv::dnn::Net net;
    std::vector<std::string> labels;
    std::unique_ptr<Parser> parser;

    int input_w;
    int input_h;
    cv::Scalar letterbox_color;
};

class Detector {
   public:
    Detector(ModelRuntime&& runtime) : _runtime(std::move(runtime)) {};

    void input(const cv::Mat& bgr);
    void forward();
    [[nodiscard]] std::vector<Detection> parse(const Thresholds&) const;

    bool is_nms_class_agnostic = true;

   private:
    [[nodiscard]] cv::Mat preprocess(const cv::Mat& bgr);
    std::vector<Detection> apply_nms_filter(const DetectionsRaw&,
                                            const Thresholds&) const;
    std::string label_by_id(std::size_t id) const;

    ModelRuntime _runtime;

    Letterbox _letterbox;
    std::optional<cv::Mat> _outputs;
};

}  // namespace vision
