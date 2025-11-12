#pragma once

#include <opencv2/opencv.hpp>

#include "../detector.h"

// YOLOv5 output shape is [1, N, 85] where:
// 1 is batch size
// N is number of predictions
// 85 is attributes for prediction
//
// Attributes for prediction shape is:
// [box_center_x, box_center_y, box_w, box_h, objectness, probability_scores...]
namespace vision {

class YOLOv5Backend : public DetectorBackend {
   public:
    YOLOv5Backend(const std::string& model_path,
                  const std::string& names_path) {
        _net = DetectorBackend::load_model(model_path);
        _class_names = DetectorBackend::load_names(names_path);
    }

    void validate(const cv::Mat& output) const override {
        const auto dimensions = output.size[2];
        if (dimensions != ATTRIBUTE_N) {
            throw std::runtime_error{
                "Unexpected output dimensions for YOLOv5: " +
                std::to_string(dimensions) + " (expected " +
                std::to_string(ATTRIBUTE_N) + ")"};
        }
    }

    Detections parse(const cv::Mat& output,
                     Thresholds thresholds) const override {
        const int rows = output.size[1];  // predictions
        const int dims = output.size[2];  // 85

        Detections result;

        auto* data = reinterpret_cast<float*>(output.data);
        for (int i = 0; i < rows; ++i, data += dims) {
            auto objectness = data[4];
            if (objectness < thresholds.objectness) {
                continue;
            }

            auto candidate = get_best_candidate(data + 5, objectness);
            if (candidate.score < thresholds.score) {
                continue;
            }

            const auto box = cv::Rect(data[0], data[1], data[2], data[3]);

            result.ids.push_back(candidate.class_id.x);
            result.scores.push_back(static_cast<float>(candidate.score));
            result.boxes.push_back(std::move(box));
        }
        return result;
    }

    void input(const cv::Mat& blob) override { _net.setInput(std::move(blob)); }

    cv::Mat forward() override { return _net.forward(); }

    const std::vector<std::string>& class_names() const { return _class_names; }

   private:
    cv::dnn::Net _net;
    std::vector<std::string> _class_names;

    const int ATTRIBUTE_N = 85;

    struct Candidate {
        cv::Point class_id;
        double score;
    };

    Candidate get_best_candidate(float* data, float objectness) const {
        // TODO get class n
        const int CLASS_N = 80;
        cv::Mat probs(1, CLASS_N, CV_32F, data);

        cv::Point max_class_id;
        double max_class_prob;
        cv::minMaxLoc(probs, nullptr, &max_class_prob, nullptr, &max_class_id);
        return {.class_id = max_class_id, .score = objectness * max_class_prob};
    }
};

}  // namespace vision
