#pragma once

#include "parser.h"

namespace vision {

class YOLOv5Parser : public Parser {
   public:
    DetectionsRaw parse(const cv::Mat& output,
                        const Thresholds& thresholds) const override {
        const int rows = output.size[1];  // predictions
        const int dims = output.size[2];  // 85

        DetectionsRaw result;

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

            result.class_ids.push_back(candidate.class_id.x);
            result.scores.push_back(static_cast<float>(candidate.score));
            result.boxes.push_back(std::move(box));
        }
        return result;
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

   private:
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

    const int ATTRIBUTE_N = 85;
};

}  // namespace vision
