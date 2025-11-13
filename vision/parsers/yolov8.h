#pragma once

#include "parser.h"

namespace vision {

class YOLOv8Parser : public Parser {
   public:
    DetectionsRaw parse(const cv::Mat& output,
                        const Thresholds& thresholds) const override {
        // validate(out);
        const float* p = output.ptr<float>(0, 4);  // first class channel
        float mn = +1e9f, mx = -1e9f;
        for (int i = 0; i < 100; ++i) {
            mn = std::min(mn, p[i]);
            mx = std::max(mx, p[i]);
        }
        std::cout << "class0 sample range: " << mn << " .. " << mx << "\n";

        const int C = output.size[1];  // 84
        const int N = output.size[2];  // 8400
        std::cout << "C: " << C << " \nN: " << N << "\n";
        const int Nc = C - 4;
        const auto* data = reinterpret_cast<const float*>(output.data);

        DetectionsRaw result;

        auto chan = [&](int c) -> const float* {
            return output.ptr<float>(0, c);
        };

        const float* pcx = chan(0);
        const float* pcy = chan(1);
        const float* pw = chan(2);
        const float* ph = chan(3);

        for (int i = 0; i < N; ++i) {
            float cx = pcx[i];
            float cy = pcy[i];
            float w = pw[i];
            float h = ph[i];

            int bestClass = -1;
            float bestScore = 0.f;
            for (int c = 0; c < Nc; ++c) {
                float s = chan(4 + c)[i];
                if (s > bestScore) {
                    bestScore = s;
                    bestClass = c;
                }
            }
            if (bestScore < 0.1) continue;

            result.class_ids.push_back(bestClass);
            result.boxes.emplace_back(cx, cy, w, h);
            result.scores.push_back(bestScore);
        }

        return result;
    }

    void validate(const cv::Mat& output) const override {
        // std::cout << output.size[1] << " " << output.size[2] << "\n";
        // const auto dimensions = output.size[2];
        // if (dimensions != ATTRIBUTE_N) {
        //     throw std::runtime_error{
        //         "Unexpected output dimensions for YOLOv5: " +
        //         std::to_string(dimensions) + " (expected " +
        //         std::to_string(ATTRIBUTE_N) + ")"};
        // }
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
