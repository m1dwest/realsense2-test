#pragma once

#include "parser.h"

namespace vision {

class YOLOv8Parser : public Parser {
   public:
    using Parser::Parser;

    DetectionsRaw parse(const cv::Mat& output,
                        const Thresholds& thresholds) const override {
        validate(output);

        const float* p = output.ptr<float>(0, 4);  // first class channel
        float mn = +1e9f, mx = -1e9f;
        for (int i = 0; i < 100; ++i) {
            mn = std::min(mn, p[i]);
            mx = std::max(mx, p[i]);
        }

        const int C = output.size[1];  // 84 // 7 for RPS
        const int N = output.size[2];  // 8400
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
        const auto actual_features = output.size[1];
        const auto actual_locations = output.size[2];
        const auto expected_features = class_num + 4;  // 4 is box points
        const auto expected_locations = (input_w / 8) * (input_h / 8) +
                                        (input_w / 16) * (input_h / 16) +
                                        (input_w / 32) * (input_h / 32);

        if (actual_features != expected_features) {
            throw std::runtime_error{
                "Unexpected features quantity for YOLOv8: " +
                std::to_string(actual_features) + " (expected " +
                std::to_string(expected_features) + ")"};
        }

        if (actual_locations != expected_locations) {
            throw std::runtime_error{
                "Unexpected locations quantity for YOLOv8: " +
                std::to_string(actual_locations) + " (expected " +
                std::to_string(expected_locations) + ")"};
        }
    }

   private:
    // downsampled grid, strides 8, 16, 23
    // (640 / 8) * (640 / 8) + (640 / 16 * 640 / 16) + (640 / 32 * 640 / 32)
    const int LOCATIONS_N = 8400;
};

}  // namespace vision
