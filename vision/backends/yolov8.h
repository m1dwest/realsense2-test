#pragma once

#include <opencv2/opencv.hpp>

#include "../detector.h"

namespace vision {

class YOLOv8Backend : public DetectorBackend {
   public:
    YOLOv8Backend(const std::string& model_path,
                  const std::string& names_path) {
        _net = DetectorBackend::load_model(model_path);
        _class_names = DetectorBackend::load_names(names_path);
    }

    void validate(const cv::Mat& output) const override {
        std::cout << output.size[1] << " " << output.size[2] << "\n";
        // const auto dimensions = output.size[2];
        // if (dimensions != ATTRIBUTE_N) {
        //     throw std::runtime_error{
        //         "Unexpected output dimensions for YOLOv5: " +
        //         std::to_string(dimensions) + " (expected " +
        //         std::to_string(ATTRIBUTE_N) + ")"};
        // }
    }

    Detections parse(const cv::Mat& output,
                     Thresholds thresholds) const override {
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

        Detections detections;

        auto chan = [&](int c) -> const float* {
            return output.ptr<float>(0, c);
        };
        // float mn = 1e9f, mx = -1e9f;
        // const float* cls0 = chan(4);
        // for (int i = 0; i < 100 && i < N; ++i) {
        //     mn = std::min(mn, cls0[i]);
        //     mx = std::max(mx, cls0[i]);
        // }
        // std::cout << "class0 logits sample range: " << mn << " .. " << mx
        //           << "\n";

        const float* pcx = chan(0);
        const float* pcy = chan(1);
        const float* pw = chan(2);
        const float* ph = chan(3);

        for (int i = 0; i < N; ++i) {
            float cx = pcx[i];
            float cy = pcy[i];
            float w = pw[i];
            float h = ph[i];
            // if (objectness < OBJ_THRESH) {
            //     continue;
            // }

            int bestClass = -1;
            float bestScore = 0.f;
            for (int c = 0; c < Nc; ++c) {
                // std::cout << chan(4 + c)[i] << " -\n";
                // float s = sigmoidf(chan(4 + c)[i]);
                float s = chan(4 + c)[i];
                if (s > bestScore) {
                    bestScore = s;
                    bestClass = c;
                }
            }
            if (bestScore < 0.1) continue;
            // std::cout << bestScore << "\n";
            // auto candidate = get_best_candidate(data + 5, objectness);
            // if (candidate.score < SCORE_THRESH) {
            //     continue;
            // }

            // const auto box = box_from_letterbox(cx, cy, w, h, _letterbox);
            // if (!box.has_value()) {
            //     continue;
            // }

            detections.ids.push_back(bestClass);
            detections.boxes.emplace_back(cx, cy, w, h);
            detections.scores.push_back(bestScore);
        }

        return detections;

        // const auto filtered = apply_nms_filter(detections, thresholds);
        // std::cout << "Detections: " << filtered.size() << "\n";
        // return filtered;

        // const auto detections =
        //     parse_yolov11_openCVDNN(outputs(), 640, 480, 640, 640);
        // std::vector<Object> result;
        // for (const auto& d : detections) {
        //     result.push_back({.class_name = _class_names[d.classId],
        //                       .score = d.score,
        //                       .box = d.box});
        // }
        // return result;
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
