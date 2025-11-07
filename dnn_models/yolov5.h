#pragma once

#include "dnn_model.h"

#include <ranges>

// YOLOv5 output shape is [1, N, 85] where:
// 1 is batch size
// N is number of predictions
// 85 is attributes for prediction
//
// Attributes for prediction shape is:
// [box_center_x, box_center_y, box_w, box_h, objectness, probability_scores...]

class YOLOv5 : public DNNModel {
   private:
    const int INPUT_W = 640;
    const int INPUT_H = 640;
    const cv::Scalar LB_COLOR = cv::Scalar(114, 114, 114);
    const int ATTRIBUTE_N = 85;

    const float OBJ_THRESH = 0.25f;
    const float SCORE_THRESH = 0.35f;  // objectness * class_probability
    const float NMS_THRESH = 0.45f;

    void validate(const cv::Mat& output) const {
        const auto dimensions = output.size[2];
        if (dimensions != ATTRIBUTE_N) {
            throw std::runtime_error{
                "Unexpected output dimensions " + std::to_string(dimensions) +
                " (expected " + std::to_string(ATTRIBUTE_N) + ")"};
        }
    }

    struct Candidate {
        cv::Point class_id;
        double score;
    };

    Candidate get_best_candidate(float* data, float objectness) const {
        cv::Mat probs(1, _class_names.size(), CV_32F, data);

        cv::Point max_class_id;
        double max_class_prob;
        cv::minMaxLoc(probs, nullptr, &max_class_prob, nullptr, &max_class_id);
        return {.class_id = max_class_id, .score = objectness * max_class_prob};
    }

    std::optional<cv::Rect> transform_from_letterbox(float lb_cx, float lb_cy,
                                                     float lb_w,
                                                     float lb_h) const {
        // YOLOv5 returns coordinates of box center and it's dimensions
        // here we calculate box coordinates in letterbox coordinate system
        float lb_x = lb_cx - lb_w / 2.0f;
        float lb_y = lb_cy - lb_h / 2.0f;

        // Remove padding and scale back
        float x0 = (lb_x - _letterbox.image_x) / _letterbox.aspect_ratio;
        float y0 = (lb_y - _letterbox.image_y) / _letterbox.aspect_ratio;
        float x1 = (lb_x + lb_w - _letterbox.image_x) / _letterbox.aspect_ratio;
        float y1 = (lb_y + lb_h - _letterbox.image_y) / _letterbox.aspect_ratio;

        // CLip
        int ix = std::max(0, (int)std::round(x0));
        int iy = std::max(0, (int)std::round(y0));
        int iw = std::min(_letterbox.src_w - ix, (int)std::round(x1 - x0));
        int ih = std::min(_letterbox.src_h - iy, (int)std::round(y1 - y0));

        return (iw <= 0 || ih <= 0)
                   ? std::nullopt
                   : std::make_optional(cv::Rect{ix, iy, iw, ih});
    }

    std::vector<Object> apply_nms_filter(
        const std::vector<int>& class_ids, const std::vector<cv::Rect>& boxes,
        const std::vector<float>& scores) const {
        std::vector<int> filtered;
        cv::dnn::NMSBoxes(boxes, scores, SCORE_THRESH, NMS_THRESH, filtered);

        const auto objects =
            filtered | std::ranges::views::transform([&, this](int index) {
                return Object{.class_name = _class_names[class_ids[index]],
                              .score = scores[index],
                              .box = boxes[index]};
            });
        return {std::begin(objects), std::end(objects)};
    }

   public:
    using DNNModel::DNNModel;

    virtual void input(const cv::Mat& color_bgr) override {
        _letterbox = this->get_letterbox(color_bgr, INPUT_W, INPUT_H, LB_COLOR);

        const auto blob = cv::dnn::blobFromImage(
            _letterbox.data, 1.0 / 255.0, cv::Size(INPUT_W, INPUT_H),
            cv::Scalar(), /*swapRB*/ true, /*crop*/ false);

        this->input_preprocessed(std::move(blob));
    }

    virtual std::vector<Object> parse() const override {
        if (!this->has_outputs()) {
            return {};
        }

        const auto& out = outputs();
        validate(out);

        const int rows = out.size[1];
        const int dims = out.size[2];

        std::vector<int> class_ids;
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;

        auto* data = reinterpret_cast<float*>(out.data);
        for (int i = 0; i < rows; ++i, data += dims) {
            auto objectness = data[4];
            if (objectness < OBJ_THRESH) {
                continue;
            }

            auto candidate = get_best_candidate(data + 5, objectness);
            if (candidate.score < SCORE_THRESH) {
                continue;
            }

            const auto box =
                transform_from_letterbox(data[0], data[1], data[2], data[3]);
            if (!box.has_value()) {
                continue;
            }

            class_ids.push_back(candidate.class_id.x);
            boxes.push_back(std::move(box.value()));
            scores.push_back(candidate.score);
        }

        return apply_nms_filter(class_ids, boxes, scores);
    }
};
