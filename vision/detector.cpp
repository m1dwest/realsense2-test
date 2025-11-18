#include "detector.h"

#include <iostream>
#include <ranges>

#include <plog/Log.h>

namespace vision {

void Detector::input(const cv::Mat& bgr) {
    auto blob = this->preprocess(bgr);
    _runtime.net.setInput(std::move(blob));
}

void Detector::forward() {
    // const auto names = _net.getUnconnectedOutLayersNames();
    // std::vector<cv::Mat> outs;
    // _net.forward(outs, names);
    //
    // // TODO
    // // for (size_t k = 0; k < outs.size(); ++k) {
    // //     const auto& m = outs[k];
    // //     std::cout << k << ") name=" << names[k] << " dims=" << m.dims
    // //               << " sizes=[" << (m.dims > 0 ? m.size[0] : 0) << ","
    // //               << (m.dims > 1 ? m.size[1] : 0) << ","
    // //               << (m.dims > 2 ? m.size[2] : 0) << "] "
    // //               << " depth=" << m.depth() << " (expect " << CV_32F
    // //               << ")\n";
    // // }
    // _outputs = outs.at(0);
    _outputs = _runtime.net.forward();
}

[[nodiscard]] std::vector<Detection> Detector::parse(
    const Thresholds& thresholds) const {
    if (!_outputs.has_value()) {
        LOG_ERROR << "No outputs from model was found to parse\n";
        return {};
    }

    const auto& data = _outputs.value();
    _runtime.parser->validate(data);

    const auto detections = _runtime.parser->parse(data, thresholds);
    return apply_nms_filter(detections, thresholds);
}

cv::Mat Detector::preprocess(const cv::Mat& bgr) {
    _letterbox = img_to_letterbox(bgr, _runtime.input_w, _runtime.input_h,
                                  _runtime.letterbox_color);

    return cv::dnn::blobFromImage(_letterbox.data, 1.0 / 255.0,
                                  cv::Size(_runtime.input_w, _runtime.input_h),
                                  cv::Scalar(), /*swapRB*/ true,
                                  /*crop*/ false);
}

std::vector<Detection> Detector::apply_nms_filter(
    const DetectionsRaw& detections, const Thresholds& thresholds) const {
    std::vector<int> filtered;
    if (is_nms_class_agnostic) {
        cv::dnn::NMSBoxes(detections.boxes, detections.scores, thresholds.score,
                          thresholds.nms, filtered);
    } else {
        std::vector<int> tmp;
        for (int c = 0; c < _runtime.labels.size(); ++c) {
            tmp.clear();
            std::vector<cv::Rect> boxes_by_class;
            boxes_by_class.reserve(detections.boxes.size());
            std::vector<float> scores_by_class;
            scores_by_class.reserve(detections.scores.size());
            std::vector<int> map_index;
            map_index.reserve(detections.boxes.size());

            for (auto i = 0; i < detections.class_ids.size(); ++i)
                if (detections.class_ids[i] == c) {
                    boxes_by_class.push_back(detections.boxes[i]);
                    scores_by_class.push_back(detections.scores[i]);
                    map_index.push_back(i);
                }
            cv::dnn::NMSBoxes(boxes_by_class, scores_by_class, thresholds.score,
                              thresholds.nms, tmp);
            for (int k : tmp) {
                filtered.push_back(map_index[k]);
            }
        }
    }

    const auto objects =
        filtered | std::ranges::views::transform([&, this](int index) {
            return Detection{
                .label = label_by_id(detections.class_ids[index]),
                .score = detections.scores[index],
                .box = box_from_letterbox(detections.boxes[index], _letterbox)
                           .value()};
        });
    // TODO filter letterbox optional
    return {std::begin(objects), std::end(objects)};
}

std::string Detector::label_by_id(std::size_t id) const {
    if (id >= _runtime.labels.size()) {
        std::cerr << "Wrong id " << id << " for class\n";
        return {};
    } else {
        return _runtime.labels[id];
    }
}

}  // namespace vision
