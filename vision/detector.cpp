#include "detector.h"

#include <fstream>
#include <ranges>

#include "backends/yolov5.h"
#include "detail/letterbox.h"

namespace vision {

Detector::Detector(DetectorConfig cfg) : _cfg(std::move(cfg)) {
    switch (_cfg.model_kind) {
        case ModelKind::YOLOv5:
            _backend = std::make_unique<YOLOv5Backend>(_cfg.model_path,
                                                       _cfg.names_path);
            return;
        default:
            throw std::runtime_error("Unsupported ModelKind");
    }
}

cv::Mat Detector::preprocess(const cv::Mat& bgr) {
    _letterbox =
        img_to_letterbox(bgr, _cfg.input_w, _cfg.input_h, _cfg.letterbox_color);

    return cv::dnn::blobFromImage(
        _letterbox.data, 1.0 / 255.0, cv::Size(_cfg.input_w, _cfg.input_h),
        cv::Scalar(), /*swapRB*/ true, /*crop*/ false);
}

// TODO non object agnostic
std::vector<DetectedObject> Detector::apply_nms_filter(
    const Detections& detections, const Thresholds& thresholds) const {
    std::vector<int> filtered;
    cv::dnn::NMSBoxes(detections.boxes, detections.scores, thresholds.score,
                      thresholds.nms, filtered);

    const auto objects =
        filtered | std::ranges::views::transform([&, this](int index) {
            return DetectedObject{
                .class_name = name_by_class_id(detections.ids[index]),
                .score = detections.scores[index],
                .box = box_from_letterbox(detections.boxes[index], _letterbox)
                           .value()};
        });
    // TODO filter letterbox optional
    return {std::begin(objects), std::end(objects)};
}

std::string Detector::name_by_class_id(std::size_t id) const {
    const auto& names = _backend->class_names();
    if (id >= names.size()) {
        std::cerr << "Wrond id " << id << " for class\n";
        return {};
    } else {
        return names[id];
    }
}

void Detector::input(const cv::Mat& bgr) {
    auto blob = this->preprocess(bgr);
    _backend->input(std::move(blob));
}

void Detector::forward() {
    _outputs = _backend->forward();
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
}

std::vector<DetectedObject> Detector::parse(
    const Thresholds& thresholds) const {
    if (!_outputs.has_value()) {
        std::cerr << "No outputs from model was found to parse\n";
        return {};
    }

    const auto& data = _outputs.value();
    _backend->validate(data);

    const auto detections = _backend->parse(data, thresholds);
    return apply_nms_filter(detections, thresholds);
}

}  // namespace vision
