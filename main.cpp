#include <chrono>
#include <string>

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

#include "render.h"
#include "vision/detector.h"

const float OBJ_THRESH = 0.25f;
const float SCORE_THRESH = 0.35f;
const float NMS_THRESH = 0.45f;

int main() {
    auto detector_cfg =
        vision::DetectorConfig{.model_kind = vision::ModelKind::YOLOv5,
                               .model_path = "yolov5s.onnx",
                               .names_path = "coco.names",
                               .input_w = 640,
                               .input_h = 640,
                               .letterbox_color = cv::Scalar(114, 114, 114)};
    auto detector = vision::Detector(detector_cfg);
    const auto thresholds = vision::Thresholds{
        .score = SCORE_THRESH, .nms = NMS_THRESH, .objectness = OBJ_THRESH};

    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    auto profile = pipe.start(cfg);
    rs2::align align_to_color(RS2_STREAM_COLOR);
    rs2::colorizer colorize;

    auto print_fps = [tp_before =
                          std::chrono::steady_clock::now()]() mutable -> void {
        using namespace std::chrono;
        const auto tp_after = steady_clock::now();
        const auto duration = duration_cast<milliseconds>(tp_after - tp_before);
        tp_before = tp_after;

        std::cout << std::fixed << std::setprecision(2)
                  << "fps: " << 1000.f / duration.count() << "\n";
    };

    while (true) {
        auto frames = pipe.wait_for_frames();

        auto aligned_frames = align_to_color.process(frames);
        auto color = aligned_frames.get_color_frame();
        auto depth = aligned_frames.get_depth_frame();
        auto depth_colorized = colorize.process(depth);

        if (!color || !depth) {
            continue;
        }

        const auto color_bgr =
            cv::Mat(color.get_height(), color.get_width(), CV_8UC3,
                    (void*)color.get_data(), cv::Mat::AUTO_STEP);
        const auto depth_z16 =
            cv::Mat(depth.get_height(), depth.get_width(), CV_16U,
                    (void*)depth.get_data(), cv::Mat::AUTO_STEP);
        const auto depth_rgb =
            cv::Mat(color.get_height(), color.get_width(), CV_8UC3,
                    (void*)depth_colorized.get_data(), cv::Mat::AUTO_STEP);

        detector.input(color_bgr);
        detector.forward();
        const auto detections = detector.parse(thresholds);
        render(profile, detections, color_bgr, depth_z16, depth_rgb);

        print_fps();
    }

    pipe.stop();
    return 0;
}
