#include <chrono>
#include <string>

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

#include "dnn_models/yolov5.h"
#include "render.h"

int main() {
    const std::string onnx_path = "yolov5s.onnx";
    const std::string names_path = "coco.names";

    auto model = std::make_shared<YOLOv5>(onnx_path, names_path);

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

        model->input(color_bgr);
        model->forward();
        const auto objects = model->parse();
        render(profile, color_bgr, depth_z16, depth_rgb);

        print_fps();
    }

    pipe.stop();
    return 0;
}
